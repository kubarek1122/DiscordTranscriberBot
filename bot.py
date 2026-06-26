from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import AppConfig, load_config
from src.logging_util import session_log
from src.messages import RECORDING_CUT_SHORT, SKRYBA
from src.prompts import KIND_LABELS_PL
from src.summarize import get_summarizer, suggest_drift
from src.pipeline import notify_pipeline_failure, resummarize_and_post, run_pipeline
from src.recording import RecordingSession, ensure_ffmpeg, finalize_audio
from src.session import (
    SessionState,
    is_heartbeat_stale,
    latest_for_guild,
    most_recent_unfinished,
    scan_unfinished,
)
from src.transcribe import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("bot")


class _DropDiscordVoiceWarnings(logging.Filter):
    """Suppress discord.py's "PyNaCl/davey is not installed, voice will NOT
    be supported" warnings. We don't do voice here — the Node sidecar does."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "voice will NOT be supported" not in record.getMessage()


logging.getLogger("discord.client").addFilter(_DropDiscordVoiceWarnings())


class TranscriberBot(commands.Bot):
    def __init__(self, cfg: AppConfig) -> None:
        intents = discord.Intents.default()
        intents.members = True
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.cfg = cfg
        self.transcriber = Transcriber(
            model=cfg.whisper.model,
            device=cfg.whisper.device,
            compute_type=cfg.whisper.compute_type,
            language=cfg.whisper.language,
        )
        self.sessions: dict[int, RecordingSession] = {}  # guild_id -> session
        self._recovery_done = False
        self._recovery_lock = asyncio.Lock()

    async def setup_hook(self) -> None:
        # Register the slash command group on the tree.
        self.tree.add_command(SkrybaGroup(self))

        # If DEV_GUILD_ID is set, sync to that guild for instant updates
        # (otherwise global sync, which can take up to an hour to propagate).
        dev_guild = os.environ.get("DEV_GUILD_ID")
        dev_guild_id: int | None = None
        if dev_guild:
            try:
                dev_guild_id = int(dev_guild)
            except ValueError:
                log.warning(
                    "DEV_GUILD_ID=%r is not a valid integer; falling back to global sync",
                    dev_guild,
                )
        if dev_guild_id is not None:
            guild_obj = discord.Object(id=dev_guild_id)
            self.tree.copy_global_to(guild=guild_obj)
            synced = await self.tree.sync(guild=guild_obj)
            log.info("synced %d slash command(s) to guild %s", len(synced), dev_guild_id)
        else:
            synced = await self.tree.sync()
            log.info("synced %d slash command(s) globally", len(synced))

    async def on_ready(self) -> None:
        log.info("logged in as %s (id=%s)", self.user, self.user and self.user.id)
        async with self._recovery_lock:
            if not self._recovery_done:
                self._recovery_done = True
                self.loop.create_task(self._run_recovery())

    async def _run_recovery(self) -> None:
        try:
            await self._recover_unfinished()
        except Exception:
            log.exception("recovery scan failed")

    async def _auto_stop_session(self, guild_id: int, reason: str) -> None:
        """Triggered by the sidecar when a recording can't continue (kicked
        from VC, voice gateway gave up). Stops the session, posts a Polish
        notice to the original text channel, and runs the pipeline on the
        captured PCM. Idempotent — pops the session under a lock-free check
        so a concurrent `/skryba stop` can't double-run."""
        session = self.sessions.pop(guild_id, None)
        if session is None:
            return
        log.warning(
            "auto-stop: guild=%s reason=%s — recording cut short by sidecar",
            guild_id, reason,
        )
        session_dir = session.session_dir
        text_channel_id = session.text_channel_id
        try:
            await session.stop()
        except Exception:
            log.exception("auto-stop: session.stop() raised")

        # Tell the user — best-effort, don't let a failed notice block the pipeline.
        try:
            channel = self.get_channel(text_channel_id)
            if channel is None:
                channel = await self.fetch_channel(text_channel_id)
            await channel.send(RECORDING_CUT_SHORT.format(reason=reason))
        except Exception:
            log.exception("auto-stop: could not post cut-short notice")

        try:
            await run_pipeline(
                session_dir=session_dir,
                cfg=self.cfg,
                transcriber=self.transcriber,
                bot=self,
            )
        except Exception as e:
            log.exception("auto-stop: pipeline failed for %s", session_dir)
            await notify_pipeline_failure(
                bot=self,
                text_channel_id=text_channel_id,
                session_dir=session_dir,
                error=e,
            )
        finally:
            session.release_log()

    async def close(self) -> None:  # type: ignore[override]
        """Graceful shutdown. Stops every active session so the recorder
        sidecar flushes PCM to disk and the on-disk state advances to
        `recorded`. The next startup's recovery scan will resume the
        pipeline and post the summary.

        Without this hook, SIGTERM tears down the IPC socket abruptly —
        the recorder eventually notices via socket close and finalizes,
        but Python's `state.stage` is left at `recording` and only
        rescued by the heartbeat-stale check on next startup. This hook
        gets us to `recorded` cleanly."""
        if self.sessions:
            log.info(
                "shutdown: stopping %d active session(s)", len(self.sessions)
            )
            for guild_id, session in list(self.sessions.items()):
                try:
                    await session.stop()
                except Exception:
                    log.exception(
                        "shutdown: error stopping session for guild %s", guild_id
                    )
                finally:
                    session.release_log()
                self.sessions.pop(guild_id, None)
        await super().close()

    async def _recover_unfinished(self) -> None:
        unfinished = scan_unfinished(self.cfg.recording.output_dir)
        if not unfinished:
            log.info("recovery: no unfinished sessions")
            return
        log.info("recovery: found %d unfinished session(s)", len(unfinished))
        for session_dir in unfinished:
            try:
                state = SessionState.load(session_dir)
            except Exception:
                log.exception("could not load %s", session_dir)
                continue

            if state.stage == "recording":
                if is_heartbeat_stale(state, self.cfg.recording.heartbeat_stale_after_s):
                    log.warning(
                        "recovering crashed recording %s (heartbeat stale)",
                        session_dir,
                    )
                    finalize_audio(session_dir)
                    state.advance(session_dir, "recorded")
                else:
                    log.info(
                        "skipping %s: stage=recording with fresh heartbeat",
                        session_dir,
                    )
                    continue

            timeout_s = self.cfg.reliability.recovery_per_session_timeout_s
            try:
                with session_log(session_dir):
                    await asyncio.wait_for(
                        run_pipeline(
                            session_dir=session_dir,
                            cfg=self.cfg,
                            transcriber=self.transcriber,
                            bot=self,
                        ),
                        timeout=timeout_s,
                    )
                log.info("recovery: completed %s", session_dir)
            except asyncio.TimeoutError as e:
                # Treat the timeout itself as a failed attempt so the retry
                # budget eventually marks chronically-stuck sessions failed.
                log.error(
                    "recovery: pipeline timed out after %ds for %s",
                    timeout_s, session_dir,
                )
                try:
                    cur = SessionState.load(session_dir)
                    cur.record_failure(session_dir, f"recovery timeout after {timeout_s}s")
                    if cur.retries >= self.cfg.reliability.max_pipeline_retries:
                        cur.advance(session_dir, "failed")
                except Exception:
                    log.exception(
                        "recovery: could not record timeout failure on %s", session_dir
                    )
                await notify_pipeline_failure(
                    bot=self,
                    text_channel_id=state.text_channel_id,
                    session_dir=session_dir,
                    error=e,
                )
            except Exception as e:
                log.exception("recovery: pipeline failed for %s", session_dir)
                await notify_pipeline_failure(
                    bot=self,
                    text_channel_id=state.text_channel_id,
                    session_dir=session_dir,
                    error=e,
                )

def _user_voice_channel(
    interaction: discord.Interaction,
) -> Optional[discord.VoiceChannel]:
    user = interaction.user
    if not isinstance(user, discord.Member):
        return None
    if user.voice is None or user.voice.channel is None:
        return None
    ch = user.voice.channel
    return ch if isinstance(ch, discord.VoiceChannel) else None


# Slash-command choices for the discussion type. Omitting the option leaves
# the kind unset so the pipeline auto-detects it from the transcript.
_KIND_CHOICES = [
    app_commands.Choice(name=KIND_LABELS_PL[k], value=k)
    for k in ("organizational", "design", "brainstorm", "general")
]


class SkrybaGroup(app_commands.Group):
    def __init__(self, bot: TranscriberBot) -> None:
        super().__init__(name="skryba", description="Sterowanie nagrywaniem rozmów")
        self.bot = bot

    @app_commands.command(
        name="start",
        description="Dołącza do twojego kanału głosowego i zaczyna nagrywać.",
    )
    @app_commands.describe(
        typ="Typ rozmowy (pomiń, aby wykryć automatycznie po nagraniu)."
    )
    @app_commands.choices(typ=_KIND_CHOICES)
    async def start(
        self,
        interaction: discord.Interaction,
        typ: Optional[app_commands.Choice[str]] = None,
    ) -> None:
        # Validate BEFORE deferring so error responses are cleanly ephemeral
        # (no public "Bot is thinking…" left hanging over a user mistake).
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return
        if guild.id in self.bot.sessions:
            await interaction.response.send_message(
                "Już nagrywam na tym serwerze. Użyj `/skryba stop`, aby zakończyć.",
                ephemeral=True,
            )
            return
        vc_channel = _user_voice_channel(interaction)
        if vc_channel is None:
            await interaction.response.send_message(
                "Musisz najpierw wejść na kanał głosowy.", ephemeral=True
            )
            return
        if interaction.channel_id is None:
            await interaction.response.send_message(
                "Nie można ustalić kanału tekstowego.", ephemeral=True
            )
            return

        # Past validation — commit to a public response.
        await interaction.response.defer(ephemeral=False)

        # The sidecar uses this callback to surface kicks / voice-gateway
        # giveups; the bot then auto-stops the session and runs the pipeline
        # on whatever PCM was captured before the cut.
        async def _on_hard_failure(reason: str) -> None:
            await self.bot._auto_stop_session(guild.id, reason)

        session = RecordingSession.create(
            voice_channel=vc_channel,
            text_channel_id=interaction.channel_id,
            cfg=self.bot.cfg,
            discussion_kind=typ.value if typ else None,
            on_hard_failure=_on_hard_failure,
        )
        try:
            await session.start()
        except Exception as e:
            log.exception("recorder start failed")
            # Public so others in the channel know recording did NOT start
            # (they were probably expecting it to after the slash command).
            await interaction.followup.send(
                f"⚠️ Nie udało się uruchomić nagrywania: `{e}`. "
                f"Sprawdź czy bot-rejestrator jest online.",
                ephemeral=False,
            )
            return

        self.bot.sessions[guild.id] = session
        typ_label = KIND_LABELS_PL[typ.value] if typ else "automatyczny"
        # Public so the whole channel sees recording has started.
        await interaction.followup.send(
            f"🔴 Nagrywam na kanale **{vc_channel.name}** (typ: {typ_label}). "
            f"Zatrzymaj komendą `/skryba stop`.",
            ephemeral=False,
        )

    @app_commands.command(
        name="stop",
        description="Kończy nagrywanie i publikuje podsumowanie.",
    )
    async def stop(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if guild is None or guild.id not in self.bot.sessions:
            await interaction.response.send_message(
                "Nie ma aktywnego nagrania na tym serwerze.", ephemeral=True
            )
            return

        # Public defer — the progress message and subsequent edits should
        # be visible to everyone in the channel.
        await interaction.response.defer(ephemeral=False)

        session = self.bot.sessions.pop(guild.id)
        session_dir = session.session_dir
        text_channel_id = session.text_channel_id
        # Captured before the pipeline overwrites it: non-None only when the
        # user explicitly chose a type at /skryba start. Drives the drift check.
        manual_kind = session.state.discussion_kind
        await session.stop()

        # Public progress message so everyone in the channel can see the
        # bot is working through transcription / summarization / posting.
        progress_msg = await interaction.followup.send(
            "⏹️ Zakończono nagrywanie. Przygotowuję transkrypcję…",
            ephemeral=False,
            wait=True,
        )

        # Polish status strings shown in the ephemeral while the pipeline runs.
        # Keys correspond to the *target* stage the pipeline is entering.
        _STAGE_LABELS = {
            "transcribed": "Transkrybuję mowę przez Whisper…",
            "summarized": "Podsumowuję rozmowę…",
            "posted": "Publikuję podsumowanie w kanale…",
        }

        async def _on_stage(stage: str) -> None:
            label = _STAGE_LABELS.get(stage)
            if label is None or progress_msg is None:
                return
            try:
                await progress_msg.edit(content=label)
            except Exception:
                # Editing the ephemeral is best-effort UX — pipeline must
                # not fail just because Discord rate-limited or the
                # interaction expired.
                pass

        try:
            await run_pipeline(
                session_dir=session_dir,
                cfg=self.bot.cfg,
                transcriber=self.bot.transcriber,
                bot=self.bot,
                on_stage=_on_stage,
            )
            if progress_msg is not None:
                try:
                    await progress_msg.edit(content="Gotowe! Podsumowanie opublikowane.")
                except Exception:
                    pass
            # If the user picked a type but the conversation drifted, suggest a
            # better fit (ephemeral, best-effort — never disturb the summary).
            if manual_kind is not None:
                await self._maybe_suggest_drift(
                    interaction, session_dir, manual_kind
                )
        except Exception as e:
            log.exception("pipeline failed for session %s", session_dir)
            # Tell both the invoker (ephemeral, may not be seen) and the
            # original channel (persistent, others can act on it via
            # /skryba kontynuuj).
            try:
                await interaction.followup.send(
                    f"Pipeline zakończył się błędem: `{e}`. "
                    f"Spróbuj `/skryba kontynuuj` lub poczekaj na restart.",
                    ephemeral=True,
                )
            except Exception:
                pass
            await notify_pipeline_failure(
                bot=self.bot,
                text_channel_id=text_channel_id,
                session_dir=session_dir,
                error=e,
            )
        finally:
            session.release_log()

    async def _maybe_suggest_drift(
        self,
        interaction: discord.Interaction,
        session_dir,
        manual_kind: str,
    ) -> None:
        """Best-effort: if the conversation drifted from the manually-chosen
        type, send the invoker an ephemeral nudge to re-run with the detected
        type. Never raises into the stop flow."""
        try:
            transcript_path = session_dir / "transcript.txt"
            transcript = (
                transcript_path.read_text(encoding="utf-8")
                if transcript_path.exists()
                else ""
            )
            if not transcript.strip():
                return
            summarizer = get_summarizer(self.bot.cfg)
            suggested = await suggest_drift(summarizer, transcript, manual_kind)
            if suggested is None:
                return
            chosen_label = KIND_LABELS_PL.get(manual_kind, manual_kind)
            suggested_label = KIND_LABELS_PL.get(suggested, suggested)
            await interaction.followup.send(
                f"💡 Oznaczono nagranie jako **{chosen_label}**, ale rozmowa "
                f"wygląda raczej na **{suggested_label}**. Aby przeliczyć "
                f"podsumowanie: `/skryba przelicz typ:{suggested_label}`.",
                ephemeral=True,
            )
        except Exception:
            log.exception("drift suggestion failed (ignored)")

    @app_commands.command(
        name="przelicz",
        description="Przelicza podsumowanie ostatniej sesji (opcjonalnie z innym typem).",
    )
    @app_commands.describe(
        typ="Typ rozmowy do użycia (pomiń, aby wykryć automatycznie)."
    )
    @app_commands.choices(typ=_KIND_CHOICES)
    async def przelicz(
        self,
        interaction: discord.Interaction,
        typ: Optional[app_commands.Choice[str]] = None,
    ) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return
        if guild.id in self.bot.sessions:
            await interaction.response.send_message(
                "Trwa nagrywanie — najpierw zakończ przez `/skryba stop`.",
                ephemeral=True,
            )
            return
        session_dir = latest_for_guild(self.bot.cfg.recording.output_dir, guild.id)
        if session_dir is None:
            await interaction.response.send_message(
                "Brak zapisanych sesji do przeliczenia.", ephemeral=True
            )
            return

        # Ephemeral — the recalculated summary itself is posted publicly by the
        # pipeline; the command's own chatter stays with the invoker.
        await interaction.response.defer(ephemeral=True)
        try:
            with session_log(session_dir):
                kind = await resummarize_and_post(
                    session_dir=session_dir,
                    cfg=self.bot.cfg,
                    bot=self.bot,
                    kind=typ.value if typ else None,
                )
        except Exception as e:
            log.exception("przelicz failed for %s", session_dir)
            await interaction.followup.send(
                f"Nie udało się przeliczyć sesji `{session_dir.name}`: `{e}`.",
                ephemeral=True,
            )
            return
        await interaction.followup.send(
            f"♻️ Przeliczono `{session_dir.name}` jako "
            f"**{KIND_LABELS_PL.get(kind, kind)}** i opublikowano nowe podsumowanie.",
            ephemeral=True,
        )

    @app_commands.command(
        name="status",
        description="Pokazuje stan nagrywania.",
    )
    async def status(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return
        session = self.bot.sessions.get(guild.id)
        if session is not None:
            started = datetime.fromisoformat(session.state.started_at)
            elapsed_s = int((datetime.now(timezone.utc) - started).total_seconds())
            m, s = divmod(elapsed_s, 60)
            h, m = divmod(m, 60)
            members = ", ".join(session.state.members.values()) or "(brak)"
            kind = session.state.discussion_kind
            typ_label = KIND_LABELS_PL.get(kind, "automatyczny") if kind else "automatyczny"
            await interaction.response.send_message(
                f"Nagrywam od {started.isoformat(timespec='seconds')} "
                f"(czas: {h:02d}:{m:02d}:{s:02d}). Typ: {typ_label}. Mówcy: {members}.",
                ephemeral=True,
            )
            return

        # Nothing live — fall back to disk so the user can see whether a
        # session is still being processed by the pipeline (in-memory state
        # is cleared by `stop()` before run_pipeline starts).
        latest = latest_for_guild(self.bot.cfg.recording.output_dir, guild.id)
        if latest is None:
            await interaction.response.send_message(
                "Brak aktywnego nagrania i żadnych zapisanych sesji.",
                ephemeral=True,
            )
            return
        try:
            state = SessionState.load(latest)
        except Exception:
            await interaction.response.send_message(
                f"Brak aktywnego nagrania. Ostatnia sesja: `{latest.name}` "
                f"(nie udało się odczytać stanu).",
                ephemeral=True,
            )
            return

        stage_pl = {
            "recording": "trwa nagrywanie (zombie? sprawdź heartbeat)",
            "recorded": "zarchiwizowane audio, oczekuje na transkrypcję",
            "transcribed": "transkrypcja gotowa, oczekuje na podsumowanie",
            "summarized": "podsumowanie gotowe, oczekuje na publikację",
            "posted": "opublikowane",
            "failed": "porzucone / nieodzyskiwalne",
        }.get(state.stage, state.stage)
        extra = ""
        if state.retries:
            extra = f" (nieudane próby: {state.retries})"
        if state.last_error and state.stage != "posted":
            extra += f"\nOstatni błąd: `{state.last_error[:200]}`"
        await interaction.response.send_message(
            f"Brak aktywnego nagrania. Ostatnia sesja `{latest.name}`: "
            f"**{stage_pl}**{extra}",
            ephemeral=True,
        )

    @app_commands.command(
        name="kontynuuj",
        description="Wznawia ostatnią niedokończoną sesję na tym serwerze.",
    )
    async def kontynuuj(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return
        if guild.id in self.bot.sessions:
            await interaction.response.send_message(
                "Trwa nagrywanie — najpierw zakończ przez `/skryba stop`.",
                ephemeral=True,
            )
            return

        session_dir = most_recent_unfinished(
            self.bot.cfg.recording.output_dir, guild.id
        )
        if session_dir is None:
            # If the user *just* hit the retry budget on a session, surface
            # that fact instead of a blank "no unfinished session" message.
            latest = latest_for_guild(
                self.bot.cfg.recording.output_dir, guild.id
            )
            if latest is not None:
                try:
                    latest_state = SessionState.load(latest)
                except Exception:
                    latest_state = None
                if latest_state is not None and latest_state.stage == "failed":
                    err = latest_state.last_error or "(brak szczegółów)"
                    await interaction.response.send_message(
                        f"Ostatnia sesja `{latest.name}` została oznaczona jako "
                        f"`failed` po wyczerpaniu prób (`{err[:120]}`). "
                        f"Użyj `/skryba porzuc`, żeby ją wyczyścić, "
                        f"albo edytuj `session.json` ręcznie.",
                        ephemeral=True,
                    )
                    return
            await interaction.response.send_message(
                "Nie znalazłem żadnej niedokończonej sesji na tym serwerze.",
                ephemeral=True,
            )
            return

        # Past validation — commit to a public response.
        await interaction.response.defer(ephemeral=False)

        state = SessionState.load(session_dir)
        # Tolerate sessions whose recording never reached `recorded` (e.g.
        # SIGKILL during capture): finalize the PCM we have on disk first.
        if state.stage == "recording":
            finalize_audio(session_dir)
            state.advance(session_dir, "recorded")
            state = SessionState.load(session_dir)

        # Public so others in the channel see the bot is picking the session
        # back up instead of silently chewing on it.
        await interaction.followup.send(
            f"▶️ Wznawiam sesję `{session_dir.name}` od etapu `{state.stage}`…",
            ephemeral=False,
        )

        try:
            with session_log(session_dir):
                await run_pipeline(
                    session_dir=session_dir,
                    cfg=self.bot.cfg,
                    transcriber=self.bot.transcriber,
                    bot=self.bot,
                )
        except Exception as e:
            log.exception("kontynuuj: pipeline failed for %s", session_dir)
            try:
                await interaction.followup.send(
                    f"Pipeline ponownie zakończył się błędem: `{e}`.",
                    ephemeral=True,
                )
            except Exception:
                pass
            await notify_pipeline_failure(
                bot=self.bot,
                text_channel_id=state.text_channel_id,
                session_dir=session_dir,
                error=e,
            )

    @app_commands.command(
        name="porzuc",
        description="Porzuca ostatnią niedokończoną sesję (oznacza ją jako failed).",
    )
    async def porzuc(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return
        if guild.id in self.bot.sessions:
            await interaction.response.send_message(
                "Trwa nagrywanie — najpierw zakończ przez `/skryba stop`.",
                ephemeral=True,
            )
            return
        session_dir = most_recent_unfinished(
            self.bot.cfg.recording.output_dir, guild.id
        )
        if session_dir is None:
            await interaction.response.send_message(
                "Nie ma niedokończonej sesji do porzucenia.", ephemeral=True
            )
            return
        try:
            state = SessionState.load(session_dir)
            state.last_error = "abandoned by user"
            state.advance(session_dir, "failed")
        except Exception as e:
            log.exception("porzuc: could not mark session failed")
            await interaction.response.send_message(
                f"Nie udało się porzucić sesji: `{e}`.", ephemeral=True
            )
            return
        # Public — abandoning a session is a notable state change.
        await interaction.response.send_message(
            f"🗑️ Sesja `{session_dir.name}` porzucona. "
            f"Pliki zostały zachowane na dysku.",
            ephemeral=False,
        )

    @app_commands.command(
        name="jaktojestbycskryba",
        description="Jak to jest być skrybą, dobrze?",
    )
    async def jaktojestbycskryba(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return
        await interaction.response.send_message(SKRYBA)


def make_bot(cfg: AppConfig) -> TranscriberBot:
    return TranscriberBot(cfg)


async def _run(cfg: AppConfig) -> None:
    bot = make_bot(cfg)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal(signame: str) -> None:
        log.info("received %s, initiating graceful shutdown", signame)
        stop_event.set()

    # SIGTERM = `docker compose stop`. SIGINT = Ctrl+C. Both should let
    # TranscriberBot.close() run so active sessions advance to `recorded`
    # before the process exits.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except NotImplementedError:
            # Windows asyncio doesn't support add_signal_handler; the
            # default discord.py KeyboardInterrupt path covers SIGINT
            # there, and SIGTERM isn't really a thing on Windows.
            pass

    async with bot:
        login_task = asyncio.create_task(
            bot.start(cfg.secrets.discord_token), name="bot.start"
        )
        stop_task = asyncio.create_task(stop_event.wait(), name="stop.wait")
        done, pending = await asyncio.wait(
            {login_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        # Surface any unexpected error from bot.start (e.g. bad token).
        for task in done:
            if task is login_task and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]


def main() -> None:
    cfg = load_config()
    ensure_ffmpeg()
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
