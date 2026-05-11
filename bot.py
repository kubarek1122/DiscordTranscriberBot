from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, voice_recv

from config import AppConfig, load_config
from src.messages import SKRYBA
from src.pipeline import run_pipeline
from src.recording import RecordingSession, ensure_ffmpeg, finalize_audio
from src.session import SessionState, is_heartbeat_stale, scan_unfinished
from src.transcribe import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("bot")


def _load_opus() -> None:
    """Force-load libopus so voice_recv can decode incoming Opus → PCM.

    discord.py auto-loads on most platforms but silently no-ops if the lib
    name doesn't match. On Ubuntu the .so is at libopus.so.0; try a few
    candidates and log the outcome."""
    if discord.opus.is_loaded():
        log.info("opus already loaded")
        return
    candidates = ("opus", "libopus.so.0", "libopus.0.dylib", "libopus-0.dll")
    for name in candidates:
        try:
            discord.opus.load_opus(name)
            if discord.opus.is_loaded():
                log.info("loaded opus from %s", name)
                return
        except Exception as e:  # noqa: PERF203
            log.debug("opus load %s failed: %s", name, e)
    log.error(
        "libopus could not be loaded — voice receive will produce empty PCM. "
        "Install libopus0 (Linux) / opus (mac) / libopus-0.dll (Windows)."
    )


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
        if dev_guild:
            guild_obj = discord.Object(id=int(dev_guild))
            self.tree.copy_global_to(guild=guild_obj)
            synced = await self.tree.sync(guild=guild_obj)
            log.info("synced %d slash command(s) to guild %s", len(synced), dev_guild)
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

            try:
                await run_pipeline(
                    session_dir=session_dir,
                    cfg=self.cfg,
                    transcriber=self.transcriber,
                    bot=self,
                )
                log.info("recovery: completed %s", session_dir)
            except Exception:
                log.exception("recovery: pipeline failed for %s", session_dir)

    async def on_voice_state_update(  # type: ignore[override]
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        session = self.sessions.get(member.guild.id)
        if session is None:
            return
        # If our bot got disconnected from voice, finalize the session.
        if (
            self.user is not None
            and member.id == self.user.id
            and before.channel is not None
            and after.channel is None
        ):
            log.warning("bot disconnected from voice, finalizing session")
            try:
                await session.stop()
            except Exception:
                log.exception("error stopping session after disconnect")
            self.sessions.pop(member.guild.id, None)
            try:
                await run_pipeline(
                    session_dir=session.session_dir,
                    cfg=self.cfg,
                    transcriber=self.transcriber,
                    bot=self,
                )
            except Exception:
                log.exception("post-disconnect pipeline failed")


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


class SkrybaGroup(app_commands.Group):
    def __init__(self, bot: TranscriberBot) -> None:
        super().__init__(name="skryba", description="Sterowanie nagrywaniem rozmów")
        self.bot = bot

    @app_commands.command(
        name="start",
        description="Dołącza do twojego kanału głosowego i zaczyna nagrywać.",
    )
    async def start(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if guild is None:
            await interaction.followup.send(
                "Komenda działa tylko na serwerze.", ephemeral=True
            )
            return

        if guild.id in self.bot.sessions:
            await interaction.followup.send(
                "Już nagrywam na tym serwerze. Użyj `/skryba stop`, aby zakończyć.",
                ephemeral=True,
            )
            return

        vc_channel = _user_voice_channel(interaction)
        if vc_channel is None:
            await interaction.followup.send(
                "Musisz najpierw wejść na kanał głosowy.", ephemeral=True
            )
            return

        try:
            voice_client = await vc_channel.connect(
                cls=voice_recv.VoiceRecvClient,
                self_deaf=False,
                self_mute=True,
            )
        except discord.ClientException:
            existing = guild.voice_client
            if (
                isinstance(existing, voice_recv.VoiceRecvClient)
                and existing.channel.id == vc_channel.id
            ):
                voice_client = existing
            else:
                await interaction.followup.send(
                    "Nie udało się dołączyć do kanału głosowego.", ephemeral=True
                )
                return
        except Exception as e:
            log.exception("voice connect failed")
            await interaction.followup.send(
                f"Błąd przy dołączaniu do kanału: `{e}`", ephemeral=True
            )
            return

        if interaction.channel_id is None:
            await interaction.followup.send(
                "Nie można ustalić kanału tekstowego.", ephemeral=True
            )
            await voice_client.disconnect(force=False)
            return

        session = RecordingSession.create(
            voice_client=voice_client,
            text_channel_id=interaction.channel_id,
            cfg=self.bot.cfg,
        )
        session.start()
        self.bot.sessions[guild.id] = session
        await interaction.followup.send(
            f"Nagrywam na kanale **{vc_channel.name}**. "
            f"Zatrzymaj komendą `/skryba stop`.",
            ephemeral=True,
        )

    @app_commands.command(
        name="stop",
        description="Kończy nagrywanie i publikuje podsumowanie.",
    )
    async def stop(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if guild is None or guild.id not in self.bot.sessions:
            await interaction.followup.send(
                "Nie ma aktywnego nagrania na tym serwerze.", ephemeral=True
            )
            return

        session = self.bot.sessions.pop(guild.id)
        session_dir = session.session_dir
        await session.stop()

        await interaction.followup.send(
            "Zakończono nagrywanie. Trwa transkrypcja i podsumowywanie...",
            ephemeral=True,
        )

        try:
            await run_pipeline(
                session_dir=session_dir,
                cfg=self.bot.cfg,
                transcriber=self.bot.transcriber,
                bot=self.bot,
            )
        except Exception as e:
            log.exception("pipeline failed for session %s", session_dir)
            try:
                await interaction.followup.send(
                    f"Pipeline zakończył się błędem: `{e}`. "
                    f"Pliki zostały zapisane w `{session_dir}` i będą wznowione "
                    f"przy następnym starcie bota.",
                    ephemeral=True,
                )
            except Exception:
                pass

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
        if session is None:
            await interaction.response.send_message(
                "Brak aktywnego nagrania.", ephemeral=True
            )
            return
        started = datetime.fromisoformat(session.state.started_at)
        elapsed_s = int((datetime.now(timezone.utc) - started).total_seconds())
        m, s = divmod(elapsed_s, 60)
        h, m = divmod(m, 60)
        members = ", ".join(session.state.members.values()) or "(brak)"
        await interaction.response.send_message(
            f"Nagrywam od {started.isoformat(timespec='seconds')} "
            f"(czas: {h:02d}:{m:02d}:{s:02d}). Mówcy: {members}.",
            ephemeral=True,
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


def main() -> None:
    cfg = load_config()
    ensure_ffmpeg()
    _load_opus()
    bot = make_bot(cfg)
    bot.run(cfg.secrets.discord_token)


if __name__ == "__main__":
    main()
