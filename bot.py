from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import discord

from config import AppConfig, load_config
from src.pipeline import run_pipeline
from src.recording import RecordingSession, ensure_ffmpeg, finalize_audio
from src.session import SessionState, is_heartbeat_stale, scan_unfinished
from src.transcribe import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("bot")


class TranscriberBot(discord.Bot):
    def __init__(self, cfg: AppConfig) -> None:
        intents = discord.Intents.default()
        intents.members = True
        intents.voice_states = True
        super().__init__(intents=intents)
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

            # If a session crashed mid-recording, treat it as recorded
            # (heartbeat is older than the configured threshold).
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


def make_bot(cfg: AppConfig) -> TranscriberBot:
    bot = TranscriberBot(cfg)

    record = bot.create_group("record", "Sterowanie nagrywaniem rozmów")

    async def _user_in_voice(ctx: discord.ApplicationContext) -> Optional[discord.VoiceChannel]:
        if not isinstance(ctx.author, discord.Member):
            return None
        if ctx.author.voice is None or ctx.author.voice.channel is None:
            return None
        return ctx.author.voice.channel  # type: ignore[return-value]

    @record.command(name="start", description="Dołącza do twojego kanału głosowego i zaczyna nagrywać.")
    async def start_cmd(ctx: discord.ApplicationContext) -> None:
        await ctx.defer(ephemeral=True)
        guild = ctx.guild
        if guild is None:
            await ctx.followup.send("Komenda działa tylko na serwerze.", ephemeral=True)
            return

        if guild.id in bot.sessions:
            await ctx.followup.send(
                "Już nagrywam na tym serwerze. Użyj `/record stop`, aby zakończyć.",
                ephemeral=True,
            )
            return

        vc_channel = await _user_in_voice(ctx)
        if vc_channel is None:
            await ctx.followup.send(
                "Musisz najpierw wejść na kanał głosowy.",
                ephemeral=True,
            )
            return

        try:
            voice_client = await vc_channel.connect()
        except discord.ClientException:
            existing = guild.voice_client
            if existing and existing.channel.id == vc_channel.id:
                voice_client = existing  # type: ignore[assignment]
            else:
                await ctx.followup.send(
                    "Nie udało się dołączyć do kanału głosowego.",
                    ephemeral=True,
                )
                return
        except Exception as e:
            log.exception("voice connect failed")
            await ctx.followup.send(
                f"Błąd przy dołączaniu do kanału: `{e}`",
                ephemeral=True,
            )
            return

        session = RecordingSession.create(
            voice_client=voice_client,
            text_channel_id=ctx.channel_id,
            cfg=bot.cfg,
        )
        session.start()
        bot.sessions[guild.id] = session
        await ctx.followup.send(
            f"Nagrywam na kanale **{vc_channel.name}**. "
            f"Zatrzymaj komendą `/record stop`.",
            ephemeral=True,
        )

    @record.command(name="stop", description="Kończy nagrywanie i publikuje podsumowanie.")
    async def stop_cmd(ctx: discord.ApplicationContext) -> None:
        await ctx.defer()
        guild = ctx.guild
        if guild is None or guild.id not in bot.sessions:
            await ctx.followup.send(
                "Nie ma aktywnego nagrania na tym serwerze.",
                ephemeral=True,
            )
            return

        session = bot.sessions.pop(guild.id)
        session_dir = session.session_dir
        await session.stop()

        await ctx.followup.send(
            "Zakończono nagrywanie. Trwa transkrypcja i podsumowywanie...",
            ephemeral=True,
        )

        try:
            await run_pipeline(
                session_dir=session_dir,
                cfg=bot.cfg,
                transcriber=bot.transcriber,
                bot=bot,
            )
        except Exception as e:
            log.exception("pipeline failed for session %s", session_dir)
            try:
                await ctx.followup.send(
                    f"Pipeline zakończył się błędem: `{e}`. "
                    f"Pliki zostały zapisane w `{session_dir}` i będą wznowione "
                    f"przy następnym starcie bota.",
                    ephemeral=True,
                )
            except Exception:
                pass

    @record.command(name="status", description="Pokazuje stan nagrywania.")
    async def status_cmd(ctx: discord.ApplicationContext) -> None:
        guild = ctx.guild
        if guild is None:
            await ctx.respond("Komenda działa tylko na serwerze.", ephemeral=True)
            return
        session = bot.sessions.get(guild.id)
        if session is None:
            await ctx.respond("Brak aktywnego nagrania.", ephemeral=True)
            return
        started = datetime.fromisoformat(session.state.started_at)
        elapsed_s = int((datetime.now(timezone.utc) - started).total_seconds())
        m, s = divmod(elapsed_s, 60)
        h, m = divmod(m, 60)
        members = ", ".join(session.state.members.values()) or "(brak)"
        await ctx.respond(
            f"Nagrywam od {started.isoformat(timespec='seconds')} "
            f"(czas: {h:02d}:{m:02d}:{s:02d}). Mówcy: {members}.",
            ephemeral=True,
        )

    @bot.event
    async def on_voice_state_update(  # noqa: ARG001
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        guild = member.guild
        session = bot.sessions.get(guild.id)
        if session is None:
            return
        # If our bot got disconnected from voice, finalize the session gracefully.
        if member.id == bot.user.id and before.channel is not None and after.channel is None:
            log.warning("bot disconnected from voice, finalizing session")
            try:
                await session.stop()
            except Exception:
                log.exception("error stopping session after disconnect")
            bot.sessions.pop(guild.id, None)
            try:
                await run_pipeline(
                    session_dir=session.session_dir,
                    cfg=bot.cfg,
                    transcriber=bot.transcriber,
                    bot=bot,
                )
            except Exception:
                log.exception("post-disconnect pipeline failed")

    return bot


def main() -> None:
    cfg = load_config()
    ensure_ffmpeg()
    bot = make_bot(cfg)
    bot.run(cfg.secrets.discord_token)


if __name__ == "__main__":
    main()
