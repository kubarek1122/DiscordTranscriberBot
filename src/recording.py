from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import discord

from config import AppConfig
from src.logging_util import attach_session_log, detach_session_log
from src.recorder_client import RecorderClient, RecorderError
from src.session import SessionState, make_session_dir

log = logging.getLogger(__name__)


def _ffmpeg_pcm_to_wav(pcm_path: Path, wav_path: Path) -> None:
    """Convert 48kHz stereo s16le raw PCM into 16kHz mono WAV for Whisper.

    Bounded: 48 kHz stereo s16le = ~11 MB/min, so a 1-hour recording is ~660 MB.
    We allow 2 s per MB of input (very generous — typical ffmpeg PCM→WAV runs at
    well over 100×) but never less than 30 s. On timeout we kill the subprocess
    and re-raise; the caller's failure/retry path takes over."""
    try:
        size_mb = max(1, pcm_path.stat().st_size // (1024 * 1024))
    except OSError:
        size_mb = 1
    timeout_s = max(30, size_mb * 2)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "s16le",
        "-ar", "48000",
        "-ac", "2",
        "-i", str(pcm_path),
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.error(
            "ffmpeg timed out after %ds converting %s (size=%dMB)",
            timeout_s, pcm_path.name, size_mb,
        )
        # Partial WAV is unusable; remove so the next pass re-tries cleanly.
        try:
            wav_path.unlink()
        except OSError:
            pass
        raise


def finalize_audio(session_dir: Path) -> None:
    """Convert every <user_id>.pcm written by the recorder sidecar into a
    16 kHz mono <user_id>.wav. Idempotent: skips users whose wav already
    exists with non-zero size, skips empty pcm files."""
    audio_dir = session_dir / "audio"
    if not audio_dir.exists():
        return
    for pcm in sorted(audio_dir.glob("*.pcm")):
        wav = pcm.with_suffix(".wav")
        if wav.exists() and wav.stat().st_size > 0:
            continue
        if pcm.stat().st_size == 0:
            log.warning("empty pcm for %s, skipping", pcm.name)
            continue
        log.info("ffmpeg: %s -> %s", pcm.name, wav.name)
        _ffmpeg_pcm_to_wav(pcm, wav)


def cleanup_audio_files(session_dir: Path) -> None:
    """Delete per-user .pcm and .wav files after a successful upload."""
    audio_dir = session_dir / "audio"
    if not audio_dir.exists():
        return
    for ext in ("*.pcm", "*.wav"):
        for p in audio_dir.glob(ext):
            try:
                p.unlink()
            except OSError:
                pass


@dataclass
class RecordingSession:
    """Owns one recording. Delegates the actual voice capture to the Node
    recorder sidecar over a Unix socket. This Python side only manages
    session state, member resolution, and the heartbeat."""

    voice_channel: discord.VoiceChannel
    text_channel_id: int
    session_dir: Path
    state: SessionState
    cfg: AppConfig
    client: RecorderClient
    # Called when the sidecar reports a hard failure (kicked from VC, voice
    # gateway gave up). The bot wires this to `_auto_stop_session` so the
    # captured PCM still goes through the pipeline.
    on_hard_failure: Callable[[str], Awaitable[None]] | None = field(default=None, repr=False)
    _heartbeat_task: asyncio.Task | None = field(default=None, repr=False)
    _speaker_task: asyncio.Task | None = field(default=None, repr=False)
    _alert_task: asyncio.Task | None = field(default=None, repr=False)
    _log_handler: logging.Handler | None = field(default=None, repr=False)
    _stopped: bool = False

    @classmethod
    def create(
        cls,
        *,
        voice_channel: discord.VoiceChannel,
        text_channel_id: int,
        cfg: AppConfig,
        on_hard_failure: Callable[[str], Awaitable[None]] | None = None,
    ) -> "RecordingSession":
        guild = voice_channel.guild
        state = SessionState.new(
            guild_id=guild.id,
            voice_channel_id=voice_channel.id,
            text_channel_id=text_channel_id,
        )
        state.summarizer_backend = cfg.summarizer.backend
        for m in voice_channel.members:
            if not m.bot:
                state.members[str(m.id)] = m.display_name

        session_dir = make_session_dir(
            cfg.recording.output_dir, guild.id, state.started_at
        )
        state.save(session_dir)
        client = RecorderClient(cfg.recorder.socket_path)
        return cls(
            voice_channel=voice_channel,
            text_channel_id=text_channel_id,
            session_dir=session_dir,
            state=state,
            cfg=cfg,
            client=client,
            on_hard_failure=on_hard_failure,
        )

    async def start(self) -> None:
        # Attach the session-scoped log file as early as possible so any
        # failures during sidecar handshake also land in session.log.
        self._log_handler = attach_session_log(self.session_dir)
        try:
            await self.client.open()
            await self.client.join(
                guild_id=self.voice_channel.guild.id,
                voice_channel_id=self.voice_channel.id,
                session_dir=self.session_dir,
                timeout_s=self.cfg.recorder.join_timeout_s,
            )
        except RecorderError:
            await self.client.close()
            detach_session_log(self._log_handler)
            self._log_handler = None
            raise
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._speaker_task = asyncio.create_task(self._speaker_loop())
        self._alert_task = asyncio.create_task(self._alert_loop())
        log.info("recording started in %s", self.session_dir)

    async def _heartbeat_loop(self) -> None:
        """Periodically refresh member display names (people may join the VC
        after recording started) and write `last_heartbeat`. The sidecar owns
        the audio file handles, so no fsync is needed here."""
        try:
            while not self._stopped:
                await asyncio.sleep(self.cfg.recording.heartbeat_interval_s)
                guild = self.voice_channel.guild
                # Pick up anyone now in the VC who wasn't there at start.
                for m in self.voice_channel.members:
                    if m.bot:
                        continue
                    key = str(m.id)
                    if key not in self.state.members:
                        self.state.members[key] = m.display_name
                self.state.heartbeat(self.session_dir)
        except asyncio.CancelledError:
            pass

    async def _speaker_loop(self) -> None:
        """Drain the sidecar's `speaker` notifications. Used only to update
        `state.members` for users who weren't in the VC roster at start time
        but spoke during the call."""
        try:
            queue = await self.client.speaker_events()
            while not self._stopped:
                msg = await queue.get()
                uid = msg.get("user_id")
                if not uid:
                    continue
                key = str(uid)
                if key not in self.state.members:
                    name = msg.get("display_name") or f"User{uid}"
                    self.state.members[key] = name
                    self.state.save(self.session_dir)
                    log.info("speaker: %s -> %s", uid, name)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("speaker_loop crashed")

    async def _alert_loop(self) -> None:
        """Drain the sidecar's alert queue. On `recording_failed`, schedule
        the bot-supplied `on_hard_failure` callback as a *separate task* and
        exit immediately.

        The detached task is important: `on_hard_failure` typically calls
        `bot._auto_stop_session` which in turn calls `session.stop()`, and
        `session.stop()` cancels-and-awaits this alert task. Doing the
        handler inline would deadlock on `await self._alert_task`.

        Once we've seen one `recording_failed` we exit — Discord doesn't
        recover voice in-place, and we don't want to double-fire."""
        try:
            queue = await self.client.alerts()
            while not self._stopped:
                msg = await queue.get()
                if msg.get("op") != "recording_failed":
                    continue
                reason = str(msg.get("reason") or "unknown")
                log.warning("alert: recording_failed reason=%s", reason)
                if self.on_hard_failure is not None:
                    asyncio.create_task(self._invoke_hard_failure(reason))
                return
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("alert_loop crashed")

    async def _invoke_hard_failure(self, reason: str) -> None:
        if self.on_hard_failure is None:
            return
        try:
            await self.on_hard_failure(reason)
        except Exception:
            log.exception("on_hard_failure callback raised")

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for task in (self._heartbeat_task, self._speaker_task, self._alert_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        try:
            reply = await self.client.leave(
                guild_id=self.voice_channel.guild.id,
                timeout_s=self.cfg.recorder.leave_timeout_s,
            )
            log.info("recorder: left, stats=%s", reply.get("stats"))
        except Exception as e:
            log.warning("recorder leave raised: %s", e)
        finally:
            await self.client.close()

        self.state.advance(self.session_dir, "recorded")
        # NOTE: log handler intentionally NOT detached here. The caller runs
        # the pipeline next and wants its logs in the same session.log file.
        # Call release_log() in a finally block after the pipeline finishes.

    def release_log(self) -> None:
        """Detach the session log handler. Idempotent. Call once after the
        pipeline has finished (success or final failure) for this session."""
        if self._log_handler is not None:
            detach_session_log(self._log_handler)
            self._log_handler = None


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or use the provided Dockerfile."
        )
