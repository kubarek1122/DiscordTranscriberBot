from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import discord
from discord.sinks import Sink

from config import AppConfig
from src.session import SessionState, make_session_dir

log = logging.getLogger(__name__)


class ChunkedPCMSink(Sink):
    """Custom sink that streams per-user raw PCM directly to disk.

    Discord delivers 48 kHz stereo signed-16 PCM frames via Sink.write(data, user).
    We append each frame to <audio_dir>/<user_id>.pcm. fsync_all() can be called
    periodically by a heartbeat to harden against power loss.
    """

    def __init__(self, audio_dir: Path) -> None:
        super().__init__()
        self.audio_dir = audio_dir
        self._files: dict[int, BinaryIO] = {}
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.observed_user_ids: set[int] = set()

    def write(self, data, user) -> None:  # py-cord calls this for each PCM frame
        f = self._files.get(user)
        if f is None:
            path = self.audio_dir / f"{user}.pcm"
            f = open(path, "ab")
            self._files[user] = f
            self.observed_user_ids.add(int(user))
        f.write(data.pcm)

    def fsync_all(self) -> None:
        for f in self._files.values():
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:  # best-effort
                pass

    def cleanup(self) -> None:
        for f in self._files.values():
            try:
                f.flush()
                os.fsync(f.fileno())
                f.close()
            except Exception:
                pass
        self._files.clear()


def _ffmpeg_pcm_to_wav(pcm_path: Path, wav_path: Path) -> None:
    """Convert 48kHz stereo s16le raw PCM into 16kHz mono WAV for Whisper."""
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
    subprocess.run(cmd, check=True)


def finalize_audio(session_dir: Path) -> None:
    """Convert every <user_id>.pcm in audio/ to a 16 kHz mono <user_id>.wav.
    Skips users whose pcm is empty or missing. Idempotent."""
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
    """Delete per-user .pcm and .wav files. Called after success when keep_audio=false."""
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
    voice_client: discord.VoiceClient
    session_dir: Path
    state: SessionState
    cfg: AppConfig
    sink: ChunkedPCMSink
    _heartbeat_task: asyncio.Task | None = None
    _stopped: bool = False

    @classmethod
    def create(
        cls,
        *,
        voice_client: discord.VoiceClient,
        text_channel_id: int,
        cfg: AppConfig,
    ) -> "RecordingSession":
        guild = voice_client.guild
        state = SessionState.new(
            guild_id=guild.id,
            voice_channel_id=voice_client.channel.id,
            text_channel_id=text_channel_id,
        )
        state.summarizer_backend = cfg.summarizer.backend
        # Prime members from the voice channel roster.
        for m in voice_client.channel.members:
            if not m.bot:
                state.members[str(m.id)] = m.display_name

        session_dir = make_session_dir(
            cfg.recording.output_dir, guild.id, state.started_at
        )
        state.save(session_dir)
        sink = ChunkedPCMSink(session_dir / "audio")
        return cls(
            voice_client=voice_client,
            session_dir=session_dir,
            state=state,
            cfg=cfg,
            sink=sink,
        )

    def start(self) -> None:
        async def _on_done(sink: Sink, *_args):  # noqa: ARG001
            log.info("sink finished cleanup for session %s", self.session_dir.name)

        self.voice_client.start_recording(self.sink, _on_done)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        log.info("recording started in %s", self.session_dir)

    async def _heartbeat_loop(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(self.cfg.recording.heartbeat_interval_s)
                self.sink.fsync_all()
                # Refresh members for anyone who joined mid-call
                guild = self.voice_client.guild
                for uid in list(self.sink.observed_user_ids):
                    key = str(uid)
                    if key not in self.state.members:
                        member = guild.get_member(uid)
                        self.state.members[key] = (
                            member.display_name if member else f"User{uid}"
                        )
                self.state.heartbeat(self.session_dir)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        try:
            if self.voice_client.recording:
                self.voice_client.stop_recording()
        except Exception as e:
            log.warning("stop_recording raised: %s", e)
        self.sink.cleanup()

        try:
            await self.voice_client.disconnect(force=False)
        except Exception as e:
            log.warning("voice disconnect raised: %s", e)

        # Final member refresh
        guild = self.voice_client.guild
        for uid in list(self.sink.observed_user_ids):
            key = str(uid)
            if key not in self.state.members:
                member = guild.get_member(uid)
                self.state.members[key] = (
                    member.display_name if member else f"User{uid}"
                )
        self.state.advance(self.session_dir, "recorded")


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or use the provided Dockerfile."
        )
