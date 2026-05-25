from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel

from src.artifacts import write_atomic

log = logging.getLogger(__name__)


@dataclass
class Segment:
    speaker: str
    start: float  # seconds from start of speaker's audio file
    end: float
    text: str


class Transcriber:
    """Lazily-loaded Whisper wrapper. One process-wide instance."""

    def __init__(self, *, model: str, device: str, compute_type: str, language: str) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    def _ensure_loaded(self) -> WhisperModel:
        if self._model is None:
            log.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                self._model_name, self._device, self._compute_type,
            )
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _transcribe_file_blocking(self, wav_path: Path, speaker: str) -> list[Segment]:
        model = self._ensure_loaded()
        segments_iter, _info = model.transcribe(
            str(wav_path),
            language=self._language,
            vad_filter=True,
            beam_size=5,
        )
        return [
            Segment(speaker=speaker, start=float(s.start), end=float(s.end), text=s.text.strip())
            for s in segments_iter
            if s.text and s.text.strip()
        ]

    async def transcribe_file(self, wav_path: Path, speaker: str) -> list[Segment]:
        async with self._lock:  # serialize: large-v3 already saturates GPU
            return await asyncio.to_thread(self._transcribe_file_blocking, wav_path, speaker)


def _segments_to_json(segments: Iterable[Segment]) -> str:
    return json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2)


def _segments_from_json(text: str) -> list[Segment]:
    return [Segment(**item) for item in json.loads(text)]


def _load_user_offset_s(audio_dir: Path, user_id: str) -> float:
    """Read the sidecar-written first-frame offset (in seconds) for a user.

    The sidecar writes `<user_id>.offset.json` the first time the user
    speaks, containing `{"offset_ms": N}` where N is the wall-clock delta
    from session start to the user's first frame. We add this to every
    segment so two speakers' segments line up on the session timeline
    instead of being interleaved by file-local time (which would shuffle
    the conversation when sorted)."""
    p = audio_dir / f"{user_id}.offset.json"
    if not p.exists():
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return max(0.0, float(data.get("offset_ms", 0)) / 1000.0)
    except (OSError, ValueError, json.JSONDecodeError):
        log.warning("could not read offset for user_id=%s; using 0", user_id)
        return 0.0


async def transcribe_session(
    transcriber: Transcriber,
    audio_dir: Path,
    members: dict[str, str],
) -> list[Segment]:
    """Transcribe each <user_id>.wav under audio_dir, persisting per-speaker segments
    immediately so a crash mid-batch doesn't redo finished users.

    `members` is {user_id: display_name}. WAV files are named "<user_id>.wav".
    Per-segment timestamps are shifted by each user's first-frame offset so
    the returned list is correctly ordered on the session timeline.
    Returns a flat list of all segments, sorted by start time.
    """
    all_segments: list[Segment] = []
    for user_id, display_name in members.items():
        wav = audio_dir / f"{user_id}.wav"
        if not wav.exists():
            log.warning("missing wav for user_id=%s name=%s", user_id, display_name)
            continue

        offset_s = _load_user_offset_s(audio_dir, user_id)

        seg_path = audio_dir / f"{user_id}.segments.json"
        if seg_path.exists():
            log.info("reusing cached segments for %s (offset=%.2fs)", display_name, offset_s)
            cached = _segments_from_json(seg_path.read_text(encoding="utf-8"))
            # The cached file might have been written before the offset
            # fix landed (pre-offset segments are file-local). Apply the
            # offset on read so re-runs of older sessions are corrected
            # without having to re-transcribe.
            for s in cached:
                s.start += offset_s
                s.end += offset_s
            all_segments.extend(cached)
            continue

        log.info(
            "transcribing %s (%s) offset=%.2fs", display_name, wav, offset_s,
        )
        segs = await transcriber.transcribe_file(wav, speaker=display_name)
        # Persist the *file-local* segments so the cache stays speaker-local
        # and we can re-apply the offset on every read (the offset file is
        # the single source of truth).
        write_atomic(seg_path, _segments_to_json(segs))
        for s in segs:
            s.start += offset_s
            s.end += offset_s
        all_segments.extend(segs)

    all_segments.sort(key=lambda s: s.start)
    return all_segments


def format_transcript(segments: list[Segment]) -> str:
    lines: list[str] = []
    for s in segments:
        ts = _fmt_ts(s.start)
        lines.append(f"[{ts}] {s.speaker}: {s.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# CLI smoke test: python -m src.transcribe <audio_dir>
if __name__ == "__main__":
    import sys
    import asyncio

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python -m src.transcribe <audio_dir>")
        sys.exit(2)
    audio = Path(sys.argv[1])
    wavs = list(audio.glob("*.wav"))
    if not wavs:
        print(f"no .wav files under {audio}")
        sys.exit(1)
    members = {w.stem: w.stem for w in wavs}

    from config import load_config
    cfg = load_config()
    t = Transcriber(
        model=cfg.whisper.model,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        language=cfg.whisper.language,
    )
    segs = asyncio.run(transcribe_session(t, audio, members))
    print(format_transcript(segs))
