from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.artifacts import write_atomic

Stage = Literal["recording", "recorded", "transcribed", "summarized", "posted"]
STAGE_ORDER: tuple[Stage, ...] = (
    "recording",
    "recorded",
    "transcribed",
    "summarized",
    "posted",
)


class SessionState(BaseModel):
    guild_id: int
    voice_channel_id: int
    text_channel_id: int
    started_at: str  # ISO8601
    stage: Stage = "recording"
    last_heartbeat: float = Field(default_factory=time.time)
    members: dict[str, str] = Field(default_factory=dict)  # user_id -> display_name
    posted_message_id: int | None = None
    summarizer_backend: str | None = None
    error: str | None = None

    @classmethod
    def new(
        cls,
        *,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
    ) -> "SessionState":
        return cls(
            guild_id=guild_id,
            voice_channel_id=voice_channel_id,
            text_channel_id=text_channel_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, session_dir: Path) -> "SessionState":
        with (session_dir / "session.json").open("r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def save(self, session_dir: Path) -> None:
        write_atomic(session_dir / "session.json", self.model_dump_json(indent=2))

    def advance(self, session_dir: Path, stage: Stage) -> None:
        if STAGE_ORDER.index(stage) < STAGE_ORDER.index(self.stage):
            return
        self.stage = stage
        self.save(session_dir)

    def heartbeat(self, session_dir: Path) -> None:
        self.last_heartbeat = time.time()
        self.save(session_dir)


def session_dirname(started_at_iso: str) -> str:
    dt = datetime.fromisoformat(started_at_iso)
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def make_session_dir(root: Path, guild_id: int, started_at_iso: str) -> Path:
    d = root / str(guild_id) / session_dirname(started_at_iso)
    d.mkdir(parents=True, exist_ok=True)
    (d / "audio").mkdir(parents=True, exist_ok=True)
    return d


def most_recent_unfinished(root: Path, guild_id: int) -> Path | None:
    """Newest session directory for `guild_id` whose stage is not `posted`.

    Used by `/skryba kontynuuj` to retry the latest failed pipeline without
    waiting for a bot restart."""
    guild_dir = root / str(guild_id)
    if not guild_dir.exists():
        return None
    for session_dir in sorted(
        (p for p in guild_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    ):
        sj = session_dir / "session.json"
        if not sj.exists():
            continue
        try:
            state = SessionState.load(session_dir)
        except Exception:
            continue
        if state.stage != "posted":
            return session_dir
    return None


def scan_unfinished(root: Path) -> list[Path]:
    """Return session dirs whose session.json has stage != 'posted'."""
    if not root.exists():
        return []
    out: list[Path] = []
    for guild_dir in root.iterdir():
        if not guild_dir.is_dir():
            continue
        for session_dir in guild_dir.iterdir():
            sj = session_dir / "session.json"
            if not sj.exists():
                continue
            try:
                state = SessionState.load(session_dir)
            except Exception:
                continue
            if state.stage != "posted":
                out.append(session_dir)
    return out


def is_heartbeat_stale(state: SessionState, max_age_s: int) -> bool:
    return (time.time() - state.last_heartbeat) > max_age_s
