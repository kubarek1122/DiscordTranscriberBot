from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def _make_handler(session_dir: Path) -> logging.FileHandler:
    """Open a FileHandler appending to <session_dir>/session.log."""
    session_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        session_dir / "session.log", mode="a", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    return handler


def attach_session_log(session_dir: Path) -> logging.Handler:
    """Manual attach. Caller must pair with detach_session_log() in a try/finally
    (or use the session_log() context manager below). Used by RecordingSession
    where the attach/detach span two slash command invocations."""
    handler = _make_handler(session_dir)
    logging.getLogger().addHandler(handler)
    return handler


def detach_session_log(handler: logging.Handler) -> None:
    try:
        logging.getLogger().removeHandler(handler)
    finally:
        try:
            handler.flush()
        finally:
            handler.close()


@contextmanager
def session_log(session_dir: Path) -> Iterator[logging.Handler]:
    """Context manager for the pipeline call sites (stop_cmd, recovery,
    kontynuuj) where the lifetime is a single await."""
    handler = attach_session_log(session_dir)
    try:
        yield handler
    finally:
        detach_session_log(handler)
