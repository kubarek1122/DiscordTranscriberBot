from __future__ import annotations

import os
import re
from pathlib import Path


def write_atomic(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mode = "wb" if isinstance(content, bytes) else "w"
    encoding = None if isinstance(content, bytes) else "utf-8"
    with open(tmp, mode, encoding=encoding) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_ACTIONS_HEADER = re.compile(
    r"^#{1,6}\s*(?:Decyzje\s+i\s+zadania|Decyzje|Zadania|Action\s*items?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANY_HEADER = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


def extract_actions_section(summary_md: str) -> str:
    """Pull the 'Decyzje i zadania' section out of a summary, falling back to empty."""
    m = _ACTIONS_HEADER.search(summary_md)
    if not m:
        return ""
    start = m.start()
    rest = summary_md[m.end():]
    next_header = _ANY_HEADER.search(rest)
    if next_header:
        return summary_md[start: m.end() + next_header.start()].rstrip() + "\n"
    return summary_md[start:].rstrip() + "\n"


def write_summary(session_dir: Path, summary_md: str) -> Path:
    path = session_dir / "summary.md"
    write_atomic(path, summary_md)
    return path


def write_transcript(session_dir: Path, transcript: str) -> Path:
    path = session_dir / "transcript.txt"
    write_atomic(path, transcript)
    return path


def write_actions(session_dir: Path, summary_md: str) -> Path:
    path = session_dir / "actions.md"
    section = extract_actions_section(summary_md)
    write_atomic(path, section or "# Decyzje i zadania\n\n_(brak)_\n")
    return path
