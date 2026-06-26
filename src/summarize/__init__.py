from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import AppConfig
from src.prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER_TEMPLATE,
    DEFAULT_KIND,
    DISCUSSION_KINDS,
    parse_kind,
)
from src.summarize.base import Summarizer

if TYPE_CHECKING:
    from src.session import SessionState

log = logging.getLogger(__name__)

# Token-bounding for the classification call: the head usually establishes the
# discussion's nature, the tail catches a late pivot (e.g. "ok, to ustalmy kto
# co robi"). Cheap and good enough — classification never needs the whole text.
_CLASSIFY_HEAD = 3000
_CLASSIFY_TAIL = 1000


def get_summarizer(cfg: AppConfig) -> Summarizer:
    backend = cfg.summarizer.backend
    if backend == "claude":
        from src.summarize.claude import ClaudeSummarizer
        if not cfg.secrets.anthropic_api_key:
            raise RuntimeError("summarizer.backend=claude but ANTHROPIC_API_KEY is not set")
        return ClaudeSummarizer(
            api_key=cfg.secrets.anthropic_api_key,
            model=cfg.summarizer.claude.model,
            max_tokens=cfg.summarizer.claude.max_tokens,
        )
    if backend == "openai":
        from src.summarize.openai_backend import OpenAISummarizer
        if not cfg.secrets.openai_api_key:
            raise RuntimeError("summarizer.backend=openai but OPENAI_API_KEY is not set")
        return OpenAISummarizer(
            api_key=cfg.secrets.openai_api_key,
            model=cfg.summarizer.openai.model,
            max_tokens=cfg.summarizer.openai.max_tokens,
        )
    if backend == "ollama":
        from src.summarize.ollama_backend import OllamaSummarizer
        return OllamaSummarizer(
            host=cfg.summarizer.ollama.host,
            model=cfg.summarizer.ollama.model,
        )
    raise ValueError(f"Unknown summarizer backend: {backend}")


def _sample_for_classify(transcript: str) -> str:
    t = transcript.strip()
    if len(t) <= _CLASSIFY_HEAD + _CLASSIFY_TAIL:
        return t
    return t[:_CLASSIFY_HEAD] + "\n…\n" + t[-_CLASSIFY_TAIL:]


async def classify_transcript(summarizer: Summarizer, transcript: str) -> str:
    """Classify a transcript into a discussion kind with one best-effort call.
    Any failure or unrecognized reply falls back to DEFAULT_KIND, so callers
    can rely on it never raising."""
    sample = _sample_for_classify(transcript)
    if not sample:
        return DEFAULT_KIND
    try:
        raw = await summarizer.summarize(
            sample,
            system_prompt=CLASSIFY_SYSTEM,
            user_template=CLASSIFY_USER_TEMPLATE,
        )
        kind = parse_kind(raw)
        log.info("classified discussion kind=%s", kind)
        return kind
    except Exception:
        log.exception("kind classification failed; defaulting to %s", DEFAULT_KIND)
        return DEFAULT_KIND


async def resolve_kind(
    summarizer: Summarizer, transcript: str, state: "SessionState"
) -> str:
    """Resolve the discussion kind for a session: a manual choice
    (`state.discussion_kind`) always wins, otherwise auto-detect."""
    if state.discussion_kind:
        return (
            state.discussion_kind
            if state.discussion_kind in DISCUSSION_KINDS
            else DEFAULT_KIND
        )
    return await classify_transcript(summarizer, transcript)


async def suggest_drift(
    summarizer: Summarizer, transcript: str, current_kind: str
) -> str | None:
    """If the transcript looks like a *different, specific* kind than
    `current_kind`, return that kind; else None. Used at stop to flag that a
    manually-chosen type may not fit. Never suggests the vague DEFAULT_KIND."""
    detected = await classify_transcript(summarizer, transcript)
    if detected != current_kind and detected != DEFAULT_KIND:
        return detected
    return None


__all__ = [
    "Summarizer",
    "get_summarizer",
    "resolve_kind",
    "classify_transcript",
    "suggest_drift",
]
