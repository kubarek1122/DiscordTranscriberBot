from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

import discord
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import AppConfig
from src.artifacts import write_actions, write_summary, write_transcript
from src.messages import PIPELINE_FAILED
from src.recording import cleanup_audio_files, finalize_audio
from src.session import STAGE_ORDER, SessionState, Stage  # noqa: F401
from src.summarize import Summarizer, get_summarizer
from src.transcribe import Transcriber, format_transcript, transcribe_session

log = logging.getLogger(__name__)

DISCORD_MAX_LEN = 1900  # leave headroom under 2000-char message cap


def _stage_at_least(state: SessionState, target: Stage) -> bool:
    return STAGE_ORDER.index(state.stage) >= STAGE_ORDER.index(target)


StageCallback = Callable[[Stage], Awaitable[None]]


async def run_pipeline(
    *,
    session_dir: Path,
    cfg: AppConfig,
    transcriber: Transcriber,
    bot: discord.Client,
    summarizer: Optional[Summarizer] = None,
    on_stage: StageCallback | None = None,
) -> None:
    """Resumable pipeline: recorded -> transcribed -> summarized -> posted.

    Idempotent w.r.t. the session.json `stage` field. To re-run from an earlier
    stage, rewind `state.stage` (and clear `posted_message_id` for re-post) on
    disk first; see src.replay.

    A failure increments `state.retries`; if the retry budget is exhausted the
    session is advanced to the terminal `failed` stage and the exception is
    re-raised so the caller can notify the user."""
    state = SessionState.load(session_dir)
    if state.stage == "failed":
        log.info("pipeline: session=%s already failed, skipping", session_dir.name)
        return
    if state.stage == "posted":
        log.info("pipeline: session=%s already posted, skipping", session_dir.name)
        return
    log.info(
        "pipeline: session=%s stage=%s retries=%d",
        session_dir.name,
        state.stage,
        state.retries,
    )

    async def _notify(stage: Stage) -> None:
        if on_stage is None:
            return
        try:
            await on_stage(stage)
        except Exception:
            log.exception("on_stage callback raised; ignoring")

    try:
        # 1. Finalize audio (pcm -> wav) -> stage=recorded
        if not _stage_at_least(state, "recorded"):
            finalize_audio(session_dir)
            state.advance(session_dir, "recorded")

        # 2. Transcribe -> stage=transcribed
        if not _stage_at_least(state, "transcribed"):
            await _notify("transcribed")
            finalize_audio(session_dir)  # idempotent; safety net
            segments = await transcribe_session(
                transcriber, session_dir / "audio", state.members
            )
            write_transcript(session_dir, format_transcript(segments))
            state.advance(session_dir, "transcribed")
        else:
            log.info("transcript already exists, skipping transcription")

        # 3. Summarize -> stage=summarized
        if not _stage_at_least(state, "summarized"):
            await _notify("summarized")
            if summarizer is None:
                summarizer = get_summarizer(cfg)
            transcript = (session_dir / "transcript.txt").read_text(encoding="utf-8")
            if not transcript.strip():
                summary_md = (
                    "## Podsumowanie\nRozmowa była zbyt krótka lub niezrozumiała, "
                    "aby ją podsumować.\n\n"
                    "## Kluczowe punkty\n_(brak)_\n\n"
                    "## Decyzje i zadania\n_(brak)_\n"
                )
            else:
                summary_md = await _summarize_with_retry(summarizer, transcript, cfg)
            write_summary(session_dir, summary_md)
            write_actions(session_dir, summary_md)
            state.summarizer_backend = summarizer.name if summarizer else state.summarizer_backend
            state.advance(session_dir, "summarized")

        # 4. Post -> stage=posted
        if not _stage_at_least(state, "posted"):
            await _notify("posted")
            await _post_with_retry(state, session_dir, bot, cfg)
            if not cfg.recording.keep_audio:
                cleanup_audio_files(session_dir)
    except BaseException as e:
        # Treat asyncio.CancelledError as a clean abort — don't burn a retry
        # for it. Anything else counts toward the budget.
        if isinstance(e, asyncio.CancelledError):
            raise
        state.record_failure(session_dir, f"{type(e).__name__}: {e}")
        log.warning(
            "pipeline: session=%s attempt failed (%d/%d): %s",
            session_dir.name,
            state.retries,
            cfg.reliability.max_pipeline_retries,
            e,
        )
        if state.retries >= cfg.reliability.max_pipeline_retries:
            log.error(
                "pipeline: session=%s exceeded retry budget, marking failed",
                session_dir.name,
            )
            state.advance(session_dir, "failed")
        raise


async def _summarize_with_retry(
    summarizer: Summarizer, transcript: str, cfg: AppConfig
) -> str:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(cfg.reliability.summarizer_retries),
        wait=wait_exponential(
            multiplier=1, min=1, max=cfg.reliability.summarizer_backoff_max_s
        ),
        reraise=True,
    ):
        with attempt:
            return await summarizer.summarize(transcript)
    raise RuntimeError("unreachable")


async def _post_with_retry(
    state: SessionState,
    session_dir: Path,
    bot: discord.Client,
    cfg: AppConfig,
) -> None:
    if state.posted_message_id is not None:
        log.info("already posted (msg=%s); skipping", state.posted_message_id)
        state.advance(session_dir, "posted")
        return

    channel = bot.get_channel(state.text_channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(state.text_channel_id)
        except Exception as e:
            log.error("cannot resolve text channel %s: %s", state.text_channel_id, e)
            raise

    summary_md = (session_dir / "summary.md").read_text(encoding="utf-8")
    body = summary_md if len(summary_md) <= DISCORD_MAX_LEN else (
        summary_md[:DISCORD_MAX_LEN] + "\n\n_(pełne podsumowanie w załączniku summary.md)_"
    )
    transcript_path = session_dir / "transcript.txt"
    summary_path = session_dir / "summary.md"
    include_full_summary = len(summary_md) > DISCORD_MAX_LEN

    def _open_files() -> list[discord.File]:
        files: list[discord.File] = []
        if transcript_path.exists():
            files.append(discord.File(str(transcript_path), filename="transcript.txt"))
        if include_full_summary:
            files.append(discord.File(str(summary_path), filename="summary.md"))
        return files

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(cfg.reliability.post_retries),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((discord.HTTPException, discord.ConnectionClosed, OSError)),
        reraise=True,
    ):
        with attempt:
            msg = await channel.send(
                content=body or "_(brak treści)_",
                files=_open_files(),
            )
            state.posted_message_id = msg.id
            state.advance(session_dir, "posted")
            return


async def notify_pipeline_failure(
    *,
    bot: discord.Client,
    text_channel_id: int,
    session_dir: Path,
    error: BaseException,
) -> None:
    """Post a Polish failure notice to the original text channel.

    Best-effort: swallows exceptions so the caller's error path is not
    derailed by a secondary failure. Truncates the error message to keep
    the channel readable."""
    err_text = str(error) or error.__class__.__name__
    if len(err_text) > 200:
        err_text = err_text[:200] + "…"
    body = PIPELINE_FAILED.format(session_name=session_dir.name, error=err_text)
    try:
        channel = bot.get_channel(text_channel_id)
        if channel is None:
            channel = await bot.fetch_channel(text_channel_id)
        await channel.send(body)
    except Exception:
        log.exception(
            "notify_pipeline_failure: could not post to channel %s", text_channel_id
        )
