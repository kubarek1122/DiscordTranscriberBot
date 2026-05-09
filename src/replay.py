from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from config import load_config
from src.artifacts import write_actions, write_summary, write_transcript
from src.recording import finalize_audio
from src.session import STAGE_ORDER, SessionState
from src.summarize import get_summarizer
from src.transcribe import Transcriber, format_transcript, transcribe_session

log = logging.getLogger(__name__)


async def replay(session_dir: Path, from_stage: str, repost: bool) -> None:
    cfg = load_config()
    state = SessionState.load(session_dir)
    log.info("session %s currently at stage=%s", session_dir.name, state.stage)

    valid = ("transcribe", "summarize", "post")
    if from_stage not in valid:
        raise SystemExit(f"--from-stage must be one of: {valid}")

    if from_stage == "transcribe":
        # Wipe cached segments so we re-run Whisper from scratch
        for sj in (session_dir / "audio").glob("*.segments.json"):
            sj.unlink()
        state.stage = "recorded"
    elif from_stage == "summarize":
        state.stage = "transcribed"
    elif from_stage == "post":
        state.stage = "summarized"
        state.posted_message_id = None

    state.save(session_dir)

    if STAGE_ORDER.index(state.stage) < STAGE_ORDER.index("recorded"):
        finalize_audio(session_dir)
        state.advance(session_dir, "recorded")

    if STAGE_ORDER.index(state.stage) < STAGE_ORDER.index("transcribed"):
        transcriber = Transcriber(
            model=cfg.whisper.model,
            device=cfg.whisper.device,
            compute_type=cfg.whisper.compute_type,
            language=cfg.whisper.language,
        )
        segs = await transcribe_session(transcriber, session_dir / "audio", state.members)
        write_transcript(session_dir, format_transcript(segs))
        state.advance(session_dir, "transcribed")

    if STAGE_ORDER.index(state.stage) < STAGE_ORDER.index("summarized"):
        summarizer = get_summarizer(cfg)
        transcript = (session_dir / "transcript.txt").read_text(encoding="utf-8")
        summary_md = await summarizer.summarize(transcript) if transcript.strip() else (
            "## Podsumowanie\n_(brak)_\n\n## Kluczowe punkty\n_(brak)_\n\n## Decyzje i zadania\n_(brak)_\n"
        )
        write_summary(session_dir, summary_md)
        write_actions(session_dir, summary_md)
        state.summarizer_backend = summarizer.name
        state.advance(session_dir, "summarized")

    print(f"\n--- {session_dir / 'summary.md'} ---\n")
    print((session_dir / "summary.md").read_text(encoding="utf-8"))

    if repost:
        print(
            "\n[replay] To re-post the summary to Discord, restart the bot — "
            "the startup recovery scan will pick this session up because "
            "stage=summarized and posted_message_id is null."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded session.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument(
        "--from-stage",
        choices=("transcribe", "summarize", "post"),
        default="summarize",
    )
    parser.add_argument(
        "--repost",
        action="store_true",
        help="Reset posted_message_id so the next bot startup re-posts.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    asyncio.run(replay(args.session_dir, args.from_stage, args.repost))


if __name__ == "__main__":
    main()
