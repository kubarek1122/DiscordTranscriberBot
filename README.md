# DiscordTranscriber

Self-hosted Discord bot that joins voice channels, transcribes Polish discussions per-speaker with local Whisper, and posts an LLM-generated summary back to the text channel.

## Quick start

1. Copy `.env.example` to `.env` and fill in your tokens.
2. Tweak `config.yaml` (default uses Claude as summarizer + faster-whisper `large-v3` on CUDA).
3. Run:

   ```bash
   docker compose up -d
   ```

4. In Discord, join a voice channel, then run `/record start`. Speak. Run `/record stop`.

## Slash commands

- `/record start` — bot joins your voice channel and starts recording.
- `/record stop` — finalize, transcribe, summarize, post the summary.
- `/record status` — show current state.

## Artifacts

For each session, `recordings/<guild_id>/<YYYY-MM-DD_HH-MM-SS>/` will contain:

- `session.json` — state file (used for crash recovery)
- `audio/<username>.wav` — per-speaker audio (deleted after success unless `recording.keep_audio: true`)
- `audio/<username>.segments.json` — Whisper segments per speaker
- `transcript.txt` — interleaved per-speaker transcript
- `summary.md` — LLM-generated Polish summary with key points + actions
- `actions.md` — extracted "Decyzje i zadania" section

## Local (non-Docker) run

```bash
python -m venv .venv
.venv\Scripts\activate           # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python bot.py
```

You'll need CUDA 12.8 + cuDNN 9.x installed on the host for GPU acceleration. Otherwise set `whisper.device: cpu` and `whisper.compute_type: int8` in `config.yaml`.

## Replaying a past session

```bash
python -m src.replay recordings/<guild>/<session> --from-stage summarize
```

Useful for re-summarizing after switching the configured backend.
