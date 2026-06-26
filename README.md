# DiscordTranscriber

Self-hosted Discord bot that joins voice channels, transcribes Polish discussions per-speaker with local Whisper, and posts an LLM-generated summary back to the text channel.

## Quick start

1. Copy `.env.example` to `.env` and fill in your tokens.
2. Tweak `config.yaml` (default uses Claude as summarizer + faster-whisper `large-v3` on CUDA).
3. Run:

   ```bash
   docker compose up -d
   ```

4. In Discord, join a voice channel, then run `/skryba start`. Speak. Run `/skryba stop`.

## Slash commands

- `/skryba start` — bot joins your voice channel and starts recording. Optional `typ:` chooses the discussion type (Organizacyjna / Projektowa / Burza mózgów / Ogólna) for a tailored summary; omit it to auto-detect.
- `/skryba stop` — finalize, transcribe, summarize, post the summary.
- `/skryba status` — show current state.
- `/skryba kontynuuj` — resume the latest unfinished session on this server.
- `/skryba porzuc` — abandon the latest unfinished session (marks it `failed`; files kept).
- `/skryba jaktojestbycskryba` — show quote form Asterix & Obelix

## Artifacts

For each session, `recordings/<guild_id>/<YYYY-MM-DD_HH-MM-SS>/` will contain:

- `session.json` — state file (used for crash recovery)
- `audio/<username>.wav` — per-speaker audio (deleted after success unless `recording.keep_audio: true`)
- `audio/<username>.segments.json` — Whisper segments per speaker
- `transcript.txt` — interleaved per-speaker transcript
- `summary.md` — LLM-generated Polish summary with key points + actions
- `actions.md` — extracted "Decyzje i zadania" section

## Local (non-Docker) run

> **Note:** `python bot.py` is only the gateway/pipeline half. Voice capture lives in the
> Node recorder sidecar (`recorder/`), which the bot reaches over a **Unix-domain socket**.
> Recording therefore requires the sidecar running and a platform with Unix sockets — i.e.
> Linux (or WSL2). On native Windows the bot starts but `/skryba start` cannot connect to the
> recorder, so use the Docker stack there. Running `bot.py` alone is only useful for the
> pipeline/replay paths that don't capture audio.

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
