# DiscordTranscriber — Design Document

A self-hosted Discord bot that joins voice channels on demand, transcribes Polish discussions per-speaker, generates a Polish summary with key points + action items, and posts it back to the text channel the command was invoked from. Originally implemented in Python with `discord.py` + `discord-ext-voice-recv`. This document captures the design so it can be re-implemented in JavaScript with a DAVE-protocol-capable voice library.

---

## 1. Goal & contract

| Item | Value |
|---|---|
| Bot trigger | Slash commands in a guild text channel |
| Input | Live Discord voice channel audio, multiple speakers |
| Language | Polish (`pl`) |
| Output (Discord) | One message in the invocation channel containing the summary, with `transcript.txt` attached |
| Output (disk) | `summary.md`, `transcript.txt`, `actions.md`, `session.json`, per-speaker audio + segments JSON |
| Hosting | User's local machine (Windows, RTX 5060 Ti 16 GB), in Docker with NVIDIA passthrough |
| Reliability bar | Survives power loss, network outage, Discord gateway/voice disconnect, summarizer outage |

---

## 2. Slash commands

Single command group: `/skryba` ("scribe" in Polish).

| Command | Description |
|---|---|
| `/skryba start` | Bot joins the invoking user's current voice channel and begins recording. Errors if user is not in VC or recording is already active in this guild. Replies ephemerally. |
| `/skryba stop` | Stops recording, runs the pipeline (transcribe → summarize → write artifacts → post), posts the summary publicly in the channel the command was issued in. Replies ephemerally with progress. |
| `/skryba status` | Shows current state (idle / recording since X with elapsed time and known speakers). Ephemeral. |
| `/skryba jaktojestbycskryba` | Easter-egg command that posts a short comedic Polish quote (content in `src/messages.py`). |

All command descriptions and user-facing strings are in Polish.

Recording state is per-guild, stored in an in-memory map keyed by `guild_id`. Only one recording per guild at a time.

---

## 3. High-level pipeline

```
/skryba start
    │
    ▼
[Voice connect]──► [Per-speaker PCM streamed to disk, heartbeat every 10 s]
                                                │
                                  /skryba stop  │  or bot voice disconnect
                                                ▼
                                          [Finalize: PCM → WAV via ffmpeg]
                                                ▼
                                          [Whisper per speaker, segments cached]
                                                ▼
                                          [Merge into interleaved transcript]
                                                ▼
                                          [LLM summary (pluggable backend)]
                                                ▼
                                          [Write summary.md, actions.md]
                                                ▼
                                          [Post message + transcript.txt attachment]
                                                ▼
                                          [Optionally delete per-user wavs]
```

Every step writes its result to disk before advancing the session `stage`. On crash + restart, a recovery scan resumes from the last completed stage.

---

## 4. Audio capture (the part that needed DAVE)

- **Discord voice format on the wire:** Opus, encrypted (DAVE / voice gateway v8 since late 2024). The voice library must implement DAVE to decrypt before decoding.
- **After decode:** 48 kHz, stereo, signed 16-bit little-endian PCM, 20 ms frames (3840 bytes per frame).
- **One stream per user.** The voice library exposes a "sink" or event callback fired with `(user, pcmBytes)` per frame. Per-user attribution comes for free; no diarization needed.
- **Storage during capture:** append raw PCM bytes to `recordings/<guild_id>/<session>/audio/<user_id>.pcm`. One open file handle per user, kept open for the duration of the session. **No buffering in memory** — every frame written immediately so a crash never loses more than the OS write-back delay.
- **Heartbeat thread/timer (every 10 s):**
  1. `fsync` every open PCM file handle.
  2. Update `session.json.last_heartbeat`.
  3. Resolve any newly-seen user IDs to display names via the guild member cache and store in `session.json.members`.
- **On stop:** stop listening, close all file handles, finalize PCM files, then disconnect from voice.

### JavaScript notes

- `@discordjs/voice` ≥ 0.18 added DAVE support; verify the installed version actually negotiates voice gateway v8 cleanly (you should not see "extra keys: {seq: N}" warnings in logs).
- Alternative: `eris` with `@snazzah/davey` (Node.js bindings to libdave) if `@discordjs/voice` lags.
- Voice receive in `@discordjs/voice`:
  ```js
  const conn = joinVoiceChannel({ channelId, guildId, adapterCreator, selfDeaf: false, selfMute: true });
  const receiver = conn.receiver;
  receiver.speaking.on('start', (userId) => {
    const opusStream = receiver.subscribe(userId, { end: { behavior: EndBehaviorType.Manual } });
    const pcmStream = opusStream.pipe(new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 }));
    pcmStream.pipe(fs.createWriteStream(`.../audio/${userId}.pcm`, { flags: 'a' }));
  });
  ```
  Make sure `selfDeaf: false` — otherwise Discord won't send audio to the bot at all.

---

## 5. Audio finalization (PCM → WAV)

After recording stops, convert each `<user_id>.pcm` to a 16 kHz mono `<user_id>.wav` using ffmpeg (Whisper performs best at 16 kHz mono):

```
ffmpeg -hide_banner -loglevel error -y \
  -f s16le -ar 48000 -ac 2 -i <user_id>.pcm \
  -ar 16000 -ac 1 <user_id>.wav
```

Idempotent: skip if the `.wav` already exists with non-zero size, skip if the `.pcm` is empty.

---

## 6. Transcription

- **Engine:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) with model `large-v3`, `device=cuda`, `compute_type=float16`, `language=pl`.
- **VAD:** built-in Silero VAD filter enabled (`vad_filter=True`) to drop silence / breathing.
- **Beam size:** 5.
- **Per-user pass:** transcribe each `<user_id>.wav` independently. Persist each user's segments to `audio/<user_id>.segments.json` **immediately after that user finishes**, so a crash partway through doesn't redo finished users.
- **Segment shape:** `{ speaker: <display_name>, start: <sec>, end: <sec>, text: <str> }`.
- **Merge:** flatten all per-user segments into one list, sort by `start`, format each line as `[HH:MM:SS] <display_name>: <text>`. Write to `transcript.txt`.

### JavaScript notes

- Native JS Whisper options are weaker than Python's. Three viable paths:
  1. **Shell out to whisper.cpp** via `child_process.spawn`. Pre-converted 16 kHz mono WAV → JSON output with `-oj`. Most portable.
  2. **`nodejs-whisper` npm package** — thin wrapper around whisper.cpp binaries.
  3. **Keep a Python sidecar** running faster-whisper, talk to it over a Unix socket or HTTP. Best quality, more moving parts.
- For an RTX 5060 Ti, whisper.cpp with `-ngl 99` and the CUDA build is the closest equivalent to faster-whisper.

---

## 7. Summarization (pluggable backend)

The summarizer is abstracted behind a one-method interface so backends can be swapped via config without code changes:

```ts
interface Summarizer {
  name: string;
  summarize(transcript: string): Promise<string>;  // returns Polish Markdown
}
```

Three implementations:

| Backend | SDK | Default model | Notes |
|---|---|---|---|
| `claude` | `@anthropic-ai/sdk` | `claude-sonnet-4-6` | Default. Uses prompt caching: the system prompt is sent with `cache_control: { type: "ephemeral" }`. Strong on Polish. |
| `openai` | `openai` | `gpt-4o` | Drop-in alternative. |
| `ollama` | `ollama` npm package | `qwen2.5:14b-instruct` (or `qwen3.5:9b` per current `config.yaml`) | Fully offline. Quality on Polish noticeably weaker than Claude/GPT unless using a 70B+ model. |

Configuration knob: `summarizer.backend: claude | openai | ollama`. Secrets only required for the selected backend.

### Polish system prompt (verbatim — keep this exactly when porting)

The prompt is in `src/prompts.py` as `POLISH_SUMMARY_SYSTEM`. The key invariants:

- Output is **entirely in Polish**, Markdown formatted, with exactly these three section headers in this order:
  1. `## Podsumowanie` — 3–6 sentences, prose, third-person, neutral tone.
  2. `## Kluczowe punkty` — 3–10 bullet points (`-`), one full sentence each.
  3. `## Decyzje i zadania` — bullets of decisions and tasks. If an owner or deadline is in the transcript, append in parentheses. If none, the literal placeholder `_(brak)_`.
- Rules: don't invent facts; skip small talk; if the transcript is too short/unclear, say so in `Podsumowanie` and put `_(brak)_` in the other two sections; no other headers; no "Here is the summary:" preamble.
- Special trigger words in the transcript: `skrybo zapisz` / `skrybo zapamiętaj` mark the preceding or following fragment as important — the model should highlight that fragment in the summary.

The user-message template wraps the transcript between `---` delimiters with a short instruction in Polish.

### Reliability

- Summarizer call wrapped in exponential-backoff retry: **3 attempts**, delays 1 s / 4 s / 16 s.
- If all retries fail, the transcript is still on disk → session stays at `stage=transcribed`, the user can re-summarize later via the replay CLI (see §11).

---

## 8. Posting to Discord

- **Where:** the text channel from which `/skryba stop` was invoked. The channel ID was stored in `session.json.text_channel_id` at `/skryba start` time, so even if the bot restarts mid-pipeline it knows where to post.
- **Body:** the full `summary.md` content. If longer than 1900 chars (Discord's hard cap is 2000), truncate the body and attach the full `summary.md` as a file with a Polish "see attachment" note.
- **Attachments:** always attach `transcript.txt`. Attach `summary.md` if body was truncated.
- **Idempotent posting (critical):**
  - `session.json.posted_message_id` is set **only after** the Discord API call succeeds (returns 200 + message id).
  - Retry on `HTTPException` / `ConnectionClosed` / `OSError` with exponential backoff, up to **5 attempts**.
  - If all retries fail, leave `stage=summarized` so the next startup recovery scan retries.
  - If `posted_message_id` is already set on a session being processed, **skip posting** and just mark `stage=posted`. Never double-post.
- **File handles must be reopened per retry attempt** — `discord.File` (or its JS equivalent) consumes the underlying stream on first send; subsequent retries with the same object will send empty bodies.

---

## 9. Session state machine & failsafes

Each session directory has a `session.json` written atomically (write to `.tmp`, then `os.replace`). Schema:

```json
{
  "guild_id": 1437846938270564362,
  "voice_channel_id": 1437846938824343635,
  "text_channel_id": 1437846938824343636,
  "started_at": "2026-05-11T19:56:36.123456+00:00",
  "stage": "recording | recorded | transcribed | summarized | posted",
  "last_heartbeat": 1747000000.0,
  "members": { "<user_id>": "<display_name>" },
  "posted_message_id": null,
  "summarizer_backend": "claude",
  "error": null
}
```

### Stage transitions (monotonic — never go backwards in the normal path)

`recording → recorded → transcribed → summarized → posted`

`advance(stage)` is a no-op if the new stage is earlier than the current one. Each stage's outputs are persisted before the stage is advanced, so every transition is recoverable.

### Failsafe mechanisms (all in scope for the JS port)

1. **Atomic writes everywhere.** `write_atomic(path, content)`: write `path.tmp`, `flush`, `fsync`, `rename`. Use for `session.json`, `summary.md`, `transcript.txt`, `actions.md`, every `*.segments.json`. No partial files ever appear.
2. **PCM streamed to disk** during capture, fsynced every heartbeat tick.
3. **Heartbeat.** Update `last_heartbeat` every 10 s. On startup, a session with `stage=recording` whose `last_heartbeat` is older than 60 s is treated as crashed and advanced to `recorded`.
4. **Startup recovery scan.** On bot ready: walk `recordings/*/*/session.json`, find any session with `stage != "posted"`, finalize audio if needed, then run the pipeline from the last completed stage. The user gets their summary even if the machine rebooted overnight.
5. **Voice disconnect handler.** If the bot is involuntarily removed from VC mid-recording: flush in-flight PCM, mark `stage=recorded`, do **not** auto-rejoin (auto-rejoin races with users moving channels), and run the pipeline. Captured audio is safe.
6. **Idle timeout** (config knob, default 300 s): if a session sits at `stage=recording` with no audio for this long after a disconnect, auto-finalize.
7. **Container restart policy.** Docker `restart: unless-stopped` so power restoration → daemon up → bot up → recovery scan runs. Zero human action.
8. **Bounded retries.** Summarizer 3×, post 5×, both exponential backoff. Failed sessions stay on disk for manual replay.

---

## 10. On-disk layout

```
recordings/
  <guild_id>/
    <YYYY-MM-DD_HH-MM-SS>/                # session dir
      session.json
      audio/
        <user_id>.pcm                     # raw 48k stereo s16le; deleted after success
        <user_id>.wav                     # 16k mono; intermediate; deleted after success
        <user_id>.segments.json           # cached Whisper output per user
      transcript.txt                      # interleaved per-speaker, [HH:MM:SS] Name: text
      summary.md                          # full LLM output
      actions.md                          # extracted "Decyzje i zadania" section
```

`recording.keep_audio: false` (default) → delete `.pcm` and `.wav` after a successful post. Set to `true` to retain audio for re-transcription.

---

## 11. Replay CLI

`src/replay.py`:

```
python -m src.replay <session-dir> [--from-stage transcribe|summarize|post] [--repost]
```

- `--from-stage transcribe`: wipe cached segments, re-run Whisper, re-summarize, write artifacts.
- `--from-stage summarize` (default): re-run summarizer only (e.g. after switching backend in config).
- `--from-stage post`: do nothing but reset `posted_message_id=null` and `stage=summarized` — the next bot startup's recovery scan will re-post.

Replay does not connect to Discord directly — it only manipulates disk. Re-posts happen through the recovery scan.

---

## 12. Configuration

### `config.yaml`

```yaml
whisper:
  model: large-v3
  device: cuda          # cuda | cpu
  compute_type: float16 # float16 on GPU, int8 on CPU
  language: pl

summarizer:
  backend: claude       # claude | openai | ollama
  claude:
    model: claude-sonnet-4-6
    max_tokens: 4096
  openai:
    model: gpt-4o
    max_tokens: 4096
  ollama:
    host: http://localhost:11434
    model: qwen3.5:9b

recording:
  output_dir: ./recordings
  keep_audio: false
  chunk_seconds: 30
  idle_timeout_s: 300
  heartbeat_interval_s: 10
  heartbeat_stale_after_s: 60

reliability:
  summarizer_retries: 3
  post_retries: 5
```

### `.env`

```
DISCORD_TOKEN=...
ANTHROPIC_API_KEY=...    # only if backend=claude
OPENAI_API_KEY=...       # only if backend=openai
DEV_GUILD_ID=...         # optional: sync slash commands to one guild for instant updates
```

---

## 13. Docker deployment

- **Base image:** `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` for GPU-accelerated Whisper. The 5060 Ti is Blackwell (sm_120) → needs CUDA ≥ 12.8 and cuDNN 9.x.
- **Host prerequisites (Windows):** Docker Desktop with WSL2 backend; NVIDIA Container Toolkit installed in the WSL distro. Verify with `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.
- **System packages installed in image:** Python 3.11, `ffmpeg`, `libsodium23`, `libopus0`. (The opus runtime library is **required** for voice receive to decode incoming Opus → PCM; without it the decoder errors out.)
- **`docker-compose.yml`:**
  - `restart: unless-stopped`
  - Bind-mount `./recordings:/app/recordings` (state must survive container rebuild for recovery to work)
  - Bind-mount `./config.yaml:/app/config.yaml:ro`
  - Named volume `whisper-models:/root/.cache/huggingface` to avoid re-downloading `large-v3` on every rebuild
  - NVIDIA GPU reservation: `deploy.resources.reservations.devices.[driver=nvidia, count=1, capabilities=[gpu]]`
  - Optional Ollama sibling service (commented out unless `summarizer.backend=ollama`)

### Node.js port specifics

- Replace `nvidia/cuda:...` base with `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` + Node 20 (apt install via NodeSource), or use `node:20-bookworm` if you call out to a Python sidecar for Whisper.
- The `libopus0` system package is still required for the JS decoder (`@discordjs/opus` / `opusscript`).
- `prism-media` is the standard npm choice for decoding Opus → PCM stream-wise.

---

## 14. Discord bot setup checklist

1. Create application at [discord.com/developers/applications](https://discord.com/developers/applications).
2. Enable intents: **Server Members Intent**, **Voice State Intent**. Message Content intent is **not** needed (slash commands only).
3. OAuth2 scopes: `bot`, `applications.commands`.
4. Permissions: `View Channels`, `Connect`, `Speak`, `Send Messages`, `Attach Files`, `Use Voice Activity`, `Use Application Commands`.
5. Invite to the server via the generated URL.
6. Put token in `.env`.

---

## 15. Verification checklist (end-to-end)

1. **GPU visible in container:** `docker compose exec bot nvidia-smi`.
2. **Opus loaded at bot startup:** look for a log line indicating libopus loaded.
3. **Voice connect works without DAVE errors:** no `WS payload has extra keys: {'seq': N}` warnings; no `OpusError: corrupted stream`.
4. **First frame logged:** the sink logs "first frame from <user>" within ~1 s of someone speaking.
5. **Per-user PCM files grow:** during a 60 s test, each speaker's `.pcm` reaches ~11 MB.
6. **`/skryba stop` produces all four artifacts:** `session.json (stage=posted)`, `summary.md`, `transcript.txt`, `actions.md`.
7. **Summary posts in the right channel** with `transcript.txt` attached.
8. **Crash recovery:** `docker compose kill bot` mid-recording → `docker compose up -d` → recovery scan completes the session and posts.
9. **Post-failure recovery:** block Discord traffic at the OS firewall while `stage=summarized` → confirm retries fail cleanly without changing `stage`. Unblock + restart → recovery scan posts exactly once (check `posted_message_id`).
10. **Multi-speaker:** transcript shows interleaved lines with both display names and monotonic timestamps.

---

## 16. Recommended JS stack (starting point)

| Concern | Recommended package |
|---|---|
| Discord client + slash commands | `discord.js` ≥ 14.16 |
| Voice (receive, DAVE) | `@discordjs/voice` ≥ 0.18 + `@discordjs/opus` + `prism-media` |
| Audio conversion | `fluent-ffmpeg` (wraps the same ffmpeg CLI) |
| Whisper | Shell out to `whisper.cpp` with `-l pl -m ggml-large-v3.bin -ngl 99 -oj`, or `nodejs-whisper` |
| Anthropic SDK | `@anthropic-ai/sdk` |
| OpenAI SDK | `openai` |
| Ollama | `ollama` (npm) |
| Config | `dotenv` + a YAML parser like `yaml` |
| Validation | `zod` for config + session schema |
| Retries | `p-retry` or hand-rolled exponential backoff |
| Logging | `pino` |

### Critical pitfalls to verify in the JS port

- **DAVE support actually negotiated.** Confirm `@discordjs/voice` version actually handles voice gateway v8 / DAVE. Look for the absence of "extra keys: seq" warnings and the absence of opus decode errors.
- **`selfDeaf: false`** when joining VC — required for receiving audio.
- **No buffering audio in memory.** Stream per-user PCM straight to disk via `fs.createWriteStream(..., { flags: 'a' })`. fsync periodically.
- **Atomic JSON writes.** `fs.promises.writeFile(tmp, ...)` then `fs.promises.rename(tmp, final)`.
- **Reopen file handles per post retry.** A `MessagePayload.create` consuming a `Readable` stream cannot be sent twice.
- **One recording per guild.** Keep an in-memory `Map<guildId, RecordingSession>` and reject `/skryba start` if it already has an entry.
- **Heartbeat + recovery scan.** Without these the bot is fragile to power loss; they are not optional.
