// One recording session = one voice channel join. Streams per-user PCM to
// disk via fs.createWriteStream(...) appended (flags: 'a') so a crash never
// loses more than the OS write-back delay.

import fs from 'node:fs';
import path from 'node:path';
import {
  joinVoiceChannel,
  EndBehaviorType,
  VoiceConnectionStatus,
  entersState,
} from '@discordjs/voice';
import prism from 'prism-media';
import { log } from './log.js';

// 48 kHz × 2 channels × 2 bytes = 192 bytes per millisecond of s16le PCM.
const BYTES_PER_MS = 192;

// How long to wait for the voice connection to reach Ready before declaring
// the join failed. The Python side's `recorder.join_timeout_s` MUST stay
// strictly above this (+ fetch overhead) or it orphans sessions on slow joins.
const READY_TIMEOUT_MS = 15_000;

// How often the no-audio watchdog checks for inactivity, in ms.
const IDLE_CHECK_INTERVAL_MS = 10_000;

// One stereo s16 sample pair = 4 bytes. All writes (frames and padding)
// MUST be a multiple of this or the L/R interleaving shifts and the rest
// of the file decodes as static.
const SAMPLE_ALIGN = 4;

// Burst boundary. Discord delivers a user's opus frames sparsely and
// bursty — even continuous speech can arrive as clusters of frames spaced
// 150–250 ms apart in wall-clock (bad mics, sender-side VAD, packet loss,
// jitter). We must NOT place each frame at its absolute wall-clock offset,
// or those inter-frame gaps get backfilled with silence and continuous
// speech is stretched into a choppy frame+silence stutter.
//
// Instead: frames arriving within BURST_GAP_MS of the previous one are
// treated as the same utterance and written back-to-back (smooth). Only a
// gap LARGER than this is a real pause — there we re-anchor the file to the
// new frame's absolute session time, which (a) inserts genuine silence and
// (b) keeps cross-speaker ordering correct at turn granularity.
//
// Tuning: conversational turn-taking pauses are typically well above this;
// within-utterance delivery gaps are below it. 600 ms is a safe middle.
const BURST_GAP_MS = 600;

// Reusable zero-filled buffer for filling inter-utterance silence gaps in
// the per-user PCM streams. Discord's voice gateway only delivers RTP
// frames while a user is actively speaking, so without this padding each
// .pcm file would be a concatenation of speech bursts with no
// representation of the silence between them — destroying session
// timing. 200 ms keeps the buffer small enough that backpressure can kick
// in mid-pad on a long silence; for short tails we `.subarray()`.
const PCM_SILENCE_CHUNK = Buffer.alloc(BYTES_PER_MS * 200);

export class RecordingSession {
  constructor({ client, guildId, voiceChannelId, sessionDir, idleTimeoutS, onSpeakerEvent, onHardFailure }) {
    this.client = client;
    this.guildId = guildId;
    this.voiceChannelId = voiceChannelId;
    this.sessionDir = sessionDir;
    this.audioDir = path.join(sessionDir, 'audio');
    this.idleTimeoutS = idleTimeoutS || 0; // 0 disables the no-audio watchdog
    this.onSpeakerEvent = onSpeakerEvent; // optional callback for IPC notifications
    this.onHardFailure = onHardFailure;   // optional callback ({reason}) on unrecoverable voice failure
    this.failureReported = false;         // dedupe: emit recording_failed at most once

    this.conn = null;
    /** @type {Map<string, fs.WriteStream>} */
    this.writers = new Map();
    /** @type {Map<string, number>} */
    this.bytesPerUser = new Map();
    /** @type {Set<string>} */
    this.subscribedUsers = new Set();
    /** @type {Map<string, import('stream').Readable>} */
    this.opusStreams = new Map();
    /** @type {Map<string, prism.opus.Decoder>} */
    this.decoders = new Map();
    /**
     * Total bytes (audio + silence padding) written to each user's PCM file.
     * Distinct from `bytesPerUser`, which counts only real audio for stats.
     * We use this to compute "where in the session does this file currently
     * end?" so the next chunk's silence pad is correct, even after a
     * decoder error / teardown / rebuild cycle (T1f).
     * @type {Map<string, number>}
     */
    this.fileBytes = new Map();
    /**
     * Wall-clock time (ms since session start) of the last frame written for
     * each user. Used to decide whether the next frame continues the current
     * burst (concatenate, smooth) or starts a new one after a real pause
     * (re-anchor to absolute session time). See BURST_GAP_MS.
     * @type {Map<string, number>}
     */
    this.lastFrameMs = new Map();
    this.totalFrames = 0;
    this.stopped = false;
    // Captured in start() so every chunk's pad-bytes can be derived from
    // (now - sessionStartMs). The byte offset N in <user>.pcm corresponds
    // to session time N / BYTES_PER_MS — making file-local time and
    // session wall-clock time equal, which is what `transcribe_session`
    // assumes when sorting segments.
    this.sessionStartMs = 0;
    // Wall-clock (ms) of the most recent decoded frame from ANY user. Drives
    // the no-audio watchdog; initialized to session start so a session that
    // never receives a single frame still times out.
    this.lastAnyFrameMs = 0;
    /** @type {NodeJS.Timeout | null} */
    this._idleTimer = null;
  }

  async start() {
    fs.mkdirSync(this.audioDir, { recursive: true });
    this.sessionStartMs = Date.now();
    this.lastAnyFrameMs = this.sessionStartMs;
    const guild = await this.client.guilds.fetch(this.guildId);
    const channel = await guild.channels.fetch(this.voiceChannelId);
    if (!channel?.isVoiceBased()) {
      throw new Error(`channel ${this.voiceChannelId} is not a voice channel`);
    }

    this.conn = joinVoiceChannel({
      channelId: this.voiceChannelId,
      guildId: this.guildId,
      adapterCreator: guild.voiceAdapterCreator,
      selfDeaf: false,   // required: deafened bots receive no audio
      selfMute: true,
    });

    // joinVoiceChannel resolves quickly; wait for the connection to be ready
    // before declaring success so the IPC ack means audio will actually flow.
    await entersState(this.conn, VoiceConnectionStatus.Ready, READY_TIMEOUT_MS);

    this._armIdleWatchdog();

    this.conn.receiver.speaking.on('start', (userId) => this._onSpeakingStart(userId));

    // When a user leaves the recorded VC, Discord frees their SSRC. Our
    // existing opus subscription is bound to that dead SSRC, so it just
    // goes silent — and when they rejoin (or move back) Discord assigns
    // a fresh SSRC, but `_onSpeakingStart` would early-return because the
    // userId is still in `subscribedUsers`. Result: post-rejoin audio is
    // silently dropped.
    //
    // Tear down on leave so the next `speaking.start` rebuilds against
    // the new SSRC. `fileBytes` survives teardown (see
    // `_teardownUserStream`), so the next chunk's pad fills the gap
    // between when they left and when they spoke again.
    this._onVoiceStateUpdate = (oldState, newState) => {
      if (this.stopped) return;
      // Ignore the recorder bot's own moves (we destroy/recreate the
      // voice connection ourselves; that's not a "user left" event).
      const selfId = this.client.user?.id;
      if (selfId && newState.id === selfId) return;
      const wasHere = oldState.channelId === this.voiceChannelId;
      const stillHere = newState.channelId === this.voiceChannelId;
      if (wasHere && !stillHere) {
        const userId = newState.id;
        if (this.subscribedUsers.has(userId)) {
          log.info(
            `recorder[${this.guildId}]: ${userId} left VC, tearing down stream so next speech re-subscribes`,
          );
          this._teardownUserStream(userId);
        }
      }
    };
    this.client.on('voiceStateUpdate', this._onVoiceStateUpdate);

    // If we get unexpectedly disconnected mid-recording, try to reconnect
    // once; if that fails too, mark the connection destroyed and notify
    // Python so the session can be wrapped up on the parent side.
    this.conn.on(VoiceConnectionStatus.Disconnected, async () => {
      if (this.stopped) return;
      try {
        await Promise.race([
          entersState(this.conn, VoiceConnectionStatus.Signalling, 5_000),
          entersState(this.conn, VoiceConnectionStatus.Connecting, 5_000),
        ]);
        log.info(`recorder[${this.guildId}]: reconnecting`);
      } catch {
        log.warn(`recorder[${this.guildId}]: unrecoverable disconnect, destroying`);
        try { this.conn.destroy(); } catch { /* ignore */ }
        this._reportHardFailure('voice_disconnect');
      }
    });

    // Discord can revoke the voice session (kick from VC, channel deletion,
    // permission change). The Destroyed state fires for both manual and
    // forced teardowns — only treat as hard failure if we didn't initiate it.
    this.conn.on(VoiceConnectionStatus.Destroyed, () => {
      if (this.stopped) return;
      log.warn(`recorder[${this.guildId}]: voice connection destroyed externally`);
      this._reportHardFailure('voice_destroyed');
    });

    log.info(
      `recorder[${this.guildId}]: joined voice channel ${this.voiceChannelId}, writing to ${this.audioDir}`,
    );
  }

  _onSpeakingStart(userId) {
    if (this.stopped) return;
    if (this.subscribedUsers.has(userId)) return;
    this.subscribedUsers.add(userId);

    const opusStream = this.conn.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.Manual },
    });
    const decoder = new prism.opus.Decoder({
      rate: 48000,
      channels: 2,
      frameSize: 960,
    });

    const pcmPath = path.join(this.audioDir, `${userId}.pcm`);
    const writer = fs.createWriteStream(pcmPath, { flags: 'a' });
    this.writers.set(userId, writer);
    if (!this.bytesPerUser.has(userId)) {
      this.bytesPerUser.set(userId, 0);
    }
    // After a decoder-error teardown (T1f) the file may already exist; pick
    // up the byte count from disk so the next pad-then-write resumes from
    // the right session offset instead of double-padding from 0.
    if (!this.fileBytes.has(userId)) {
      let existing = 0;
      try {
        existing = fs.statSync(pcmPath).size;
      } catch (_) {
        // File doesn't exist yet on first subscribe — that's fine.
      }
      this.fileBytes.set(userId, existing);
    }
    // Track the upstreams per user so a decoder/opus error can tear down
    // just this user's pipeline without disturbing other speakers.
    this.opusStreams.set(userId, opusStream);
    this.decoders.set(userId, decoder);

    const offsetMs = Math.max(0, Date.now() - this.sessionStartMs);
    log.info(
      `recorder[${this.guildId}]: first frame from ${userId} (+${offsetMs}ms) -> ${pcmPath}`,
    );

    if (this.onSpeakerEvent) {
      this._resolveDisplayName(userId).then((displayName) => {
        try {
          this.onSpeakerEvent({
            user_id: userId,
            display_name: displayName,
          });
        } catch { /* best-effort */ }
      });
    }

    opusStream.pipe(decoder);
    decoder.on('data', (chunk) => {
      if (this.stopped) return;
      // The writer may have been torn down by an error handler between two
      // frames; check before touching it.
      if (!this.writers.has(userId)) return;

      this.lastAnyFrameMs = Date.now(); // feed the no-audio watchdog
      const nowMs = Date.now() - this.sessionStartMs;
      const last = this.lastFrameMs.get(userId);

      // Decide whether this frame starts a NEW burst. The very first frame
      // for a user (last === undefined), or a frame arriving more than
      // BURST_GAP_MS after the previous one, re-anchors the file to absolute
      // session time. Frames within a burst are written back-to-back so
      // sparse/bursty delivery of continuous speech isn't stretched into a
      // stutter (see BURST_GAP_MS).
      const newBurst = last === undefined || nowMs - last > BURST_GAP_MS;
      if (newBurst) {
        const fileBytes = this.fileBytes.get(userId) || 0;
        // Pad with silence so this burst begins at its true session offset.
        let padBytes = Math.floor(nowMs * BYTES_PER_MS) - fileBytes;
        padBytes -= padBytes % SAMPLE_ALIGN; // keep stereo pairs intact
        while (padBytes > 0) {
          const n = Math.min(padBytes, PCM_SILENCE_CHUNK.length);
          const slice =
            n === PCM_SILENCE_CHUNK.length
              ? PCM_SILENCE_CHUNK
              : PCM_SILENCE_CHUNK.subarray(0, n);
          const okPad = writer.write(slice);
          this.fileBytes.set(userId, (this.fileBytes.get(userId) || 0) + n);
          padBytes -= n;
          // Backpressure mid-pad: stop here. The frame write below pauses
          // the decoder and resumes on drain; the frame is still buffered
          // (not lost). The remaining pad is recomputed next data event.
          if (!okPad) break;
        }
      }
      this.lastFrameMs.set(userId, nowMs);

      this.totalFrames += 1;
      this.bytesPerUser.set(userId, this.bytesPerUser.get(userId) + chunk.length);
      const ok = writer.write(chunk);
      this.fileBytes.set(
        userId,
        (this.fileBytes.get(userId) || 0) + chunk.length,
      );
      if (!ok) {
        // Disk pressure or slow fs — pause the decoder so PCM doesn't pile
        // up in the Node heap until OOM. Resume once the OS write buffer
        // has drained.
        decoder.pause();
        writer.once('drain', () => {
          if (!this.stopped && this.decoders.get(userId) === decoder) {
            decoder.resume();
          }
        });
      }
    });
    decoder.on('error', (e) => {
      log.warn(
        `recorder[${this.guildId}]: decoder error for ${userId}: ${e?.message || e}; tearing down user stream`,
      );
      this._teardownUserStream(userId);
    });
    opusStream.on('error', (e) => {
      log.warn(
        `recorder[${this.guildId}]: opus stream error for ${userId}: ${e?.message || e}; tearing down user stream`,
      );
      this._teardownUserStream(userId);
    });
  }

  /**
   * Arm the no-audio watchdog: if no decoded frame arrives from ANY user for
   * `idleTimeoutS` seconds, report a hard failure so Python finalizes the
   * session on whatever PCM was captured. A no-op when idleTimeoutS <= 0.
   */
  _armIdleWatchdog() {
    if (this.idleTimeoutS <= 0) return;
    const idleMs = this.idleTimeoutS * 1000;
    this._idleTimer = setInterval(() => {
      if (this.stopped) return;
      if (Date.now() - this.lastAnyFrameMs > idleMs) {
        log.warn(
          `recorder[${this.guildId}]: no audio for ${this.idleTimeoutS}s, reporting idle_timeout`,
        );
        this._clearIdleWatchdog();
        this._reportHardFailure('idle_timeout');
      }
    }, IDLE_CHECK_INTERVAL_MS);
    // Don't let the watchdog keep the event loop alive on its own.
    this._idleTimer.unref?.();
  }

  _clearIdleWatchdog() {
    if (this._idleTimer) {
      clearInterval(this._idleTimer);
      this._idleTimer = null;
    }
  }

  _reportHardFailure(reason) {
    if (this.failureReported || this.stopped) return;
    this.failureReported = true;
    if (this.onHardFailure) {
      try {
        this.onHardFailure({ reason });
      } catch (e) {
        log.warn(`recorder[${this.guildId}]: onHardFailure callback threw:`, e?.message || e);
      }
    }
  }

  /**
   * Tear down one user's opus→decoder→writer chain after an error so that
   * (a) the writer is closed and the PCM file isn't left half-open, and
   * (b) the next `speaking start` event for the same user creates a fresh
   * pipeline. Idempotent.
   */
  _teardownUserStream(userId) {
    const opusStream = this.opusStreams?.get(userId);
    const decoder = this.decoders?.get(userId);
    const writer = this.writers.get(userId);
    try { opusStream?.unpipe?.(decoder); } catch { /* ignore */ }
    try { opusStream?.destroy?.(); } catch { /* ignore */ }
    try { decoder?.destroy?.(); } catch { /* ignore */ }
    if (writer) {
      try { writer.end(); } catch { /* ignore */ }
    }
    this.opusStreams?.delete(userId);
    this.decoders?.delete(userId);
    this.writers.delete(userId);
    this.subscribedUsers.delete(userId);
    // Drop lastFrameMs so the next frame after a rejoin is treated as a new
    // burst and re-anchored to absolute session time (it would be anyway,
    // since the gap exceeds BURST_GAP_MS — but this keeps intent explicit).
    this.lastFrameMs.delete(userId);
    // Intentionally NOT deleting fileBytes / bytesPerUser: the file
    // still exists on disk and the next `speaking start` for this user
    // must resume padding from the current file end, not from 0.
  }

  async _resolveDisplayName(userId) {
    try {
      const guild = await this.client.guilds.fetch(this.guildId);
      const member = await guild.members.fetch(userId);
      return member.displayName || member.user.username;
    } catch {
      return `User${userId}`;
    }
  }

  async stop() {
    if (this.stopped) return this._stats();
    this.stopped = true;
    this._clearIdleWatchdog();

    // Detach the long-lived client listener so finished sessions don't
    // leak references back to the discord.js Client.
    if (this._onVoiceStateUpdate) {
      try {
        this.client.off('voiceStateUpdate', this._onVoiceStateUpdate);
      } catch { /* ignore */ }
      this._onVoiceStateUpdate = null;
    }

    // Tear down subscriptions
    for (const userId of this.subscribedUsers) {
      try {
        const sub = this.conn?.receiver?.subscriptions?.get?.(userId);
        sub?.destroy?.();
      } catch { /* ignore */ }
    }

    // Close all writers and fsync them
    const closes = [];
    for (const [userId, writer] of this.writers) {
      closes.push(new Promise((resolve) => {
        writer.end(() => {
          try {
            fs.fsyncSync(writer.fd);
          } catch { /* fd may already be closed */ }
          resolve();
        });
      }));
    }
    await Promise.all(closes);

    try { this.conn?.destroy(); } catch { /* ignore */ }
    log.info(
      `recorder[${this.guildId}]: stopped (total_frames=${this.totalFrames}, users_with_audio=${this.writers.size})`,
    );
    return this._stats();
  }

  _stats() {
    const bytes = {};
    for (const [k, v] of this.bytesPerUser) bytes[k] = v;
    return {
      total_frames: this.totalFrames,
      users_with_audio: this.writers.size,
      bytes_per_user: bytes,
    };
  }
}
