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
import { planPad, BYTES_PER_MS, SAMPLE_ALIGN } from './timeline.js';

// How long to wait for the voice connection to reach Ready before declaring
// the join failed. The Python side's `recorder.join_timeout_s` MUST stay
// strictly above this (+ fetch overhead) or it orphans sessions on slow joins.
const READY_TIMEOUT_MS = 15_000;

// How often the no-audio watchdog checks for inactivity, in ms.
const IDLE_CHECK_INTERVAL_MS = 10_000;

// Silence placement is driven by Discord's `speaking` start/end VAD events
// (see _onSpeakingStart / _onSpeakingEnd): we pad the real gap at the start of
// each speech burst, then write that burst's frames back-to-back. This is a
// pure safety net for the case where Discord never emits `speaking 'end'`
// (lenient VAD, held-open mic): a wall-clock gap between consecutive frames
// larger than this forces a re-anchor so a genuine multi-second pause isn't
// concatenated away. Kept generous so ordinary sparse/DTX delivery within an
// utterance (~150–250 ms inter-frame) is NEVER mistaken for a pause — that
// false re-anchoring is exactly what used to shred speech into a
// 20 ms-on / 200 ms-off stutter.
const SAFETY_RESYNC_MS = 1500;

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
     * Session-time (ms) of the last frame written for each user. Feeds the
     * SAFETY_RESYNC_MS fallback in `planPad`.
     * @type {Map<string, number>}
     */
    this.lastFrameMs = new Map();
    /**
     * Per-user speech-burst state, driven by Discord's `speaking` events:
     *   isSpeaking   — currently inside a Discord-detected speech period
     *   burstAnchorMs — session-time the current burst began (its true onset)
     *   needPad      — a burst just started; pad the real silence before its
     *                  first decoded frame, then concatenate the rest
     * @type {Map<string, boolean>} */
    this.isSpeaking = new Map();
    /** @type {Map<string, number>} */
    this.burstAnchorMs = new Map();
    /** @type {Map<string, boolean>} */
    this.needPad = new Map();
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
    this.conn.receiver.speaking.on('end', (userId) => this._onSpeakingEnd(userId));

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

    // Mark the start of a speech burst (Discord's own VAD). Runs on EVERY
    // start — even when already subscribed — so the next decoded frame pads
    // the real silence since the previous burst, then frames concatenate.
    // The pad is deferred to the first frame, so a spurious start with no
    // audio inserts nothing.
    if (!this.isSpeaking.get(userId)) {
      this.isSpeaking.set(userId, true);
      this.burstAnchorMs.set(userId, Math.max(0, Date.now() - this.sessionStartMs));
      this.needPad.set(userId, true);
    }

    // Build the opus→decoder→writer pipeline once per user (kept alive for the
    // whole session via EndBehaviorType.Manual).
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

      // How much silence (if any) to insert before this frame so the file
      // stays aligned to session time. Pads at burst starts and across long
      // unmarked pauses; returns 0 for normal in-burst frames (incl. sparse
      // DTX delivery) so continuous speech is never shredded. See planPad.
      let padBytes = planPad({
        nowMs,
        fileBytes: this.fileBytes.get(userId) || 0,
        lastFrameMs: this.lastFrameMs.get(userId),
        needPad: this.needPad.get(userId) === true,
        burstAnchorMs: this.burstAnchorMs.get(userId) || 0,
        safetyMs: SAFETY_RESYNC_MS,
      });
      this.needPad.set(userId, false);
      while (padBytes > 0) {
        const n = Math.min(padBytes, PCM_SILENCE_CHUNK.length);
        const slice =
          n === PCM_SILENCE_CHUNK.length
            ? PCM_SILENCE_CHUNK
            : PCM_SILENCE_CHUNK.subarray(0, n);
        const okPad = writer.write(slice);
        this.fileBytes.set(userId, (this.fileBytes.get(userId) || 0) + n);
        padBytes -= n;
        // Backpressure mid-pad: stop here. The frame write below pauses the
        // decoder and resumes on drain (the frame is still buffered, not
        // lost). Any unwritten pad is reclaimed at the next burst's re-anchor.
        if (!okPad) break;
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

  _onSpeakingEnd(userId) {
    if (this.stopped) return;
    // Discord's VAD says this user stopped. Close the burst; the silence until
    // the next burst is padded lazily when that burst's first frame arrives.
    this.isSpeaking.set(userId, false);
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
    // Drop per-user burst state so the next `speaking start` after a rejoin
    // re-anchors cleanly (pads the real away-gap before its first frame).
    this.lastFrameMs.delete(userId);
    this.isSpeaking.delete(userId);
    this.burstAnchorMs.delete(userId);
    this.needPad.delete(userId);
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
