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

// Reusable zero-filled buffer for filling inter-utterance silence gaps in
// the per-user PCM streams. Discord's voice gateway only delivers RTP
// frames while a user is actively speaking, so without this padding each
// .pcm file would be a concatenation of speech bursts with no
// representation of the silence between them — destroying session
// timing. 200 ms keeps the buffer small enough that backpressure can kick
// in mid-pad on a long silence; for short tails we `.subarray()`.
const PCM_SILENCE_CHUNK = Buffer.alloc(BYTES_PER_MS * 200);

export class RecordingSession {
  constructor({ client, guildId, voiceChannelId, sessionDir, onSpeakerEvent, onHardFailure }) {
    this.client = client;
    this.guildId = guildId;
    this.voiceChannelId = voiceChannelId;
    this.sessionDir = sessionDir;
    this.audioDir = path.join(sessionDir, 'audio');
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
    this.totalFrames = 0;
    this.stopped = false;
    // Captured in start() so every chunk's pad-bytes can be derived from
    // (now - sessionStartMs). The byte offset N in <user>.pcm corresponds
    // to session time N / BYTES_PER_MS — making file-local time and
    // session wall-clock time equal, which is what `transcribe_session`
    // assumes when sorting segments.
    this.sessionStartMs = 0;
  }

  async start() {
    fs.mkdirSync(this.audioDir, { recursive: true });
    this.sessionStartMs = Date.now();
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
    await entersState(this.conn, VoiceConnectionStatus.Ready, 15_000);

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

      // Pad with silence so the file's byte position matches session time.
      // The chunk represents 20 ms of audio ending at `wallMs`; we want
      // the file to be exactly `wallMs` ms long after this chunk lands.
      const wallMs = Date.now() - this.sessionStartMs;
      const targetBytes = wallMs * BYTES_PER_MS;
      const currentBytes = this.fileBytes.get(userId) || 0;
      let padBytes = targetBytes - chunk.length - currentBytes;
      // During active speech, frames arrive every ~20 ms and padBytes is
      // ~0 (modulo jitter); after a silence it's the full pause length.
      // Clamp negatives — over-arrival is harmless, we just skip padding.
      while (padBytes > 0) {
        const slice =
          padBytes >= PCM_SILENCE_CHUNK.length
            ? PCM_SILENCE_CHUNK
            : PCM_SILENCE_CHUNK.subarray(0, padBytes);
        const okPad = writer.write(slice);
        this.fileBytes.set(
          userId,
          (this.fileBytes.get(userId) || 0) + slice.length,
        );
        padBytes -= slice.length;
        if (!okPad) {
          // Backpressure mid-pad: stop padding for now, let drain re-fire
          // the decoder. The remaining pad will be re-computed on the
          // next data event (which will see a still-large wallMs gap).
          break;
        }
      }

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
