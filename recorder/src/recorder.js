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
    this.totalFrames = 0;
    this.stopped = false;
  }

  async start() {
    fs.mkdirSync(this.audioDir, { recursive: true });
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
    this.bytesPerUser.set(userId, 0);
    // Track the upstreams per user so a decoder/opus error can tear down
    // just this user's pipeline without disturbing other speakers.
    this.opusStreams.set(userId, opusStream);
    this.decoders.set(userId, decoder);

    log.info(`recorder[${this.guildId}]: first frame from ${userId} -> ${pcmPath}`);

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
      this.totalFrames += 1;
      this.bytesPerUser.set(userId, this.bytesPerUser.get(userId) + chunk.length);
      const ok = writer.write(chunk);
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
