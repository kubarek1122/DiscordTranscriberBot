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
  constructor({ client, guildId, voiceChannelId, sessionDir, onSpeakerEvent }) {
    this.client = client;
    this.guildId = guildId;
    this.voiceChannelId = voiceChannelId;
    this.sessionDir = sessionDir;
    this.audioDir = path.join(sessionDir, 'audio');
    this.onSpeakerEvent = onSpeakerEvent; // optional callback for IPC notifications

    this.conn = null;
    /** @type {Map<string, fs.WriteStream>} */
    this.writers = new Map();
    /** @type {Map<string, number>} */
    this.bytesPerUser = new Map();
    /** @type {Set<string>} */
    this.subscribedUsers = new Set();
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
    // once; if that fails too, mark the connection destroyed and let stop()
    // wrap up gracefully.
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
      }
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
      this.totalFrames += 1;
      this.bytesPerUser.set(userId, this.bytesPerUser.get(userId) + chunk.length);
      writer.write(chunk);
    });
    decoder.on('error', (e) => {
      log.warn(`recorder[${this.guildId}]: decoder error for ${userId}:`, e?.message || e);
    });
    opusStream.on('error', (e) => {
      log.warn(`recorder[${this.guildId}]: opus stream error for ${userId}:`, e?.message || e);
    });
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
