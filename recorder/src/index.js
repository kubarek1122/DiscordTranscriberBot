// Entry point. Logs in as the recorder bot, opens the Unix-socket IPC server,
// dispatches join/leave messages from the Python bot to RecordingSession.

import { Client, GatewayIntentBits, Partials } from 'discord.js';
import { startIpcServer } from './ipc.js';
import { RecordingSession } from './recorder.js';
import { log } from './log.js';

const SOCKET_PATH = process.env.IPC_SOCKET || '/tmp/skryba/recorder.sock';
const TOKEN = process.env.RECORDER_DISCORD_TOKEN;

if (!TOKEN) {
  log.error('RECORDER_DISCORD_TOKEN is not set');
  process.exit(2);
}

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMembers, // resolve display names for the speaker event
  ],
  partials: [Partials.GuildMember],
});

/** @type {Map<string, RecordingSession>} guild_id -> session */
const sessions = new Map();

// `clientReady` replaces the deprecated `ready` event from discord.js
// v14.18 onward; v15 will remove the legacy name entirely.
client.once('clientReady', () => {
  log.info(`logged in as ${client.user?.tag} (id=${client.user?.id})`);
});

client.on('error', (e) => log.error('client error', e));

async function onJoin({ guild_id, voice_channel_id, session_dir, send }) {
  if (!guild_id || !voice_channel_id || !session_dir) {
    return { op: 'error', error: 'join: missing required fields' };
  }
  if (sessions.has(guild_id)) {
    return { op: 'error', guild_id, error: 'already recording in this guild' };
  }
  const session = new RecordingSession({
    client,
    guildId: guild_id,
    voiceChannelId: voice_channel_id,
    sessionDir: session_dir,
    onSpeakerEvent: ({ user_id, display_name }) => {
      send({ op: 'speaker', guild_id, user_id, display_name });
    },
    onHardFailure: ({ reason }) => {
      // Tell Python the recording was cut short by something Discord-side
      // (kicked, channel deleted, voice gateway gave up). Python decides
      // whether to keep the PCM and run the pipeline anyway.
      send({ op: 'recording_failed', guild_id, reason });
    },
  });
  try {
    await session.start();
  } catch (e) {
    log.error(`join[${guild_id}] failed`, e);
    return { op: 'error', guild_id, error: e?.message || String(e) };
  }
  sessions.set(guild_id, session);
  return { op: 'joined', guild_id };
}

async function onLeave({ guild_id }) {
  const session = sessions.get(guild_id);
  if (!session) {
    return { op: 'error', guild_id, error: 'no active session' };
  }
  const stats = await session.stop();
  sessions.delete(guild_id);
  return { op: 'left', guild_id, stats };
}

async function onDisconnect() {
  // Python lost the IPC connection (crash, restart). Tear every session
  // down cleanly so audio that was captured isn't lost. Python's recovery
  // scan will pick the sessions up via their session.json on its next start.
  for (const [gid, session] of [...sessions]) {
    try {
      await session.stop();
    } catch (e) {
      log.warn(`stop on disconnect failed for ${gid}`, e);
    }
    sessions.delete(gid);
  }
}

// Wire up IPC first so we surface bind errors immediately;
// only then attempt to log in.
startIpcServer({
  socketPath: SOCKET_PATH,
  handlers: { onJoin, onLeave, onDisconnect },
});

await client.login(TOKEN);

// Graceful shutdown: SIGTERM from `docker compose stop` etc.
const shutdown = async (sig) => {
  log.warn(`received ${sig}, shutting down`);
  await onDisconnect();
  try { await client.destroy(); } catch { /* ignore */ }
  process.exit(0);
};
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
