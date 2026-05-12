// Unix-socket IPC server. Newline-delimited JSON.
//
// Python → Node:
//   {"op":"join","guild_id":"...","voice_channel_id":"...","session_dir":"..."}
//   {"op":"leave","guild_id":"..."}
//
// Node → Python:
//   {"op":"joined","guild_id":"..."}
//   {"op":"left","guild_id":"...","stats":{...}}
//   {"op":"speaker","guild_id":"...","user_id":"...","display_name":"..."}
//   {"op":"error","guild_id":"...","error":"..."}
//   {"op":"hello","version":1}

import net from 'node:net';
import fs from 'node:fs';
import path from 'node:path';
import { log } from './log.js';

export const PROTOCOL_VERSION = 1;

export function startIpcServer({ socketPath, handlers }) {
  const dir = path.dirname(socketPath);
  fs.mkdirSync(dir, { recursive: true });
  try { fs.unlinkSync(socketPath); } catch (_) { /* not present */ }

  let activeSocket = null;

  const server = net.createServer((sock) => {
    if (activeSocket) {
      log.warn('ipc: rejecting second connection');
      sock.end(JSON.stringify({ op: 'error', error: 'already connected' }) + '\n');
      return;
    }
    activeSocket = sock;
    log.info('ipc: python connected');

    const send = (msg) => {
      if (!sock.writable) return;
      sock.write(JSON.stringify(msg) + '\n');
    };
    send({ op: 'hello', version: PROTOCOL_VERSION });

    let buf = '';
    sock.on('data', async (chunk) => {
      buf += chunk.toString('utf8');
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let msg;
        try {
          msg = JSON.parse(line);
        } catch (e) {
          send({ op: 'error', error: `invalid json: ${e.message}` });
          continue;
        }
        try {
          if (msg.op === 'join') {
            const res = await handlers.onJoin({ ...msg, send });
            send(res);
          } else if (msg.op === 'leave') {
            const res = await handlers.onLeave({ ...msg, send });
            send(res);
          } else {
            send({ op: 'error', error: `unknown op: ${msg.op}` });
          }
        } catch (e) {
          log.error('ipc: handler threw', e);
          send({
            op: 'error',
            guild_id: msg.guild_id,
            error: e?.message || String(e),
          });
        }
      }
    });

    const cleanup = (reason) => {
      if (activeSocket !== sock) return;
      activeSocket = null;
      log.warn(`ipc: python disconnected (${reason})`);
      handlers.onDisconnect?.();
    };

    sock.on('close', () => cleanup('close'));
    sock.on('error', (e) => {
      log.warn('ipc: socket error', e);
      cleanup('error');
    });
  });

  server.listen(socketPath, () => {
    try { fs.chmodSync(socketPath, 0o660); } catch (_) { /* best-effort */ }
    log.info(`ipc: listening on ${socketPath}`);
  });

  server.on('error', (e) => log.error('ipc: server error', e));

  return server;
}
