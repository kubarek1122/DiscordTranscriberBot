from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class RecorderError(RuntimeError):
    pass


class RecorderClient:
    """Async client for the Node voice-receive sidecar over a Unix socket.

    Newline-delimited JSON. One client per recording session — open() before
    a session starts, close() after it ends. The sidecar enforces single
    connection, so callers must not interleave sessions on one client.
    """

    def __init__(self, socket_path: Path | str) -> None:
        self._socket_path = str(socket_path)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # Three streams of inbound messages, separated so callers can wait on
        # exactly what they care about:
        #   _reply_q     — command replies + hello + bad-protocol errors
        #   _unsolicited — `speaker` events (informational)
        #   _alerts      — `recording_failed` and other "you should react now"
        #                  pushes from the sidecar
        self._unsolicited: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._alerts: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._hello_seen = False
        self._reply_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def open(self) -> None:
        log.info("recorder-client: connecting to %s", self._socket_path)
        self._reader, self._writer = await asyncio.open_unix_connection(
            path=self._socket_path
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        # First inbound message should be {"op":"hello","version":1}; wait
        # briefly so we surface "wrong protocol" early.
        try:
            msg = await asyncio.wait_for(self._reply_q.get(), timeout=5.0)
        except asyncio.TimeoutError as e:
            raise RecorderError("recorder did not send hello within 5s") from e
        if msg.get("op") != "hello":
            raise RecorderError(f"expected hello, got {msg}")
        self._hello_seen = True
        log.info("recorder-client: handshake ok (version=%s)", msg.get("version"))

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    log.warning("recorder-client: bad json: %r", line)
                    continue
                op = msg.get("op")
                # Asynchronous notifications go on side queues so they don't
                # get confused with command replies.
                if op == "speaker":
                    await self._unsolicited.put(msg)
                elif op == "recording_failed":
                    log.warning(
                        "recorder-client: sidecar reported recording_failed: %s",
                        msg,
                    )
                    await self._alerts.put(msg)
                else:
                    await self._reply_q.put(msg)
        except Exception:
            log.exception("recorder-client: read loop crashed")
        finally:
            # Wake up any pending RPCs with a synthetic error so they don't
            # hang forever if the sidecar dies.
            await self._reply_q.put(
                {"op": "error", "error": "recorder connection closed"}
            )

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._writer is None:
            raise RecorderError("recorder-client: not connected")
        line = (json.dumps(payload) + "\n").encode("utf-8")
        self._writer.write(line)
        await self._writer.drain()

    async def _request(
        self,
        payload: dict[str, Any],
        *,
        expect_op: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        await self._send(payload)
        try:
            reply = await asyncio.wait_for(self._reply_q.get(), timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise RecorderError(
                f"recorder did not reply to {payload.get('op')} within {timeout_s}s"
            ) from e
        if reply.get("op") == "error":
            raise RecorderError(reply.get("error") or "unknown recorder error")
        if reply.get("op") != expect_op:
            raise RecorderError(f"expected {expect_op}, got {reply}")
        return reply

    async def join(
        self,
        *,
        guild_id: int,
        voice_channel_id: int,
        session_dir: Path,
        timeout_s: float,
    ) -> dict[str, Any]:
        return await self._request(
            {
                "op": "join",
                "guild_id": str(guild_id),
                "voice_channel_id": str(voice_channel_id),
                "session_dir": str(session_dir),
            },
            expect_op="joined",
            timeout_s=timeout_s,
        )

    async def leave(self, *, guild_id: int, timeout_s: float) -> dict[str, Any]:
        return await self._request(
            {"op": "leave", "guild_id": str(guild_id)},
            expect_op="left",
            timeout_s=timeout_s,
        )

    async def speaker_events(self) -> asyncio.Queue[dict[str, Any]]:
        """Caller can `await client.speaker_events().get()` in a background task
        to learn of newly-detected speakers (informational only)."""
        return self._unsolicited

    async def alerts(self) -> asyncio.Queue[dict[str, Any]]:
        """Caller can `await client.alerts().get()` to learn of unrecoverable
        sidecar events that warrant ending the current session early
        (e.g. recording_failed)."""
        return self._alerts

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
