// Regression test for the backpressure bug that compressed the session
// timeline: the old padding loop wrote one ~200 ms chunk and dropped the rest
// of a gap the moment `writer.write()` returned false, so multi-second silences
// collapsed and every speaker's audio slid toward t=0. `UserPcmWriter` must now
// drain the full silence debt (and the frame) across `drain` events.

import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';

import { UserPcmWriter, SILENCE_CHUNK_BYTES } from '../src/pump.js';

// A WriteStream stand-in with a real high-water mark: write() returns false
// once the in-flight buffer exceeds `hwm`, and `flush()` simulates the OS
// draining that buffer and emitting 'drain'. Counts every byte handed to it.
function fakeWriter(hwm = 16384) {
  const w = new EventEmitter();
  w.written = 0;
  w._buffered = 0;
  w.write = (buf) => {
    w.written += buf.length;
    w._buffered += buf.length;
    return w._buffered < hwm;
  };
  w.flush = () => {
    w._buffered = 0;
    w.emit('drain');
  };
  return w;
}

function fakeSource() {
  const d = { _paused: false };
  d.pause = () => { d._paused = true; };
  d.resume = () => { d._paused = false; };
  d.isPaused = () => d._paused;
  return d;
}

function makeWriter(extra = {}) {
  const writer = fakeWriter();
  const source = fakeSource();
  let fileBytes = 0;
  const pump = new UserPcmWriter({
    writer,
    source,
    silence: Buffer.alloc(SILENCE_CHUNK_BYTES),
    onBytes: (n) => { fileBytes += n; },
    isStopped: () => false,
    ...extra,
  });
  return { pump, writer, source, getFileBytes: () => fileBytes };
}

// Drive simulated drains until the flush is idle (or a guard trips).
function pumpToIdle(source, writer) {
  let guard = 0;
  while (source.isPaused() && guard++ < 100000) writer.flush();
  assert.ok(guard < 100000, 'flush did not converge');
}

test('long silence debt is written in full under backpressure', () => {
  const { pump, writer, source, getFileBytes } = makeWriter();
  const debt = 500_000; // ~2.6 s of 48 kHz stereo silence — many chunks
  const frame = Buffer.alloc(3840); // one 20 ms stereo frame

  pump.enqueue(debt, frame);
  pumpToIdle(source, writer);

  assert.equal(pump.debt, 0, 'all silence debt drained');
  assert.equal(pump.frame, null, 'frame written');
  assert.equal(writer.written, debt + frame.length, 'every byte landed, none dropped');
  assert.equal(getFileBytes(), debt + frame.length, 'onBytes tracks actual writes');
  assert.equal(source.isPaused(), false, 'source resumed once idle');
});

test('a short pad that fits the buffer needs no drain cycle', () => {
  const { pump, writer, source } = makeWriter();
  const frame = Buffer.alloc(3840);

  pump.enqueue(0, frame);

  assert.equal(writer.written, frame.length);
  assert.equal(source.isPaused(), false, 'never paused for a sub-hwm write');
});

test('silence is written before the frame, exactly once', () => {
  // Capture write order: every silence chunk must precede the frame so the
  // file stays aligned (offset == session time).
  const writer = fakeWriter();
  const source = fakeSource();
  const order = [];
  const frame = Buffer.alloc(3840);
  const pump = new UserPcmWriter({
    writer,
    source,
    silence: Buffer.alloc(SILENCE_CHUNK_BYTES),
  });
  const realWrite = writer.write;
  writer.write = (buf) => {
    order.push(buf === frame ? 'frame' : 'silence');
    return realWrite(buf);
  };

  pump.enqueue(SILENCE_CHUNK_BYTES * 3, frame);
  pumpToIdle(source, writer);

  assert.equal(order.filter((x) => x === 'frame').length, 1, 'frame written once');
  assert.equal(order[order.length - 1], 'frame', 'frame written last');
  assert.ok(order.slice(0, -1).every((x) => x === 'silence'), 'all padding precedes the frame');
});

test('dispose() drops the queue and ignores a late drain', () => {
  const { pump, writer, source, getFileBytes } = makeWriter();
  pump.enqueue(500_000, Buffer.alloc(3840));
  const writtenBeforeDispose = writer.written;
  assert.ok(source.isPaused(), 'backed up and paused mid-pad');

  pump.dispose();
  writer.flush(); // a late drain after teardown must not write anything

  assert.equal(pump.debt, 0);
  assert.equal(pump.frame, null);
  assert.equal(writer.written, writtenBeforeDispose, 'no writes after dispose');
  assert.ok(getFileBytes() <= writtenBeforeDispose);
});
