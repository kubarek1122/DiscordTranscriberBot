import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planPad, BYTES_PER_MS, SAMPLE_ALIGN } from '../src/timeline.js';

const SAFETY = 1500;

test('burst start pads file up to the burst onset (initial join silence)', () => {
  const pad = planPad({
    nowMs: 1830,
    fileBytes: 0,
    lastFrameMs: undefined,
    needPad: true,
    burstAnchorMs: 1830,
    safetyMs: SAFETY,
  });
  assert.equal(pad, 1830 * BYTES_PER_MS);
  assert.equal(pad % SAMPLE_ALIGN, 0);
});

test('in-burst frame ~20 ms after the previous one inserts no silence', () => {
  const pad = planPad({
    nowMs: 2020,
    fileBytes: 2000 * BYTES_PER_MS,
    lastFrameMs: 2000,
    needPad: false,
    burstAnchorMs: 1830,
    safetyMs: SAFETY,
  });
  assert.equal(pad, 0);
});

test('sparse DTX delivery (~220 ms gap, < safety) is NOT shredded', () => {
  // This is the exact case the old code mishandled: a sparse frame mid-utterance.
  const pad = planPad({
    nowMs: 2220,
    fileBytes: 2000 * BYTES_PER_MS,
    lastFrameMs: 2000,
    needPad: false,
    burstAnchorMs: 1830,
    safetyMs: SAFETY,
  });
  assert.equal(pad, 0);
});

test('long unmarked pause (> safety) re-anchors and pads the real gap', () => {
  // Discord never emitted speaking "end"; a 2 s gap must become real silence.
  const pad = planPad({
    nowMs: 4000,
    fileBytes: 2000 * BYTES_PER_MS,
    lastFrameMs: 2000,
    needPad: false,
    burstAnchorMs: 0,
    safetyMs: SAFETY,
  });
  assert.equal(pad, (4000 - 2000) * BYTES_PER_MS);
});

test('never returns negative when the file is already past the anchor', () => {
  const pad = planPad({
    nowMs: 1000,
    fileBytes: 300000, // 300000 / 192 ≈ 1562 ms, already past a 1000 ms anchor
    lastFrameMs: undefined,
    needPad: true,
    burstAnchorMs: 1000,
    safetyMs: SAFETY,
  });
  assert.equal(pad, 0);
});

test('pad is always a whole number of sample-aligned bytes', () => {
  const pad = planPad({
    nowMs: 1000,
    fileBytes: 2, // force a non-aligned raw difference
    lastFrameMs: undefined,
    needPad: true,
    burstAnchorMs: 1000,
    safetyMs: SAFETY,
  });
  assert.equal(pad % SAMPLE_ALIGN, 0);
  assert.equal(pad, 1000 * BYTES_PER_MS - 2 - ((1000 * BYTES_PER_MS - 2) % SAMPLE_ALIGN));
});
