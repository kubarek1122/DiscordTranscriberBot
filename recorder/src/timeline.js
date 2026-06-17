// Pure timeline arithmetic for the recorder, isolated so it can be unit-tested
// without a live Discord voice connection. This is the logic that decides how
// much silence to insert before a decoded frame — the part that historically
// shredded audio when it mis-fired.

// 48 kHz × 2 channels × 2 bytes = 192 bytes per millisecond of s16le PCM.
export const BYTES_PER_MS = 192;

// One stereo s16 sample pair = 4 bytes. All writes (frames and padding) MUST
// be a multiple of this or the L/R interleaving shifts and the rest of the
// file decodes as static.
export const SAMPLE_ALIGN = 4;

/**
 * Decide how many bytes of silence to insert before writing the current
 * decoded frame, so that byte offset N in a user's .pcm maps to session time
 * `N / BYTES_PER_MS` ms (the invariant `transcribe_session` relies on).
 *
 * Pure: all state is passed in. Returns a non-negative, sample-aligned count.
 *
 *  - `needPad` (a Discord `speaking` burst just started): align the file to
 *    the burst's true onset (`burstAnchorMs`), inserting the real silence
 *    since the previous burst. This is the normal, authoritative path.
 *  - otherwise, if the gap since the previous frame exceeds `safetyMs`: a long
 *    real pause for which Discord never emitted `speaking 'end'` — re-anchor to
 *    now so we don't compress a multi-second silence into nothing.
 *  - otherwise: a continuation of the current burst (including sparse DTX
 *    delivery) — write the frame back-to-back with NO padding. This is what
 *    prevents the "20 ms speech / 200 ms silence" shredding.
 *
 * @param {object} p
 * @param {number} p.nowMs                session-time of this frame (ms)
 * @param {number} p.fileBytes            bytes already written to the file
 * @param {number|undefined} p.lastFrameMs session-time of the previous frame
 * @param {boolean} p.needPad             a burst just started
 * @param {number} p.burstAnchorMs        session-time the burst began
 * @param {number} p.safetyMs             max intra-burst gap before a re-anchor
 * @param {number} [p.bytesPerMs]
 * @param {number} [p.sampleAlign]
 * @returns {number} pad bytes (>= 0, multiple of sampleAlign)
 */
export function planPad({
  nowMs,
  fileBytes,
  lastFrameMs,
  needPad,
  burstAnchorMs,
  safetyMs,
  bytesPerMs = BYTES_PER_MS,
  sampleAlign = SAMPLE_ALIGN,
}) {
  let anchorMs;
  if (needPad) {
    anchorMs = burstAnchorMs;
  } else if (lastFrameMs !== undefined && nowMs - lastFrameMs > safetyMs) {
    anchorMs = nowMs;
  } else {
    return 0; // continuation of the burst — concatenate, no silence
  }
  let padBytes = Math.floor(anchorMs * bytesPerMs) - fileBytes;
  if (padBytes <= 0) return 0; // audio already at/past the anchor; never negative
  padBytes -= padBytes % sampleAlign;
  return padBytes;
}
