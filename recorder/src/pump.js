// Per-user PCM writer. Drains pending silence padding (in chunk-sized pieces)
// followed by one decoded frame to a Node writable, honoring stream
// backpressure so a long inter-burst gap is materialized in full instead of
// being dropped on the first `write() === false`. Isolated from recorder.js
// (no Discord deps) so it is unit-testable with plain stream fakes, like
// timeline.js.
//
// The regression this guards against: the old in-line loop did `if (!ok)
// break;` mid-pad and never wrote the rest, so multi-second silences collapsed
// to ~200 ms, the session timeline compressed (a 50-min call became ~18 min),
// and every speaker's audio slid toward t=0 — fusing distinct speakers into a
// false overlap.

// 200 ms of 48 kHz stereo s16le PCM. Big enough that disk I/O dominates, small
// enough that backpressure can pace a long pad without one giant buffered write.
export const SILENCE_CHUNK_BYTES = 192 * 200;

export class UserPcmWriter {
  /**
   * @param {object} o
   * @param {{write(buf:Buffer):boolean, once(ev:string,cb:()=>void):void}} o.writer
   * @param {{pause():void, resume():void, isPaused():boolean}} o.source
   *   The paced upstream (the opus decoder). Paused while the writer drains so
   *   PCM doesn't pile up in the heap; resumed once the queue is empty.
   * @param {Buffer} o.silence reusable zero-filled buffer; its length is the pad chunk size.
   * @param {(n:number)=>void} [o.onBytes] called with bytes actually written (to advance fileBytes).
   * @param {()=>boolean} [o.isStopped] bail predicate; when true, no further writes.
   */
  constructor({ writer, source, silence, onBytes, isStopped }) {
    this.writer = writer;
    this.source = source;
    this.silence = silence;
    this.onBytes = onBytes || (() => {});
    this.isStopped = isStopped || (() => false);
    this.debt = 0; // silence bytes still owed before the frame
    this.frame = null; // the single decoded frame awaiting write
    this._draining = false; // a one-shot 'drain' listener is registered
    this._disposed = false; // pipeline torn down; ignore late drains
  }

  /**
   * Queue `padBytes` of silence (written first, to stay aligned to session
   * time) then `frame`, and flush. At most one frame is ever outstanding: the
   * caller pauses its source via us the moment a write backs up, so a new frame
   * cannot arrive while one is pending.
   * @param {number} padBytes
   * @param {Buffer} frame
   */
  enqueue(padBytes, frame) {
    if (this._disposed) return;
    if (padBytes > 0) this.debt += padBytes;
    this.frame = frame;
    this._flush();
  }

  _flush() {
    if (this._disposed || this.isStopped()) return;
    const { writer, silence } = this;

    // 1) Silence debt first.
    while (this.debt > 0) {
      const n = Math.min(this.debt, silence.length);
      const slice = n === silence.length ? silence : silence.subarray(0, n);
      const ok = writer.write(slice);
      this.debt -= n;
      this.onBytes(n);
      if (!ok) return this._pauseUntilDrain();
    }

    // 2) Then the frame.
    if (this.frame) {
      const f = this.frame;
      this.frame = null;
      const ok = writer.write(f);
      this.onBytes(f.length);
      if (!ok) return this._pauseUntilDrain();
    }

    // Queue fully flushed — let the source flow again.
    if (this.source.isPaused()) this.source.resume();
  }

  _pauseUntilDrain() {
    if (!this.source.isPaused()) this.source.pause();
    if (this._draining) return; // only one listener at a time
    this._draining = true;
    this.writer.once('drain', () => {
      this._draining = false;
      if (this._disposed || this.isStopped()) return;
      this._flush();
    });
  }

  /** Abandon the queue after a teardown so a late 'drain' can't write to a
   * closed writer. The file end (tracked via onBytes) is accurate for what
   * actually landed, so a rebuilt pipeline re-pads the remaining gap. */
  dispose() {
    this._disposed = true;
    this.debt = 0;
    this.frame = null;
  }
}
