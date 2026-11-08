/** Same sfc32/splitmix32 streams as the pinned simulator, with explicit state. */
export interface RngState { a: number; b: number; c: number; d: number }

export class Rng {
  private s: RngState;
  constructor(seed: number) {
    let x = seed >>> 0;
    const split = () => {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s = { a: split(), b: split(), c: split(), d: split() };
    for (let i = 0; i < 12; i++) this.nextU32();
  }
  static restore(s: RngState): Rng {
    const rng = new Rng(0);
    rng.s = { ...s };
    return rng;
  }
  snapshot(): RngState { return { ...this.s }; }
  nextU32(): number {
    const s = this.s;
    const t = (((s.a + s.b) >>> 0) + s.d) >>> 0;
    s.d = (s.d + 1) >>> 0;
    s.a = s.b ^ (s.b >>> 9);
    s.b = (s.c + (s.c << 3)) >>> 0;
    s.c = ((s.c << 21) | (s.c >>> 11)) >>> 0;
    s.c = (s.c + t) >>> 0;
    return t;
  }
  next(): number { return this.nextU32() / 4294967296; }
  int(n: number): number {
    if (!Number.isInteger(n) || n <= 0) throw new Error(`Invalid RNG bound: ${n}`);
    return Math.floor(this.next() * n);
  }
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr;
  }
  fork(salt: number): Rng {
    return new Rng((this.nextU32() ^ Math.imul(salt + 1, 0x9e3779b1)) >>> 0);
  }
}
