import { describe, expect, it } from 'vitest';
import { Rng } from '../src/rng.ts';

describe('Rng', () => {
  /**
   * The golden vector.
   *
   * These six integers are the foundation of every reproducibility claim in the project: the
   * weights a network initialises with, the order a batch is shuffled into, and the points a
   * dataset generates all descend from this sequence. If this test fails, every stored run and
   * every pinned loss number has been invalidated — that is a decision to make deliberately and
   * record in the commit message, not a number to update until the test goes green.
   *
   * Exact integer arithmetic throughout, so this holds on every engine. The engine-scoped
   * caveat in §4 applies to `normal()` and to nothing else here.
   */
  it('produces its pinned sequence for seed 4417', () => {
    const rng = new Rng(4417);
    expect(Array.from({ length: 6 }, () => rng.u32())).toEqual([
      55303081, 2544064971, 1253366596, 4207743915, 143669195, 1700000921,
    ]);
  });

  it('replays identically from the same seed', () => {
    const a = new Rng(99);
    const b = new Rng(99);
    for (let i = 0; i < 1000; i++) expect(a.u32()).toBe(b.u32());
  });

  it('diverges immediately for adjacent seeds', () => {
    // Seeds are set by a slider, so 4417 and 4418 are one drag apart. A generator whose
    // low-order seed bits barely perturb the stream would make "try another seed" useless.
    const a = new Rng(4417);
    const b = new Rng(4418);
    expect(a.u32()).not.toBe(b.u32());
  });

  it('stays inside [0, 1) over a long run', () => {
    const rng = new Rng(7);
    for (let i = 0; i < 200_000; i++) {
      const v = rng.float();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('spreads uniformly across ten buckets', () => {
    const rng = new Rng(11);
    const buckets = new Array<number>(10).fill(0);
    const draws = 100_000;
    for (let i = 0; i < draws; i++) buckets[Math.floor(rng.float() * 10)]!++;
    for (const count of buckets) expect(Math.abs(count - draws / 10)).toBeLessThan(draws * 0.01);
  });

  it('keeps int(n) inside range', () => {
    const rng = new Rng(3);
    for (let i = 0; i < 50_000; i++) {
      const v = rng.int(7);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(7);
      expect(Number.isInteger(v)).toBe(true);
    }
  });

  it('draws a standard normal', () => {
    // He and Glorot initialisation both scale this, so a mis-scaled normal would show up as
    // every network in the project being hard to train, with nothing obviously wrong.
    const rng = new Rng(5);
    const n = 200_000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const v = rng.normal();
      sum += v;
      sumSq += v * v;
    }
    const mean = sum / n;
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(Math.sqrt(sumSq / n - mean * mean) - 1)).toBeLessThan(0.01);
  });

  it('shuffles into a permutation, in place', () => {
    const rng = new Rng(13);
    const a = Int32Array.from({ length: 500 }, (_, i) => i);
    rng.shuffle(a);
    expect(Array.from(a).sort((p, q) => p - q)).toEqual(Array.from({ length: 500 }, (_, i) => i));
    // And it actually moved things — an identity "shuffle" would pass the check above.
    expect(Array.from(a)).not.toEqual(Array.from({ length: 500 }, (_, i) => i));
  });

  it('shuffles every position, not just the tail', () => {
    // A Fisher–Yates written with the loop bound one out leaves index 0 fixed forever. The
    // symptom is a batch whose first sample never changes, which is invisible in a loss curve.
    const seen = new Set<number>();
    for (let trial = 0; trial < 200; trial++) {
      const a = Int32Array.from({ length: 8 }, (_, i) => i);
      new Rng(trial).shuffle(a);
      seen.add(a[0] as number);
    }
    expect(seen.size).toBe(8);
  });
});
