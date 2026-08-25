/**
 * The five generators added in slice 3, and the properties that make each one worth having.
 *
 * `moons` keeps its own file from slice 0. These are checked for the same things: they replay,
 * they balance their classes, and — where it is the reason the set exists — the geometry is
 * asserted rather than assumed.
 */

import { describe, expect, it } from 'vitest';
import { sample, type Dataset } from '@neurallab/core';
import { GENERATORS, blobs, checkerboard, circles, spirals, xor } from '../src/generators.ts';

/** Brute force: is there any direction along which a threshold separates the classes? */
function linearlySeparable(ds: Dataset, directions = 720): boolean {
  for (let k = 0; k < directions; k++) {
    const angle = (k / directions) * Math.PI * 2;
    const ux = Math.cos(angle);
    const uy = Math.sin(angle);
    let max0 = -Infinity;
    let min1 = Infinity;
    for (let i = 0; i < ds.n; i++) {
      const p = sample(ds, i);
      const proj = (p[0] as number) * ux + (p[1] as number) * uy;
      if (ds.y![i] === 0) max0 = Math.max(max0, proj);
      else min1 = Math.min(min1, proj);
    }
    if (max0 < min1) return true;
  }
  return false;
}

describe('every generator', () => {
  it('replays byte-for-byte from the same seed', () => {
    for (const [key, gen] of Object.entries(GENERATORS)) {
      const a = gen.build({ n: 120, noise: 0.15, seed: 4417 });
      const b = gen.build({ n: 120, noise: 0.15, seed: 4417 });
      expect(Array.from(a.x), key).toEqual(Array.from(b.x));
      expect(Array.from(a.y!), key).toEqual(Array.from(b.y!));
    }
  });

  it('describes itself completely', () => {
    for (const [key, gen] of Object.entries(GENERATORS)) {
      const ds = gen.build({ n: 120, seed: 1 });
      expect(ds.featureNames, key).toHaveLength(ds.dim);
      expect(ds.classNames, key).toHaveLength(ds.classes);
      expect(ds.x, key).toHaveLength(ds.n * ds.dim);
      expect(ds.y, key).toHaveLength(ds.n);
      expect(ds.name.length, key).toBeGreaterThan(0);
    }
  });

  it('produces only finite coordinates and valid labels', () => {
    for (const [key, gen] of Object.entries(GENERATORS)) {
      for (const n of [4, 41, 500]) {
        const ds = gen.build({ n, noise: 0.6, seed: 3 });
        for (const v of ds.x) expect(Number.isFinite(v), key).toBe(true);
        for (const c of ds.y!) {
          expect(c, key).toBeGreaterThanOrEqual(0);
          expect(c, key).toBeLessThan(ds.classes);
        }
      }
    }
  });

  it('uses every class it declares', () => {
    // A set that names three classes and produces two makes the output panel draw a dead bar
    // and the softmax carry an output that can never be right.
    for (const [key, gen] of Object.entries(GENERATORS)) {
      const ds = gen.build({ n: 240, seed: 5 });
      const seen = new Set(Array.from(ds.y!));
      expect(seen.size, key).toBe(ds.classes);
    }
  });

  it('spreads further as noise rises', () => {
    // Digits is excluded on purpose, not loosened around: it is real pixel data with nothing to
    // jitter, `noise` is accepted and ignored (see digits.ts's own comment on why), and asserting
    // "more noise spreads it further" against a parameter the generator never reads would be
    // asserting something that was never true rather than relaxing something that was.
    for (const [key, gen] of Object.entries(GENERATORS)) {
      if (key === 'digits') continue;
      const tight = spread(gen.build({ n: 240, noise: 0, seed: 8 }));
      const loose = spread(gen.build({ n: 240, noise: 0.5, seed: 8 }));
      expect(loose, key).toBeGreaterThan(tight);
    }
  });
});

describe('circles', () => {
  it('is not linearly separable', () => {
    // The reason this set exists: no half-plane encloses anything. Moons can be separated by a
    // sufficiently bent line and a reader might not notice the bend is essential.
    expect(linearlySeparable(circles({ n: 200, noise: 0, seed: 2 }))).toBe(false);
  });

  it('puts each class on its own radius', () => {
    const ds = circles({ n: 200, noise: 0, seed: 2 });
    for (let i = 0; i < ds.n; i++) {
      const p = sample(ds, i);
      const r = Math.hypot(p[0] as number, p[1] as number);
      expect(r).toBeCloseTo(ds.y![i] === 0 ? 0.4 : 1, 5);
    }
  });
});

describe('blobs', () => {
  it('has three classes, evenly filled', () => {
    const ds = blobs({ n: 240, seed: 1 });
    expect(ds.classes).toBe(3);
    const counts = [0, 0, 0];
    for (const c of ds.y!) counts[c] = (counts[c] as number) + 1;
    for (const n of counts) expect(n).toBe(80);
  });

  it('separates its clusters at the default noise', () => {
    // Not a hard requirement of the maths, but if the blobs overlap at default settings the set
    // stops being the easy multi-class case it is there to be.
    const ds = blobs({ n: 240, noise: 0.15, seed: 1 });
    // Flat, two entries per class — the same layout the rest of the project uses, and it keeps
    // `noUncheckedIndexedAccess` from turning a centroid into four assertions.
    const sums = new Float64Array(6);
    const counts = new Int32Array(3);
    for (let i = 0; i < ds.n; i++) {
      const p = sample(ds, i);
      const c = ds.y![i] as number;
      sums[c * 2] = (sums[c * 2] as number) + (p[0] as number);
      sums[c * 2 + 1] = (sums[c * 2 + 1] as number) + (p[1] as number);
      counts[c] = (counts[c] as number) + 1;
    }
    for (let c = 0; c < 3; c++) {
      sums[c * 2] = (sums[c * 2] as number) / (counts[c] as number);
      sums[c * 2 + 1] = (sums[c * 2 + 1] as number) / (counts[c] as number);
    }
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const dx = (sums[a * 2] as number) - (sums[b * 2] as number);
        const dy = (sums[a * 2 + 1] as number) - (sums[b * 2 + 1] as number);
        expect(Math.hypot(dx, dy), `blob ${a} vs ${b}`).toBeGreaterThan(1);
      }
    }
  });
});

describe('checkerboard', () => {
  it('labels by the cell a point started in', () => {
    /*
     * The class comes from the pre-jitter cell. Computed after the jitter the boundaries would
     * be perfectly sharp at any noise, and the noise slider would do nothing visible — which is
     * the wrong lesson about noise.
     */
    const ds = checkerboard({ n: 400, noise: 0, seed: 4 });
    for (let i = 0; i < ds.n; i++) {
      const p = sample(ds, i);
      const expected = (((Math.floor(p[0] as number) + Math.floor(p[1] as number)) % 2) + 2) % 2;
      expect(ds.y![i]).toBe(expected);
    }
  });

  it('blurs its boundaries when noise is added', () => {
    // The same check as above, but now some points *should* disagree with their own cell.
    const ds = checkerboard({ n: 400, noise: 0.3, seed: 4 });
    let crossed = 0;
    for (let i = 0; i < ds.n; i++) {
      const p = sample(ds, i);
      const cell = (((Math.floor(p[0] as number) + Math.floor(p[1] as number)) % 2) + 2) % 2;
      if (cell !== ds.y![i]) crossed++;
    }
    expect(crossed).toBeGreaterThan(0);
  });

  it('is not linearly separable', () => {
    expect(linearlySeparable(checkerboard({ n: 300, noise: 0, seed: 6 }))).toBe(false);
  });
});

describe('spirals', () => {
  it('is not linearly separable', () => {
    expect(linearlySeparable(spirals({ n: 200, noise: 0, seed: 7 }))).toBe(false);
  });

  it('winds outward from the centre', () => {
    const ds = spirals({ n: 200, noise: 0, seed: 7 });
    const first = sample(ds, 0);
    const last = sample(ds, Math.floor(ds.n / 2) - 1);
    expect(Math.hypot(first[0] as number, first[1] as number)).toBeLessThan(0.1);
    expect(Math.hypot(last[0] as number, last[1] as number)).toBeCloseTo(1, 1);
  });

  it('puts the two arms opposite each other', () => {
    // Half a turn apart at every radius. If the phase offset were lost the two arms would sit
    // on top of one another and the set would be unlearnable rather than hard.
    const ds = spirals({ n: 200, noise: 0, seed: 7 });
    const half = Math.floor(ds.n / 2);
    const a = sample(ds, 10);
    const b = sample(ds, half + 10);
    expect(a[0]).toBeCloseTo(-(b[0] as number), 4);
    expect(a[1]).toBeCloseTo(-(b[1] as number), 4);
  });
});

describe('xor', () => {
  it('is not linearly separable', () => {
    expect(linearlySeparable(xor({ n: 200, noise: 0, seed: 1 }))).toBe(false);
  });
});

function spread(ds: Dataset): number {
  let mx = 0;
  let my = 0;
  for (let i = 0; i < ds.n; i++) {
    mx += ds.x[i * 2] as number;
    my += ds.x[i * 2 + 1] as number;
  }
  mx /= ds.n;
  my /= ds.n;
  let total = 0;
  for (let i = 0; i < ds.n; i++) {
    total += Math.hypot((ds.x[i * 2] as number) - mx, (ds.x[i * 2 + 1] as number) - my);
  }
  return total / ds.n;
}
