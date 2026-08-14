import { describe, expect, it } from 'vitest';
import { sample } from '@neurallab/core';
import { GENERATORS, isGeneratorKey, moons } from '../src/generators.ts';

describe('moons', () => {
  it('replays byte-for-byte from the same seed', () => {
    const a = moons({ seed: 4417 });
    const b = moons({ seed: 4417 });
    expect(Array.from(a.x)).toEqual(Array.from(b.x));
    expect(Array.from(a.y!)).toEqual(Array.from(b.y!));
  });

  it('differs for a different seed', () => {
    expect(Array.from(moons({ seed: 1 }).x)).not.toEqual(Array.from(moons({ seed: 2 }).x));
  });

  it('balances the two classes', () => {
    for (const n of [4, 40, 41, 240, 999]) {
      const ds = moons({ n });
      const upper = Array.from(ds.y!).filter((c) => c === 0).length;
      expect(Math.abs(upper - (ds.n - upper))).toBeLessThanOrEqual(1);
    }
  });

  it('puts every noiseless point on one of the two arcs', () => {
    // Noise 0 means the geometry is exactly the two circles. If an index or an offset is wrong
    // this is where it shows, rather than in a training curve four slices later.
    const ds = moons({ n: 60, noise: 0, seed: 1 });
    for (let i = 0; i < ds.n; i++) {
      const p = sample(ds, i);
      const [cx, cy] = ds.y![i] === 0 ? [0, 0] : [1, 0.5];
      const r = Math.hypot((p[0] as number) - cx, (p[1] as number) - cy);
      expect(r).toBeCloseTo(1, 5);
    }
  });

  it('is not linearly separable', () => {
    /*
     * The reason this dataset is the project's default, asserted rather than assumed.
     *
     * Slice 2 opens with challenge 1: a network with no hidden layer cannot solve this, and it
     * is supposed to visibly fail. If the geometry ever drifted — a smaller offset, a different
     * radius — the two arcs would come apart, challenge 1 would quietly start succeeding, and
     * the lesson would be gone with nothing failing to say so.
     *
     * Brute force: project onto 720 directions, and for each ask whether any threshold splits
     * the classes perfectly. A perceptron finds a separating line if one exists, so if none of
     * these directions admits one, no line does.
     */
    const ds = moons({ n: 200, noise: 0, seed: 3 });
    let separable = false;

    for (let k = 0; k < 720 && !separable; k++) {
      const angle = (k / 720) * Math.PI * 2;
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
      // A gap between the classes along this direction is a separating line.
      if (max0 < min1) separable = true;
    }

    expect(separable).toBe(false);
  });

  it('spreads further as noise rises', () => {
    const tight = spread(moons({ n: 240, noise: 0, seed: 8 }));
    const loose = spread(moons({ n: 240, noise: 0.4, seed: 8 }));
    expect(loose).toBeGreaterThan(tight);
  });

  it('survives a sample count below the minimum', () => {
    const ds = moons({ n: 1 });
    expect(ds.n).toBe(4);
    for (const v of ds.x) expect(Number.isFinite(v)).toBe(true);
  });

  it('describes itself completely', () => {
    const ds = moons();
    expect(ds.featureNames).toHaveLength(ds.dim);
    expect(ds.classNames).toHaveLength(ds.classes);
    expect(ds.x).toHaveLength(ds.n * ds.dim);
    expect(ds.y).toHaveLength(ds.n);
  });
});

describe('GENERATORS', () => {
  it('names every entry it exposes', () => {
    for (const [key, gen] of Object.entries(GENERATORS)) {
      expect(isGeneratorKey(key)).toBe(true);
      expect(gen.label.length).toBeGreaterThan(0);
      expect(gen.build({ n: 40, seed: 1 }).n).toBe(40);
    }
  });

  it('rejects a key it does not have', () => {
    expect(isGeneratorKey('spirals')).toBe(false);
    expect(isGeneratorKey('toString')).toBe(false);
  });
});

/** Mean distance from the centroid — a scalar that has to grow with noise. */
function spread(ds: { x: Float32Array; n: number; dim: number }): number {
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
