import { describe, expect, it } from 'vitest';
import { Rng } from '../src/rng.ts';
import {
  bounds2d,
  classCounts,
  fitStandardiser,
  sample,
  split,
  standardise,
  type Dataset,
} from '../src/dataset.ts';

/** A tiny labelled set with a deliberately lopsided class balance: 12 of class 0, 4 of class 1. */
function lopsided(): Dataset {
  const n = 16;
  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    x[i * 2] = i;
    x[i * 2 + 1] = i * 2;
    y[i] = i < 12 ? 0 : 1;
  }
  return {
    name: 'lopsided',
    x, y, n, dim: 2, classes: 2,
    featureNames: ['a', 'b'],
    classNames: ['many', 'few'],
  };
}

describe('split', () => {
  it('partitions every sample exactly once', () => {
    const ds = lopsided();
    const { train, val } = split(ds, 0.7, new Rng(1));
    const all = [...train, ...val].sort((a, b) => a - b);
    expect(all).toEqual(Array.from({ length: ds.n }, (_, i) => i));
  });

  it('stratifies, so a rare class reaches both sides', () => {
    /*
     * The reason this is not a plain shuffle-and-slice. With 12/4 and a 70% cut, taking the
     * first 70% of one shuffle can easily hand validation zero samples of the rare class —
     * and its recall is then undefined while accuracy looks fine. Challenge 8 asks a reader to
     * trust the validation curve, so the split under it has to be sound.
     */
    for (let seed = 0; seed < 50; seed++) {
      const ds = lopsided();
      const { train, val } = split(ds, 0.7, new Rng(seed));
      const inTrain = train.filter((i) => ds.y![i] === 1).length;
      const inVal = val.filter((i) => ds.y![i] === 1).length;
      expect(inTrain).toBeGreaterThan(0);
      expect(inVal).toBeGreaterThan(0);
    }
  });

  it('keeps proportions close to the requested fraction', () => {
    const ds = lopsided();
    const { train } = split(ds, 0.75, new Rng(2));
    expect(train.length / ds.n).toBeGreaterThan(0.6);
    expect(train.length / ds.n).toBeLessThan(0.9);
  });

  it('handles an unlabelled set as one bucket', () => {
    const base = lopsided();
    const ds: Dataset = { ...base, y: null, classes: 0, classNames: [] };
    const { train, val } = split(ds, 0.7, new Rng(3));
    expect(train.length + val.length).toBe(ds.n);
    expect(val.length).toBeGreaterThan(0);
  });

  it('rejects a fraction that would empty a side', () => {
    expect(() => split(lopsided(), 0, new Rng(1))).toThrow();
    expect(() => split(lopsided(), 1, new Rng(1))).toThrow();
  });
});

describe('standardiser', () => {
  it('is fitted over the given rows only', () => {
    /*
     * The leak test. Feature `a` runs 0..15 over the whole set; fitted over rows 0..3 its mean
     * must be 1.5, not 7.5. Fitting over everything and splitting afterwards is the classic
     * quiet mistake, and it makes every validation number in the project slightly too good.
     */
    const ds = lopsided();
    const s = fitStandardiser(ds, Int32Array.from([0, 1, 2, 3]));
    expect(s.mean[0]).toBeCloseTo(1.5, 6);
    expect(s.mean[1]).toBeCloseTo(3.0, 6);
  });

  it('produces zero mean and unit sd on the rows it was fitted to', () => {
    const ds = lopsided();
    const rows = Int32Array.from({ length: ds.n }, (_, i) => i);
    const z = standardise(ds, fitStandardiser(ds, rows));
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < z.n; i++) {
      const v = sample(z, i)[0] as number;
      sum += v;
      sumSq += v * v;
    }
    expect(sum / z.n).toBeCloseTo(0, 5);
    expect(Math.sqrt(sumSq / z.n)).toBeCloseTo(1, 5);
  });

  it('survives a constant feature instead of producing NaN', () => {
    // A zero-spread column divided by its own sd is NaN, and one NaN reaches every weight in a
    // single forward pass. Much easier to stop here than to trace back from there.
    const n = 4;
    const ds: Dataset = {
      name: 'flat',
      x: Float32Array.from([1, 5, 1, 5, 1, 5, 1, 5]),
      y: null, n, dim: 2, classes: 0,
      featureNames: ['constant', 'alsoConstant'],
      classNames: [],
    };
    const s = fitStandardiser(ds, Int32Array.from([0, 1, 2, 3]));
    expect(s.sd[0]).toBe(1);
    const z = standardise(ds, s);
    for (const v of z.x) expect(Number.isFinite(v)).toBe(true);
  });

  it('leaves the original untouched', () => {
    const ds = lopsided();
    const before = Float32Array.from(ds.x);
    standardise(ds, fitStandardiser(ds, Int32Array.from([0, 1, 2])));
    expect(Array.from(ds.x)).toEqual(Array.from(before));
  });
});

describe('bounds2d', () => {
  it('measures the real extent', () => {
    const box = bounds2d(lopsided());
    expect(box.minX).toBe(0);
    expect(box.maxX).toBe(15);
    expect(box.minY).toBe(0);
    expect(box.maxY).toBe(30);
  });

  it('widens a degenerate axis so the box is still drawable', () => {
    const ds: Dataset = {
      name: 'line', x: Float32Array.from([1, 0, 1, 1, 1, 2]), y: null,
      n: 3, dim: 2, classes: 0, featureNames: ['x', 'y'], classNames: [],
    };
    const box = bounds2d(ds);
    expect(box.maxX - box.minX).toBeGreaterThan(0.5);
  });
});

describe('classCounts', () => {
  it('counts each class', () => {
    expect(Array.from(classCounts(lopsided()))).toEqual([12, 4]);
  });
});
