import { describe, expect, it } from 'vitest';
import { DIGITS_PER_CLASS, DIGITS_TOTAL, digits } from '../src/digits.ts';

/**
 * The one hand-checked test that matters for this file: the base64 blob has to round-trip to the
 * *exact* bytes of a real, independently-verifiable source row, not merely be internally
 * self-consistent. `KNOWN_ROW` is the literal first line of the UCI `optdigits.tra` file this
 * project's data was built from — copied by hand from the source, not derived from `digits.ts`
 * itself, so a bug that corrupted every row identically (an off-by-one in the packing loop, a
 * wrong stride) would still be caught here even though every *other* check in this file would
 * keep passing.
 */
const KNOWN_ROW = [
  0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11, 2, 0, 0, 5, 16, 3, 0, 5,
  7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0, 0, 0, 0, 6, 14,
  7, 1, 0, 0,
];
const KNOWN_LABEL = 0;

describe('digits', () => {
  it('round-trips the known first row of the source file byte-for-byte', () => {
    const ds = digits({ n: DIGITS_TOTAL });
    let found = -1;
    for (let i = 0; i < ds.n; i++) {
      if ((ds.y as Int32Array)[i] !== KNOWN_LABEL) continue;
      let matches = true;
      for (let k = 0; k < 64; k++) {
        if (ds.x[i * 64 + k] !== KNOWN_ROW[k]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        found = i;
        break;
      }
    }
    expect(found).toBeGreaterThanOrEqual(0);
  });

  it('is exactly balanced — 120 of each digit — when every row is requested', () => {
    const ds = digits({ n: DIGITS_TOTAL });
    const counts = new Array(10).fill(0);
    for (let i = 0; i < ds.n; i++) counts[(ds.y as Int32Array)[i] as number]++;
    for (const c of counts) expect(c).toBe(DIGITS_PER_CLASS);
  });

  it('is deterministic in its seed', () => {
    const a = digits({ n: 200, seed: 7 });
    const b = digits({ n: 200, seed: 7 });
    expect(a.x).toEqual(b.x);
    expect(a.y).toEqual(b.y);
  });

  it('a different seed reorders the rows', () => {
    const a = digits({ n: 200, seed: 1 });
    const b = digits({ n: 200, seed: 2 });
    expect(a.x).not.toEqual(b.x);
  });

  it('clamps n defensively rather than throwing', () => {
    expect(digits({ n: 999999 }).n).toBe(DIGITS_TOTAL);
    expect(digits({ n: 0 }).n).toBeGreaterThan(0);
    expect(digits({ n: -50 }).n).toBeGreaterThan(0);
  });

  it('describes its own shape correctly', () => {
    const ds = digits({ n: 50 });
    expect(ds.dim).toBe(64);
    expect(ds.classes).toBe(10);
    expect(ds.featureNames).toHaveLength(64);
    expect(ds.classNames).toEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
    expect(ds.name).toBe('Digits');
  });

  it('accepts (and ignores) noise, the same GeneratorOptions shape every generator takes', () => {
    expect(() => digits({ n: 20, noise: 0.9, seed: 1 })).not.toThrow();
  });
});
