import { describe, expect, it } from 'vitest';
import { Rng } from '@neurallab/core';
import {
  alphaAt,
  bmu,
  bmu2,
  createSom,
  neighbourhoodStrength,
  sigmaAt,
  type Schedule,
} from '../src/som.ts';

describe('createSom', () => {
  it('sizes every buffer from cols, rows and dim', () => {
    const som = createSom(4, 3, 5, 'hex', new Rng(1));
    expect(som.W).toHaveLength(4 * 3 * 5);
    expect(som.hits).toHaveLength(4 * 3);
    expect(som.neighbours).toHaveLength(4 * 3 * 6);
    expect(Array.from(som.hits).every((h) => h === 0)).toBe(true);
  });

  it('draws weights in [0, 1) — matching the colour cube, not standardised data', () => {
    const som = createSom(6, 6, 3, 'hex', new Rng(4417));
    for (const w of som.W) {
      expect(w).toBeGreaterThanOrEqual(0);
      expect(w).toBeLessThan(1);
    }
  });

  it('replays exactly from the same seed', () => {
    const a = createSom(4, 4, 3, 'hex', new Rng(4417));
    const b = createSom(4, 4, 3, 'hex', new Rng(4417));
    expect(Array.from(a.W)).toEqual(Array.from(b.W));
  });

  it('rejects a degenerate map', () => {
    expect(() => createSom(0, 3, 3, 'hex', new Rng(1))).toThrow();
    expect(() => createSom(3, 3, 0, 'hex', new Rng(1))).toThrow();
  });
});

describe('bmu', () => {
  it('picks the node whose weights exactly match the sample', () => {
    const som = createSom(2, 2, 2, 'rect', new Rng(1));
    // Node 2 (col 0, row 1) is forced to sit exactly on the query point.
    som.W.set([0.9, 0.9], 2 * 2);
    expect(bmu(som, [0.9, 0.9])).toBe(2);
  });

  it('breaks ties toward the lower index — first strictly-better wins, not last', () => {
    const som = createSom(2, 1, 1, 'rect', new Rng(1));
    som.W.set([0.5, 0.5]);
    expect(bmu(som, [0.5])).toBe(0);
  });
});

describe('bmu2', () => {
  it('returns the best and second-best as distinct nodes even on an exact tie', () => {
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0.5, 0.5, 0.9]);
    const [best, second] = bmu2(som, [0.5]);
    expect(best).toBe(0);
    expect(second).toBe(1);
    expect(best).not.toBe(second);
  });

  it('agrees with bmu on which node is best', () => {
    const som = createSom(4, 4, 3, 'hex', new Rng(7));
    const x = [0.3, 0.6, 0.1];
    const [best] = bmu2(som, x);
    expect(best).toBe(bmu(som, x));
  });
});

describe('neighbourhoodStrength', () => {
  it('is exactly 1 at distance 0, for any positive sigma', () => {
    expect(neighbourhoodStrength(0, 3)).toBeCloseTo(1, 10);
    expect(neighbourhoodStrength(0, 0.001)).toBeCloseTo(1, 10);
  });

  it('falls off as a Gaussian — hand-checked at d = σ', () => {
    // exp(-σ²/(2σ²)) = exp(-0.5)
    expect(neighbourhoodStrength(2, 2)).toBeCloseTo(Math.exp(-0.5), 10);
  });

  it('is monotonically decreasing in d', () => {
    const values = [0, 1, 2, 3, 4].map((d) => neighbourhoodStrength(d, 2));
    for (let i = 1; i < values.length; i++) {
      expect(values[i] as number).toBeLessThan(values[i - 1] as number);
    }
  });
});

describe('schedules', () => {
  const base: Schedule = { alpha0: 0.5, sigma0: 6, decay: 'exponential', steps: 1000 };

  it('starts at v0 and never exceeds it', () => {
    expect(alphaAt(base, 0)).toBeCloseTo(0.5, 6);
    expect(sigmaAt(base, 0)).toBeCloseTo(6, 6);
  });

  it('exponential reaches v0·e⁻³ (about 5%) exactly at the horizon', () => {
    expect(alphaAt(base, 1000)).toBeCloseTo(0.5 * Math.exp(-3), 10);
  });

  it('linear reaches the floor, not exactly zero, at and past the horizon', () => {
    const s: Schedule = { ...base, decay: 'linear' };
    expect(sigmaAt(s, 1000)).toBeGreaterThan(0);
    expect(sigmaAt(s, 1000)).toBeLessThan(1e-3);
    // Past the horizon clamps rather than reversing — going negative would mean the lattice
    // starts *widening* again, which no schedule here promises.
    expect(sigmaAt(s, 5000)).toBe(sigmaAt(s, 1000));
  });

  it('inverse halves by the horizon and never reaches zero', () => {
    const s: Schedule = { ...base, decay: 'inverse' };
    expect(alphaAt(s, 1000)).toBeCloseTo(0.25, 10);
    expect(alphaAt(s, 1_000_000)).toBeGreaterThan(0);
  });

  it('a step before t=0 clamps rather than extrapolating backwards', () => {
    expect(alphaAt(base, -50)).toBeCloseTo(alphaAt(base, 0), 10);
  });
});
