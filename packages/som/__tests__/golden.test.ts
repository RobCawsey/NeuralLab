/**
 * The golden run — the SOM side's answer to `packages/mlp/__tests__/golden.test.ts`.
 *
 * The same protocol is run by `scripts/som.ts`, which prints the map as real terminal colour and
 * asserts the same figures, so drift is caught by whichever happens to be run first.
 *
 * **If this fails, say which in the commit message and update the value in the same commit.**
 * A failure means either the change is wrong, or the change is deliberate — invariant 7.
 *
 * Quantisation error is pinned from step 300 to step 3000, not from step 0. `scripts/som.ts`'s
 * own comment has the full measured reason: `createSom`'s weights are drawn uniform in [0, 1) to
 * match the colour cube, so a fresh random map is already 144 points from the *exact* data
 * distribution — a strong quantiser with zero structure behind it (topographic error 0.97 at
 * step 0). Training's real, monotonic improvement shows up once the initial reorganisation has
 * settled, which is what this file pins.
 */

import { describe, expect, it } from 'vitest';
import { Rng } from '@neurallab/core';
import { colourCube } from '@neurallab/data';
import { createSom, type Som } from '../src/som.ts';
import { createSomTrainer, somStep } from '../src/train.ts';
import { quantisationError, topographicError } from '../src/metrics.ts';
import type { Schedule } from '../src/som.ts';

const PROTOCOL = {
  seed: 4417,
  weightSeed: 1,
  drawSeed: 2,
  n: 1500,
  cols: 12,
  rows: 12,
  steps: 3000,
  alpha0: 0.5,
  sigma0: 6,
  decay: 'exponential' as const,
};

function checksumOf(som: Som): number {
  let checksum = 0;
  const bits = new Uint32Array(som.W.buffer, som.W.byteOffset, som.W.length);
  for (let i = 0; i < bits.length; i++) checksum = (checksum ^ (bits[i] as number)) >>> 0;
  return checksum;
}

function goldenRun(steps = PROTOCOL.steps): { som: Som; qe: number; te: number; checksum: number } {
  const data = colourCube({ n: PROTOCOL.n, seed: PROTOCOL.seed });
  const allRows = Int32Array.from({ length: data.n }, (_, i) => i);
  const som = createSom(PROTOCOL.cols, PROTOCOL.rows, data.dim, 'hex', new Rng(PROTOCOL.weightSeed));
  const schedule: Schedule = {
    alpha0: PROTOCOL.alpha0,
    sigma0: PROTOCOL.sigma0,
    decay: PROTOCOL.decay,
    steps: PROTOCOL.steps,
  };
  const trainer = createSomTrainer(som, allRows, schedule, new Rng(PROTOCOL.drawSeed));
  for (let i = 0; i < steps; i++) somStep(trainer, data);
  return {
    som,
    qe: quantisationError(som, data, allRows),
    te: topographicError(som, data, allRows),
    checksum: checksumOf(som),
  };
}

describe('golden run', () => {
  it('reaches its pinned quantisation and topographic error', () => {
    const run = goldenRun();
    expect(run.qe).toBeCloseTo(0.1166, 4);
    expect(run.te).toBeCloseTo(0.0753, 4);
  });

  it('reaches its pinned weights', () => {
    // Same reasoning as the MLP side: two maps can agree on QE to four decimals without agreeing
    // weight for weight. XOR-ing the raw float32 bits catches any change to any weight.
    expect(goldenRun().checksum).toBe(214067);
  });

  it('quantisation error actually fell once the lattice had organised, rather than the pin describing a broken run', () => {
    const early = goldenRun(300);
    const late = goldenRun();
    expect(late.qe).toBeLessThan(early.qe * 0.5);
  });

  it('the random baseline is topographically incoherent, which is why QE is not pinned against it', () => {
    const data = colourCube({ n: PROTOCOL.n, seed: PROTOCOL.seed });
    const allRows = Int32Array.from({ length: data.n }, (_, i) => i);
    const som = createSom(PROTOCOL.cols, PROTOCOL.rows, data.dim, 'hex', new Rng(PROTOCOL.weightSeed));
    expect(topographicError(som, data, allRows)).toBeGreaterThan(0.8);
  });

  it('replays exactly', () => {
    const a = goldenRun();
    const b = goldenRun();
    expect(a.checksum).toBe(b.checksum);
    expect(a.qe).toBe(b.qe);
  });

  it('chunking the run into uneven bursts gives bit-identical results to running it straight through', () => {
    // The same invariant `apps/web/__tests__/build.test.ts` checks for the MLP worker, checked
    // here before a SOM worker exists to need it: the training loop itself must not care how its
    // steps are grouped, because that is what makes chunking it safe later.
    const data = colourCube({ n: PROTOCOL.n, seed: PROTOCOL.seed });
    const allRows = Int32Array.from({ length: data.n }, (_, i) => i);
    const schedule: Schedule = {
      alpha0: PROTOCOL.alpha0,
      sigma0: PROTOCOL.sigma0,
      decay: PROTOCOL.decay,
      steps: PROTOCOL.steps,
    };

    const straight = createSom(PROTOCOL.cols, PROTOCOL.rows, data.dim, 'hex', new Rng(PROTOCOL.weightSeed));
    const straightTrainer = createSomTrainer(straight, allRows, schedule, new Rng(PROTOCOL.drawSeed));
    for (let i = 0; i < PROTOCOL.steps; i++) somStep(straightTrainer, data);

    const chunked = createSom(PROTOCOL.cols, PROTOCOL.rows, data.dim, 'hex', new Rng(PROTOCOL.weightSeed));
    const chunkedTrainer = createSomTrainer(chunked, allRows, schedule, new Rng(PROTOCOL.drawSeed));
    const bursts = [7, 113, 1, 979, 1900]; // sums to 3000, deliberately uneven
    expect(bursts.reduce((a, b) => a + b, 0)).toBe(PROTOCOL.steps);
    for (const burst of bursts) {
      for (let i = 0; i < burst; i++) somStep(chunkedTrainer, data);
    }

    expect(checksumOf(chunked)).toBe(checksumOf(straight));
  });
});
