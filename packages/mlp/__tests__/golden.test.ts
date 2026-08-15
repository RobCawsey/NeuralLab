/**
 * The golden run.
 *
 * Gradient check proves the maths is *right*. This proves it is *unchanged*. Refactoring the
 * training loop is safe exactly when these numbers do not move, and every "harmless" tidy-up
 * gets checked against them.
 *
 * **If this fails, say which in the commit message and update the value in the same commit.**
 * A failure means either the change is wrong, or the change is deliberate — it is never a number
 * to nudge until the test goes green. Invariant 7.
 *
 * The same protocol is run by `npm run train`, which asserts the same figures, so drift is
 * caught by whichever happens to be run first.
 */

import { describe, expect, it } from 'vitest';
import { Rng, fitStandardiser, split, standardise, type Dataset } from '@neurallab/core';
import { moons } from '@neurallab/data';
import { createNet, createScratch, initialise } from '../src/net.ts';
import { createTrainer, evaluateRows, trainStep } from '../src/train.ts';

const PROTOCOL = {
  seed: 4417,
  weightSeed: 1,
  n: 240,
  noise: 0.15,
  trainFraction: 0.7,
  hidden: [8, 8],
  learningRate: 0.1,
  batchSize: 16,
  steps: 400,
} as const;

function goldenRun(): {
  trainLoss: number;
  trainAccuracy: number;
  checksum: number;
  epochs: number;
  data: Dataset;
} {
  const data = moons({ n: PROTOCOL.n, noise: PROTOCOL.noise, seed: PROTOCOL.seed });
  const parts = split(data, PROTOCOL.trainFraction, new Rng(PROTOCOL.seed ^ 0x5f3759df));
  const z = standardise(data, fitStandardiser(data, parts.train));

  const net = createNet({
    shape: [z.dim, ...PROTOCOL.hidden, 2],
    hidden: 'tanh',
    output: 'softmax',
    loss: 'crossEntropy',
  });
  initialise(net, 'glorot', new Rng(PROTOCOL.weightSeed));

  const trainer = createTrainer(
    net,
    parts.train,
    { learningRate: PROTOCOL.learningRate, batchSize: PROTOCOL.batchSize, optimiser: 'sgd' },
    new Rng(PROTOCOL.weightSeed),
  );
  for (let i = 0; i < PROTOCOL.steps; i++) trainStep(trainer, z);

  const result = evaluateRows(net, z, parts.train, createScratch(net));

  let checksum = 0;
  for (const layer of net.layers) {
    for (const view of [layer.W, layer.b]) {
      const bits = new Uint32Array(view.buffer, view.byteOffset, view.length);
      for (let i = 0; i < bits.length; i++) checksum = (checksum ^ (bits[i] as number)) >>> 0;
    }
  }

  return {
    trainLoss: result.loss,
    trainAccuracy: result.accuracy,
    checksum,
    epochs: trainer.epoch,
    data,
  };
}

describe('golden run', () => {
  it('reaches its pinned loss', () => {
    expect(goldenRun().trainLoss).toBeCloseTo(0.1007, 4);
  });

  it('reaches its pinned weights', () => {
    /*
     * The loss alone is a weak guard: two different weight vectors can agree to four decimals,
     * so a change that reordered the updates could leave it untouched. XOR-ing the raw float32
     * bits catches any change to any weight at all.
     */
    expect(goldenRun().checksum).toBe(2217589195);
  });

  it('actually learned, rather than merely reproducing', () => {
    // A protocol pinned to a broken number is still pinned. This says the number is a good one.
    const run = goldenRun();
    expect(run.trainAccuracy).toBeGreaterThan(0.95);
    expect(run.epochs).toBe(38);
  });

  it('replays exactly', () => {
    const a = goldenRun();
    const b = goldenRun();
    expect(a.trainLoss).toBe(b.trainLoss);
    expect(a.checksum).toBe(b.checksum);
  });
});
