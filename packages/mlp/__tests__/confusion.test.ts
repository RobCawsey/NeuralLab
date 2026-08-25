import { describe, expect, it } from 'vitest';
import { createNet, createScratch } from '../src/net.ts';
import { confusionAt, confusionMatrix, topConfusions } from '../src/confusion.ts';
import type { Dataset } from '@neurallab/core';

/**
 * Hand-worked, the same way the forward pass's own first test was in slice 1: a network whose
 * weights are all zero always outputs its bias vector as the logits, so setting the bias by hand
 * makes the prediction for *every* row the same known class — no training, no randomness, an
 * expected matrix that can be written down before the test runs rather than read off whatever
 * the code happens to produce.
 */
describe('confusionMatrix', () => {
  it('a network that always predicts class 1 confuses every other class into it', () => {
    // 3 classes, 1 input feature (irrelevant — weights are zero). Actual labels: 0, 0, 1, 2.
    const net = createNet({ shape: [1, 3], hidden: 'linear', output: 'softmax', loss: 'crossEntropy' });
    net.layers[0]!.b[1] = 5; // logit 1 dominates regardless of input — always predicts class 1

    const ds: Dataset = {
      name: 'hand-worked',
      x: new Float32Array([0, 1, 2, 3]),
      y: new Int32Array([0, 0, 1, 2]),
      n: 4,
      dim: 1,
      classes: 3,
      featureNames: ['x'],
      classNames: ['a', 'b', 'c'],
    };
    const rows = new Int32Array([0, 1, 2, 3]);
    const scratch = createScratch(net);

    const m = confusionMatrix(net, ds, rows, scratch);

    expect(m.classes).toBe(3);
    expect(m.total).toBe(4);
    expect(m.correct).toBe(1); // only the one actual-class-1 row is right

    expect(confusionAt(m, 0, 1)).toBe(2); // both actual-0 rows predicted as 1
    expect(confusionAt(m, 1, 1)).toBe(1); // the actual-1 row, correctly
    expect(confusionAt(m, 2, 1)).toBe(1); // the actual-2 row predicted as 1

    // Every other cell in a 3×3 grid is zero — nine cells, three checked above, six left.
    let nonZero = 0;
    for (const c of m.counts) if (c !== 0) nonZero++;
    expect(nonZero).toBe(3);
  });

  it('topConfusions ranks the off-diagonal pairs worst first and never includes the diagonal', () => {
    const net = createNet({ shape: [1, 3], hidden: 'linear', output: 'softmax', loss: 'crossEntropy' });
    net.layers[0]!.b[1] = 5;
    const ds: Dataset = {
      name: 'hand-worked',
      x: new Float32Array([0, 1, 2, 3]),
      y: new Int32Array([0, 0, 1, 2]),
      n: 4,
      dim: 1,
      classes: 3,
      featureNames: ['x'],
      classNames: ['a', 'b', 'c'],
    };
    const scratch = createScratch(net);
    const m = confusionMatrix(net, ds, new Int32Array([0, 1, 2, 3]), scratch);

    const top = topConfusions(m, 5);
    expect(top[0]).toEqual({ actual: 0, predicted: 1, count: 2 });
    expect(top.some((p) => p.actual === p.predicted)).toBe(false);
    expect(top).toHaveLength(2); // only (0→1, count 2) and (2→1, count 1) are off-diagonal and nonzero
  });

  it('does not touch the network passed to it', () => {
    const net = createNet({ shape: [1, 3], hidden: 'linear', output: 'softmax', loss: 'crossEntropy' });
    net.layers[0]!.b[1] = 5;
    const before = Float32Array.from(net.layers[0]!.b);
    const ds: Dataset = {
      name: 'hand-worked',
      x: new Float32Array([0, 1]),
      y: new Int32Array([0, 1]),
      n: 2,
      dim: 1,
      classes: 3,
      featureNames: ['x'],
      classNames: ['a', 'b', 'c'],
    };
    confusionMatrix(net, ds, new Int32Array([0, 1]), createScratch(net));
    expect(net.layers[0]!.b).toEqual(before);
  });
});
