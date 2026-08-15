/**
 * The stepper's trace — and the one property the whole teaching screen rests on.
 *
 * §6: *not an illustration of the algorithm — the algorithm, paused.* That claim is only true
 * while a traced step and an untraced step do exactly the same thing. The moment tracing changes
 * a weight, the screen stops showing what the app does and starts showing what it does when
 * somebody is watching.
 */

import { describe, expect, it } from 'vitest';
import { Rng, fitStandardiser, split, standardise, type Dataset } from '@neurallab/core';
import { moons } from '@neurallab/data';
import { createNet, createScratch, flattenWeights, forward, initialise, type Net } from '../src/net.ts';
import { createTrainer, evaluateRows, trainStep } from '../src/train.ts';
import { createTraceScratch, largest, peak } from '../src/trace.ts';
import { sampleLoss } from '../src/loss.ts';

function prepared(): { z: Dataset; rows: Int32Array } {
  const data = moons({ n: 240, noise: 0.15, seed: 4417 });
  const parts = split(data, 0.7, new Rng(4417 ^ 0x5f3759df));
  return { z: standardise(data, fitStandardiser(data, parts.train)), rows: parts.train };
}

function build(z: Dataset, hidden: number[] = [8, 8], act: 'tanh' | 'relu' = 'tanh'): Net {
  const net = createNet({
    shape: [z.dim, ...hidden, Math.max(2, z.classes)],
    hidden: act,
    output: 'softmax',
    loss: 'crossEntropy',
  });
  initialise(net, 'glorot', new Rng(1));
  return net;
}

describe('tracing cannot change the run', () => {
  it('leaves bit-identical weights after 200 steps', () => {
    /*
     * The load-bearing test of slice 5. Two runs from the same seed, one tracing every step and
     * one tracing none, compared on every weight — not on the loss, which could agree while the
     * weights differed.
     */
    const { z, rows } = prepared();

    const plain = build(z);
    const plainTrainer = createTrainer(plain, rows, { learningRate: 0.1, batchSize: 16, optimiser: 'sgd' }, new Rng(1));
    for (let i = 0; i < 200; i++) trainStep(plainTrainer, z);

    const traced = build(z);
    const tracedTrainer = createTrainer(traced, rows, { learningRate: 0.1, batchSize: 16, optimiser: 'sgd' }, new Rng(1));
    const into = createTraceScratch(traced);
    for (let i = 0; i < 200; i++) trainStep(tracedTrainer, z, { trace: { indexInBatch: i % 16, into } });

    expect(Array.from(flattenWeights(traced))).toEqual(Array.from(flattenWeights(plain)));
    expect(tracedTrainer.step).toBe(plainTrainer.step);
    expect(tracedTrainer.epoch).toBe(plainTrainer.epoch);
    expect(tracedTrainer.cursor).toBe(plainTrainer.cursor);
  });

  it('returns identical metrics', () => {
    const { z, rows } = prepared();

    const plain = build(z);
    const plainTrainer = createTrainer(plain, rows, { learningRate: 0.1, batchSize: 16, optimiser: 'sgd' }, new Rng(1));

    const traced = build(z);
    const tracedTrainer = createTrainer(traced, rows, { learningRate: 0.1, batchSize: 16, optimiser: 'sgd' }, new Rng(1));
    const into = createTraceScratch(traced);

    for (let i = 0; i < 50; i++) {
      const a = trainStep(plainTrainer, z);
      const b = trainStep(tracedTrainer, z, { trace: { indexInBatch: 0, into } });
      expect(b.loss).toBe(a.loss);
      expect(b.lossMin).toBe(a.lossMin);
      expect(b.lossMax).toBe(a.lossMax);
      expect(b.step).toBe(a.step);
    }
  });

  it('still reaches the golden run while tracing every step', () => {
    // The strongest form of the same claim: the pinned number survives being watched.
    const { z, rows } = prepared();
    const net = build(z);
    const trainer = createTrainer(net, rows, { learningRate: 0.1, batchSize: 16, optimiser: 'sgd' }, new Rng(1));
    const into = createTraceScratch(net);
    for (let i = 0; i < 400; i++) trainStep(trainer, z, { trace: { indexInBatch: 3, into } });

    const result = evaluateRows(net, z, rows, createScratch(net));
    expect(result.loss).toBeCloseTo(0.1007, 4);
    expect(result.accuracy).toBeCloseTo(0.9702, 4);
    expect(trainer.epoch).toBe(38);
  });
});

describe('what the trace records', () => {
  const { z, rows } = prepared();

  function traceOnce(hidden: number[] = [8, 8], act: 'tanh' | 'relu' = 'tanh', index = 0) {
    const net = build(z, hidden, act);
    const trainer = createTrainer(net, rows, { learningRate: 0.1, batchSize: 16, optimiser: 'sgd' }, new Rng(1));
    const into = createTraceScratch(net);
    const metrics = trainStep(trainer, z, { trace: { indexInBatch: index, into } });
    return { net, trace: metrics.trace!, trainer };
  }

  it('records every layer, forward and backward', () => {
    const { trace } = traceOnce();
    expect(trace.forward).toHaveLength(3);
    expect(trace.backward).toHaveLength(3);
    expect(trace.forward.map((l) => l.units)).toEqual([8, 8, 2]);
    expect(trace.output).toHaveLength(2);
  });

  it('matches a forward pass run independently', () => {
    /*
     * The trace is a recording, not a second calculation. Running `forward` on the same input
     * with the weights the trace captured must reproduce the activations it reported.
     */
    const { trace } = traceOnce();
    const net = createNet({
      shape: [2, 8, 8, 2],
      hidden: 'tanh',
      output: 'softmax',
      loss: 'crossEntropy',
    });
    for (let l = 0; l < net.layers.length; l++) {
      net.layers[l]!.W.set(trace.forward[l]!.W);
      net.layers[l]!.b.set(trace.forward[l]!.b);
    }
    const out = forward(net, trace.input, createScratch(net));
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(trace.output[i] as number, 12);
  });

  it('reports the loss the output actually earns', () => {
    const { trace } = traceOnce();
    expect(trace.loss).toBeCloseTo(sampleLoss('crossEntropy', trace.output, trace.target), 12);
  });

  it('flags the fused output layer and no other', () => {
    /*
     * Under softmax + cross-entropy the Jacobian cancels and `dz = a − onehot(t)` directly. It is
     * the one place the chain rule does not appear as two visible factors, so the screen says so
     * rather than leaving a reader hunting for a δ × a′ that is not there.
     */
    const { trace } = traceOnce();
    expect(trace.backward.map((l) => l.fused)).toEqual([false, false, true]);

    const output = trace.backward[2]!;
    expect(output.derivative).toHaveLength(0);
    expect(output.deltaIn).toHaveLength(0);
    for (let i = 0; i < trace.output.length; i++) {
      const expected = (trace.output[i] as number) - (i === trace.target ? 1 : 0);
      expect(output.dz[i]).toBeCloseTo(expected, 12);
    }
  });

  it('records δz as δin × the derivative on hidden layers', () => {
    // The chain rule, visible. If these three strips ever stopped agreeing the screen would be
    // showing a multiplication that did not happen.
    const { trace } = traceOnce();
    for (const layer of trace.backward.filter((l) => !l.fused)) {
      for (let u = 0; u < layer.units; u++) {
        expect(layer.dz[u]).toBeCloseTo(
          (layer.deltaIn[u] as number) * (layer.derivative[u] as number),
          12,
        );
      }
    }
  });

  it('uses the relu derivative a reader can check by eye', () => {
    const { trace } = traceOnce([8, 8], 'relu');
    const hidden = trace.backward[0]!;
    const forwardLayer = trace.forward[0]!;
    for (let u = 0; u < hidden.units; u++) {
      expect(hidden.derivative[u]).toBe((forwardLayer.a[u] as number) > 0 ? 1 : 0);
    }
  });

  it('leaves δ out empty for the input layer', () => {
    // Layer 0 computes it and nothing reads it — there is no layer below to receive it.
    expect(traceOnce().trace.backward[0]!.deltaOut).toHaveLength(0);
  });

  it('reports Δw as the change applied to the weight, not the gradient', () => {
    /*
     * Negated and scaled by the learning rate, so a reader comparing the strip to the weight
     * before and after sees them agree. Showing the raw gradient would have the sign backwards.
     */
    const { net, trace } = traceOnce();
    for (let l = 0; l < net.layers.length; l++) {
      const before = trace.forward[l]!.W;
      const after = net.layers[l]!.W;
      const delta = trace.deltaW[l]!;
      for (let i = 0; i < after.length; i++) {
        expect((before[i] as number) + (delta[i] as number)).toBeCloseTo(after[i] as number, 6);
      }
    }
  });

  it('describes the batch it belongs to', () => {
    const { trace } = traceOnce([8, 8], 'tanh', 5);
    expect(trace.batchSize).toBe(16);
    expect(trace.indexInBatch).toBe(5);
    expect(trace.step).toBe(1);
    expect(trace.row).toBeGreaterThanOrEqual(0);
  });

  it('clamps an index past the end of the batch', () => {
    // "Trace sample 12" on a batch of 8 traces the last one rather than throwing or silently
    // tracing nothing.
    const { trace } = traceOnce([4], 'tanh', 99);
    expect(trace.indexInBatch).toBe(15);
  });
});

describe('strip helpers', () => {
  it('finds the peak magnitude, ignoring sign', () => {
    expect(peak([0.1, -0.9, 0.3])).toBeCloseTo(0.9, 12);
    expect(peak([])).toBe(0);
  });

  it('ignores non-finite values rather than returning NaN', () => {
    expect(peak([1, NaN, 2])).toBe(2);
  });

  it('finds the largest entry and keeps its sign', () => {
    expect(largest([0.1, -0.9, 0.3])).toEqual({ index: 1, value: -0.9 });
  });
});
