/**
 * A recording of one sample's journey through the network, for the stepper.
 *
 * §6 of the design document: *not an illustration of the algorithm — the algorithm, paused.* So
 * every number here is produced by `forward` and `backward`, the same functions the worker drains
 * at full speed, on the same weights, in the same order. Nothing in this file computes anything
 * about a network; it copies.
 *
 * **The trace cannot change the run, and that is structural rather than careful.** It is taken
 * after the batch has been accumulated and averaged but before `sgdStep`, using its *own* scratch
 * and its *own* gradient buffers. The real ones are read for `deltaW` and never written. A test
 * asserts a traced step and an untraced step leave bit-identical weights, because the moment that
 * stops being true the teaching screen becomes a lie.
 */

import { sample, type Dataset } from '@neurallab/core';
import { backward, createGrads, zeroGrads, type Grads } from './backward.ts';
import { derivativeFromOutput } from './loss.ts';
import { createScratch, forward, type Net, type Scratch } from './net.ts';
import type { Activation } from './activations.ts';

export interface LayerForward {
  readonly units: number;
  readonly act: Activation;
  /** Pre-activation. */
  readonly z: Float64Array;
  /** Post-activation — what the next layer sees. */
  readonly a: Float64Array;
  /** The weights feeding this layer, `units × inputs` row-major. */
  readonly W: Float32Array;
  readonly b: Float32Array;
}

export interface LayerBackward {
  readonly units: number;
  readonly act: Activation;
  /**
   * True for the output layer under softmax + cross-entropy, where the Jacobian cancels and
   * `dz = a − onehot(t)` directly.
   *
   * Worth flagging rather than hiding: it is the one place in the backward pass where the chain
   * rule does not appear as two visible factors, and a reader looking for `δ × a′` and not
   * finding it deserves to be told why instead of concluding they have misunderstood.
   */
  readonly fused: boolean;
  /** δ arriving from the layer above. Empty when `fused`. */
  readonly deltaIn: Float64Array;
  /** The layer's own derivative, in terms of its output. Empty when `fused`. */
  readonly derivative: Float64Array;
  /** `dL/dz` for this layer — what the stepper draws as δ. */
  readonly dz: Float64Array;
  /** `Wᵀdz` — what this layer hands to the one below. Empty for layer 0. */
  readonly deltaOut: Float64Array;
}

export interface StepTrace {
  /** The step this trace belongs to — the one that has just been applied. */
  readonly step: number;
  /** Row in the dataset, and where it sat in the batch. */
  readonly row: number;
  readonly indexInBatch: number;
  readonly batchSize: number;

  readonly input: Float64Array;
  readonly target: number;
  readonly forward: readonly LayerForward[];
  readonly output: Float64Array;
  readonly loss: number;
  readonly backward: readonly LayerBackward[];

  /**
   * The update actually applied, per layer: `−lr × mean gradient over the batch`.
   *
   * The batch's, not this sample's. Everything above describes one sample because that is what a
   * reader can follow; the update is the batch's because that is what happened. Conflating them
   * would show a Δw that never touched the weights.
   */
  readonly deltaW: readonly Float64Array[];
  readonly deltaB: readonly Float64Array[];
  readonly learningRate: number;
}

/** Reusable buffers, so opening the stepper does not allocate a network's worth of arrays. */
export interface TraceScratch {
  readonly scratch: Scratch;
  readonly grads: Grads;
}

export function createTraceScratch(net: Net): TraceScratch {
  return { scratch: createScratch(net), grads: createGrads(net) };
}

/**
 * Record one sample, using the weights as they stand.
 *
 * Called from `trainStep` between averaging the gradients and applying them, which is the only
 * moment where both halves are true at once: the weights are still the ones the forward pass
 * used, and `applied` already holds the batch mean the update is about to use.
 */
export function captureTrace(
  net: Net,
  ds: Dataset,
  row: number,
  indexInBatch: number,
  batchSize: number,
  step: number,
  learningRate: number,
  applied: Grads,
  into: TraceScratch,
): StepTrace {
  const x = sample(ds, row);
  const target = ds.y === null ? 0 : (ds.y[row] as number);

  zeroGrads(into.grads);
  const output = forward(net, x, into.scratch);
  const loss = backward(net, x, target, into.scratch, into.grads);

  const forwardLayers: LayerForward[] = net.layers.map((layer, l) => ({
    units: layer.units,
    act: layer.act,
    z: Float64Array.from(into.scratch.z[l] as Float64Array),
    a: Float64Array.from(into.scratch.a[l] as Float64Array),
    W: Float32Array.from(layer.W),
    b: Float32Array.from(layer.b),
  }));

  const last = net.layers.length - 1;
  const backwardLayers: LayerBackward[] = net.layers.map((layer, l) => {
    const fused = l === last && layer.act === 'softmax';
    const a = into.scratch.a[l] as Float64Array;

    // Recomputed for display from the values already recorded, not measured a second time —
    // `backward` folds the multiply into one expression and keeps no separate copy.
    const derivative = fused ? new Float64Array(0) : new Float64Array(layer.units);
    if (!fused) {
      for (let u = 0; u < layer.units; u++) {
        derivative[u] = derivativeFromOutput(layer.act, a[u] as number);
      }
    }

    return {
      units: layer.units,
      act: layer.act,
      fused,
      deltaIn:
        fused || l === last
          ? new Float64Array(0)
          : Float64Array.from(into.grads.dIn[l + 1] as Float64Array),
      derivative,
      dz: Float64Array.from(into.grads.dz[l] as Float64Array),
      // Layer 0 computes this and nothing reads it — there is no layer below to receive it.
      deltaOut: l === 0 ? new Float64Array(0) : Float64Array.from(into.grads.dIn[l] as Float64Array),
    };
  });

  return {
    step,
    row,
    indexInBatch,
    batchSize,
    input: Float64Array.from(x),
    target,
    forward: forwardLayers,
    output: Float64Array.from(output),
    loss,
    backward: backwardLayers,
    // Negated and scaled here so the stepper shows the number that lands on the weight, not the
    // gradient it was derived from. A reader comparing a strip to a weight should see them agree.
    deltaW: applied.dW.map((g) => {
      const out = new Float64Array(g.length);
      for (let i = 0; i < g.length; i++) out[i] = -learningRate * (g[i] as number);
      return out;
    }),
    deltaB: applied.db.map((g) => {
      const out = new Float64Array(g.length);
      for (let i = 0; i < g.length; i++) out[i] = -learningRate * (g[i] as number);
      return out;
    }),
    learningRate,
  };
}

/** Largest absolute value in a strip — the stepper normalises each strip against its own range. */
export function peak(values: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    const v = Math.abs(values[i] as number);
    if (Number.isFinite(v) && v > max) max = v;
  }
  return max;
}

/** Index and value of the largest-magnitude entry, for "the biggest single update was…". */
export function largest(values: ArrayLike<number>): { index: number; value: number } {
  let index = 0;
  let best = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = Math.abs(values[i] as number);
    if (Number.isFinite(v) && v > best) {
      best = v;
      index = i;
    }
  }
  return { index, value: (values[index] as number) ?? 0 };
}
