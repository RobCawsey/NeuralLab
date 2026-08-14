/**
 * Backpropagation. The artefact this whole project exists to show.
 *
 * Written out by hand, immediately readable against `forward` in `net.ts`, and deliberately not
 * produced by an autodiff engine — invariant 6. The stepper in slice 5 pauses *this* function
 * between its stages rather than illustrating it, so every intermediate it needs is a named
 * buffer here rather than a local.
 *
 * The whole algorithm is three lines of arithmetic repeated per layer, backwards:
 *
 *   dz  = dIn(from the layer above) ⊙ act'(a)     what this layer's pre-activation wants
 *   dW += dz ⊗ input                              the gradient for its own weights
 *   dIn = Wᵀ dz                                   what it hands to the layer below
 */

import { derivativeFromOutput, outputDelta, sampleLoss, type LossKind } from './loss.ts';
import type { Dense, Net, Scratch } from './net.ts';

export interface Grads {
  /** Same shape as each layer's `W`. Accumulated across a batch, then divided once. */
  readonly dW: Float64Array[];
  readonly db: Float64Array[];
  /** `dL/dz` per layer — the value the stepper draws as "δ". */
  readonly dz: Float64Array[];
  /** `dL/d(input)` per layer — what this layer sends to the one below it. */
  readonly dIn: Float64Array[];
}

export function createGrads(net: Net): Grads {
  return {
    dW: net.layers.map((l) => new Float64Array(l.W.length)),
    db: net.layers.map((l) => new Float64Array(l.b.length)),
    dz: net.layers.map((l) => new Float64Array(l.units)),
    dIn: net.layers.map((l) => new Float64Array(l.inputs)),
  };
}

/** Zero the accumulators. `dz` and `dIn` are fully overwritten each pass and need no clearing. */
export function zeroGrads(grads: Grads): void {
  for (const g of grads.dW) g.fill(0);
  for (const g of grads.db) g.fill(0);
}

/**
 * One backward pass for one sample, accumulating into `grads`. Returns that sample's loss.
 *
 * `forward` must have been called with this same `scratch` and this same input first — the
 * backward pass reads the activations it left behind. Calling them out of order produces
 * gradients for a different sample than the one you think, which trains *almost* correctly and
 * is very hard to see.
 */
export function backward(
  net: Net,
  input: ArrayLike<number>,
  target: number,
  scratch: Scratch,
  grads: Grads,
): number {
  const last = net.layers.length - 1;
  const output = scratch.a[last] as Float64Array;
  const loss = sampleLoss(net.loss, output, target);

  // The output layer starts the chain, and it is the only layer whose δ comes from the loss
  // rather than from the layer above.
  outputDelta(net.loss, (net.layers[last] as Dense).act, output, target, grads.dz[last] as Float64Array);

  for (let l = last; l >= 0; l--) {
    const layer = net.layers[l] as Dense;
    const dz = grads.dz[l] as Float64Array;
    const dW = grads.dW[l] as Float64Array;
    const db = grads.db[l] as Float64Array;
    const dIn = grads.dIn[l] as Float64Array;
    // Layer 0's input is the sample; every other layer's is the activation below it.
    const lowerA = l === 0 ? input : (scratch.a[l - 1] as Float64Array);

    // The output layer's dz is already set by outputDelta; every other layer's arrives as the
    // dIn of the layer above and still needs its own activation derivative applied.
    if (l !== last) {
      const fromAbove = grads.dIn[l + 1] as Float64Array;
      const a = scratch.a[l] as Float64Array;
      for (let u = 0; u < layer.units; u++) {
        dz[u] = (fromAbove[u] as number) * derivativeFromOutput(layer.act, a[u] as number);
      }
    }

    dIn.fill(0);
    for (let u = 0; u < layer.units; u++) {
      const d = dz[u] as number;
      const row = u * layer.inputs;
      db[u] = (db[u] as number) + d;
      for (let i = 0; i < layer.inputs; i++) {
        dW[row + i] = (dW[row + i] as number) + d * (lowerA[i] as number);
        // Wᵀdz, folded into the same loop rather than transposing the matrix. Layer 0 computes
        // this too and nothing reads it — the cost is one layer's worth of multiply-adds, and
        // the alternative is a special case in the middle of the clearest loop in the project.
        dIn[i] = (dIn[i] as number) + (layer.W[row + i] as number) * d;
      }
    }
  }

  return loss;
}

/** Scale accumulated gradients to a mean. Called once per batch, not once per sample. */
export function scaleGrads(grads: Grads, factor: number): void {
  for (const g of grads.dW) for (let i = 0; i < g.length; i++) g[i] = (g[i] as number) * factor;
  for (const g of grads.db) for (let i = 0; i < g.length; i++) g[i] = (g[i] as number) * factor;
}

/** Euclidean norm of one layer's weight gradient — the gradient-flow bars in slice 7. */
export function gradNorm(grads: Grads, layer: number): number {
  const g = grads.dW[layer];
  if (!g) return 0;
  let sum = 0;
  for (let i = 0; i < g.length; i++) sum += (g[i] as number) * (g[i] as number);
  return Math.sqrt(sum);
}

/* ---------------- optimiser ---------------- */

export type OptimiserKind = 'sgd';

/**
 * Plain stochastic gradient descent: `w -= lr * dw`.
 *
 * Momentum and Adam arrive in slice 7. Adding them now would mean three code paths and no way
 * to tell which one a disappointing loss curve came from — and slice 7 pairs them with the
 * diagnostics that make the difference visible.
 */
export function sgdStep(net: Net, grads: Grads, learningRate: number): void {
  for (let l = 0; l < net.layers.length; l++) {
    const layer = net.layers[l] as Dense;
    const dW = grads.dW[l] as Float64Array;
    const db = grads.db[l] as Float64Array;
    for (let i = 0; i < layer.W.length; i++) {
      layer.W[i] = (layer.W[i] as number) - learningRate * (dW[i] as number);
    }
    for (let i = 0; i < layer.b.length; i++) {
      layer.b[i] = (layer.b[i] as number) - learningRate * (db[i] as number);
    }
  }
}

/** True when any weight has stopped being a number — challenge 3's outcome, detected. */
export function hasDiverged(net: Net): boolean {
  for (const layer of net.layers) {
    for (let i = 0; i < layer.W.length; i++) {
      if (!Number.isFinite(layer.W[i] as number)) return true;
    }
  }
  return false;
}

export type { LossKind };
