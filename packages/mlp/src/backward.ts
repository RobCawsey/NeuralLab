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

export const OPTIMISERS = ['sgd', 'momentum', 'adam'] as const;
export type OptimiserKind = (typeof OPTIMISERS)[number];

export function isOptimiserKind(v: string): v is OptimiserKind {
  return (OPTIMISERS as readonly string[]).includes(v);
}

/**
 * Fixed hyperparameters for momentum and Adam.
 *
 * Not exposed as sliders. §6's guided flow has already made the case that a beginner does not
 * need a fourth dial before they have used the first three, and 0.9 / 0.999 / 1e-8 are the
 * values every reference implementation ships with — there is no lesson in retuning them.
 */
const MOMENTUM_BETA = 0.9;
const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;

/**
 * Per-parameter state an optimiser carries between steps. Shaped like `Grads`, because momentum
 * and Adam are exactly that: one more array per parameter, updated from the gradient instead of
 * replacing it.
 */
export interface OptimiserState {
  kind: OptimiserKind;
  /** Momentum's velocity, or Adam's first moment. Unused and empty for `sgd`. */
  readonly mW: Float64Array[];
  readonly mB: Float64Array[];
  /** Adam's second moment. Unused and empty for `momentum` and `sgd`. */
  readonly vW: Float64Array[];
  readonly vB: Float64Array[];
  /** Adam's step count, for bias correction. Meaningless for the other two. */
  t: number;
}

export function createOptimiserState(net: Net, kind: OptimiserKind): OptimiserState {
  const zeros = (): Float64Array[] => net.layers.map((l) => new Float64Array(l.W.length));
  const zerosB = (): Float64Array[] => net.layers.map((l) => new Float64Array(l.b.length));
  return { kind, mW: zeros(), mB: zerosB(), vW: zeros(), vB: zerosB(), t: 0 };
}

/**
 * Switch an existing optimiser's kind, resetting its state to zero.
 *
 * A run mid-flight that changes optimiser cannot keep Adam's second moment and call it momentum's
 * velocity — the two numbers mean different things, and a state built for one kind applied under
 * another is not "warm-started", it is wrong. Zeroing is the same discontinuity switching from
 * SGD to Adam already has on step 1, just arriving one step later.
 */
export function resetOptimiserState(state: OptimiserState, kind: OptimiserKind): void {
  state.kind = kind;
  state.t = 0;
  for (const arrays of [state.mW, state.mB, state.vW, state.vB]) {
    for (const a of arrays) a.fill(0);
  }
}

/**
 * Apply one update, in place, according to `state.kind`. `sgdStep` still exists on its own below
 * for anything that wants plain SGD without carrying optimiser state — the golden run does.
 *
 * **Momentum** here is the un-scaled form most references mean by "SGD with momentum":
 * `v ← β·v + g`, `w ← w − lr·v`. The exponential-moving-average form (`v ← β·v + (1−β)·g`) is
 * closer to what Adam's first moment does below, and mixing the two conventions in one file is
 * exactly the kind of thing that turns into a silent factor-of-ten learning-rate bug.
 *
 * **Adam** is standard, bias-corrected: `m`/`v` are moving averages of the gradient and its
 * square, divided by `1 − βᵗ` because both start at zero and are biased toward it early on. The
 * bias-correction terms are the one part of this that is easy to misplace, the loss still goes
 * down without them, and §13 names that as the specific risk — which is why `adam.test.ts`
 * checks a step by hand rather than by watching a loss curve fall.
 */
export function applyUpdate(net: Net, grads: Grads, state: OptimiserState, learningRate: number): void {
  if (state.kind === 'sgd') {
    sgdStep(net, grads, learningRate);
    return;
  }

  state.t++;
  const bc1 = 1 - Math.pow(ADAM_BETA1, state.t);
  const bc2 = 1 - Math.pow(ADAM_BETA2, state.t);

  for (let l = 0; l < net.layers.length; l++) {
    const layer = net.layers[l] as Dense;
    const dW = grads.dW[l] as Float64Array;
    const db = grads.db[l] as Float64Array;
    const mW = state.mW[l] as Float64Array;
    const mB = state.mB[l] as Float64Array;

    if (state.kind === 'momentum') {
      for (let i = 0; i < layer.W.length; i++) {
        mW[i] = MOMENTUM_BETA * (mW[i] as number) + (dW[i] as number);
        layer.W[i] = (layer.W[i] as number) - learningRate * (mW[i] as number);
      }
      for (let i = 0; i < layer.b.length; i++) {
        mB[i] = MOMENTUM_BETA * (mB[i] as number) + (db[i] as number);
        layer.b[i] = (layer.b[i] as number) - learningRate * (mB[i] as number);
      }
      continue;
    }

    // adam
    const vW = state.vW[l] as Float64Array;
    const vB = state.vB[l] as Float64Array;
    for (let i = 0; i < layer.W.length; i++) {
      const g = dW[i] as number;
      mW[i] = ADAM_BETA1 * (mW[i] as number) + (1 - ADAM_BETA1) * g;
      vW[i] = ADAM_BETA2 * (vW[i] as number) + (1 - ADAM_BETA2) * g * g;
      const mHat = (mW[i] as number) / bc1;
      const vHat = (vW[i] as number) / bc2;
      layer.W[i] = (layer.W[i] as number) - (learningRate * mHat) / (Math.sqrt(vHat) + ADAM_EPSILON);
    }
    for (let i = 0; i < layer.b.length; i++) {
      const g = db[i] as number;
      mB[i] = ADAM_BETA1 * (mB[i] as number) + (1 - ADAM_BETA1) * g;
      vB[i] = ADAM_BETA2 * (vB[i] as number) + (1 - ADAM_BETA2) * g * g;
      const mHat = (mB[i] as number) / bc1;
      const vHat = (vB[i] as number) / bc2;
      layer.b[i] = (layer.b[i] as number) - (learningRate * mHat) / (Math.sqrt(vHat) + ADAM_EPSILON);
    }
  }
}

/** Plain stochastic gradient descent: `w -= lr * dw`. What the golden run is pinned against. */
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
