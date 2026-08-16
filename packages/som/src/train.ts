/**
 * One training step: draw a sample, find its BMU, drag every node toward it by
 * `α(t) · h(d, t)`. `d` is a lattice distance, not a data distance — the whole algorithm, per §3
 * of the design document.
 */

import { Rng, sample, type Dataset } from '@neurallab/core';
import { latticeDistance } from './lattice.ts';
import { alphaAt, bmu, neighbourhoodStrength, sigmaAt, sqDistance, type Schedule, type Som } from './som.ts';
import type { SomStepTrace } from './trace.ts';

export interface SomTrainer {
  readonly som: Som;
  readonly schedule: Schedule;
  /** Row indices into the dataset this trainer draws from — the SOM's own train/val split. */
  readonly rows: Int32Array;
  readonly rng: Rng;
  step: number;
}

export function createSomTrainer(som: Som, rows: Int32Array, schedule: Schedule, rng: Rng): SomTrainer {
  return { som, schedule, rows, rng, step: 0 };
}

export interface StepResult {
  readonly rowIndex: number;
  readonly bmuIndex: number;
  /** Present only when a trace was asked for — the stepper's recording of this step. */
  readonly trace?: SomStepTrace;
}

export interface SomStepOptions {
  /** Record this step for the stepper — §8's teaching screen, the SOM side of it. */
  readonly trace?: boolean;
}

/**
 * One step is one sample, not a minibatch.
 *
 * The MLP side batches because a gradient is an average over several rows and invariant 2 asks
 * for a fixed unit of work per step; a SOM update has no such average to take — each sample drags
 * the lattice on its own, immediately, which is the "winner take most" behaviour the algorithm
 * is. Samples are drawn **with replacement**, uniformly from `rows`, which is the classical
 * Kohonen training loop and not a shuffled pass over the set: `steps` is the schedule's own
 * horizon, and a fixed-length shuffled epoch would tie "how many times has this row been seen"
 * to `rows.length` in a way the schedule already accounts for on its own.
 */
export function somStep(trainer: SomTrainer, ds: Dataset, options: SomStepOptions = {}): StepResult {
  const { som, schedule, rows, rng } = trainer;
  const rowIndex = rows[rng.int(rows.length)] as number;
  const x = sample(ds, rowIndex);

  const winner = bmu(som, x);
  som.hits[winner] = (som.hits[winner] as number) + 1;

  const alpha = alphaAt(schedule, trainer.step);
  const sigma = sigmaAt(schedule, trainer.step);

  const n = som.cols * som.rows;

  // Only allocated when asked for. A trace is small at these map sizes — a 24×24 map is 576
  // floats per array — so unlike the MLP side there is no reusable scratch object to thread
  // through; the cost of allocating fresh arrays on the rare step someone opens the stepper for
  // is not worth the bookkeeping that would save it.
  const tracing = options.trace === true;
  const traceDistances = tracing ? new Float32Array(n) : undefined;
  const traceStrength = tracing ? new Float32Array(n) : undefined;
  const traceBefore = tracing ? Float32Array.from(som.W) : undefined;

  for (let node = 0; node < n; node++) {
    const d = latticeDistance(som.cols, som.topology, winner, node);
    const h = neighbourhoodStrength(d, sigma);
    if (tracing) {
      traceDistances![node] = Math.sqrt(sqDistance(som, node, x));
      traceStrength![node] = h;
    }
    // Below this, `rate * (x[k] - w)` is far under a Float32 weight's precision at the value
    // ranges this kernel trains on — skipping it is a throughput win with nothing to lose. A
    // test drives a node far enough from the winner, at a small enough σ, to land in exactly
    // this branch and confirms its weights are untouched.
    if (h < 1e-7) continue;
    const base = node * som.dim;
    const rate = alpha * h;
    for (let k = 0; k < som.dim; k++) {
      const w = som.W[base + k] as number;
      som.W[base + k] = w + rate * ((x[k] as number) - w);
    }
  }

  trainer.step++;

  if (!tracing) return { rowIndex, bmuIndex: winner };

  return {
    rowIndex,
    bmuIndex: winner,
    trace: {
      step: trainer.step,
      row: rowIndex,
      input: Float32Array.from(x),
      bmu: winner,
      alpha,
      sigma,
      distances: traceDistances!,
      strength: traceStrength!,
      before: traceBefore!,
    },
  };
}
