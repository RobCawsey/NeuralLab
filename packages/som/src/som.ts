/**
 * The map itself, and the pieces of one training step that do not touch a dataset row:
 * the best-matching unit, the neighbourhood function, and the decay schedules.
 *
 * `mlp` and `som` never import each other — see `packages/core/src/dataset.ts`. This file's only
 * import from outside the package is `Rng`, the same rule the MLP kernel follows.
 */

import { Rng } from '@neurallab/core';
import { buildNeighbours, type Topology } from './lattice.ts';

export type { Topology };

export interface Som {
  readonly cols: number;
  readonly rows: number;
  readonly dim: number;
  readonly topology: Topology;
  /** `cols * rows * dim`, node-major: node `i`'s weights are `W[i*dim .. i*dim+dim)`. */
  W: Float32Array;
  /** Samples won, per node — cumulative across the run, reset only by `createSom`. */
  hits: Int32Array;
  /** Six slots per node, `-1` past the edge. Built once here; see `lattice.ts`. */
  readonly neighbours: Int32Array;
}

/**
 * A fresh map, weights drawn uniform in `[0, 1)`.
 *
 * That range is not normalised against the data — it is a deliberate match to the colour cube,
 * this package's own demonstration set, whose three features already live in `[0, 1)`. A reader
 * pointing the kernel at data with a different range gets a map that visibly has to travel before
 * it organises, which is itself honest: nothing here claims to standardise the input, unlike the
 * MLP side. Sizing `W`, `hits` and `neighbours` from `cols`/`rows`/`dim` is the only place their
 * shape is decided, matching invariant 3's rule that a buffer's owner is the one place that sizes
 * it.
 */
export function createSom(cols: number, rows: number, dim: number, topology: Topology, rng: Rng): Som {
  if (cols < 1 || rows < 1 || dim < 1) {
    throw new Error(`degenerate map: cols=${cols} rows=${rows} dim=${dim}`);
  }
  const W = new Float32Array(cols * rows * dim);
  for (let i = 0; i < W.length; i++) W[i] = rng.float();
  return {
    cols,
    rows,
    dim,
    topology,
    W,
    hits: new Int32Array(cols * rows),
    neighbours: buildNeighbours(cols, rows, topology),
  };
}

/**
 * Squared Euclidean distance between a node's weight vector and a sample — no `sqrt` to compare.
 * Exported for the stepper's "distances" stage, which needs it for every node, not just the
 * winner `bmu` keeps.
 */
export function sqDistance(som: Som, node: number, x: ArrayLike<number>): number {
  const base = node * som.dim;
  let sum = 0;
  for (let k = 0; k < som.dim; k++) {
    const d = (som.W[base + k] as number) - (x[k] as number);
    sum += d * d;
  }
  return sum;
}

/** The best-matching unit — the node whose weights are nearest `x`. */
export function bmu(som: Som, x: ArrayLike<number>): number {
  let best = 0;
  let bestSq = Infinity;
  const n = som.cols * som.rows;
  for (let i = 0; i < n; i++) {
    const sq = sqDistance(som, i, x);
    if (sq < bestSq) {
      bestSq = sq;
      best = i;
    }
  }
  return best;
}

/**
 * The best **and** second-best matching units, for topographic error — TE needs to know whether
 * the runner-up is a lattice neighbour of the winner, which a single `bmu()` call cannot answer.
 * One pass, not two calls to `bmu` with the winner excluded, so a tie between the top two nodes
 * cannot silently pick the same node for both.
 */
export function bmu2(som: Som, x: ArrayLike<number>): [best: number, second: number] {
  let best = 0;
  let second = 0;
  let bestSq = Infinity;
  let secondSq = Infinity;
  const n = som.cols * som.rows;
  for (let i = 0; i < n; i++) {
    const sq = sqDistance(som, i, x);
    if (sq < bestSq) {
      second = best;
      secondSq = bestSq;
      best = i;
      bestSq = sq;
    } else if (sq < secondSq) {
      second = i;
      secondSq = sq;
    }
  }
  return [best, second];
}

/**
 * `h(d, σ)` — the Gaussian neighbourhood function. `d` is a lattice distance from `lattice.ts`,
 * never a data-space one; conflating the two is §3's whole warning restated as code.
 *
 * `σ` is floored well above zero by `sigmaAt` before it ever reaches here, so `d = 0` gives
 * exactly 1 rather than the `0 / 0` a literal zero would produce — see the note on `decayAt`.
 */
export function neighbourhoodStrength(d: number, sigma: number): number {
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

export type Decay = 'exponential' | 'linear' | 'inverse';

export interface Schedule {
  readonly alpha0: number;
  readonly sigma0: number;
  readonly decay: Decay;
  /** The horizon both decays are written against — see `decayAt`. */
  readonly steps: number;
}

/**
 * A value below which nothing here decays. Not a tuning constant — a guard against `sigmaAt`
 * ever returning exactly zero, which would make `neighbourhoodStrength` divide by zero at
 * `d = 0` and hand every remaining step a `NaN` weight update that trains nothing and says
 * nothing about why. `alphaAt` gets the same floor for symmetry, though it does not strictly
 * need one: `alpha = 0` just stops the map moving, no division involved.
 */
const DECAY_FLOOR = 1e-6;

/**
 * Shared by `alphaAt` and `sigmaAt`. `t` is clamped into `[0, schedule.steps]` first, so a step
 * requested past the horizon reads as "fully decayed" rather than reversing the curve.
 *
 * Written against an explicit `steps` horizon rather than a per-step multiplier, because a
 * multiplier makes "how fast does it cool" depend on how long you happen to run — which is
 * exactly the confusion challenge 10 exists to create *deliberately*, so the schedule itself
 * should not create it by accident.
 */
function decayAt(v0: number, decay: Decay, t: number, steps: number): number {
  const clamped = Math.max(0, Math.min(t, steps));
  const frac = steps > 0 ? clamped / steps : 1;
  let v: number;
  switch (decay) {
    case 'exponential':
      // ~5% of v0 at the horizon (v0 · e⁻³), not v0/e. A SOM needs to have actually finished
      // cooling by the end of its own horizon — both to stop moving and, for σ specifically, to
      // let nodes fit individual data points rather than staying smoothed together — and e⁻¹
      // (37% remaining) measured visibly under-converged on the colour cube: a 12×12 map still
      // had a *higher* quantisation error after 3 000 steps than its random initial weights,
      // because the neighbourhood was still wide enough at the end to keep the lattice smoothed
      // into a manifold rather than letting individual nodes settle onto the data.
      v = v0 * Math.exp(-3 * frac);
      break;
    case 'linear':
      // Reaches exactly 0 at the horizon without the floor below — the one decay that needs it.
      v = v0 * (1 - frac);
      break;
    case 'inverse':
      // v0/2 at the horizon, v0/3 at twice it — classic Kohonen inverse-time decay, asymptotic
      // rather than reaching zero on its own.
      v = v0 / (1 + frac);
      break;
  }
  return Math.max(v, DECAY_FLOOR);
}

export function alphaAt(schedule: Schedule, t: number): number {
  return decayAt(schedule.alpha0, schedule.decay, t, schedule.steps);
}

export function sigmaAt(schedule: Schedule, t: number): number {
  return decayAt(schedule.sigma0, schedule.decay, t, schedule.steps);
}
