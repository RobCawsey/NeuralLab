/**
 * Everything the page knows about the Kohonen half, mirroring `run/state.ts`'s shape for the
 * MLP — one plain object, no state library, §12.
 *
 * **Trained on this thread, unlike the MLP.** Slice 4 moved MLP training into a worker because a
 * frame budget and backprop over hundreds of weights do not share a thread comfortably. A SOM
 * step has no backward pass — a nearest-node search and a linear pull over at most a few hundred
 * prototypes — and measures at roughly 240 000 steps/s on a 12×12 map, about twenty times the
 * MLP's own pre-worker throughput. A full 20 000-step run finishes inside a single frame budget
 * with room to spare, so there is nothing here a worker would buy back.
 */

import { Rng, sample, type Dataset } from '@neurallab/core';
import { blobs, circles, colourCube, moons, spirals, xor } from '@neurallab/data';
import {
  bmu,
  createSom,
  createSomTrainer,
  quantisationError,
  somStep,
  topographicError,
  type Decay,
  type Som,
  type SomTrainer,
  type Topology,
} from '@neurallab/som';
import type { SomHistoryPoint } from '../render/somchart.ts';

/**
 * Every generator SOM can train on. Reuses the MLP's own labelled sets — a SOM simply never
 * reads `y` — rather than building parallel unlabelled versions of shapes that already exist;
 * §3's own rule is that "unlabelled" describes training, not the file. The colour cube is the
 * one built for this half specifically and stays the default: three weights *are* a colour.
 */
export const SOM_DATASETS = {
  colourCube: { label: 'Colour cube', build: (n: number, seed: number) => colourCube({ n, seed }) },
  moons: { label: 'Two moons', build: (n: number, seed: number) => moons({ n, seed }) },
  circles: { label: 'Concentric circles', build: (n: number, seed: number) => circles({ n, seed }) },
  blobs: { label: 'Three blobs', build: (n: number, seed: number) => blobs({ n, seed }) },
  spirals: { label: 'Two spirals', build: (n: number, seed: number) => spirals({ n, seed }) },
  xor: { label: 'XOR', build: (n: number, seed: number) => xor({ n, seed }) },
} satisfies Record<string, { label: string; build: (n: number, seed: number) => Dataset }>;

export type SomDatasetKey = keyof typeof SOM_DATASETS;

export function isSomDatasetKey(k: string): k is SomDatasetKey {
  return Object.prototype.hasOwnProperty.call(SOM_DATASETS, k);
}

export interface SomConfig {
  readonly dataset: SomDatasetKey;
  readonly n: number;
  readonly seed: number;
  readonly cols: number;
  readonly rows: number;
  readonly topology: Topology;
  readonly weightSeed: number;
  readonly drawSeed: number;
  readonly alpha0: number;
  readonly sigma0: number;
  readonly decay: Decay;
  readonly targetSteps: number;
}

export interface SomState {
  dataset: SomDatasetKey;
  n: number;
  seed: number;
  cols: number;
  rows: number;
  topology: Topology;
  weightSeed: number;
  drawSeed: number;
  alpha0: number;
  sigma0: number;
  decay: Decay;
  targetSteps: number;

  data: Dataset;
  /** SOM trains unsupervised on every row — there is no split to hold any of it back. */
  rows_: Int32Array;
  som: Som;
  trainer: SomTrainer;
  history: SomHistoryPoint[];
  running: boolean;
  stepsPerSecond: number;
  /** The most recent BMU, for the ring and the readout — set by a step or a hover probe. */
  lastBmu: number;
  hoverNode: number;
}

const DEFAULTS: SomConfig = {
  dataset: 'colourCube',
  n: 1500,
  seed: 4417,
  cols: 12,
  rows: 12,
  topology: 'hex',
  weightSeed: 1,
  drawSeed: 2,
  alpha0: 0.5,
  sigma0: 6,
  decay: 'exponential',
  targetSteps: 3000,
};

/** How often QE/TE are measured — the same reasoning as `evalEvery` on the MLP side: a fixed
 * interval is wrong at both ends of the step range, so this scales with the run's own length. */
export function somEvalEvery(targetSteps: number): number {
  return Math.max(1, Math.round(targetSteps / 100));
}

export function somConfig(s: SomState): SomConfig {
  return {
    dataset: s.dataset,
    n: s.n,
    seed: s.seed,
    cols: s.cols,
    rows: s.rows,
    topology: s.topology,
    weightSeed: s.weightSeed,
    drawSeed: s.drawSeed,
    alpha0: s.alpha0,
    sigma0: s.sigma0,
    decay: s.decay,
    targetSteps: s.targetSteps,
  };
}

function buildSomData(config: SomConfig): Dataset {
  return SOM_DATASETS[config.dataset].build(config.n, config.seed);
}

export function createSomState(): SomState {
  const s: SomState = {
    ...DEFAULTS,
    data: undefined as unknown as Dataset,
    rows_: new Int32Array(0),
    som: undefined as unknown as Som,
    trainer: undefined as unknown as SomTrainer,
    history: [],
    running: false,
    stepsPerSecond: 0,
    lastBmu: -1,
    hoverNode: -1,
  };
  rebuildSomData(s);
  rebuildSom(s);
  return s;
}

export function rebuildSomData(s: SomState): void {
  s.data = buildSomData(somConfig(s));
  s.rows_ = Int32Array.from({ length: s.data.n }, (_, i) => i);
}

/** Rebuild the map and its trainer from the current configuration. Clears the run. */
export function rebuildSom(s: SomState): void {
  s.som = createSom(s.cols, s.rows, s.data.dim, s.topology, new Rng(s.weightSeed));
  s.trainer = createSomTrainer(
    s.som,
    s.rows_,
    { alpha0: s.alpha0, sigma0: s.sigma0, decay: s.decay, steps: s.targetSteps },
    new Rng(s.drawSeed),
  );
  resetSomRun(s);
}

export function resetSomRun(s: SomState): void {
  s.history = [];
  s.running = false;
  s.stepsPerSecond = 0;
  s.lastBmu = -1;
  recordSomPoint(s);
}

function recordSomPoint(s: SomState): void {
  s.history.push({
    step: s.trainer.step,
    qe: quantisationError(s.som, s.data, s.rows_),
    te: topographicError(s.som, s.data, s.rows_),
  });
}

/** Run steps until `untilStep`, recording history at `somEvalEvery` intervals. Main-thread. */
export function runSomSteps(s: SomState, untilStep: number): void {
  const every = somEvalEvery(s.targetSteps);
  while (s.trainer.step < untilStep) {
    const r = somStep(s.trainer, s.data);
    s.lastBmu = r.bmuIndex;
    if (s.trainer.step % every === 0 || s.trainer.step === untilStep) recordSomPoint(s);
  }
}

/** The node nearest a raw data-space point — for hovering the scatter-less colour cube probe. */
export function somBmuAt(s: SomState, point: ArrayLike<number>): number {
  return bmu(s.som, point);
}

export function somSampleRow(s: SomState, row: number): Float32Array {
  return sample(s.data, row);
}

/* ---------------- URL persistence ---------------- */

export function readSomUrl(s: SomState, search: string): void {
  const q = new URLSearchParams(search);

  const dataset = q.get('sdata');
  if (dataset !== null && isSomDatasetKey(dataset)) s.dataset = dataset;

  const topology = q.get('topo');
  if (topology === 'hex' || topology === 'rect') s.topology = topology;

  const decay = q.get('decay');
  if (decay === 'exponential' || decay === 'linear' || decay === 'inverse') s.decay = decay;

  s.n = clampInt(q.get('sn'), 100, 3000, s.n);
  s.seed = clampInt(q.get('sseed'), 1, 9999, s.seed);
  s.cols = clampInt(q.get('cols'), 3, 24, s.cols);
  s.rows = clampInt(q.get('rows'), 3, 24, s.rows);
  s.weightSeed = clampInt(q.get('swseed'), 1, 9999, s.weightSeed);
  s.alpha0 = clampFloat(q.get('alpha0'), 0.01, 1, s.alpha0);
  s.sigma0 = clampFloat(q.get('sigma0'), 0.5, 30, s.sigma0);
  s.targetSteps = clampInt(q.get('ssteps'), 100, 20000, s.targetSteps);
}

export function writeSomUrl(s: SomState, q: URLSearchParams): void {
  q.set('sdata', s.dataset);
  q.set('sn', String(s.n));
  q.set('sseed', String(s.seed));
  q.set('cols', String(s.cols));
  q.set('rows', String(s.rows));
  q.set('topo', s.topology);
  q.set('swseed', String(s.weightSeed));
  q.set('alpha0', String(s.alpha0));
  q.set('sigma0', String(s.sigma0));
  q.set('decay', s.decay);
  q.set('ssteps', String(s.targetSteps));
}

function clampInt(raw: string | null, lo: number, hi: number, fallback: number): number {
  if (raw === null) return fallback;
  const v = Number.parseInt(raw, 10);
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
}

function clampFloat(raw: string | null, lo: number, hi: number, fallback: number): number {
  if (raw === null) return fallback;
  const v = Number.parseFloat(raw);
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
}
