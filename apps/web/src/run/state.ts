/**
 * Everything the page knows, in one plain object.
 *
 * From slice 4 the page does **not** own the trainer — the worker does. What lives here is a
 * mirror: a network whose weights are replaced from each report, used to draw the graph and to
 * answer the probe. It is never trained on this thread, and nothing here calls `trainStep`.
 *
 * No state library, and that is settled rather than pending — §12. One writer, one render pass.
 */

import { forward, type Net, type Scratch } from '@neurallab/mlp';
import { createScratch } from '@neurallab/mlp';
import type { Dataset, Split, Standardiser } from '@neurallab/core';
import { isGeneratorKey, GENERATORS, type GeneratorKey } from '@neurallab/data';
import { isActivation, isInitScheme, parseHidden, type Activation, type InitScheme } from '@neurallab/mlp';
import { buildData, buildNet, type DataConfig, type NetConfig } from './build.ts';
import type { RunPoint, TrainSetup } from '../workers/protocol.ts';

export type AppStage = 'guided' | 'explorer' | 'lab';
export type NetKind = 'mlp' | 'som';

export interface AppState {
  stage: AppStage;
  net: NetKind;
  dataset: GeneratorKey;
  n: number;
  noise: number;
  seed: number;
  trainFraction: number;
  /** Hidden layer widths. Empty means input wired straight to output — challenge 1. */
  hidden: number[];
  hiddenAct: Activation;
  init: InitScheme;
  /** Separate from `seed`, so reinitialising the weights does not move the data. */
  weightSeed: number;
  learningRate: number;
  batchSize: number;
  targetSteps: number;

  data: Dataset;
  parts: Split;
  standardiser: Standardiser;
  /** The standardised copy — the only one the network ever sees. */
  z: Dataset;
  isVal: Uint8Array;

  /** A mirror of the worker's network, for drawing. Never trained here. */
  model: Net;
  scratch: Scratch;

  /** Everything the chart draws, one point per evaluation. Produced by the worker. */
  points: RunPoint[];
  step: number;
  epoch: number;
  running: boolean;
  stepsPerSecond: number;
  diverged: boolean;
  /** True between asking the worker to rebuild and its `ready`. */
  rebuilding: boolean;

  /** The point the forward pass is evaluated at, in **data** coordinates. */
  probe: [number, number];
}

/**
 * How often the full train/validation sets are measured — derived from the run's length.
 *
 * A fixed interval is wrong at both ends. Measuring every step is 400 forward passes against the
 * 16 a step itself does. A fixed 10 was fine for a 400-step run and badly wrong for a 20 000-step
 * one: 2 000 points for a 300 px chart — six per pixel — costing about 45% of the run, measured
 * as 2 546 steps/s against 6 377 once this scaled.
 */
export function evalEvery(targetSteps: number): number {
  return Math.max(1, Math.round(targetSteps / 200));
}

const DEFAULTS = {
  stage: 'explorer' as AppStage,
  net: 'mlp' as NetKind,
  dataset: 'moons' as GeneratorKey,
  n: 240,
  noise: 0.15,
  seed: 4417,
  trainFraction: 0.7,
  hidden: [8, 8],
  hiddenAct: 'tanh' as Activation,
  init: 'glorot' as InitScheme,
  weightSeed: 1,
  learningRate: 0.1,
  batchSize: 16,
  targetSteps: 400,
};

export function createState(): AppState {
  const s = {
    ...DEFAULTS,
    hidden: [...DEFAULTS.hidden],
    data: undefined as unknown as Dataset,
    parts: undefined as unknown as Split,
    standardiser: undefined as unknown as Standardiser,
    z: undefined as unknown as Dataset,
    isVal: new Uint8Array(0),
    model: undefined as unknown as Net,
    scratch: undefined as unknown as Scratch,
    points: [] as RunPoint[],
    step: 0,
    epoch: 0,
    running: false,
    stepsPerSecond: 0,
    diverged: false,
    rebuilding: false,
    probe: [0, 0] as [number, number],
  };
  rebuildData(s);
  rebuildNet(s);
  return s;
}

export function dataConfig(s: AppState): DataConfig {
  return {
    dataset: s.dataset,
    n: s.n,
    noise: s.noise,
    seed: s.seed,
    trainFraction: s.trainFraction,
  };
}

export function netConfig(s: AppState): NetConfig {
  return {
    hidden: [...s.hidden],
    hiddenAct: s.hiddenAct,
    init: s.init,
    weightSeed: s.weightSeed,
  };
}

/** Everything the worker needs to reproduce this run exactly. */
export function trainSetup(s: AppState, generation: number): TrainSetup {
  return {
    generation,
    data: dataConfig(s),
    net: netConfig(s),
    train: { learningRate: s.learningRate, batchSize: s.batchSize },
    evalEvery: evalEvery(s.targetSteps),
  };
}

export function rebuildData(s: AppState): void {
  const built = buildData(dataConfig(s));
  s.data = built.data;
  s.parts = built.parts;
  s.standardiser = built.standardiser;
  s.z = built.z;
  s.isVal = built.isVal;
}

/** Rebuild the mirror network. The worker rebuilds its own from the same configuration. */
export function rebuildNet(s: AppState): void {
  s.model = buildNet(netConfig(s), s.data.dim, s.data.classes);
  s.scratch = createScratch(s.model);
  resetRun(s);
}

/** Forget the run. The worker is told separately; this is only the page's view of it. */
export function resetRun(s: AppState): void {
  s.points = [];
  s.step = 0;
  s.epoch = 0;
  s.running = false;
  s.stepsPerSecond = 0;
  s.diverged = false;
}

/**
 * Evaluate the mirror network at the probe point.
 *
 * The probe is held in data coordinates because that is what the reader is pointing at, and
 * standardised here — the network only ever sees standardised inputs, so a forward pass on raw
 * coordinates would be answering a different question from the one the scatter is asking.
 */
export function evaluateProbe(s: AppState): Float64Array {
  return forward(s.model, probeInput(s), s.scratch);
}

/** The standardised probe, for the input column of the network graph. */
export function probeInput(s: AppState): Float32Array {
  const { mean, sd } = s.standardiser;
  const x = new Float32Array(s.data.dim);
  for (let k = 0; k < s.data.dim; k++) {
    const raw = k === 0 ? s.probe[0] : k === 1 ? s.probe[1] : 0;
    x[k] = (raw - (mean[k] as number)) / (sd[k] as number);
  }
  return x;
}

/* ---------------- URL persistence ---------------- */

/**
 * Everything needed to reproduce this screen, and nothing else. §8.
 *
 * Read defensively: these values are user-writable and outlive the code reading them. A junk
 * parameter degrades to its default rather than throwing on boot.
 */
export function readUrl(s: AppState, search: string): void {
  const q = new URLSearchParams(search);

  const data = q.get('data');
  if (data !== null && isGeneratorKey(data)) s.dataset = data;

  const net = q.get('net');
  if (net === 'mlp' || net === 'som') s.net = net;

  const stage = q.get('stage');
  if (stage === 'guided' || stage === 'explorer' || stage === 'lab') s.stage = stage;

  const act = q.get('act');
  if (act !== null && isActivation(act) && act !== 'softmax') s.hiddenAct = act;

  const init = q.get('init');
  if (init !== null && isInitScheme(init)) s.init = init;

  const arch = q.get('arch');
  if (arch !== null) s.hidden = parseHidden(arch);

  s.n = clampInt(q.get('n'), 20, 1000, s.n);
  s.seed = clampInt(q.get('seed'), 1, 9999, s.seed);
  s.weightSeed = clampInt(q.get('wseed'), 1, 9999, s.weightSeed);
  s.batchSize = clampInt(q.get('batch'), 1, 256, s.batchSize);
  s.targetSteps = clampInt(q.get('steps'), 100, 20000, s.targetSteps);
  s.noise = clampFloat(q.get('noise'), 0, 0.6, s.noise);
  s.trainFraction = clampFloat(q.get('split'), 0.5, 0.9, s.trainFraction);
  // Up to 500, because challenge 3 needs a rate that visibly destroys the network and the
  // measured figure for that is in the hundreds, not the single digits §6 first guessed.
  s.learningRate = clampFloat(q.get('lr'), 0.0001, 500, s.learningRate);
}

export function writeUrl(s: AppState): string {
  const q = new URLSearchParams();
  q.set('net', s.net);
  q.set('stage', s.stage);
  q.set('data', s.dataset);
  q.set('n', String(s.n));
  q.set('noise', s.noise.toFixed(2));
  q.set('split', s.trainFraction.toFixed(2));
  q.set('seed', String(s.seed));
  q.set('arch', s.hidden.join('-'));
  q.set('act', s.hiddenAct);
  q.set('init', s.init);
  q.set('wseed', String(s.weightSeed));
  q.set('lr', String(s.learningRate));
  q.set('batch', String(s.batchSize));
  q.set('steps', String(s.targetSteps));
  return '?' + q.toString();
}

/** The step count this dataset actually needs — measured, not preferred. See GENERATORS. */
export function suggestedSteps(key: GeneratorKey): number {
  return GENERATORS[key].steps;
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
