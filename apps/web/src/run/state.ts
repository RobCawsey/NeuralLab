/**
 * Everything the page knows, in one plain object.
 *
 * No state library, and that is settled rather than pending — §12 of the design document. There
 * is one writer and one render pass; the day a panel's DOM needs reconciling rather than
 * retexturing is the day that gets revisited.
 */

import {
  fitStandardiser,
  split,
  standardise,
  Rng,
  type Dataset,
  type Split,
  type Standardiser,
} from '@neurallab/core';
import { GENERATORS, isGeneratorKey, type GeneratorKey } from '@neurallab/data';
import {
  createNet,
  createScratch,
  createTrainer,
  forward,
  initialise,
  isActivation,
  isInitScheme,
  parseHidden,
  type Activation,
  type InitScheme,
  type Net,
  type Scratch,
  type Trainer,
} from '@neurallab/mlp';
import type { EvalPoint, HistoryPoint } from '../render/chart.ts';

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
  /** Total steps the run stops at, so the progress bar and the chart have an end. */
  targetSteps: number;
  /** Rebuilt by `rebuildData`; never assigned from outside. */
  data: Dataset;
  parts: Split;
  standardiser: Standardiser;
  /**
   * The dataset the network actually trains on — standardised once, here.
   *
   * Kept beside the raw one rather than replacing it: the scatter draws data coordinates
   * because that is what the reader is pointing at, and the network only ever sees standardised
   * inputs. Two views of one set, and the panels are explicit about which they hold.
   */
  z: Dataset;
  /** Rebuilt by `rebuildNet`. */
  model: Net;
  scratch: Scratch;
  trainer: Trainer;
  running: boolean;
  history: HistoryPoint[];
  evals: EvalPoint[];
  /** Wall-clock, for the steps/s readout. */
  startedAt: number;
  elapsedMs: number;
  /** The point the forward pass is evaluated at, in **data** coordinates. */
  probe: [number, number];
}

/**
 * How often the full train/validation sets are measured.
 *
 * Every step would be wrong twice over: it is 168 forward passes against the 16 the step itself
 * did, so the run would spend ten times longer measuring than training, and the chart would be
 * denser than the pixels available to draw it.
 */
export const EVAL_EVERY = 10;

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
    model: undefined as unknown as Net,
    scratch: undefined as unknown as Scratch,
    trainer: undefined as unknown as Trainer,
    running: false,
    history: [] as HistoryPoint[],
    evals: [] as EvalPoint[],
    startedAt: 0,
    elapsedMs: 0,
    probe: [0, 0] as [number, number],
  };
  rebuildData(s);
  rebuildNet(s);
  return s;
}

/**
 * Regenerate the dataset and everything derived from it.
 *
 * The split gets its **own** Rng, seeded from the same number. Sharing one generator with the
 * dataset would make the split depend on how many draws the generator happened to take, so
 * changing the sample count would silently reshuffle the split as well — two things moving when
 * the reader moved one.
 */
export function rebuildData(s: AppState): void {
  const gen = GENERATORS[s.dataset];
  s.data = gen.build({ n: s.n, noise: s.noise, seed: s.seed });
  s.parts = split(s.data, s.trainFraction, new Rng(s.seed ^ 0x5f3759df));
  s.standardiser = fitStandardiser(s.data, s.parts.train);
  s.z = standardise(s.data, s.standardiser);
}

/**
 * Rebuild the network from the current architecture and dataset.
 *
 * Input width and output width come from the data, not from a control: a network whose output
 * count disagrees with the number of classes is not a configuration a reader should be able to
 * reach by dragging a slider.
 *
 * Softmax on the output and cross-entropy for the loss, always. Slice 7 can offer alternatives;
 * offering them now would be four dead radio buttons.
 */
export function rebuildNet(s: AppState): void {
  const outputs = Math.max(2, s.data.classes);
  s.model = createNet({
    shape: [s.data.dim, ...s.hidden, outputs],
    hidden: s.hiddenAct,
    output: 'softmax',
    loss: 'crossEntropy',
  });
  initialise(s.model, s.init, new Rng(s.weightSeed));
  s.scratch = createScratch(s.model);
  resetRun(s);
}

/**
 * Throw away the run, keeping the network and the data.
 *
 * The trainer's shuffle Rng is seeded from the weight seed, so "reinitialise and train again"
 * replays exactly — the same weights *and* the same batch order. Seeding it from the data seed
 * instead would make changing the noise slider also reshuffle the batches.
 */
export function resetRun(s: AppState): void {
  s.trainer = createTrainer(
    s.model,
    s.parts.train,
    { learningRate: s.learningRate, batchSize: s.batchSize },
    new Rng(s.weightSeed),
  );
  s.running = false;
  s.history = [];
  s.evals = [];
  s.startedAt = 0;
  s.elapsedMs = 0;
}

/**
 * Evaluate the network at the probe point.
 *
 * The probe is held in data coordinates because that is what the reader is pointing at, and
 * standardised here — the network only ever sees standardised inputs, so a forward pass on raw
 * coordinates would be answering a different question from the one the scatter is asking.
 */
export function evaluateProbe(s: AppState): Float64Array {
  const { mean, sd } = s.standardiser;
  const x = new Float32Array(s.data.dim);
  for (let k = 0; k < s.data.dim; k++) {
    const raw = k === 0 ? s.probe[0] : k === 1 ? s.probe[1] : 0;
    x[k] = (raw - (mean[k] as number)) / (sd[k] as number);
  }
  return forward(s.model, x, s.scratch);
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
 * Everything needed to reproduce this screen, and nothing else. §8 of the design document.
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
  s.targetSteps = clampInt(q.get('steps'), 50, 20000, s.targetSteps);
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
