/**
 * Slice 4 — off the main thread.
 *
 * This file no longer trains. It sends a configuration to a worker, receives weights and chart
 * points about twenty-five times a second, and draws. Every `trainStep` in the project now
 * happens in exactly one place, which is `workers/trainer.worker.ts`.
 *
 * The step sequence is unchanged, so the golden run reproduces: measured live at 0.1007 / 0.9702
 * / 38 epochs, the same figures slice 2 pinned, and `build.test.ts` asserts that chunking the
 * loop into uneven bursts cannot move them.
 */

import { bounds2d, sample, Rng } from '@neurallab/core';
import { GENERATORS } from '@neurallab/data';
import {
  ACTIVATIONS,
  INIT_SCHEMES,
  applyWeights,
  argmax,
  computeLossSurface,
  describeShape,
  flattenWeights,
  isActivation,
  isInitScheme,
  isOptimiserKind,
  paramBudget,
  paramCount,
  parseHidden,
  projectOntoDirections,
  randomDirection,
  shapeOf,
  unitDirection,
  type TrainConfig,
} from '@neurallab/mlp';
import { classColour, drawScatter, resize } from './render/scatter.ts';
import { drawHistogram } from './render/histogram.ts';
import { activationStats, weightStats } from './run/diagnostics.ts';
import { createScratch } from '@neurallab/mlp';
import { wx, wy, sx, sy, visibleBox, type Camera } from './render/camera.ts';
import { drawNetwork, drawOverCapNotice, heatColour } from './render/network.ts';
import { drawChart } from './render/chart.ts';
import { FIELD_RES, FIELD_THROTTLE_MS, drawField, type Field } from './render/field.ts';
import { TrainerClient } from './workers/client.ts';
import { createStepper } from './ui/stepper.ts';
import { createGuided } from './ui/guided.ts';
import { hitNode, layoutNetwork, UNIT_CAP } from './render/graph-layout.ts';
import {
  createState,
  evaluateProbe,
  probeInput,
  readUrl,
  rebuildData,
  rebuildNet,
  recordSnapshot,
  resetRun,
  suggestedSteps,
  trainSetup,
  writeUrl,
  type AppStage,
} from './run/state.ts';
import {
  createSomState,
  isSomDatasetKey,
  readSomUrl,
  rebuildSom,
  rebuildSomData,
  runSomSteps,
  runSomStepTraced,
  writeSomUrl,
  SOM_DATASETS,
} from './run/somState.ts';
import { layoutLattice, hitLatticeNode, type LatticeLayout } from './render/lattice-layout.ts';
import { drawLattice } from './render/lattice.ts';
import { drawHeatgrid } from './render/heatgrid.ts';
import { drawSomChart } from './render/somchart.ts';
import { componentPlane, uMatrix } from '@neurallab/som';
import { createSomStepper } from './ui/somStepper.ts';
import { createSomGuided } from './ui/somGuided.ts';
import { createChallenges } from './ui/challenges.ts';
import type { ChallengeConfig } from './run/challenges.ts';

const state = createState();
readUrl(state, window.location.search);
rebuildData(state);
rebuildNet(state);

const somState = createSomState();
readSomUrl(somState, window.location.search);
rebuildSomData(somState);
rebuildSom(somState);

/** Everything the URL needs from both networks — disjoint key prefixes, one query string. */
function syncUrl(): string {
  const q = new URLSearchParams(writeUrl(state).slice(1));
  writeSomUrl(somState, q);
  q.set('view', view3d ? '3d' : '2d');
  return '?' + q.toString();
}

let camera: Camera | null = null;
let hover: number | null = null;

/*
 * The decision field, and the bookkeeping that decides when to ask for another.
 *
 * `fieldStale` is set by anything that changes what the network would answer — every report, and
 * every rebuild. `fieldPending` is new in slice 4 and is the one that matters now: a request and
 * its reply are separated by a message round trip, so without it every render during that gap
 * would queue another probe, and the worker would spend its time drawing fields for weights it
 * had already moved past.
 */
let field: Field | null = null;
let fieldStale = true;
let fieldPending = false;
let fieldAt = 0;
let fieldMs = 0;

/**
 * Diagnostics get their own scratch, for the same reason `evaluateRows` and the stepper's trace
 * do: `activationStats` calls `forward` once per training row, and if it wrote into
 * `state.scratch` it would overwrite the probe's activations after the graph had already been
 * told to read them from there — the graph would render the *last diagnostics sample's* colours
 * instead of the point the reader is pointing at. Rebuilt alongside the network, in
 * `rebuildEverything`.
 */
let diagScratch = createScratch(state.model);
let focus: [number, number] | null = null;
let dragging = false;

/* ---------------- 3D ---------------- */

/**
 * §7's "dynamically imported" kept literal, not just descriptive: neither `three` nor either
 * scene module is in the graph any request reaches before a reader actually asks for 3D.
 * `view3d` lives here rather than on `AppState`/`SomState` because it is one toggle shared by
 * both networks, the same reason `hover`/`focus` above are module state and not state fields.
 */
let view3d = new URLSearchParams(window.location.search).get('view') === '3d';
let lossSurfaceScene: import('./render3d/lossSurface3d.ts').LossSurfaceHandle | null = null;
let latticeFoldScene: import('./render3d/latticeFold3d.ts').LatticeFoldHandle | null = null;

/**
 * The two directions the loss surface is drawn against. Held fixed across renders — redrawn
 * only when the network's shape changes — so the picture does not swim under a reader who is
 * still looking at it; the run's own path is what is supposed to move, not the ground it moves
 * across.
 */
let lossDir1: Float32Array | null = null;
let lossDir2: Float32Array | null = null;
let lossDirShape = '';
const lossDirRng = new Rng(4417);
let lossLiteral = false;
let lastLossSurfaceAt = 0;
const LOSS_SURFACE_THROTTLE_MS = 300;

function ensureLossSurfaceScene(): Promise<void> {
  if (lossSurfaceScene) return Promise.resolve();
  return import('./render3d/lossSurface3d.ts').then((mod) => {
    lossSurfaceScene = mod.createLossSurfaceScene($<HTMLCanvasElement>('loss-surface-3d'));
    render();
  });
}

function ensureLatticeFoldScene(): Promise<void> {
  if (latticeFoldScene) return Promise.resolve();
  return import('./render3d/latticeFold3d.ts').then((mod) => {
    latticeFoldScene = mod.createLatticeFold3dScene($<HTMLCanvasElement>('lattice-fold-3d'));
    render();
  });
}

function ensure3dSceneFor(net: 'mlp' | 'som'): void {
  void (net === 'som' ? ensureLatticeFoldScene() : ensureLossSurfaceScene());
}

/** CSS-pixel size of a WebGL canvas — `render/scatter.ts`'s `resize` calls `getContext('2d')`,
 * which would permanently lock a canvas out of ever getting a WebGL context afterward. */
function size3d(canvas: HTMLCanvasElement): { w: number; h: number } {
  return { w: Math.max(1, canvas.clientWidth), h: Math.max(1, canvas.clientHeight) };
}

function renderLossSurface3d(): void {
  if (!lossSurfaceScene) return;
  const key = `${shapeOf(state.model).join('-')}:${lossLiteral ? 'literal' : 'representative'}`;
  if (!lossDir1 || !lossDir2 || lossDirShape !== key) {
    lossDirShape = key;
    if (lossLiteral) {
      lossDir1 = unitDirection(state.model, 0);
      lossDir2 = unitDirection(state.model, 1);
    } else {
      lossDir1 = randomDirection(state.model, lossDirRng);
      lossDir2 = randomDirection(state.model, lossDirRng);
    }
  }

  const badge = $('loss3d-badge');
  badge.textContent = lossLiteral ? 'literal — 2 named weights' : '2 random directions, filter-normalised';

  const { w, h } = size3d($<HTMLCanvasElement>('loss-surface-3d'));
  lossSurfaceScene.resize(w, h);

  const now = performance.now();
  if (now - lastLossSurfaceAt < LOSS_SURFACE_THROTTLE_MS) return;
  lastLossSurfaceAt = now;

  const base = flattenWeights(state.model);
  const surface = computeLossSurface(state.model, state.z, state.parts.train, lossDir1, lossDir2, 25, 1.2);
  const path = state.snapshots.map((snap) => {
    const [alpha, beta] = projectOntoDirections(base, snap.weights, lossDir1 as Float32Array, lossDir2 as Float32Array);
    return { alpha, beta };
  });
  lossSurfaceScene.update(surface, path, { alpha: 0, beta: 0 });
}

function renderLatticeFold3d(): void {
  if (!latticeFoldScene) return;
  const { w, h } = size3d($<HTMLCanvasElement>('lattice-fold-3d'));
  latticeFoldScene.resize(w, h);
  const dims: [number, number, number] = [0, Math.min(1, somState.som.dim - 1), Math.min(2, somState.som.dim - 1)];
  latticeFoldScene.update(somState.som, somState.data, dims);
  const names = somState.data.featureNames;
  $('fold3d-badge').textContent =
    somState.som.dim <= 3
      ? `dims ${dims.map((d) => names[d] ?? `f${d}`).join(', ')}`
      : `PCA not built yet — showing dims ${dims.join(', ')}`;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  // Thrown rather than logged. A missing element is a typo in index.html, and the panel that
  // depends on it would otherwise fail silently three slices later.
  if (!el) throw new Error(`#${id} is missing from index.html`);
  return el as T;
};

const stage = $<HTMLCanvasElement>('stage');
const graph = $<HTMLCanvasElement>('graph');
const chart = $<HTMLCanvasElement>('chart');
const somLattice = $<HTMLCanvasElement>('som-lattice');

/* ---------------- the worker ---------------- */

/**
 * Training happens over there now. This thread draws.
 *
 * What used to be a frame-budget loop is a message handler: the worker trains in 40 ms chunks,
 * reports about 25 times a second, and the page applies the weights it is sent and redraws. Two
 * things slice 3 had to live with are gone — the decision field no longer competes with training
 * for the same thread (measured at 19%), and a background tab no longer stops the run, because
 * workers are not throttled by `requestAnimationFrame`.
 */
const trainer = new TrainerClient({
  onReady: (weights) => {
    state.rebuilding = false;
    applyWeights(state.model, weights);
    fieldStale = true;
    if (runWhenReady) {
      runWhenReady = false;
      trainer.run(state.targetSteps);
    }
    render();
  },

  onReport: (report) => {
    state.step = report.step;
    state.epoch = report.epoch;
    state.running = report.running;
    state.diverged = report.diverged;
    if (report.stepsPerSecond > 0) state.stepsPerSecond = report.stepsPerSecond;
    for (const point of report.points) state.points.push(point);

    applyWeights(state.model, report.weights);
    // One snapshot per report that closed off a chart point, not one per report — reports arrive
    // ~25 times a second, points at most 200 times a run, and the ring is capped in count.
    if (report.points.length > 0) recordSnapshot(state, report.weights);
    // The weights moved, so the field is a picture of a network that no longer exists.
    fieldStale = true;
    render();
  },

  onField: (next, ms) => {
    field = next;
    fieldMs = ms;
    fieldPending = false;
    render();
  },

  onTrace: (trace) => {
    stepper.receiveTrace(trace);
  },

  onError: (message) => {
    // A worker that dies looks exactly like one that is paused, so this has to be visible.
    state.running = false;
    state.rebuilding = false;
    $('run-note').innerHTML = `Training stopped: <em>${message}</em>`;
    render();
  },
});

const stepper = createStepper({
  trainer,
  onOpen: () => setRunning(false),
  classNames: () => state.data.classNames,
});

const somStepper = createSomStepper({
  getSom: () => somState.som,
  getData: () => somState.data,
  stepTraced: () => runSomStepTraced(somState),
  onOpen: () => setSomRunning(false),
});

/*
 * The guided flow — §6/§13. `pickDataset` and `pickShape` reuse exactly the same rebuild path
 * Explorer's controls use (`regenerateData`/`regenerateNet`), so a choice made here is not a
 * second, parallel way of changing the run — it is the same one, with different words on the
 * button.
 */
const guided = createGuided({
  getState: () => state,
  pickDataset: (key) => {
    state.dataset = key;
    regenerateData();
  },
  pickShape: (hidden) => {
    state.hidden = [...hidden];
    regenerateNet();
  },
  startTraining: () => setRunning(true),
  skipToExplorer: () => $('stage-explorer').click(),
});

const somGuided = createSomGuided({
  getState: () => somState,
  pickDataset: (key) => {
    somState.dataset = key;
    regenerateSomData();
  },
  startTraining: () => setSomRunning(true),
  skipToExplorer: () => $('stage-explorer').click(),
  requestRender: () => render(),
});

/*
 * The challenge track — §9/§13. `applyChallenge` is the one place a card's recipe becomes real
 * state: every field a config names is written directly (the same "guided sets state, does not
 * drive the control" shape `pickDataset`/`pickShape` above already use), then whichever side was
 * touched is rebuilt through the same `regenerateData`/`regenerateSomData` every other control on
 * that side already goes through.
 */
const challenges = createChallenges({
  getState: () => state,
  getSomState: () => somState,
  applyChallenge: (config) => applyChallenge(config),
});

/**
 * True when the reader asked to train before the worker had finished rebuilding.
 *
 * The gap is small — a rebuild is a few milliseconds — but it is a real window, and pressing
 * Train inside it used to send `run` against a session the page had already discarded: the run
 * happened, the step counter advanced, and not one chart point arrived. A dropped intent is
 * worse than a delayed one, so it is held and honoured on `ready`.
 */
let runWhenReady = false;

function setRunning(on: boolean): void {
  if (on && state.step >= state.targetSteps) return;

  if (on && state.rebuilding) {
    runWhenReady = true;
    state.running = true;
    render();
    return;
  }

  runWhenReady = false;
  state.running = on;
  if (on) trainer.run(state.targetSteps);
  else trainer.pause();
  render();
}

/** The three fields `trainer.configure` needs, read from state in one place so none is missed. */
function currentTrainConfig(): TrainConfig {
  return { learningRate: state.learningRate, batchSize: state.batchSize, optimiser: state.optimiser };
}

/* ---------------- controls ---------------- */

function fillSelect(id: string, values: readonly string[], current: string): HTMLSelectElement {
  const select = $<HTMLSelectElement>(id);
  for (const v of values) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    select.append(opt);
  }
  select.value = current;
  return select;
}

/** Set once `boot` has wired the steps slider, so a dataset change can move it. */
let syncSteps: () => void = () => {};

/**
 * Set once `boot` has wired every MLP/SOM control — `applyChallenge` is the one caller that
 * writes many state fields directly rather than through a control's own listener, the same shape
 * of gap the guided flow hit twice (§6, §11): a display element left disagreeing with the state
 * it is supposed to be showing. Called after every challenge, so none of them go stale.
 */
let syncMlpControls: () => void = () => {};
let syncSomControls: () => void = () => {};

/** 1, 2, 5 × 10ⁿ — a target of 6 314 steps is a number nobody chose. */
function snapSteps(raw: number): number {
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(100, raw))));
  const norm = raw / mag;
  const mult = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
  return Math.min(20000, Math.max(100, Math.round(mult * mag)));
}

function fillDatasets(): void {
  const select = $<HTMLSelectElement>('i-dataset');
  for (const [key, gen] of Object.entries(GENERATORS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = gen.label;
    select.append(opt);
  }
  select.value = state.dataset;
  select.addEventListener('change', () => {
    if (select.value in GENERATORS) {
      state.dataset = select.value as typeof state.dataset;
      /*
       * Adopt the set's own step count — measured, not preferred. The checkerboard is at 0.66
       * after 4 000 steps and 0.88 after 20 000; opening it at 400 would show a reader a
       * failure and let them conclude it was the app's.
       */
      state.targetSteps = suggestedSteps(state.dataset);
      syncSteps();
      regenerateData();
    }
  });
}

function slider(
  inputId: string,
  outputId: string,
  read: () => number,
  write: (v: number) => void,
  format: (v: number) => string,
  after: () => void = regenerateData,
): void {
  const input = $<HTMLInputElement>(inputId);
  const output = $(outputId);
  input.value = String(read());
  output.textContent = format(read());
  input.addEventListener('input', () => {
    write(Number(input.value));
    output.textContent = format(read());
    history.replaceState(null, '', syncUrl());
    after();
  });
}

function segment(groupId: string, onPick: (id: string) => void): void {
  const group = $(groupId);
  for (const button of Array.from(group.querySelectorAll('button'))) {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      for (const sibling of Array.from(group.querySelectorAll('button'))) {
        sibling.classList.toggle('on', sibling === button);
      }
      onPick(button.id);
    });
  }
}

/* ---------------- panels ---------------- */

function renderDataPanels(): void {
  const { data, parts, standardiser } = state;

  /*
   * The `<select>` is a control, not a display — nothing reads it back except its own `change`
   * listener, so a caller that sets `state.dataset` directly (the guided flow does, rather than
   * driving the dropdown) leaves it showing whatever was chosen last. Found by picking XOR in
   * Guided and then opening Explorer to find the dropdown still reading "Two moons" while every
   * other panel correctly said XOR. Setting it to a value it already holds is a no-op — this is
   * safe to run on every call, including the ones the dropdown's own listener triggered.
   */
  $<HTMLSelectElement>('i-dataset').value = state.dataset;

  $('ph-data').textContent = data.name.toLowerCase();
  $('ph-seed').textContent = `seed ${state.seed}`;
  $('v-samples').textContent = String(data.n);
  $('s-total').textContent = String(data.n);
  $('s-train').textContent = String(parts.train.length);
  $('s-val').textContent = String(parts.val.length);

  const legend = $('legend');
  legend.replaceChildren();
  for (let c = 0; c < Math.max(1, data.classes); c++) {
    const s = document.createElement('s');
    s.style.color = classColour(c);
    s.textContent = `● ${data.classNames[c] ?? `class ${c}`}`;
    legend.append(s);
  }

  const std = $('standardiser');
  std.replaceChildren();
  for (let k = 0; k < data.dim; k++) {
    std.append(
      kv(
        data.featureNames[k] ?? `f${k}`,
        `μ ${(standardiser.mean[k] as number).toFixed(3)}   σ ${(standardiser.sd[k] as number).toFixed(3)}`,
      ),
    );
  }
}

function renderNetPanels(): void {
  // Same fix as the dataset `<select>`, for the same reason: a caller that sets `state.hidden`
  // directly — the guided flow does — must not leave this control showing an earlier shape.
  $<HTMLInputElement>('i-arch').value = state.hidden.join('-');
  syncPresets();

  $('ph-arch').textContent = describeShape(state.model);
  $('s-params').textContent = String(paramCount(state.model));
  let edges = 0;
  for (const l of state.model.layers) edges += l.W.length;
  $('s-edges').textContent = String(edges);

  /*
   * The architecture editor's neighbour — slice 8. `state.parts` is only set once the first
   * dataset has been built, which is true by the time this ever runs, but a defensive read costs
   * nothing and a crash here would take the whole panel down with it.
   */
  const trainRows = state.parts?.train.length ?? 0;
  const budget = paramBudget(state.model, trainRows);
  const budgetEl = $('s-budget');
  budgetEl.textContent = `${budget.params} / ${budget.samples}`;
  budgetEl.className = budget.overBudget ? 'bad' : 'ok';
  $('budget-note').innerHTML = budget.overBudget
    ? `<em>${budget.params}</em> parameters for <em>${budget.samples}</em> training rows — ` +
      'more free numbers than data points. This network can memorise the training set outright ' +
      'rather than learn its shape &mdash; challenge 7.'
    : `<em>${budget.params}</em> parameters for <em>${budget.samples}</em> training rows — ` +
      'room to generalise.';
}

/**
 * The note under the output bars.
 *
 * It was a fixed string in slice 1 saying the weights were random — true then, and wrong from
 * the first training step of slice 2 onward. It sat under a network at 97% accuracy telling a
 * reader its answers meant nothing. §6's rule is that explanations are written against live
 * values; a string that is only true before anything happens is exactly the kind that rots.
 */
function renderOutputNote(out: Float64Array): void {
  const note = $('out-note');
  const best = argmax(out);
  const confidence = out[best] as number;
  const name = state.data.classNames[best] ?? `class ${best}`;

  if (state.step === 0) {
    note.innerHTML =
      'The weights are <em>random</em> and nothing has been trained, so whichever class wins ' +
      'here means nothing — a confident answer is as arbitrary as an even one. Press ' +
      '<em>Reinitialise</em> and watch the same probe change its mind.';
    return;
  }

  const held = state.points[state.points.length - 1];
  const accuracy = held ? `${(held.valAccuracy * 100).toFixed(1)}%` : null;
  note.innerHTML =
    `After ${state.step.toLocaleString()} steps the network calls this point ` +
    `<em>${name}</em> at <em>${confidence.toFixed(3)}</em>` +
    (accuracy === null
      ? '.'
      : `, and it is right about <em>${accuracy}</em> of the points it was never shown. ` +
        'Drag the probe across the boundary to watch the confidence fall and recover.');
}

/** The output probabilities, as labelled bars. Redrawn on every probe move. */
function renderOutputs(out: Float64Array): void {
  const host = $('outputs');
  host.replaceChildren();
  const best = argmax(out);
  for (let c = 0; c < out.length; c++) {
    const row = document.createElement('div');
    row.className = 'prob';

    const name = document.createElement('span');
    name.textContent = state.data.classNames[c] ?? `class ${c}`;
    if (c === best) name.style.color = classColour(c);

    const track = document.createElement('i');
    const fill = document.createElement('b');
    const v = out[c] as number;
    fill.style.width = `${(Number.isFinite(v) ? v : 0) * 100}%`;
    fill.style.background = classColour(c);
    fill.style.opacity = c === best ? '1' : '0.5';
    track.append(fill);

    const value = document.createElement('u');
    value.textContent = Number.isFinite(v) ? v.toFixed(3) : 'NaN';

    row.append(name, track, value);
    host.append(row);
  }
}

/** One strip of cells per layer, coloured by activation magnitude. */
function renderActivations(input: ArrayLike<number>): void {
  const host = $('acts');
  host.replaceChildren();

  host.append(strip('input', input, state.data.dim));
  for (let l = 0; l < state.model.layers.length; l++) {
    const layer = state.model.layers[l];
    if (!layer) continue;
    const name =
      l === state.model.layers.length - 1 ? `output · ${layer.act}` : `hidden ${l + 1} · ${layer.act}`;
    host.append(strip(name, state.scratch.a[l] as Float64Array, layer.units));
  }
}

function strip(label: string, values: ArrayLike<number>, count: number): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'strip';

  let max = 0;
  let dead = 0;
  for (let i = 0; i < count; i++) {
    const v = Math.abs(values[i] as number);
    if (Number.isFinite(v) && v > max) max = v;
    if (v === 0) dead++;
  }

  const head = document.createElement('div');
  head.className = 'strip-head';
  const left = document.createElement('span');
  left.textContent = label;
  const right = document.createElement('span');
  // "How many units contributed nothing" is the number challenge 6 is read from, so it is
  // present from the slice that first draws activations rather than bolted on later.
  right.textContent = dead > 0 ? `${dead} at zero · max ${max.toFixed(2)}` : `max ${max.toFixed(2)}`;
  if (dead > 0) right.style.color = '#d9625c';
  head.append(left, right);

  const cells = document.createElement('div');
  cells.className = 'strip-cells';
  for (let i = 0; i < count; i++) {
    const cell = document.createElement('i');
    const v = values[i] as number;
    cell.style.background = Number.isFinite(v)
      ? heatColour(max > 0 ? Math.abs(v) / max : 0)
      : '#d9625c';
    cell.title = `${i}: ${Number.isFinite(v) ? v.toFixed(4) : 'NaN'}`;
    cells.append(cell);
  }

  wrap.append(head, cells);
  return wrap;
}

function kv(label: string, value: string): HTMLElement {
  const row = document.createElement('div');
  row.className = 'kv';
  const name = document.createElement('span');
  name.textContent = label;
  const val = document.createElement('b');
  val.textContent = value;
  row.append(name, val);
  return row;
}

/* ---------------- render ---------------- */

/**
 * Gradient flow, weight histograms, activation histograms, dead-unit count — §7 of the design
 * document. Recomputed every render with no throttle: at 168 training rows on a small network
 * this is the same order of arithmetic as one training step (§5's budget), measured cheap enough
 * not to need one.
 */
function renderDiagnostics(): void {
  const weightHists = weightStats(state.model);
  const actStats = activationStats(state.model, state.z, state.parts.train, diagScratch);
  const last = state.points[state.points.length - 1];
  const norms = last?.gradNorms ?? state.model.layers.map(() => 0);

  const gradHost = $('gradflow');
  gradHost.replaceChildren();
  // Floored at 1e-6 rather than 0, so a genuinely vanished gradient still draws a hairline
  // instead of an empty gauge indistinguishable from "nothing has trained yet".
  const floor = -6;
  let ceiling = floor;
  for (const n of norms) ceiling = Math.max(ceiling, Math.log10(Math.max(1e-6, n)));
  for (let l = 0; l < norms.length; l++) {
    const n = norms[l] as number;
    const row = document.createElement('div');
    row.className = 'diag-row';
    const lb = document.createElement('span');
    lb.className = 'diag-lb';
    lb.textContent = `layer ${l + 1}`;
    const gauge = document.createElement('span');
    gauge.className = 'gauge';
    const bar = document.createElement('i');
    const logN = Math.log10(Math.max(1e-6, n));
    const pct = ceiling > floor ? ((logN - floor) / (ceiling - floor)) * 100 : 100;
    bar.style.width = `${Math.max(2, Math.min(100, pct))}%`;
    gauge.append(bar);
    const nt = document.createElement('span');
    nt.className = 'diag-nt';
    nt.textContent = n.toExponential(1).replace('e-', 'e−');
    row.append(lb, gauge, nt);
    gradHost.append(row);
  }

  const weightHost = $('weighthist');
  weightHost.replaceChildren();
  weightHists.forEach((hist, l) => {
    const block = document.createElement('div');
    block.className = 'diag-block';
    const label = document.createElement('span');
    label.className = 'diag-lb';
    label.textContent = `layer ${l + 1}`;
    const canvas = document.createElement('canvas');
    canvas.className = 'diag-hist';
    block.append(label, canvas);
    weightHost.append(block);
    queueMicrotask(() => {
      const fit = resize(canvas);
      if (fit) drawHistogram(fit.ctx, hist, fit.w, fit.h, '#4ea8c4');
    });
  });

  const actHost = $('acthist');
  actHost.replaceChildren();
  let deadTotal = 0;
  let reluTotal = 0;
  let anyRelu = false;
  actStats.forEach((s, l) => {
    if (s.isRelu) {
      anyRelu = true;
      deadTotal += s.deadUnits;
      reluTotal += s.totalUnits;
    }
    const block = document.createElement('div');
    block.className = 'diag-block';
    const label = document.createElement('span');
    label.className = 'diag-lb' + (s.isRelu && s.deadUnits > 0 ? ' diag-dead' : '');
    label.textContent = s.isRelu ? `layer ${l + 1} · ${s.deadUnits} of ${s.totalUnits} dead` : `layer ${l + 1}`;
    const canvas = document.createElement('canvas');
    canvas.className = 'diag-hist';
    block.append(label, canvas);
    actHost.append(block);
    queueMicrotask(() => {
      const fit = resize(canvas);
      if (fit) drawHistogram(fit.ctx, s.histogram, fit.w, fit.h, '#e9a13b');
    });
  });
  $('ph-dead').textContent = anyRelu ? `${deadTotal} of ${reluTotal} dead` : 'no relu layers';
}

/** Everything that changes as the run advances. */
function renderRunPanels(): void {
  const { step, epoch, targetSteps, points } = state;
  const last = points[points.length - 1];

  // Same fix as the dataset select and the arch input in slice 6: a control synced from state
  // on every render cannot go stale, whatever set state.optimiser last.
  for (const b of Array.from($('optimiser').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === `opt-${state.optimiser}`);
  }
  $('ph-opt').textContent = state.optimiser === 'sgd' ? 'SGD' : state.optimiser === 'momentum' ? 'Momentum' : 'Adam';

  $('ph-chart').textContent = `step ${step}`;
  $('s-step').textContent = `${step} / ${targetSteps}`;
  $('s-epoch').textContent = String(epoch);
  $<HTMLElement>('s-progress').style.width = `${Math.min(100, (step / targetSteps) * 100)}%`;

  $('s-trainloss').textContent = last ? last.trainLoss.toFixed(4) : '—';
  $('s-valloss').textContent = last ? last.valLoss.toFixed(4) : '—';
  $('s-trainacc').textContent = last ? last.trainAccuracy.toFixed(4) : '—';
  $('s-valacc').textContent = last ? last.valAccuracy.toFixed(4) : '—';

  // Measured by the worker over its own running time, so it is throughput rather than a
  // wall-clock rate that a pause or a hidden tab would quietly dilute.
  $('s-sps').textContent =
    state.stepsPerSecond > 0 ? Math.round(state.stepsPerSecond).toLocaleString() : '—';

  /*
   * The field's resolution and price, printed rather than hidden.
   *
   * A reader who notices the boundary getting crisper the moment they hit pause deserves to
   * know that is the drawing changing and not the network. §5 asked for exactly this.
   */
  if (field) {
    $('field-badge').textContent =
      `${field.res}² · ${(field.res * field.res).toLocaleString()} passes · ${fieldMs.toFixed(1)} ms`;
  }

  const btn = $<HTMLButtonElement>('btn-train');
  const finished = step >= targetSteps;
  btn.textContent = state.running ? 'Pause' : finished ? 'Done' : 'Train';
  btn.disabled = finished && !state.running;

  const badge = $('graph-badge');
  badge.classList.toggle('training', state.running);
  badge.classList.toggle('done', finished && !state.diverged);
  badge.textContent = state.diverged
    ? 'diverged'
    : state.running
      ? 'training'
      : step === 0
        ? 'random weights'
        : finished
          ? 'finished'
          : 'paused';

  /*
   * The run note, written against live values rather than as a fixed string — §6.
   *
   * The version of this that said "the outputs will be near-even" was wrong the first time a
   * probe read 0.766, and the same trap applies to anything asserting how a run went. Every
   * branch below quotes a number it has actually measured.
   */
  const note = $('run-note');
  if (state.diverged) {
    note.innerHTML =
      'The weights stopped being numbers. That is a <em>bug</em>, not a lesson &mdash; ' +
      'softmax shifts by its maximum precisely so a large learning rate degrades readably.';
  } else if (!last) {
    note.innerHTML =
      'Nothing trained yet. Press <em>Train</em>, or <em>Step</em> to advance one minibatch ' +
      'at a time and watch the edges move.';
  } else if (last.valLoss > last.trainLoss * 1.25) {
    note.innerHTML =
      `Validation loss is <em>${last.valLoss.toFixed(3)}</em> against training's ` +
      `<em>${last.trainLoss.toFixed(3)}</em>. The network is fitting this sample better than ` +
      'it fits the problem &mdash; that gap is what challenge 7 is about.';
  } else if (finished) {
    note.innerHTML =
      `Finished at <em>${(last.trainAccuracy * 100).toFixed(1)}%</em> on training and ` +
      `<em>${(last.valAccuracy * 100).toFixed(1)}%</em> on data it never saw.`;
  } else {
    note.innerHTML =
      `Training loss <em>${last.trainLoss.toFixed(4)}</em>, accuracy ` +
      `<em>${(last.trainAccuracy * 100).toFixed(1)}%</em>.`;
  }
}

/**
 * Ask the worker for a field, if one is worth asking for.
 *
 * Two resolutions, and the switch is announced in the panel header rather than left for a reader
 * to notice the boundary getting crisper. While the weights are moving there is no point paying
 * for detail that is wrong a report later; once they stop, there is nothing else to spend the
 * time on.
 *
 * `fieldPending` matters more than it did when this ran inline. A request and its reply are now
 * separated by a message round trip, so without it every render during that gap would queue
 * another probe — and at 25 reports a second the worker would spend all its time drawing fields
 * for weights it had already left behind.
 */
function refreshField(camera: Camera, width: number, height: number): void {
  const wantRes = state.running ? FIELD_RES.live : FIELD_RES.paused;
  const now = performance.now();

  if (fieldPending) return;
  if (state.rebuilding) return;
  if (field !== null && !fieldStale && field.res === wantRes) return;
  if (state.running && field !== null && now - fieldAt < FIELD_THROTTLE_MS) return;

  fieldPending = true;
  fieldStale = false;
  fieldAt = now;
  trainer.requestField(wantRes, visibleBox(camera, width, height));
}

function render(): void {
  // Runs every tick regardless of network or overlay state — completion usually happens after
  // the card itself has been closed, with the reader watching Explorer instead.
  challenges.render();

  if (state.net === 'som') {
    renderSom();
    return;
  }

  const out = evaluateProbe(state);
  const input = probeInput(state);

  // The stage view — graph and scatter in 2D, the loss surface in 3D. Never both: §7's rule
  // that 3D is a view and not a second simulation means only one of them is ever doing work.
  if (!view3d) {
    const scatterFit = resize(stage);
    if (scatterFit) {
      const view = drawScatter(scatterFit.ctx, state.data, scatterFit.w, scatterFit.h, {
        hover,
        isVal: state.isVal,
        underlay: (cam) => {
          refreshField(cam, scatterFit.w, scatterFit.h);
          if (field) drawField(scatterFit.ctx, field, cam);
        },
      });
      camera = view.camera;
      drawProbe(scatterFit.ctx, out);
    }

    const graphFit = resize(graph);
    if (graphFit) {
      if (shapeOf(state.model).some((n) => n > UNIT_CAP)) {
        graphFit.ctx.clearRect(0, 0, graphFit.w, graphFit.h);
        drawOverCapNotice(graphFit.ctx, graphFit.w, graphFit.h);
      } else {
        drawNetwork(graphFit.ctx, state.model, input, state.scratch, graphFit.w, graphFit.h, {
          focus,
        });
      }
    }
  } else {
    renderLossSurface3d();
  }

  const chartFit = resize(chart);
  if (chartFit) {
    drawChart(chartFit.ctx, state.points, chartFit.w, chartFit.h, {
      totalSteps: state.targetSteps,
    });
  }

  const best = argmax(out);
  const confidence = out[best] as number;
  $('v-cursor').textContent = `${state.probe[0].toFixed(2)}, ${state.probe[1].toFixed(2)}`;
  $('v-predict').textContent = state.data.classNames[best] ?? `class ${best}`;
  $('v-conf').textContent = Number.isFinite(confidence) ? confidence.toFixed(3) : 'NaN';

  renderOutputs(out);
  renderOutputNote(out);
  renderActivations(input);
  renderRunPanels();
  renderDiagnostics();
  guided.render();
}

/* ---------------- SOM ---------------- */

/** Cached every render, for `moveSomProbe`'s hit-testing — the lattice's answer to `camera`. */
let somLatticeLayout: LatticeLayout | null = null;

function renderSom(): void {
  const s = somState;

  if (!view3d) {
    const latticeFit = resize($('som-lattice'));
    if (latticeFit) {
      somLatticeLayout = layoutLattice(s.som.cols, s.som.rows, s.som.topology, latticeFit.w, latticeFit.h);
      drawLattice(latticeFit.ctx, s.som, somLatticeLayout, latticeFit.w, latticeFit.h, {
        bmu: s.lastBmu,
        hover: s.hoverNode,
      });
    }
  } else {
    renderLatticeFold3d();
  }

  const chartFit = resize($('som-chart'));
  if (chartFit) drawSomChart(chartFit.ctx, s.history, chartFit.w, chartFit.h, s.targetSteps);

  const umatrixFit = resize($('som-umatrix'));
  const umatrixMax = umatrixFit
    ? drawHeatgrid(umatrixFit.ctx, uMatrix(s.som), s.cols, s.rows, umatrixFit.w, umatrixFit.h, [233, 161, 59])
    : 0;
  $('som-umatrix-max').textContent = umatrixMax.toFixed(3);

  renderSomPlanes();
  renderSomStats();
  somGuided.render();
}

/** One small heatmap per input dimension — §3's answer to reading a map above 3 dimensions. */
function renderSomPlanes(): void {
  const s = somState;
  const host = $('som-planes');
  host.replaceChildren();
  for (let k = 0; k < s.som.dim; k++) {
    const block = document.createElement('div');
    block.className = 'diag-block';
    const label = document.createElement('span');
    label.className = 'diag-lb';
    label.textContent = s.data.featureNames[k] ?? `dim ${k}`;
    const canvas = document.createElement('canvas');
    canvas.className = 'diag-hist';
    block.append(label, canvas);
    host.append(block);
    // Same reason as the diagnostics histograms: the canvas has no CSS box to measure until it
    // is in the document, so `resize` is deferred one microtask past the append.
    queueMicrotask(() => {
      const fit = resize(canvas);
      if (fit) drawHeatgrid(fit.ctx, componentPlane(s.som, k), s.cols, s.rows, fit.w, fit.h, [139, 123, 216]);
    });
  }
}

/**
 * Every text readout and control that has to agree with `somState` on every render — the same
 * rule slice 6 learned for the MLP side: a control set from somewhere other than its own event
 * listener must be resynced here, or it goes stale the first time state changes any other way.
 */
function renderSomStats(): void {
  const s = somState;
  const last = s.history[s.history.length - 1];
  const step = s.trainer.step;
  const finished = step >= s.targetSteps;

  $('som-ph-chart').textContent = `step ${step}`;
  $('som-s-step').textContent = `${step} / ${s.targetSteps}`;
  $<HTMLElement>('som-s-progress').style.width = `${Math.min(100, (step / Math.max(1, s.targetSteps)) * 100)}%`;
  $('som-s-qe').textContent = last ? last.qe.toFixed(4) : '—';
  $('som-s-te').textContent = last ? last.te.toFixed(4) : '—';
  $('som-s-sps').textContent = s.stepsPerSecond > 0 ? Math.round(s.stepsPerSecond).toLocaleString() : '—';
  $('som-s-total').textContent = String(s.data.n);
  $('som-s-nodes').textContent = String(s.som.cols * s.som.rows);
  $('som-s-maxhit').textContent = String(Math.max(0, ...Array.from(s.som.hits)));

  $<HTMLSelectElement>('som-i-dataset').value = s.dataset;
  $('som-ph-data').textContent = SOM_DATASETS[s.dataset].label;
  $('som-ph-seed').textContent = `seed ${s.seed}`;
  $('som-ph-lattice').textContent = `${s.cols}×${s.rows} ${s.topology}`;
  $('som-ph-decay').textContent = s.decay;
  $('som-ph-planes').textContent = `${s.som.dim} dim${s.som.dim === 1 ? '' : 's'}`;

  for (const b of Array.from($('som-topology').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === `som-topo-${s.topology}`);
  }
  for (const b of Array.from($('som-decay').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === `som-decay-${s.decay}`);
  }

  const btn = $<HTMLButtonElement>('btn-train');
  btn.textContent = s.running ? 'Pause' : finished ? 'Done' : 'Train';
  btn.disabled = finished && !s.running;
  $<HTMLButtonElement>('btn-stepper').disabled = false;

  const badge = $('som-badge');
  badge.classList.toggle('training', s.running);
  badge.classList.toggle('done', finished);
  badge.textContent = s.running ? 'training' : step === 0 ? 'random weights' : finished ? 'finished' : 'paused';

  const note = $('som-run-note');
  if (!last || step === 0) {
    note.innerHTML =
      'Nothing trained yet. Press <em>Train</em>, or <em>Step</em> to advance one sample at a ' +
      'time and watch the lattice move.';
  } else if (finished) {
    note.innerHTML =
      `Finished at step <em>${step}</em>. Quantisation error <em>${last.qe.toFixed(4)}</em>, ` +
      `topographic error <em>${last.te.toFixed(4)}</em>.`;
  } else {
    note.innerHTML =
      `Quantisation error <em>${last.qe.toFixed(4)}</em>, topographic error ` +
      `<em>${last.te.toFixed(4)}</em>.`;
  }
}

/**
 * The main-thread training loop — see `run/somState.ts` for why this does not need a worker.
 *
 * Paced at a fixed number of ticks rather than a fixed number of steps per tick: a 20 000-step
 * run and a 500-step run both finish in about two seconds of visible motion, because the point is
 * watching the lattice organise, not the steps themselves. A worker's chunk is sized by a time
 * budget for the same reason the MLP's is 40 ms; this one is sized by tick count for a reason
 * specific to a run that is fast enough to have no *other* constraint.
 *
 * **`setTimeout`, not `requestAnimationFrame` — found by running it, not chosen up front.** The
 * first version used `requestAnimationFrame` and hung at step 0 the moment the tab lost paint
 * visibility, exactly the limitation §3/slice 3 hit for the MLP before slice 4 moved training to
 * a worker: `rAF` simply does not fire in a hidden tab. Reintroducing that limitation on purpose
 * would be an odd way to have learned it once already. `setTimeout` is not exempt from background
 * throttling either — browsers clamp it to roughly one tick a second once hidden — but "paces
 * slower while nobody is watching" is correct behaviour; "never moves again" is not.
 */
const SOM_TICKS = 120;
const SOM_TICK_MS = 16;
let somTimer: ReturnType<typeof setTimeout> | null = null;

function somPump(): void {
  const s = somState;
  if (!s.running) {
    somTimer = null;
    return;
  }
  const target = s.targetSteps;
  const already = s.trainer.step;
  if (already >= target) {
    s.running = false;
    somTimer = null;
    render();
    return;
  }

  const perTick = Math.max(1, Math.ceil(target / SOM_TICKS));
  const untilStep = Math.min(target, already + perTick);
  const started = performance.now();
  runSomSteps(s, untilStep);
  const elapsed = (performance.now() - started) / 1000;
  if (elapsed > 0) s.stepsPerSecond = (untilStep - already) / elapsed;

  // Finished-ness has to be settled *before* this pump's own render, or the last frame of a run
  // paints with `running` still true — "training" forever, one render short of correct — and
  // nothing schedules the render that would have fixed it, because nothing schedules another tick.
  if (s.trainer.step >= target) {
    s.running = false;
    somTimer = null;
    render();
    return;
  }

  render();
  somTimer = setTimeout(somPump, SOM_TICK_MS);
}

function setSomRunning(on: boolean): void {
  const s = somState;
  if (on && s.trainer.step >= s.targetSteps) return;
  s.running = on;
  if (on && somTimer === null) somTimer = setTimeout(somPump, SOM_TICK_MS);
  render();
}

function somStepOnce(): void {
  const s = somState;
  if (s.trainer.step >= s.targetSteps) return;
  s.running = false;
  runSomSteps(s, s.trainer.step + 1);
  render();
}

function regenerateSomData(): void {
  setSomRunning(false);
  rebuildSomData(somState);
  rebuildSom(somState);
  somState.hoverNode = -1;
  render();
  history.replaceState(null, '', syncUrl());
}

function regenerateSom(): void {
  setSomRunning(false);
  rebuildSom(somState);
  somState.hoverNode = -1;
  render();
  history.replaceState(null, '', syncUrl());
}

/** The probe itself, drawn over the scatter as a ring in the predicted class's colour. */
function drawProbe(ctx: CanvasRenderingContext2D, out: Float64Array): void {
  if (!camera) return;
  const px = sx(camera, state.probe[0]);
  const py = sy(camera, state.probe[1]);
  const best = argmax(out);

  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = '#5c5871';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px, 0);
  ctx.lineTo(px, ctx.canvas.height);
  ctx.moveTo(0, py);
  ctx.lineTo(ctx.canvas.width, py);
  ctx.stroke();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(px, py, 8, 0, Math.PI * 2);
  ctx.strokeStyle = '#0e0d15';
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.strokeStyle = classColour(best);
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(px, py, 2, 0, Math.PI * 2);
  ctx.fillStyle = '#e4e2ec';
  ctx.fill();
}

/**
 * Rebuild both sides from the current configuration.
 *
 * The page rebuilds its mirror and the worker rebuilds its own, from the *same* config through
 * the same `buildData`/`buildNet`. Neither sends the other a dataset — they agree because there
 * is one implementation, not because anything is kept in sync.
 *
 * `rebuilding` is set until the worker's `ready` arrives. Without it the page would ask for a
 * field against the network it is about to replace, and the answer would arrive looking current.
 */
function rebuildEverything(options: { data: boolean }): void {
  setRunning(false);
  if (options.data) {
    rebuildData(state);
    centreProbe();
    hover = null;
  }
  // The output width follows the class count, so new data means a new network too. Rebuilding
  // from the same weight seed keeps "change the noise" from also meaning "reroll the weights".
  rebuildNet(state);
  resetRun(state);
  diagScratch = createScratch(state.model);

  state.rebuilding = true;
  field = null;
  fieldStale = true;
  fieldPending = false;
  focus = null;

  trainer.reset(trainSetup(state, trainer.nextGeneration()), Math.max(2, state.data.classes));

  if (options.data) renderDataPanels();
  renderNetPanels();
  render();
  history.replaceState(null, '', syncUrl());
}

function regenerateData(): void {
  rebuildEverything({ data: true });
}

function regenerateNet(): void {
  rebuildEverything({ data: false });
}

/**
 * Reconfigure the app to one challenge card's recipe.
 *
 * Every field the config names is written straight onto `state`/`somState`, the same shape
 * `pickDataset`/`pickShape` already use for the guided flow — a card is not a parallel way of
 * changing the run, it is the same state everything else writes. `regenerateData`/
 * `regenerateSomData` are always the ones called, never the narrower `regenerateNet`/
 * `regenerateSom`: every challenge that touches a side names that side's dataset too, so the
 * wider rebuild is always correct and there is no second branch to get wrong.
 *
 * Challenge 12 is the one card that writes both sides at once — an MLP config and a SOM config
 * in the same object — which is exactly why `touchesMlp`/`touchesSom` are independent rather than
 * read off `config.net`.
 */
function applyChallenge(config: ChallengeConfig): void {
  let touchesMlp = false;
  if (config.dataset !== undefined) { state.dataset = config.dataset; touchesMlp = true; }
  if (config.n !== undefined) { state.n = config.n; touchesMlp = true; }
  if (config.trainFraction !== undefined) { state.trainFraction = config.trainFraction; touchesMlp = true; }
  if (config.hidden !== undefined) { state.hidden = [...config.hidden]; touchesMlp = true; }
  if (config.hiddenAct !== undefined) { state.hiddenAct = config.hiddenAct; touchesMlp = true; }
  if (config.init !== undefined) { state.init = config.init; touchesMlp = true; }
  if (config.learningRate !== undefined) { state.learningRate = config.learningRate; touchesMlp = true; }
  if (config.targetSteps !== undefined) { state.targetSteps = config.targetSteps; touchesMlp = true; }

  let touchesSom = false;
  if (config.somDataset !== undefined) { somState.dataset = config.somDataset; touchesSom = true; }
  if (config.cols !== undefined) { somState.cols = config.cols; touchesSom = true; }
  if (config.rows !== undefined) { somState.rows = config.rows; touchesSom = true; }
  if (config.topology !== undefined) { somState.topology = config.topology; touchesSom = true; }
  if (config.decay !== undefined) { somState.decay = config.decay; touchesSom = true; }
  if (config.somTargetSteps !== undefined) { somState.targetSteps = config.somTargetSteps; touchesSom = true; }
  if (touchesSom) somState.scheduleSteps = config.scheduleSteps ?? somState.targetSteps;

  state.net = config.net;
  document.body.dataset['net'] = state.net;
  state.stage = 'explorer';
  document.body.dataset['stage'] = state.stage;
  for (const b of Array.from($('nets').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === (state.net === 'som' ? 'net-som' : 'net-mlp'));
  }
  for (const b of Array.from($('stages').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === 'stage-explorer');
  }
  if (view3d) ensure3dSceneFor(state.net);

  if (touchesMlp) regenerateData();
  if (touchesSom) regenerateSomData();
  syncMlpControls();
  syncSomControls();

  history.replaceState(null, '', syncUrl());
  render();
}

function centreProbe(): void {
  const box = bounds2d(state.data);
  state.probe = [(box.minX + box.maxX) / 2, (box.minY + box.maxY) / 2];
}

/* ---------------- pointer ---------------- */

function nearest(px: number, py: number): number | null {
  if (!camera) return null;
  const { data } = state;
  const worldX = wx(camera, px);
  const worldY = wy(camera, py);
  const reach = 10 / camera.scale;
  let best: number | null = null;
  let bestDist = reach * reach;
  for (let i = 0; i < data.n; i++) {
    const p = sample(data, i);
    const dx = (p[0] as number) - worldX;
    const dy = (p[1] as number) - worldY;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function moveProbe(event: PointerEvent): void {
  if (!camera) return;
  const rect = stage.getBoundingClientRect();
  state.probe = [
    wx(camera, event.clientX - rect.left),
    wy(camera, event.clientY - rect.top),
  ];
  render();
}

stage.addEventListener('pointerdown', (event) => {
  dragging = true;
  stage.setPointerCapture(event.pointerId);
  moveProbe(event);
});

stage.addEventListener('pointermove', (event) => {
  if (dragging) {
    moveProbe(event);
    return;
  }
  const rect = stage.getBoundingClientRect();
  const found = nearest(event.clientX - rect.left, event.clientY - rect.top);
  if (found !== hover) {
    hover = found;
    render();
  }
});

stage.addEventListener('pointerup', (event) => {
  dragging = false;
  if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
});

stage.addEventListener('pointerleave', () => {
  if (hover !== null && !dragging) {
    hover = null;
    render();
  }
});

// Hovering a node dims every edge not wired into it. The graph at 8-8 is 96 edges, which is
// legible in aggregate and not individually; this is how one weight gets looked at.
graph.addEventListener('pointermove', (event) => {
  const rect = graph.getBoundingClientRect();
  // Recomputed rather than cached from the last draw: the canvas can resize between a draw and
  // a pointer event, and a stale layout hit-tests against nodes that are no longer there.
  const layout = layoutNetwork(shapeOf(state.model), graph.clientWidth, graph.clientHeight);
  const found = hitNode(layout, event.clientX - rect.left, event.clientY - rect.top);
  if ((found?.[0] ?? -1) !== (focus?.[0] ?? -1) || (found?.[1] ?? -1) !== (focus?.[1] ?? -1)) {
    focus = found;
    render();
  }
});

graph.addEventListener('pointerleave', () => {
  if (focus !== null) {
    focus = null;
    render();
  }
});

somLattice.addEventListener('pointermove', (event) => {
  const rect = somLattice.getBoundingClientRect();
  const layout = layoutLattice(
    somState.som.cols,
    somState.som.rows,
    somState.som.topology,
    somLattice.clientWidth,
    somLattice.clientHeight,
  );
  const found = hitLatticeNode(layout, event.clientX - rect.left, event.clientY - rect.top);
  if (found !== somState.hoverNode) {
    somState.hoverNode = found;
    render();
  }
});

somLattice.addEventListener('pointerleave', () => {
  if (somState.hoverNode !== -1) {
    somState.hoverNode = -1;
    render();
  }
});

/* ---------------- narrow chassis ---------------- */

function drawer(buttonId: string, panelId: string): void {
  $(buttonId).addEventListener('click', () => {
    const panel = $(panelId);
    const opening = !panel.classList.contains('open');
    closeDrawers();
    panel.classList.toggle('open', opening);
    $('scrim').hidden = !opening;
  });
}

function closeDrawers(): void {
  $('panel-left').classList.remove('open');
  $('panel-right').classList.remove('open');
  $('scrim').hidden = true;
}

/* ---------------- boot ---------------- */

function boot(): void {
  fillDatasets();

  slider('i-n', 'v-n', () => state.n, (v) => (state.n = v), (v) => String(v));
  slider('i-noise', 'v-noise', () => state.noise, (v) => (state.noise = v), (v) => v.toFixed(2));
  slider('i-seed', 'v-seed', () => state.seed, (v) => (state.seed = v), (v) => String(v));
  slider(
    'i-split',
    'v-split',
    () => state.trainFraction,
    (v) => (state.trainFraction = v),
    (v) => `${Math.round(v * 100)}%`,
  );

  // Softmax is the output layer's job and is not offered as a hidden activation — a softmax
  // hidden layer is a real thing and not a thing anybody should reach for by accident.
  const act = fillSelect('i-act', ACTIVATIONS.filter((a) => a !== 'softmax'), state.hiddenAct);
  act.addEventListener('change', () => {
    if (isActivation(act.value)) {
      state.hiddenAct = act.value;
      regenerateNet();
    }
  });

  const init = fillSelect('i-init', INIT_SCHEMES, state.init);
  init.addEventListener('change', () => {
    if (isInitScheme(init.value)) {
      state.init = init.value;
      regenerateNet();
    }
  });

  const arch = $<HTMLInputElement>('i-arch');
  arch.value = state.hidden.join('-');
  arch.addEventListener('input', () => {
    const parsed = parseHidden(arch.value);
    // Empty is valid — it is challenge 1. Anything that parses to nothing while containing
    // characters is not, and says so rather than silently reverting.
    const invalid = arch.value.trim() !== '' && parsed.length === 0;
    arch.classList.toggle('bad', invalid);
    if (invalid) return;
    state.hidden = parsed;
    regenerateNet();
  });

  segment('presets', (id) => {
    const widths: Record<string, number[]> = {
      'arch-none': [],
      'arch-one': [8],
      'arch-two': [8, 8],
      'arch-deep': [8, 8, 8],
    };
    state.hidden = widths[id] ?? [];
    arch.classList.remove('bad');
    regenerateNet();
  });

  segment('stages', (id) => {
    state.stage = id.replace('stage-', '') as AppStage;
    document.body.dataset['stage'] = state.stage;
    history.replaceState(null, '', syncUrl());
  });

  segment('nets', (id) => {
    state.net = id === 'net-som' ? 'som' : 'mlp';
    document.body.dataset['net'] = state.net;
    history.replaceState(null, '', syncUrl());
    if (view3d) ensure3dSceneFor(state.net);
    render();
  });

  segment('views', (id) => {
    view3d = id === 'view-3d';
    document.body.dataset['view'] = view3d ? '3d' : '2d';
    history.replaceState(null, '', syncUrl());
    if (view3d) ensure3dSceneFor(state.net);
    render();
  });

  $('loss3d-badge').addEventListener('click', () => {
    lossLiteral = !lossLiteral;
    render();
  });

  /*
   * The learning-rate slider is logarithmic, and it has to be.
   *
   * The interesting range is 1e-4 to 500 — challenge 4 lives at the bottom and challenge 3 needs
   * the top, because the measured figure that actually destroys this network is in the hundreds,
   * not the 3.0 §6 first guessed. Linear, the entire useful region below 1 would be the first
   * 0.2% of the track.
   */
  const lr = $<HTMLInputElement>('i-lr');
  lr.value = String(Math.log10(state.learningRate));
  const showLr = (): void => {
    lr.value = String(Math.log10(state.learningRate));
    $('v-lr').textContent =
      state.learningRate >= 1 ? state.learningRate.toFixed(1) : state.learningRate.toFixed(4);
    $('lr-note').innerHTML =
      state.learningRate > 20
        ? 'Large enough to destroy the network rather than train it &mdash; challenge 3.'
        : state.learningRate < 0.001
          ? 'Small enough that float32 loses some updates entirely &mdash; challenge 4.'
          : '';
  };
  lr.addEventListener('input', () => {
    state.learningRate = Number(Math.pow(10, Number(lr.value)).toPrecision(3));
    trainer.configure(currentTrainConfig());
    showLr();
    history.replaceState(null, '', syncUrl());
    render();
  });
  showLr();

  /*
   * Optimiser. A segmented control, not a slider — SGD / Momentum / Adam is a choice among three
   * named things, and momentum's β and Adam's β1/β2/ε are fixed rather than exposed (§7 of the
   * design document): a beginner does not need a fourth and fifth dial before the first three
   * have taught them anything.
   *
   * Switching mid-run does not rebuild — `trainer.configure` reaches `setTrainConfig` in the
   * worker, which resets only the optimiser's own state (Adam's moments cannot mean anything
   * under a `momentum` label) and leaves the step count, the chart, and the weights exactly
   * where they were.
   */
  segment('optimiser', (id) => {
    const kind = id.replace('opt-', '');
    if (!isOptimiserKind(kind)) return;
    state.optimiser = kind;
    trainer.configure(currentTrainConfig());
    history.replaceState(null, '', syncUrl());
    render();
  });

  slider('i-batch', 'v-batch', () => state.batchSize, (v) => (state.batchSize = v), (v) => String(v),
    () => {
      // Changing the batch size mid-run would make the chart's two halves incomparable, so the
      // run restarts rather than quietly changing what a step means.
      regenerateNet();
    });

  // Logarithmic, snapped to a readable figure — nobody wants a target of 6 314 steps.
  const steps = $<HTMLInputElement>('i-steps');
  const showSteps = (): void => {
    steps.value = String(Math.log10(state.targetSteps));
    $('v-steps').textContent = state.targetSteps.toLocaleString();
  };
  steps.addEventListener('input', () => {
    state.targetSteps = snapSteps(Math.pow(10, Number(steps.value)));
    $('v-steps').textContent = state.targetSteps.toLocaleString();
    history.replaceState(null, '', syncUrl());
    render();
  });
  showSteps();
  syncSteps = showSteps;

  syncMlpControls = (): void => {
    $<HTMLInputElement>('i-n').value = String(state.n);
    $('v-n').textContent = String(state.n);
    $<HTMLInputElement>('i-split').value = String(state.trainFraction);
    $('v-split').textContent = `${Math.round(state.trainFraction * 100)}%`;
    act.value = state.hiddenAct;
    init.value = state.init;
    showLr();
    showSteps();
  };

  /* ---------------- SOM controls ---------------- */

  const somDatasetSelect = $<HTMLSelectElement>('som-i-dataset');
  for (const [key, d] of Object.entries(SOM_DATASETS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = d.label;
    somDatasetSelect.append(opt);
  }
  somDatasetSelect.value = somState.dataset;
  somDatasetSelect.addEventListener('change', () => {
    if (isSomDatasetKey(somDatasetSelect.value)) {
      somState.dataset = somDatasetSelect.value;
      regenerateSomData();
    }
  });

  slider('som-i-n', 'som-v-n', () => somState.n, (v) => (somState.n = v), (v) => String(v), regenerateSomData);
  slider(
    'som-i-seed', 'som-v-seed', () => somState.seed, (v) => (somState.seed = v), (v) => String(v),
    regenerateSomData,
  );
  slider(
    'som-i-cols', 'som-v-cols', () => somState.cols, (v) => (somState.cols = v), (v) => String(v),
    regenerateSom,
  );
  slider(
    'som-i-rows', 'som-v-rows', () => somState.rows, (v) => (somState.rows = v), (v) => String(v),
    regenerateSom,
  );
  slider(
    'som-i-alpha0', 'som-v-alpha0', () => somState.alpha0, (v) => (somState.alpha0 = v),
    (v) => v.toFixed(2), regenerateSom,
  );
  slider(
    'som-i-sigma0', 'som-v-sigma0', () => somState.sigma0, (v) => (somState.sigma0 = v),
    (v) => v.toFixed(1), regenerateSom,
  );

  segment('som-topology', (id) => {
    somState.topology = id === 'som-topo-rect' ? 'rect' : 'hex';
    regenerateSom();
  });

  // Every schedule control restarts the run rather than bending it mid-flight — the same rule
  // the MLP side applies to batch size: changing what a step *means* partway through would make
  // the QE/TE chart's two halves describe two different experiments.
  segment('som-decay', (id) => {
    somState.decay = id === 'som-decay-linear' ? 'linear' : id === 'som-decay-inverse' ? 'inverse' : 'exponential';
    regenerateSom();
  });

  const somSteps = $<HTMLInputElement>('som-i-steps');
  const showSomSteps = (): void => {
    somSteps.value = String(Math.log10(somState.targetSteps));
    $('som-v-steps').textContent = somState.targetSteps.toLocaleString();
  };
  somSteps.addEventListener('input', () => {
    somState.targetSteps = snapSteps(Math.pow(10, Number(somSteps.value)));
    // Ordinary use keeps the schedule's own horizon equal to the run length — only a challenge
    // deliberately pulls them apart, and dragging this slider is not that.
    somState.scheduleSteps = somState.targetSteps;
    showSomSteps();
    regenerateSom();
  });
  showSomSteps();

  syncSomControls = (): void => {
    $<HTMLInputElement>('som-i-cols').value = String(somState.cols);
    $('som-v-cols').textContent = String(somState.cols);
    $<HTMLInputElement>('som-i-rows').value = String(somState.rows);
    $('som-v-rows').textContent = String(somState.rows);
    for (const b of Array.from($('som-topology').querySelectorAll('button'))) {
      b.classList.toggle('on', b.id === (somState.topology === 'rect' ? 'som-topo-rect' : 'som-topo-hex'));
    }
    for (const b of Array.from($('som-decay').querySelectorAll('button'))) {
      b.classList.toggle('on', b.id === `som-decay-${somState.decay}`);
    }
    showSomSteps();
  };

  $('btn-train').addEventListener('click', () => {
    if (state.net === 'som') setSomRunning(!somState.running);
    else setRunning(!state.running);
  });

  /*
   * One step is "run until one more than where you are".
   *
   * There is no separate `step` message, because a second entry point into the training loop is
   * a second place for the sequence to diverge from the golden run. The worker cannot tell the
   * difference between this and a normal run that happens to end quickly, which is the point.
   */
  $('btn-step').addEventListener('click', () => {
    if (state.net === 'som') {
      somStepOnce();
      return;
    }
    if (state.step >= state.targetSteps) return;
    if (state.rebuilding) {
      // Same window as Train, and the same answer: hold the intent rather than drop it.
      runWhenReady = false;
      return;
    }
    state.running = false;
    trainer.run(state.step + 1);
  });

  $('btn-reset').addEventListener('click', () => {
    // Back to step zero with the *same* weights, so a run can be repeated exactly. Reinitialise
    // is the button that changes them.
    if (state.net === 'som') regenerateSom();
    else regenerateNet();
  });

  $('btn-stepper').addEventListener('click', () => {
    if (state.net === 'som') somStepper.open();
    else stepper.open();
  });
  $('st-close').addEventListener('click', () => stepper.close());

  $('btn-challenges').addEventListener('click', () => challenges.open());

  const reinitWeights = (): void => {
    if (state.net === 'som') {
      somState.weightSeed = 1 + ((somState.weightSeed * 7919 + 13) % 9999);
      regenerateSom();
      return;
    }
    state.weightSeed = 1 + ((state.weightSeed * 7919 + 13) % 9999);
    regenerateNet();
  };
  // One button per network — each panel needs its own for layout, but both do the same thing,
  // so they share one handler rather than one copying the other's logic.
  $('btn-reinit').addEventListener('click', reinitWeights);
  $('som-btn-reinit').addEventListener('click', reinitWeights);

  $('btn-resample').addEventListener('click', () => {
    if (state.net === 'som') {
      somState.seed = 1 + ((somState.seed * 7919 + 13) % 9999);
      $<HTMLInputElement>('som-i-seed').value = String(somState.seed);
      $('som-v-seed').textContent = String(somState.seed);
      regenerateSomData();
      return;
    }
    state.seed = 1 + ((state.seed * 7919 + 13) % 9999);
    $<HTMLInputElement>('i-seed').value = String(state.seed);
    $('v-seed').textContent = String(state.seed);
    regenerateData();
  });

  drawer('btn-panel-left', 'panel-left');
  drawer('btn-panel-right', 'panel-right');
  $('scrim').addEventListener('click', closeDrawers);

  /*
   * Slice 3 had to pause here when the tab went away, because `requestAnimationFrame` does not
   * fire in a background tab and training stopped whether the app agreed or not. A worker is not
   * throttled by visibility, so the run now survives a tab switch and no handler is needed. The
   * page simply stops redrawing, which is what should happen.
   */

  window.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement && event.target.type === 'text') return;
    // Space has to match on `code` — its `key` is a literal space.
    if (event.code === 'Space') {
      event.preventDefault();
      $('btn-train').click();
    }
    if (event.key === '.') $('btn-step').click();
    if (event.key === 'r' || event.key === 'R') $('btn-resample').click();
    if (event.key === 'w' || event.key === 'W') reinitWeights();
    if (event.key === 's' || event.key === 'S') $('btn-stepper').click();
    if (event.key === 'c' || event.key === 'C') $('btn-challenges').click();
    if (event.key === '2') $('view-2d').click();
    if (event.key === '3') $('view-3d').click();

    if (challenges.isOpen()) {
      if (event.key === 'Escape') challenges.close();
      return;
    }
    if (stepper.isOpen()) {
      // Scoped to the overlay being open, so arrow keys do not hijack the sliders behind it.
      if (event.key === 'ArrowRight') $('st-next').click();
      if (event.key === 'ArrowLeft') $('st-prev').click();
      if (event.key === 'Escape') stepper.close();
      return;
    }
    if (somStepper.isOpen()) {
      if (event.key === 'ArrowRight') $('som-st-next').click();
      if (event.key === 'ArrowLeft') $('som-st-prev').click();
      if (event.key === 'Escape') somStepper.close();
      return;
    }
    if (event.key === 'Escape') closeDrawers();
  });

  document.body.dataset['stage'] = state.stage;
  document.body.dataset['net'] = state.net;
  document.body.dataset['view'] = view3d ? '3d' : '2d';
  for (const b of Array.from($('stages').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === `stage-${state.stage}`);
  }
  for (const b of Array.from($('views').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === (view3d ? 'view-3d' : 'view-2d'));
  }
  if (view3d) ensure3dSceneFor(state.net);
  // No syncPresets() here — renderNetPanels() below calls it, along with everything else that
  // has to agree with state.hidden before the first paint.

  $('hint').textContent = 'space train · . step · R resample · W reinitialise';

  centreProbe();
  state.rebuilding = true;
  trainer.init(trainSetup(state, trainer.nextGeneration()), Math.max(2, state.data.classes));

  renderDataPanels();
  renderNetPanels();
  render();
  window.addEventListener('resize', render);
}

function syncPresets(): void {
  const key = state.hidden.join('-');
  const map: Record<string, string> = { '': 'arch-none', '8': 'arch-one', '8-8': 'arch-two', '8-8-8': 'arch-deep' };
  const wanted = map[key];
  for (const b of Array.from($('presets').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === wanted);
  }
}

boot();
