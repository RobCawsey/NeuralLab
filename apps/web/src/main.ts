/**
 * Slice 2 — it learns.
 *
 * Backpropagation, SGD, and the loss chart. Training runs on the main thread in a `Step` budget
 * per frame; slice 4 moves it into a worker. The step *sequence* does not depend on how the
 * steps are grouped into frames — invariant 2 — so the golden number holds whatever the frame
 * rate is, and a test asserts it.
 */

import { bounds2d, sample } from '@neurallab/core';
import { GENERATORS } from '@neurallab/data';
import {
  ACTIVATIONS,
  INIT_SCHEMES,
  argmax,
  describeShape,
  evaluateRows,
  isActivation,
  isInitScheme,
  paramCount,
  parseHidden,
  shapeOf,
  trainStep,
} from '@neurallab/mlp';
import { classColour, drawScatter, resize } from './render/scatter.ts';
import { wx, wy, sx, sy, type Camera } from './render/camera.ts';
import { drawNetwork, drawOverCapNotice, heatColour } from './render/network.ts';
import { drawChart } from './render/chart.ts';
import { hitNode, layoutNetwork, UNIT_CAP } from './render/graph-layout.ts';
import {
  EVAL_EVERY,
  createState,
  evaluateProbe,
  probeInput,
  readUrl,
  rebuildData,
  rebuildNet,
  resetRun,
  writeUrl,
  type AppStage,
} from './run/state.ts';

const state = createState();
readUrl(state, window.location.search);
rebuildData(state);
rebuildNet(state);

let camera: Camera | null = null;
let hover: number | null = null;
let focus: [number, number] | null = null;
let dragging = false;

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

/* ---------------- the run loop ---------------- */

/**
 * Milliseconds of training per frame.
 *
 * A budget rather than a fixed step count, because a step on 2-8-8-2 and a step on 2-64-64-2
 * differ by twenty times and a fixed count would make one of them stutter. This does *not*
 * compromise reproducibility: how many whole steps happen this frame changes nothing about
 * their order or their arithmetic. Invariant 2's rule is that a step is never scaled by frame
 * time, and none of them is.
 */
const FRAME_BUDGET_MS = 8;

let frameHandle = 0;

function tick(): void {
  frameHandle = 0;
  if (!state.running) return;

  const started = performance.now();
  let ran = 0;

  while (state.trainer.step < state.targetSteps && performance.now() - started < FRAME_BUDGET_MS) {
    const metrics = trainStep(state.trainer, state.z);
    state.history.push({
      step: metrics.step,
      loss: metrics.loss,
      lossMin: metrics.lossMin,
      lossMax: metrics.lossMax,
    });
    if (metrics.step % EVAL_EVERY === 0 || metrics.step === state.targetSteps) {
      recordEval();
    }
    ran++;
    // Challenge 3's outcome. A network of NaN cannot be trained further and every panel reading
    // it would show NaN, so the run stops and says why rather than continuing silently.
    if (state.trainer.diverged) {
      state.running = false;
      break;
    }
  }

  if (ran > 0) state.elapsedMs = performance.now() - state.startedAt;
  if (state.trainer.step >= state.targetSteps) state.running = false;

  render();
  if (state.running) schedule();
}

function schedule(): void {
  if (frameHandle === 0) frameHandle = requestAnimationFrame(tick);
}

function recordEval(): void {
  const tr = evaluateRows(state.model, state.z, state.parts.train, state.scratch);
  const va = evaluateRows(state.model, state.z, state.parts.val, state.scratch);
  state.evals.push({
    step: state.trainer.step,
    trainLoss: tr.loss,
    valLoss: va.loss,
    trainAccuracy: tr.accuracy,
    valAccuracy: va.accuracy,
  });
}

function setRunning(on: boolean): void {
  if (on && state.trainer.step >= state.targetSteps) return;
  state.running = on;
  if (on) {
    state.startedAt = performance.now() - state.elapsedMs;
    schedule();
  }
  render();
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
    history.replaceState(null, '', writeUrl(state));
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
  $('ph-arch').textContent = describeShape(state.model);
  $('s-params').textContent = String(paramCount(state.model));
  let edges = 0;
  for (const l of state.model.layers) edges += l.W.length;
  $('s-edges').textContent = String(edges);
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

/** Everything that changes as the run advances. */
function renderRunPanels(): void {
  const { trainer, targetSteps, evals } = state;
  const last = evals[evals.length - 1];

  $('ph-chart').textContent = `step ${trainer.step}`;
  $('s-step').textContent = `${trainer.step} / ${targetSteps}`;
  $('s-epoch').textContent = String(trainer.epoch);
  $<HTMLElement>('s-progress').style.width = `${Math.min(100, (trainer.step / targetSteps) * 100)}%`;

  $('s-trainloss').textContent = last ? last.trainLoss.toFixed(4) : '—';
  $('s-valloss').textContent = last ? last.valLoss.toFixed(4) : '—';
  $('s-trainacc').textContent = last ? last.trainAccuracy.toFixed(4) : '—';
  $('s-valacc').textContent = last ? last.valAccuracy.toFixed(4) : '—';

  const seconds = state.elapsedMs / 1000;
  $('s-sps').textContent = seconds > 0.2 ? Math.round(trainer.step / seconds).toLocaleString() : '—';

  const btn = $<HTMLButtonElement>('btn-train');
  const finished = trainer.step >= targetSteps;
  btn.textContent = state.running ? 'Pause' : finished ? 'Done' : 'Train';
  btn.disabled = finished && !state.running;

  const badge = $('graph-badge');
  badge.classList.toggle('training', state.running);
  badge.classList.toggle('done', finished && !trainer.diverged);
  badge.textContent = trainer.diverged
    ? 'diverged'
    : state.running
      ? 'training'
      : trainer.step === 0
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
  if (trainer.diverged) {
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

function render(): void {
  const out = evaluateProbe(state);
  const input = probeInput(state);

  const scatterFit = resize(stage);
  if (scatterFit) {
    const view = drawScatter(scatterFit.ctx, state.data, scatterFit.w, scatterFit.h, hover);
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

  const chartFit = resize(chart);
  if (chartFit) {
    drawChart(chartFit.ctx, state.history, state.evals, chartFit.w, chartFit.h, {
      totalSteps: state.targetSteps,
    });
  }

  const best = argmax(out);
  const confidence = out[best] as number;
  $('v-cursor').textContent = `${state.probe[0].toFixed(2)}, ${state.probe[1].toFixed(2)}`;
  $('v-predict').textContent = state.data.classNames[best] ?? `class ${best}`;
  $('v-conf').textContent = Number.isFinite(confidence) ? confidence.toFixed(3) : 'NaN';

  renderOutputs(out);
  renderActivations(input);
  renderRunPanels();
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

function regenerateData(): void {
  setRunning(false);
  rebuildData(state);
  // The output width follows the class count, so new data means a new network. Rebuilding from
  // the same weight seed keeps "change the noise" from also meaning "reroll the weights".
  rebuildNet(state);
  centreProbe();
  hover = null;
  renderDataPanels();
  renderNetPanels();
  render();
  history.replaceState(null, '', writeUrl(state));
}

function regenerateNet(): void {
  setRunning(false);
  rebuildNet(state);
  focus = null;
  renderNetPanels();
  render();
  history.replaceState(null, '', writeUrl(state));
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
    syncPresets();
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
    arch.value = state.hidden.join('-');
    arch.classList.remove('bad');
    regenerateNet();
  });

  segment('stages', (id) => {
    state.stage = id.replace('stage-', '') as AppStage;
    document.body.dataset['stage'] = state.stage;
    history.replaceState(null, '', writeUrl(state));
  });

  segment('nets', (id) => {
    state.net = id === 'net-som' ? 'som' : 'mlp';
    document.body.dataset['net'] = state.net;
    history.replaceState(null, '', writeUrl(state));
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
    state.trainer.config = { learningRate: state.learningRate, batchSize: state.batchSize };
    showLr();
    history.replaceState(null, '', writeUrl(state));
    render();
  });
  showLr();

  slider('i-batch', 'v-batch', () => state.batchSize, (v) => (state.batchSize = v), (v) => String(v),
    () => {
      // Changing the batch size mid-run would make the chart's two halves incomparable, so the
      // run restarts rather than quietly changing what a step means.
      resetRun(state);
      render();
    });

  slider('i-steps', 'v-steps', () => state.targetSteps, (v) => (state.targetSteps = v),
    (v) => String(v), () => render());

  $('btn-train').addEventListener('click', () => setRunning(!state.running));

  $('btn-step').addEventListener('click', () => {
    setRunning(false);
    if (state.trainer.step >= state.targetSteps) return;
    const m = trainStep(state.trainer, state.z);
    state.history.push({ step: m.step, loss: m.loss, lossMin: m.lossMin, lossMax: m.lossMax });
    recordEval();
    render();
  });

  $('btn-reset').addEventListener('click', () => {
    // Back to step zero with the *same* weights, so a run can be repeated exactly. Reinitialise
    // is the button that changes them.
    rebuildNet(state);
    render();
  });

  $('btn-reinit').addEventListener('click', () => {
    state.weightSeed = 1 + ((state.weightSeed * 7919 + 13) % 9999);
    regenerateNet();
  });

  $('btn-resample').addEventListener('click', () => {
    state.seed = 1 + ((state.seed * 7919 + 13) % 9999);
    $<HTMLInputElement>('i-seed').value = String(state.seed);
    $('v-seed').textContent = String(state.seed);
    regenerateData();
  });

  drawer('btn-panel-left', 'panel-left');
  drawer('btn-panel-right', 'panel-right');
  $('scrim').addEventListener('click', closeDrawers);

  window.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement && event.target.type === 'text') return;
    // Space has to match on `code` — its `key` is a literal space.
    if (event.code === 'Space') {
      event.preventDefault();
      $('btn-train').click();
    }
    if (event.key === '.') $('btn-step').click();
    if (event.key === 'r' || event.key === 'R') $('btn-resample').click();
    if (event.key === 'w' || event.key === 'W') $('btn-reinit').click();
    if (event.key === 'Escape') closeDrawers();
  });

  document.body.dataset['stage'] = state.stage;
  document.body.dataset['net'] = state.net;
  for (const b of Array.from($('stages').querySelectorAll('button'))) {
    b.classList.toggle('on', b.id === `stage-${state.stage}`);
  }
  syncPresets();

  $('hint').textContent = 'space train · . step · R resample · W reinitialise';

  centreProbe();
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
