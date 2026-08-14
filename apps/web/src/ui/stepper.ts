/**
 * The stepper — §6's teaching screen, and the reason this project exists.
 *
 * **Not an illustration of the algorithm — the algorithm, paused.** Every number drawn here comes
 * from one `StepTrace`, produced by the same `trainStep` the worker drains at full speed
 * (`trace.test.ts` proves tracing cannot move a weight). This file does no arithmetic about the
 * network — it pages through a recording and writes prose against the values already in it.
 *
 * A trace covers one **whole step** — forward through every layer, backward through every layer,
 * the update already applied. "Stepping" is a cursor into that recording, not a re-run of the
 * network one layer at a time: requesting a new trace from the worker only happens when the
 * reader pages past the last stage of the one they are looking at.
 */

import { largest, peak, type StepTrace } from '@neurallab/mlp';
import { drawStrip, drawTiles } from '../render/stepper.ts';
import { resize } from '../render/scatter.ts';
import type { TrainerClient } from '../workers/client.ts';

type StageKind = 'sample' | 'forward' | 'output' | 'backward' | 'update';

interface Stage {
  readonly kind: StageKind;
  readonly layer: number;
  readonly label: string;
  readonly sub: string;
}

/**
 * `sample` + one stage per forward layer (the last folded into `output`) + one stage per hidden
 * layer's backward pass + `update`. For 2-8-8-2 that is 1 + 3 + 2 + 1 = 7, matching Fig 8.2.
 *
 * The output layer's backward step is not separate: under softmax + cross-entropy it is a fused
 * `dz = a − onehot(t)`, computed from the loss with no intervening derivative, so folding it into
 * the stage that already shows the loss is where that fact belongs rather than a stage that would
 * otherwise be empty.
 */
function buildStages(trace: StepTrace): Stage[] {
  const L = trace.forward.length;
  const stages: Stage[] = [
    { kind: 'sample', layer: -1, label: 'Take a sample', sub: `x = [${formatVec(trace.input, 2)}]` },
  ];
  for (let l = 0; l < L; l++) {
    if (l < L - 1) {
      stages.push({
        kind: 'forward',
        layer: l,
        label: `Forward → layer ${l + 1}`,
        sub: `${trace.forward[l]!.units} units · ${trace.forward[l]!.act}`,
      });
    } else {
      stages.push({
        kind: 'output',
        layer: l,
        label: 'Output and loss',
        sub: `${trace.loss.toFixed(4)} cross-entropy`,
      });
    }
  }
  for (let l = L - 2; l >= 0; l--) {
    stages.push({
      kind: 'backward',
      layer: l,
      label: `Backward ← layer ${l + 1}`,
      sub: 'δ in → δ out',
    });
  }
  stages.push({
    kind: 'update',
    layer: -1,
    label: 'Apply the update',
    sub: `lr ${trace.learningRate}`,
  });
  return stages;
}

function euclidNorm(values: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (Number.isFinite(v)) sum += v * v;
  }
  return Math.sqrt(sum);
}

function formatVec(values: ArrayLike<number>, dp: number): string {
  const parts: string[] = [];
  for (let i = 0; i < values.length; i++) parts.push((values[i] as number).toFixed(dp));
  return parts.join(', ');
}

/** How many entries of a derivative strip are exactly zero — a dead relu unit, made countable. */
function countZero(values: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < values.length; i++) if (values[i] === 0) n++;
  return n;
}

export interface StepperOptions {
  readonly trainer: TrainerClient;
  /** Called before the overlay opens, so the caller can pause the run underneath it. */
  readonly onOpen: () => void;
  readonly classNames: () => readonly string[];
}

export interface StepperController {
  open(): void;
  close(): void;
  readonly isOpen: () => boolean;
  /** Wired into the trainer's onTrace event by the caller. */
  receiveTrace(trace: StepTrace): void;
}

export function createStepper(opts: StepperOptions): StepperController {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from index.html`);
    return el as T;
  };

  const root = $('stepper');
  const stagesEl = $('st-stages');
  const detailEl = $('st-detail');
  const explainEl = $('st-explain');
  const whereEl = $('st-where');

  /** Every trace requested this session, so `back` can step across a step boundary. */
  let history: StepTrace[] = [];
  let traceIndex = -1;
  let stageIndex = 0;
  let open = false;
  let awaitingFirst = false;

  function current(): StepTrace | null {
    return traceIndex >= 0 ? (history[traceIndex] ?? null) : null;
  }

  function stagesForCurrent(): Stage[] {
    const t = current();
    return t ? buildStages(t) : [];
  }

  function render(): void {
    const trace = current();
    if (!trace) {
      detailEl.replaceChildren(loadingPane());
      stagesEl.replaceChildren();
      explainEl.replaceChildren();
      whereEl.textContent = 'requesting a step from the worker…';
      return;
    }

    const stages = stagesForCurrent();
    const stage = stages[stageIndex] ?? stages[0]!;

    whereEl.textContent =
      `step ${trace.step} · sample ${trace.indexInBatch + 1} of ${trace.batchSize} · ` +
      `batch ${trace.batchSize}`;

    renderStageList(stages, stage);
    renderDetail(trace, stage);
    renderExplain(trace, stage);

    $<HTMLButtonElement>('st-prev').disabled = stageIndex === 0 && traceIndex === 0;
    $<HTMLButtonElement>('st-next').disabled = false;
  }

  function loadingPane(): HTMLElement {
    const p = document.createElement('div');
    p.className = 'st-pb';
    p.innerHTML = '<p class="prose dim">Running one real step…</p>';
    return p;
  }

  function renderStageList(stages: Stage[], active: Stage): void {
    stagesEl.replaceChildren();
    const activeAt = stages.indexOf(active);
    stages.forEach((s, i) => {
      const row = document.createElement('div');
      row.className = 'st-stage' + (i === activeAt ? ' on' : i < activeAt ? ' done' : '');
      const b = document.createElement('b');
      b.textContent = `${i + 1} · ${s.label}`;
      const span = document.createElement('span');
      span.textContent = s.sub;
      row.append(b, span);
      row.addEventListener('click', () => {
        stageIndex = i;
        render();
      });
      stagesEl.append(row);
    });
  }

  function heading(text: string, sub?: string): HTMLElement {
    const h = document.createElement('div');
    h.className = 'st-sub';
    h.textContent = sub ? `${text} — ${sub}` : text;
    return h;
  }

  function stripRow(label: string, note: string, draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void): HTMLElement {
    const row = document.createElement('div');
    row.className = 'st-row';
    const lb = document.createElement('span');
    lb.className = 'st-lb';
    lb.textContent = label;
    const canvas = document.createElement('canvas');
    canvas.className = 'st-strip';
    const nt = document.createElement('span');
    nt.className = 'st-nt';
    nt.textContent = note;
    row.append(lb, canvas, nt);
    queueMicrotask(() => {
      const fit = resize(canvas);
      if (fit) draw(fit.ctx, fit.w, fit.h);
    });
    return row;
  }

  function tilesBlock(W: ArrayLike<number>, units: number, inputs: number, height: number): HTMLElement {
    const canvas = document.createElement('canvas');
    canvas.className = 'st-tiles';
    canvas.style.height = `${height}px`;
    queueMicrotask(() => {
      const fit = resize(canvas);
      if (fit) drawTiles(fit.ctx, W, units, inputs, fit.w, fit.h);
    });
    return canvas;
  }

  function renderDetail(trace: StepTrace, stage: Stage): void {
    const body = document.createElement('div');
    body.className = 'st-pb';
    const names = opts.classNames();

    if (stage.kind === 'sample') {
      body.append(heading('input', `${trace.input.length}-dimensional`));
      body.append(
        stripRow('x', formatVec(trace.input, 3), (ctx, w, h) => drawStrip(ctx, trace.input, w, h, { signed: true })),
      );
      const kv = document.createElement('div');
      kv.className = 'kv';
      kv.innerHTML = `<span>true class</span><b class="am">${names[trace.target] ?? `class ${trace.target}`}</b>`;
      body.append(kv);
    } else if (stage.kind === 'forward') {
      const l = stage.layer;
      const layer = trace.forward[l]!;
      body.append(heading('activation', `layer ${l + 1} · ${layer.act}`));
      body.append(
        stripRow('a', `max ${peak(layer.a).toFixed(3)}`, (ctx, w, h) =>
          drawStrip(ctx, layer.a, w, h, { peak: peak(layer.a) }),
        ),
      );
      body.append(heading('weights feeding this layer', `${layer.units} × ${layer.W.length / layer.units}`));
      body.append(tilesBlock(layer.W, layer.units, layer.W.length / layer.units, 96));
    } else if (stage.kind === 'output') {
      const out = trace.output;
      body.append(heading('output', 'softmax'));
      body.append(stripRow('p', formatVec(out, 3), (ctx, w, h) => drawStrip(ctx, out, w, h, { peak: 1 })));

      const kv = document.createElement('div');
      kv.className = 'kv';
      kv.innerHTML =
        `<span>loss</span><b class="am">${trace.loss.toFixed(4)}</b>`;
      body.append(kv);

      const back = trace.backward[trace.backward.length - 1]!;
      body.append(heading('δ = a − onehot(t)', 'fused: softmax + cross-entropy'));
      body.append(
        stripRow('δ', `‖δ‖ ${euclidNorm(back.dz).toFixed(3)}`, (ctx, w, h) =>
          drawStrip(ctx, back.dz, w, h, { signed: true }),
        ),
      );
    } else if (stage.kind === 'backward') {
      const l = stage.layer;
      const back = trace.backward[l]!;
      body.append(heading('δ arriving', `from layer ${l + 2}`));
      body.append(
        stripRow('δ in', `‖δ‖ ${euclidNorm(back.deltaIn).toFixed(3)}`, (ctx, w, h) =>
          drawStrip(ctx, back.deltaIn, w, h, { signed: true }),
        ),
      );
      body.append(heading("layer's own derivative", back.act));
      body.append(
        stripRow(
          "a'",
          back.act === 'relu' ? `${countZero(back.derivative)} at zero` : `max ${peak(back.derivative).toFixed(2)}`,
          (ctx, w, h) => drawStrip(ctx, back.derivative, w, h, { peak: Math.max(1, peak(back.derivative)) }),
        ),
      );
      body.append(heading('δz = δ ⊙ a′'));
      body.append(
        stripRow('δz', `‖δz‖ ${euclidNorm(back.dz).toFixed(3)}`, (ctx, w, h) =>
          drawStrip(ctx, back.dz, w, h, { signed: true }),
        ),
      );
      if (l > 0) {
        body.append(heading('δ sent further back', `to layer ${l}`));
        body.append(
          stripRow('δ out', `‖δ‖ ${euclidNorm(back.deltaOut).toFixed(3)}`, (ctx, w, h) =>
            drawStrip(ctx, back.deltaOut, w, h, { signed: true }),
          ),
        );
      }
      const dW = trace.deltaW[l]!;
      const units = trace.forward[l]!.units;
      body.append(heading('Δw for this layer', 'learning rate already applied'));
      body.append(tilesBlock(dW, units, dW.length / units, 96));
      const big = largest(dW);
      const kv = document.createElement('div');
      kv.className = 'kv';
      kv.innerHTML = `<span>largest single update</span><b class="am">w[${big.index}] ${big.value >= 0 ? '+' : ''}${big.value.toExponential(2)}</b>`;
      body.append(kv);
    } else {
      // update
      body.append(heading('every weight this step touched'));
      for (let l = 0; l < trace.deltaW.length; l++) {
        const dW = trace.deltaW[l]!;
        const big = largest(dW);
        const row = document.createElement('div');
        row.className = 'kv';
        row.innerHTML =
          `<span>layer ${l + 1}</span><b>largest ${big.value >= 0 ? '+' : ''}${big.value.toExponential(2)} · ‖Δw‖ ${euclidNorm(dW).toFixed(4)}</b>`;
        body.append(row);
      }
      const note = document.createElement('p');
      note.className = 'st-note';
      note.innerHTML =
        `Already applied — the network you closed this screen to open was already using these ` +
        `weights. Press <em>next</em> to run another step.`;
      body.append(note);
    }

    detailEl.replaceChildren(body);
  }

  function renderExplain(trace: StepTrace, stage: Stage): void {
    const wrap = document.createElement('div');
    const p = document.createElement('div');
    p.className = 'prose';
    p.innerHTML = explanationFor(trace, stage, opts.classNames());
    wrap.append(p);
    explainEl.replaceChildren(wrap);
  }

  function explanationFor(trace: StepTrace, stage: Stage, names: readonly string[]): string {
    const target = names[trace.target] ?? `class ${trace.target}`;

    switch (stage.kind) {
      case 'sample':
        return (
          `Sample <b>${trace.row}</b> from the training set, sample ${trace.indexInBatch + 1} ` +
          `of a batch of ${trace.batchSize}. Its true class is <b>${target}</b>. Everything on ` +
          `this screen is what the network does with this one point.`
        );
      case 'forward': {
        const layer = trace.forward[stage.layer]!;
        const inputs = layer.W.length / layer.units;
        return (
          `Layer ${stage.layer + 1} takes ${inputs} numbers from below and produces ` +
          `${layer.units} activations through <b>${layer.act}</b>. The largest here is ` +
          `<b>${peak(layer.a).toFixed(3)}</b>. This is exactly the weight tile drawn below — the ` +
          `same numbers the network graph colours as edges.`
        );
      }
      case 'output': {
        const back = trace.backward[trace.backward.length - 1]!;
        return (
          `The output layer's softmax turns its logits into a distribution; cross-entropy charges ` +
          `<b>${trace.loss.toFixed(4)}</b> for putting <b>${(trace.output[trace.target] as number).toFixed(3)}</b> ` +
          `on the true class. Under softmax and cross-entropy together the derivative simplifies to ` +
          `<b>a − onehot(target)</b> directly — there is no separate factor to multiply, which is why ` +
          `this stage shows δ arriving already finished rather than two strips being combined.`
        );
      }
      case 'backward': {
        const back = trace.backward[stage.layer]!;
        const passing =
          back.act === 'relu' ? back.derivative.length - countZero(back.derivative) : back.derivative.length;
        return (
          `The error arriving has magnitude <b>${euclidNorm(back.deltaIn).toFixed(3)}</b>. Layer ` +
          `${stage.layer + 1} multiplies it, unit by unit, by its own derivative` +
          (back.act === 'relu'
            ? ` — for relu that is 1 where the unit fired and 0 where it did not, so ` +
              `<b>${passing} of ${back.derivative.length}</b> units pass anything back at all.`
            : `.`) +
          ` What leaves for the layer below has magnitude <b>${euclidNorm(back.deltaOut).toFixed(3)}</b>` +
          (stage.layer === 0 ? ', though there is no layer below to receive it.' : '.')
        );
      }
      case 'update': {
        let worstLayer = 0;
        let worstValue = 0;
        for (let l = 0; l < trace.deltaW.length; l++) {
          const big = largest(trace.deltaW[l]!);
          if (Math.abs(big.value) > Math.abs(worstValue)) {
            worstValue = big.value;
            worstLayer = l;
          }
        }
        return (
          `Every weight moves by <b>−lr × (mean gradient over the batch)</b>, not by this one ` +
          `sample's gradient alone — the batch's ${trace.batchSize} samples were averaged before ` +
          `any of this was applied. The largest single change anywhere in the network was in ` +
          `layer <b>${worstLayer + 1}</b>, by <b>${worstValue.toExponential(2)}</b>.`
        );
      }
    }
  }

  function requestNext(): void {
    awaitingFirst = history.length === 0;
    opts.trainer.requestTrace(0);
  }

  /*
   * "Next" past the end of a trace's stages is the one place this UI does something other than
   * page through a recording — it asks the worker for the next real step. `awaitingFirst` guards
   * against a reader mashing the button while that request is still in flight.
   */
  function next(): void {
    const stages = stagesForCurrent();
    if (stageIndex < stages.length - 1) {
      stageIndex++;
      render();
    } else if (!awaitingFirst) {
      requestNext();
      render();
    }
  }

  function prev(): void {
    if (stageIndex > 0) {
      stageIndex--;
      render();
    } else if (traceIndex > 0) {
      traceIndex--;
      stageIndex = stagesForCurrent().length - 1;
      render();
    }
  }

  function runToEnd(): void {
    const stages = stagesForCurrent();
    stageIndex = Math.max(0, stages.length - 1);
    render();
  }

  $('st-next').addEventListener('click', next);
  $('st-prev').addEventListener('click', prev);
  $('st-run').addEventListener('click', runToEnd);

  return {
    open(): void {
      if (open) return;
      opts.onOpen();
      open = true;
      history = [];
      traceIndex = -1;
      stageIndex = 0;
      root.hidden = false;
      requestNext();
      render();
    },
    close(): void {
      open = false;
      root.hidden = true;
    },
    isOpen: () => open,
    receiveTrace(trace: StepTrace): void {
      if (!open) return;
      // Cap the history so a long stepper session does not grow without bound — 50 steps back
      // is far past what anyone pages through by hand.
      history.push(trace);
      if (history.length > 50) history.shift();
      traceIndex = history.length - 1;
      stageIndex = 0;
      awaitingFirst = false;
      render();
    },
  };
}
