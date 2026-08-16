/**
 * The guided flow's DOM — four steps, no hyperparameters, ~8 seconds.
 *
 * The step *data* lives in `run/guided.ts` and is checked by vitest without a browser; this file
 * is the renderer that data was built for, and it is exercised live rather than unit-tested, the
 * same split as `ui/stepper.ts`.
 *
 * Progress is controller-local rather than in `AppState`, because it is a fact about *this
 * viewing session* of the flow, not about the run — switching to Explorer and back must not
 * restart anything, on either side of that boundary.
 */

import { GENERATORS, type GeneratorKey } from '@neurallab/data';
import { evaluateRows } from '@neurallab/mlp';
import { bounds2d } from '@neurallab/core';
import { computeField, drawField, type Field } from '../render/field.ts';
import { fitCamera, padBox } from '../render/camera.ts';
import { resize } from '../render/scatter.ts';
import { MLP_FLOW, SHAPES, compareAfterword, stepStatus, type ShapeChoice } from '../run/guided.ts';
import { choiceButton, skipButton, stepEl } from './guidedShared.ts';
import type { AppState } from '../run/state.ts';

interface Snapshot {
  readonly accuracy: number;
  readonly field: Field;
}

/** The low-resolution field a thumbnail needs. Nobody reads this one at 128² — it is 84 px tall. */
const SNAPSHOT_RES = 28;

function captureSnapshot(state: AppState): Snapshot {
  const result = evaluateRows(state.model, state.z, state.parts.val, state.scratch);
  const box = padBox(bounds2d(state.data));
  const field = computeField(
    state.model,
    state.scratch,
    state.standardiser,
    box,
    SNAPSHOT_RES,
    Math.max(2, state.data.classes),
  );
  return { accuracy: result.accuracy, field };
}

function drawSnapshot(canvas: HTMLCanvasElement, snap: Snapshot): void {
  const fit = resize(canvas);
  if (!fit) return;
  fit.ctx.clearRect(0, 0, fit.w, fit.h);
  const camera = fitCamera(snap.field.box, fit.w, fit.h, 4);
  drawField(fit.ctx, snap.field, camera);
}

export interface GuidedOptions {
  readonly getState: () => AppState;
  /** Sets `state.dataset` and rebuilds the data side — the caller already knows how. */
  readonly pickDataset: (key: GeneratorKey) => void;
  /** Sets `state.hidden` and rebuilds the network side. */
  readonly pickShape: (hidden: readonly number[]) => void;
  readonly startTraining: () => void;
  readonly skipToExplorer: () => void;
}

export interface GuidedController {
  /** Called from the app's own render() every tick — cheap, and idempotent. */
  render(): void;
}

export function createGuided(opts: GuidedOptions): GuidedController {
  const panelEl = document.getElementById('guided');
  const compareEl = document.getElementById('guided-compare');
  if (!panelEl || !compareEl) throw new Error('#guided or #guided-compare is missing from index.html');
  // Reassigned to a binding TypeScript can prove non-null inside the closures below — the guard
  // above narrows this scope, but not the nested `function` declarations that capture it.
  const panel: HTMLElement = panelEl;
  const compare: HTMLElement = compareEl;

  let current = 0;
  let before: Snapshot | null = null;
  let after: Snapshot | null = null;

  /*
   * Progress is advanced *before* the rebuild is triggered, not after.
   *
   * `opts.pickDataset`/`opts.pickShape` call straight through to `regenerateData`/`regenerateNet`,
   * which end with a synchronous `render()` — and that `render()` calls back into this module's
   * own `render()` before either function here returns. Updating `current` afterwards would have
   * the panel repaint once with the stale step and never again, since nothing else prompts a
   * second render. Found by clicking a dataset and watching step 1 stay "on" instead of "done".
   */
  function pickDataset(key: GeneratorKey): void {
    before = null;
    after = null;
    if (current < 1) current = 1;
    opts.pickDataset(key);
  }

  function pickShape(shape: ShapeChoice): void {
    current = 2;
    opts.pickShape(shape.hidden);
    // The network the caller just built is exactly the "before" picture — random weights, step
    // zero — and it will never be this untouched again, so the snapshot has to happen now,
    // after the rebuild above has run and while `state.model` still holds it.
    before = captureSnapshot(opts.getState());
    after = null;
    opts.startTraining();
  }


  function dataStepBody(): HTMLElement {
    const state = opts.getState();
    const wrap = document.createElement('div');
    wrap.className = 'gd-choices';
    for (const [key, gen] of Object.entries(GENERATORS)) {
      wrap.append(
        choiceButton(gen.label, gen.blurb, state.dataset === key, () =>
          pickDataset(key as GeneratorKey),
        ),
      );
    }
    return wrap;
  }

  function shapeStepBody(): HTMLElement {
    const state = opts.getState();
    const wrap = document.createElement('div');
    wrap.className = 'gd-choices';
    for (const shape of SHAPES) {
      const on = state.hidden.length === shape.hidden.length && state.hidden.every((v, i) => v === shape.hidden[i]);
      wrap.append(choiceButton(shape.label, ' ', on, () => pickShape(shape)));
    }
    const note = document.createElement('p');
    note.className = 'gd-note';
    note.innerHTML = 'One hidden layer of 8 is <em>a curve</em>. The name <em>hidden layer</em> shows up later, in Explorer.';
    const frag = document.createElement('div');
    frag.append(wrap, note);
    return frag;
  }

  function watchStepBody(): HTMLElement {
    const state = opts.getState();
    const wrap = document.createElement('div');
    const gauge = document.createElement('span');
    gauge.className = 'gauge';
    const i = document.createElement('i');
    const pct = state.targetSteps > 0 ? Math.min(100, (state.step / state.targetSteps) * 100) : 0;
    i.style.width = `${pct}%`;
    gauge.append(i);
    const kv = document.createElement('div');
    kv.className = 'kv';
    kv.innerHTML = `<span>step</span><b class="am">${state.step.toLocaleString()} / ${state.targetSteps.toLocaleString()}</b>`;
    wrap.append(gauge, kv);
    return wrap;
  }

  function compareStepBody(): HTMLElement {
    const p = document.createElement('p');
    p.className = 'gd-note';
    p.textContent = 'See the before-and-after in the panel on the right.';
    return p;
  }

  function renderPanel(): void {
    panel.replaceChildren();
    const bodies = [dataStepBody, shapeStepBody, watchStepBody, compareStepBody];
    MLP_FLOW.forEach((step, i) =>
      panel.append(stepEl(i, stepStatus(i, current), step.title, bodies[i]!)),
    );
    panel.append(skipButton(opts.skipToExplorer));
  }

  function renderCompare(): void {
    compare.replaceChildren();
    const wrap = document.createElement('div');
    wrap.className = 'gd-compare';

    const pair = document.createElement('div');
    pair.className = 'gd-pair';

    const shot = (label: string, snap: Snapshot | null, cls: string): HTMLElement => {
      const box = document.createElement('div');
      box.className = `gd-shot ${cls}`;
      const span = document.createElement('span');
      span.textContent = label;
      const canvas = document.createElement('canvas');
      box.append(span, canvas);
      if (snap) {
        const kv = document.createElement('div');
        kv.className = 'kv';
        kv.innerHTML = `<span>right</span><b class="${cls === 'now' ? 'am' : ''}">${(snap.accuracy * 100).toFixed(1)}%</b>`;
        box.append(kv);
        queueMicrotask(() => drawSnapshot(canvas, snap));
      }
      return box;
    };

    pair.append(shot('its first guess', before, 'first'), shot('now', after, 'now'));
    wrap.append(pair);

    const note = document.createElement('p');
    note.className = 'gd-note';
    if (before && after) {
      note.innerHTML = compareAfterword(before.accuracy, after.accuracy, opts.getState().targetSteps);
    } else if (before) {
      note.textContent = 'Training — this panel fills in once it finishes.';
    } else {
      note.textContent = 'Pick a shape to begin.';
    }
    wrap.append(note);

    compare.append(wrap);
  }

  return {
    render(): void {
      const state = opts.getState();

      // Auto-advance out of "watch it learn" the moment the run actually finishes — not a
      // moment before, since `after` is a snapshot of the network the run left behind.
      if (current === 2 && state.step > 0 && state.step >= state.targetSteps && after === null) {
        after = captureSnapshot(state);
        current = 3;
      }

      renderPanel();
      renderCompare();
    },
  };
}
