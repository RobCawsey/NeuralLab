/**
 * The map's guided flow — slice 11, the SOM half of §6/§13. Four steps, no hyperparameters:
 * pick data, watch a flat sheet fold into it, see what it kept apart, label it.
 *
 * Shares `run/guided.ts`'s `GuidedFlow` type and `ui/guidedShared.ts`'s step-card vocabulary with
 * `ui/guided.ts`; does not share its controller, because the state underneath — `SomState`, not
 * `AppState` — is a different shape entirely. See the note in `guidedShared.ts`.
 */

import { topographicError, uMatrix, nodeLabels } from '@neurallab/som';
import { classColour, resize } from '../render/scatter.ts';
import { drawHeatgrid } from '../render/heatgrid.ts';
import { layoutLattice } from '../render/lattice-layout.ts';
import { SOM_FLOW, labelAfterword, regionsAfterword, stepStatus } from '../run/guided.ts';
import { choiceButton, skipButton, stepEl } from './guidedShared.ts';
import { SOM_DATASETS, type SomDatasetKey, type SomState } from '../run/somState.ts';

export interface SomGuidedOptions {
  readonly getState: () => SomState;
  readonly pickDataset: (key: SomDatasetKey) => void;
  readonly startTraining: () => void;
  readonly skipToExplorer: () => void;
  /**
   * "Reveal labels" is the one step in this flow that changes nothing about the run — no rebuild
   * to piggyback a render on, unlike `pickDataset`. Without this, the button would flip
   * `labelsRevealed` and the screen would sit unchanged until some *other* event happened to call
   * render — found by clicking it and watching nothing happen.
   */
  readonly requestRender: () => void;
}

export interface SomGuidedController {
  render(): void;
}

export function createSomGuided(opts: SomGuidedOptions): SomGuidedController {
  const panelEl = document.getElementById('som-guided');
  const compareEl = document.getElementById('som-guided-compare');
  if (!panelEl || !compareEl) throw new Error('#som-guided or #som-guided-compare is missing from index.html');
  const panel: HTMLElement = panelEl;
  const compare: HTMLElement = compareEl;

  let current = 0;
  let labelsRevealed = false;

  // Same ordering rule slice 6 learned for the MLP flow: advance *before* the rebuild that
  // triggers a synchronous render, or the panel repaints once with the stale step and nothing
  // prompts a second repaint.
  function pickDataset(key: SomDatasetKey): void {
    labelsRevealed = false;
    if (current < 1) current = 1;
    opts.pickDataset(key);
    opts.startTraining();
  }

  function dataStepBody(): HTMLElement {
    const state = opts.getState();
    const wrap = document.createElement('div');
    wrap.className = 'gd-choices';
    for (const [key, d] of Object.entries(SOM_DATASETS)) {
      wrap.append(
        choiceButton(d.label, key === 'colourCube' ? 'The classic — the map is the picture.' : 'Reused, unlabelled, from the MLP side.', state.dataset === key, () =>
          pickDataset(key as SomDatasetKey),
        ),
      );
    }
    return wrap;
  }

  function watchStepBody(): HTMLElement {
    const state = opts.getState();
    const wrap = document.createElement('div');
    const gauge = document.createElement('span');
    gauge.className = 'gauge';
    const i = document.createElement('i');
    const pct = state.targetSteps > 0 ? Math.min(100, (state.trainer.step / state.targetSteps) * 100) : 0;
    i.style.width = `${pct}%`;
    gauge.append(i);
    const kv = document.createElement('div');
    kv.className = 'kv';
    kv.innerHTML = `<span>step</span><b class="am">${state.trainer.step.toLocaleString()} / ${state.targetSteps.toLocaleString()}</b>`;
    wrap.append(gauge, kv);
    return wrap;
  }

  function regionsStepBody(): HTMLElement {
    const wrap = document.createElement('div');
    if (!labelsRevealed) {
      const p = document.createElement('p');
      p.className = 'gd-note';
      p.textContent = 'The U-matrix on the right shows the ridges — see the panel there.';
      wrap.append(p);
      const btn = document.createElement('button');
      btn.className = 'gd-choice';
      btn.style.marginTop = '6px';
      const b = document.createElement('b');
      b.textContent = 'Reveal labels';
      btn.append(b);
      btn.addEventListener('click', () => {
        labelsRevealed = true;
        current = 3;
        opts.requestRender();
      });
      wrap.append(btn);
    } else {
      const p = document.createElement('p');
      p.className = 'gd-note';
      p.textContent = 'See the label vote in the panel on the right.';
      wrap.append(p);
    }
    return wrap;
  }

  function renderPanel(): void {
    panel.replaceChildren();
    const bodies = [dataStepBody, watchStepBody, regionsStepBody, () => document.createElement('div')];
    SOM_FLOW.forEach((step, i) =>
      panel.append(stepEl(i, stepStatus(i, current), step.title, i === 3 ? null : bodies[i]!)),
    );
    panel.append(skipButton(opts.skipToExplorer));
  }

  function drawUMatrixSnapshot(canvas: HTMLCanvasElement): void {
    const state = opts.getState();
    const fit = resize(canvas);
    if (!fit) return;
    drawHeatgrid(fit.ctx, uMatrix(state.som), state.som.cols, state.som.rows, fit.w, fit.h, [233, 161, 59]);
  }

  function drawLabelSnapshot(canvas: HTMLCanvasElement): void {
    const state = opts.getState();
    const fit = resize(canvas);
    if (!fit) return;
    const labels = nodeLabels(state.som, state.data, state.rows_);
    const layout = layoutLattice(state.som.cols, state.som.rows, state.som.topology, fit.w, fit.h);
    fit.ctx.clearRect(0, 0, fit.w, fit.h);
    for (let i = 0; i < labels.length; i++) {
      const x = layout.xy[i * 2] as number;
      const y = layout.xy[i * 2 + 1] as number;
      const cls = labels[i] as number;
      fit.ctx.beginPath();
      fit.ctx.arc(x, y, layout.nodeRadius, 0, Math.PI * 2);
      fit.ctx.fillStyle = cls < 0 ? '#2c2a3a' : classColour(cls);
      fit.ctx.fill();
    }
  }

  function renderCompare(): void {
    compare.replaceChildren();
    if (current < 2) {
      const note = document.createElement('p');
      note.className = 'gd-note';
      note.textContent = 'Pick a dataset to begin.';
      compare.append(note);
      return;
    }

    const state = opts.getState();
    const wrap = document.createElement('div');
    wrap.className = 'gd-compare';

    const box = document.createElement('div');
    box.className = 'gd-shot now';
    const label = document.createElement('span');
    label.textContent = labelsRevealed ? 'labels, by majority vote' : 'the U-matrix';
    const canvas = document.createElement('canvas');
    box.append(label, canvas);
    wrap.append(box);
    queueMicrotask(() => (labelsRevealed ? drawLabelSnapshot(canvas) : drawUMatrixSnapshot(canvas)));

    const note = document.createElement('p');
    note.className = 'gd-note';
    note.innerHTML = labelsRevealed
      ? labelAfterword(nodeLabels(state.som, state.data, state.rows_), state.data.classNames)
      : regionsAfterword(Math.max(0, ...Array.from(uMatrix(state.som))), topographicError(state.som, state.data, state.rows_));
    wrap.append(note);

    compare.append(wrap);
  }

  return {
    render(): void {
      const state = opts.getState();
      // Auto-advance out of "watch" the moment the run actually finishes, the same rule
      // `ui/guided.ts` uses for the MLP side — never a moment before, since the U-matrix a reader
      // is about to be shown is a picture of the network the run left behind.
      if (current === 1 && state.trainer.step > 0 && state.trainer.step >= state.targetSteps) {
        current = 2;
      }
      renderPanel();
      renderCompare();
    },
  };
}
