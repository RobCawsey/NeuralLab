/**
 * The SOM stepper — the map's answer to `ui/stepper.ts`. §8: five stages, not seven —
 * **sample → distances → BMU → neighbourhood → update** — because there is no forward/backward
 * split to page through, only one operator each.
 *
 * No worker round trip. The MLP stepper requests a trace and waits for `onTrace`, because
 * training happens on a different thread; SOM training happens on this one (`run/somState.ts`),
 * so "request a trace" is a direct, synchronous call to `somStep(..., { trace: true })`.
 */

import type { Som, SomStepTrace } from '@neurallab/som';
import type { Dataset } from '@neurallab/core';
import { drawLattice } from '../render/lattice.ts';
import { layoutLattice } from '../render/lattice-layout.ts';
import { drawHeatgrid } from '../render/heatgrid.ts';
import { drawInputSpace } from '../render/inputspace.ts';
import { resize } from '../render/scatter.ts';

type StageKind = 'sample' | 'distances' | 'bmu' | 'neighbourhood' | 'update';

interface Stage {
  readonly kind: StageKind;
  readonly label: string;
  readonly sub: string;
}

function buildStages(trace: SomStepTrace): Stage[] {
  return [
    { kind: 'sample', label: 'Take a sample', sub: `x = [${formatVec(trace.input, 2)}]` },
    { kind: 'distances', label: 'Measure distances', sub: '‖x − w‖ per node' },
    { kind: 'bmu', label: 'Find the BMU', sub: `node ${trace.bmu}` },
    { kind: 'neighbourhood', label: 'Spread the neighbourhood', sub: `σ ${trace.sigma.toFixed(2)}` },
    { kind: 'update', label: 'Apply the update', sub: `α ${trace.alpha.toFixed(3)}` },
  ];
}

function formatVec(values: ArrayLike<number>, dp: number): string {
  const parts: string[] = [];
  for (let i = 0; i < values.length; i++) parts.push((values[i] as number).toFixed(dp));
  return parts.join(', ');
}

function peak(values: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (Number.isFinite(v) && v > max) max = v;
  }
  return max;
}

export interface SomStepperOptions {
  readonly getSom: () => Som;
  readonly getData: () => Dataset;
  /** Runs one real step, traced, and returns the trace — `somStep(trainer, data, {trace:true})`. */
  readonly stepTraced: () => SomStepTrace;
  readonly onOpen: () => void;
}

export interface SomStepperController {
  open(): void;
  close(): void;
  readonly isOpen: () => boolean;
}

export function createSomStepper(opts: SomStepperOptions): SomStepperController {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from index.html`);
    return el as T;
  };

  const root = $('som-stepper');
  const stagesEl = $('som-st-stages');
  const latticeCanvas = $<HTMLCanvasElement>('som-st-lattice');
  const inputCanvas = $<HTMLCanvasElement>('som-st-input');
  const kvEl = $('som-st-kv');
  const explainEl = $('som-st-explain');
  const whereEl = $('som-st-where');
  const inputLbEl = $('som-st-input-lb');

  let history: SomStepTrace[] = [];
  let traceIndex = -1;
  let stageIndex = 0;
  let open = false;

  function current(): SomStepTrace | null {
    return traceIndex >= 0 ? (history[traceIndex] ?? null) : null;
  }

  function render(): void {
    const trace = current();
    if (!trace) return;
    const som = opts.getSom();
    const stages = buildStages(trace);
    const stage = stages[stageIndex] ?? (stages[0] as Stage);

    whereEl.textContent = `step ${trace.step} · row ${trace.row}`;
    renderStageList(stages, stage);
    renderViews(som, trace, stage);
    renderKv(som, trace, stage);
    renderExplain(trace, stage);

    $<HTMLButtonElement>('som-st-prev').disabled = stageIndex === 0 && traceIndex === 0;
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

  /** Node weights as they stood *before* this step, for the "sample"/"distances" stages — the
   * update has not happened yet at those points in the story, but `som.W` already reflects it. */
  function beforeSom(som: Som, trace: SomStepTrace): Som {
    return { ...som, W: trace.before };
  }

  function renderViews(som: Som, trace: SomStepTrace, stage: Stage): void {
    const preUpdate = stage.kind !== 'update';
    const drawnSom = preUpdate ? beforeSom(som, trace) : som;

    inputLbEl.textContent = `input space · dims 0, 1`;

    const latticeFit = resize(latticeCanvas);
    if (latticeFit) {
      const layout = layoutLattice(som.cols, som.rows, som.topology, latticeFit.w, latticeFit.h);
      if (stage.kind === 'distances') {
        drawHeatgrid(latticeFit.ctx, trace.distances, som.cols, som.rows, latticeFit.w, latticeFit.h, [233, 161, 59]);
      } else if (stage.kind === 'neighbourhood') {
        drawHeatgrid(latticeFit.ctx, trace.strength, som.cols, som.rows, latticeFit.w, latticeFit.h, [78, 168, 196]);
      } else {
        drawLattice(latticeFit.ctx, drawnSom, layout, latticeFit.w, latticeFit.h, {
          bmu: stage.kind === 'sample' ? null : trace.bmu,
        });
      }
    }

    const inputFit = resize(inputCanvas);
    if (inputFit) {
      drawInputSpace(inputFit.ctx, drawnSom, opts.getData(), inputFit.w, inputFit.h, {
        probe: trace.input,
        bmuBefore:
          stage.kind === 'update' ? { node: trace.bmu, weights: trace.before.subarray(trace.bmu * som.dim, trace.bmu * som.dim + som.dim) } : null,
      });
    }
  }

  function renderKv(som: Som, trace: SomStepTrace, stage: Stage): void {
    kvEl.replaceChildren();
    const rows: [string, string][] = [];
    if (stage.kind === 'sample') {
      rows.push(['row', String(trace.row)], ['x', `[${formatVec(trace.input, 3)}]`]);
    } else if (stage.kind === 'distances') {
      rows.push(['max ‖x − w‖', peak(trace.distances).toFixed(3)], ['bmu (nearest)', `node ${trace.bmu}`]);
    } else if (stage.kind === 'bmu') {
      rows.push(['winner', `node ${trace.bmu}`], ['hits so far', String(som.hits[trace.bmu])]);
    } else if (stage.kind === 'neighbourhood') {
      rows.push(['σ', trace.sigma.toFixed(3)], ['h at bmu', (trace.strength[trace.bmu] as number).toFixed(3)]);
    } else {
      rows.push(['α', trace.alpha.toFixed(4)], ['α · h at bmu', (trace.alpha * (trace.strength[trace.bmu] as number)).toFixed(4)]);
    }
    for (const [label, value] of rows) {
      const row = document.createElement('div');
      row.className = 'kv';
      const span = document.createElement('span');
      span.textContent = label;
      const b = document.createElement('b');
      b.className = 'am';
      b.textContent = value;
      row.append(span, b);
      kvEl.append(row);
    }
  }

  function renderExplain(trace: SomStepTrace, stage: Stage): void {
    const p = document.createElement('div');
    p.className = 'prose';
    p.innerHTML = explanationFor(trace, stage);
    explainEl.replaceChildren(p);
  }

  function explanationFor(trace: SomStepTrace, stage: Stage): string {
    switch (stage.kind) {
      case 'sample':
        return (
          `Row <b>${trace.row}</b>, drawn at random from the training set — the SOM never looks ` +
          `at a batch, one sample drags the lattice at a time. Everything on this screen is what ` +
          `the map does with this one point.`
        );
      case 'distances':
        return (
          `Every node's weight vector is compared to the sample in <b>data space</b> — the heatmap ` +
          `is <code>‖x − w‖</code> for all ${trace.distances.length} nodes at once, brightest ` +
          `where a node is furthest away. The nearest one, by this same measure, is about to win.`
        );
      case 'bmu':
        return (
          `Node <b>${trace.bmu}</b> is nearest the sample in data space and becomes the ` +
          `<b>best-matching unit</b> — ringed on the lattice. Its hit count, the running total of ` +
          `samples it has won, just went up by one — see the panel on the left.`
        );
      case 'neighbourhood':
        return (
          `<code>h(d, σ)</code> falls off from the BMU by <b>lattice</b> distance, not data ` +
          `distance — the heatmap is centred on node ${trace.bmu} and fades with σ = ` +
          `<b>${trace.sigma.toFixed(2)}</b>. A node several lattice-steps away barely moves this ` +
          `step, however close it happens to sit in data space.`
        );
      case 'update':
        return (
          `Every node moves toward the sample by <b>α · h(d, σ)</b> — the BMU itself the most, ` +
          `farther nodes barely at all. The input-space panel draws the BMU's own step: the faint ` +
          `dot is where it stood, the line is where <b>α = ${trace.alpha.toFixed(3)}</b> just took it.`
        );
    }
  }

  function requestNext(): SomStepTrace {
    const trace = opts.stepTraced();
    history.push(trace);
    if (history.length > 50) history.shift();
    traceIndex = history.length - 1;
    stageIndex = 0;
    return trace;
  }

  function next(): void {
    const trace = current();
    const stages = trace ? buildStages(trace) : [];
    if (stageIndex < stages.length - 1) {
      stageIndex++;
    } else {
      requestNext();
    }
    render();
  }

  function prev(): void {
    if (stageIndex > 0) {
      stageIndex--;
      render();
    } else if (traceIndex > 0) {
      traceIndex--;
      stageIndex = buildStages(history[traceIndex] as SomStepTrace).length - 1;
      render();
    }
  }

  function runToEnd(): void {
    const trace = current();
    if (!trace) return;
    stageIndex = buildStages(trace).length - 1;
    render();
  }

  $('som-st-next').addEventListener('click', next);
  $('som-st-prev').addEventListener('click', prev);
  $('som-st-run').addEventListener('click', runToEnd);
  $('som-st-close').addEventListener('click', () => {
    open = false;
    root.hidden = true;
  });

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
  };
}
