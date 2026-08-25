/**
 * The challenge track's DOM — §9's concept ladder, twelve cards, four phases. Card *data* lives
 * in `run/challenges.ts` and is checked by vitest without a browser; this file is the renderer
 * that data was built for, exercised live instead, the same split every other `ui/*` controller
 * in this project already follows.
 *
 * Nothing here is locked. "Frontier" only decides which card opens by default and which ones are
 * dimmed for guidance — every card stays clickable, because a reader who already knows the
 * material should not have to replay the ladder to reach the part they came for.
 */

import { paramBudget } from '@neurallab/mlp';
import { uMatrix } from '@neurallab/som';
import { CHALLENGES, type Challenge, type ChallengeConfig, type ChallengeOutcome } from '../run/challenges.ts';
import { loadProgress, saveProgress } from '../run/progress.ts';
import type { AppState } from '../run/state.ts';
import type { SomState } from '../run/somState.ts';

export interface ChallengesOptions {
  readonly getState: () => AppState;
  readonly getSomState: () => SomState;
  /** Reconfigures the app to a card's recipe and switches to Explorer — the caller already knows how. */
  readonly applyChallenge: (config: ChallengeConfig) => void;
}

export interface ChallengesController {
  open(): void;
  close(): void;
  readonly isOpen: () => boolean;
  /** Called from the app's own render() every tick — cheap, and idempotent, like the guided flow. */
  render(): void;
}

export function createChallenges(opts: ChallengesOptions): ChallengesController {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from index.html`);
    return el as T;
  };

  const root = $('challenges');
  const chip = $('ch-chip');
  const dotsEl = $('ch-dots');
  const listEl = $('ch-list');
  const toolbarChip = $('challenge-count');

  let open = false;
  let openId: number | null = null;
  let activeId: number | null = null;
  let done = loadProgress();
  // Challenge 12's own two-halves completion — the only card where "finished" means something
  // happened on *each* network, not one run reaching its target.
  let ch12Mlp = false;
  let ch12Som = false;

  function frontier(): number | null {
    for (const c of CHALLENGES) if (!done.has(c.id)) return c.id;
    return null;
  }

  function markDone(id: number): void {
    if (done.has(id)) return;
    done = new Set(done).add(id);
    saveProgress(done);
  }

  /** Whatever the currently-active card's afterword might read, gathered from live state. */
  function outcomeFor(c: Challenge): ChallengeOutcome {
    const state = opts.getState();
    const som = opts.getSomState();

    if (c.id === 12) {
      return { finished: ch12Mlp && ch12Som, mlpFinished: ch12Mlp, somFinished: ch12Som };
    }

    if (c.config.net === 'som') {
      const finished = activeId === c.id && som.trainer.step >= som.targetSteps && som.history.length > 0;
      if (som.history.length === 0) return { finished: false };
      const first = som.history[0]!;
      const last = som.history[som.history.length - 1]!;
      const nodes = som.som.cols * som.som.rows;
      return {
        finished,
        qeStart: first.qe,
        qeEnd: last.qe,
        teStart: first.te,
        teEnd: last.te,
        uMax: Math.max(0, ...Array.from(uMatrix(som.som))),
        nodes,
      };
    }

    const finished = activeId === c.id && state.step >= state.targetSteps && state.points.length > 0;
    const last = state.points[state.points.length - 1];
    if (!last) return { finished: false };
    const budget = paramBudget(state.model, state.parts.train.length);
    return {
      finished,
      trainAccuracy: last.trainAccuracy,
      valAccuracy: last.valAccuracy,
      trainLoss: last.trainLoss,
      gradNorms: last.gradNorms,
      paramCount: budget.params,
      trainRows: budget.samples,
      overBudget: budget.overBudget,
    };
  }

  function dotsRow(): void {
    dotsEl.replaceChildren();
    for (const c of CHALLENGES) {
      const dot = document.createElement('span');
      dot.className = 'ch-dot' + (done.has(c.id) ? ' done' : '');
      dot.title = c.title;
      dotsEl.append(dot);
    }
  }

  function cardHeader(c: Challenge, isDone: boolean): HTMLElement {
    const btn = document.createElement('button');
    btn.className = 'ch-h';
    const n = document.createElement('span');
    n.className = 'ch-n';
    n.textContent = String(c.id);
    const t = document.createElement('span');
    t.className = 'ch-t';
    t.textContent = c.title;
    const check = document.createElement('span');
    check.className = 'ch-check';
    check.textContent = isDone ? '✓' : '';
    btn.append(n, t, check);
    btn.addEventListener('click', () => {
      openId = openId === c.id ? null : c.id;
      render();
    });
    return btn;
  }

  function cardBody(c: Challenge): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'ch-open-body';

    const setup = document.createElement('div');
    setup.className = 'ch-setup';
    setup.textContent = c.setup;
    wrap.append(setup);

    const doneWhen = document.createElement('p');
    doneWhen.className = 'ch-done-when';
    doneWhen.innerHTML = `<em>Done when:</em> ${c.doneWhen}`;
    wrap.append(doneWhen);

    const tags = document.createElement('div');
    tags.className = 'ch-tags';
    const chipEl = document.createElement('span');
    chipEl.className = 'chip';
    chipEl.textContent = c.concept;
    tags.append(chipEl);
    wrap.append(tags);

    const start = document.createElement('button');
    start.className = 'pri';
    start.style.width = '100%';
    start.textContent = 'Start this challenge →';
    start.addEventListener('click', () => {
      activeId = c.id;
      if (c.id === 12) {
        ch12Mlp = false;
        ch12Som = false;
      }
      opts.applyChallenge(c.config);
      close();
    });
    wrap.append(start);

    const note = document.createElement('p');
    note.className = 'ch-note';
    note.innerHTML = c.afterword(outcomeFor(c));
    wrap.append(note);

    return wrap;
  }

  function renderList(): void {
    listEl.replaceChildren();
    const inner = document.createElement('div');
    inner.className = 'ch-list-inner';
    const front = frontier();

    let lastPhase = '';
    for (const c of CHALLENGES) {
      if (c.phase !== lastPhase) {
        lastPhase = c.phase;
        const ph = document.createElement('div');
        ph.className = 'ch-phase';
        ph.textContent = c.phase;
        inner.append(ph);
      }
      const isDone = done.has(c.id);
      const isAhead = !isDone && front !== null && c.id > front;
      const card = document.createElement('div');
      card.className = 'ch-card' + (isDone ? ' done' : '') + (isAhead ? ' ahead' : '');
      card.append(cardHeader(c, isDone));
      if (openId === c.id) card.append(cardBody(c));
      inner.append(card);
    }
    listEl.append(inner);
  }

  function render(): void {
    dotsRow();
    chip.textContent = `${done.size} of 12 concepts`;
    toolbarChip.textContent = `${done.size} / 12`;

    // Completion is checked every tick, open or not, because the run that finishes it usually
    // finishes after the overlay has already been closed — the reader is watching Explorer.
    if (activeId !== null) {
      const c = CHALLENGES.find((x) => x.id === activeId);
      if (c) {
        if (c.id === 12) {
          const state = opts.getState();
          const som = opts.getSomState();
          if (state.step > 0 && state.step >= state.targetSteps) ch12Mlp = true;
          if (som.trainer.step > 0 && som.trainer.step >= som.targetSteps) ch12Som = true;
          if (ch12Mlp && ch12Som) markDone(12);
        } else {
          const o = outcomeFor(c);
          if (o.finished) markDone(c.id);
        }
      }
    }

    if (open) renderList();
  }

  $('ch-close').addEventListener('click', () => close());

  function close(): void {
    open = false;
    root.hidden = true;
  }

  return {
    open(): void {
      if (open) return;
      open = true;
      if (openId === null) openId = activeId ?? frontier() ?? 1;
      root.hidden = false;
      render();
    },
    close,
    isOpen: () => open,
    render,
  };
}
