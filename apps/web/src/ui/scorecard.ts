/**
 * The model scorecard's DOM — slice 16. Data and the pass/fail decision live in
 * `run/scorecard.ts`; this file only ever reads them and reacts to a click, the same split every
 * other `ui/*` controller in this project already follows.
 */

import { badgeEarned, SCORECARD_BAR, SCORECARD_STEPS, type ScorecardSeedResult } from '../run/scorecard.ts';

export interface ScorecardOptions {
  readonly getResults: () => readonly ScorecardSeedResult[];
  readonly isRunning: () => boolean;
  readonly start: () => void;
}

export interface ScorecardController {
  open(): void;
  close(): void;
  readonly isOpen: () => boolean;
  render(): void;
}

export function createScorecard(opts: ScorecardOptions): ScorecardController {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from index.html`);
    return el as T;
  };

  const root = $('scorecard');
  const body = $('scorecard-body');
  let open = false;

  function pct(v: number): string {
    return `${(v * 100).toFixed(1)}%`;
  }

  function render(): void {
    body.replaceChildren();
    const inner = document.createElement('div');
    inner.className = 'sc-inner';

    const intro = document.createElement('p');
    intro.className = 'sc-intro';
    intro.innerHTML =
      `Retrains <em>this exact configuration</em> — the same hidden layers, activation, ` +
      `initialisation, learning rate, batch size and optimiser Explorer has set right now — five ` +
      `times on Digits, each with a different held-out validation split and a different weight ` +
      `initialisation. ${SCORECARD_STEPS.toLocaleString()} steps each. The badge asks a sharper ` +
      `question than any one run can answer: not "did this work", but "does it work reliably".`;
    inner.append(intro);

    const results = opts.getResults();
    const running = opts.isRunning();

    const btn = document.createElement('button');
    btn.className = 'pri sc-start';
    btn.textContent = running ? 'Running…' : results.length > 0 ? 'Run again' : 'Run the scorecard';
    btn.disabled = running;
    btn.addEventListener('click', () => opts.start());
    inner.append(btn);

    if (results.length > 0) {
      const rows = document.createElement('div');
      rows.className = 'sc-rows';
      for (const r of results) {
        const row = document.createElement('div');
        row.className = 'sc-row' + (r.valAccuracy === null ? ' pending' : '');
        const label = document.createElement('span');
        label.textContent = `seed ${r.seed}`;
        const value = document.createElement('span');
        value.textContent = r.valAccuracy === null ? '…' : pct(r.valAccuracy);
        row.append(label, value);
        rows.append(row);
      }
      inner.append(rows);

      const finished = results.filter((r): r is { seed: number; valAccuracy: number } => r.valAccuracy !== null);
      if (finished.length > 0) {
        const accs = finished.map((r) => r.valAccuracy);
        const mean = accs.reduce((a, b) => a + b, 0) / accs.length;
        const min = Math.min(...accs);
        const max = Math.max(...accs);
        const summary = document.createElement('div');
        summary.className = 'sc-summary';
        summary.innerHTML =
          `<span>mean <b>${pct(mean)}</b></span><span>worst <b>${pct(min)}</b></span>` +
          `<span>best <b>${pct(max)}</b></span>`;
        inner.append(summary);
      }

      const badge = badgeEarned(results);
      if (badge !== null) {
        const b = document.createElement('div');
        b.className = 'sc-badge ' + (badge ? 'earned' : 'missed');
        b.textContent = badge
          ? `Badge earned — every seed cleared ${pct(SCORECARD_BAR)}.`
          : `Badge not earned — at least one seed fell short of ${pct(SCORECARD_BAR)}.`;
        inner.append(b);
      }
    }

    body.append(inner);
  }

  $('scorecard-close').addEventListener('click', () => close());

  function close(): void {
    open = false;
    root.hidden = true;
  }

  return {
    open(): void {
      open = true;
      root.hidden = false;
      render();
    },
    close,
    isOpen: () => open,
    render,
  };
}
