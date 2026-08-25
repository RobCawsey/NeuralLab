/**
 * Runs — slice 15. Save, list, reopen, share, and the toolbar's own "is the server even there"
 * dot. Every network call goes through `api.ts`'s `ApiResult`, so nothing in this file ever needs
 * a `try`/`catch` of its own — a failure is a value, not an exception, all the way up.
 *
 * §10's own measured claim — kill the server mid-session and training continues, no dialog, no
 * unhandled rejection, one amber dot in the toolbar — is exactly what this controller is built
 * to be true of: nothing it does can block or interrupt a running train loop, because nothing
 * outside this file and `api.ts` ever awaits it.
 */

import { checkHealth, listRuns, reopenRun, reopenShared, saveRun, shareRun, type RunDetail, type RunSummary, type SaveRunInput } from '../api.ts';

export interface RunsOptions {
  /** Reads current state into a save payload, or null if there is nothing worth saving yet. */
  readonly gatherSave: () => SaveRunInput | null;
  /** Reconfigures the app to a reopened/shared run's recipe and starts training toward its step. */
  readonly applyRun: (detail: RunDetail) => void;
  /** Transient feedback for Save/Share, reusing the toolbar's own hint strip. */
  readonly flashMessage: (msg: string) => void;
}

export interface RunsController {
  open(): void;
  close(): void;
  readonly isOpen: () => boolean;
  save(): Promise<void>;
  /** Applies a run reached via `?shared=<token>` on boot, before the overlay has ever opened. */
  openSharedFromUrl(token: string): Promise<void>;
  startHealthPoll(): void;
}

const HEALTH_POLL_MS = 15000;

export function createRuns(opts: RunsOptions): RunsController {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from index.html`);
    return el as T;
  };

  const root = $('runs');
  const body = $('runs-body');
  const dot = $('conn-dot');
  let open = false;
  let connected = true;

  function setConnected(ok: boolean): void {
    if (ok === connected) return;
    connected = ok;
    dot.hidden = ok;
  }

  function fmtLoss(v: number): string {
    return Number.isFinite(v) ? v.toFixed(4) : '—';
  }

  function fmtWhen(iso: string): string {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
  }

  function row(summary: RunSummary): HTMLElement {
    const r = document.createElement('div');
    r.className = 'run-row';

    const left = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'run-title';
    title.textContent = summary.title ?? `${summary.dataset} · ${summary.net}`;
    const meta = document.createElement('div');
    meta.className = 'run-meta';
    meta.textContent = `${summary.net} · ${summary.dataset} · final ${fmtLoss(summary.finalLoss)} · ${fmtWhen(summary.createdAt)}`;
    left.append(title, meta);

    const actions = document.createElement('div');
    actions.className = 'run-actions';

    const reopenBtn = document.createElement('button');
    reopenBtn.textContent = 'Reopen';
    reopenBtn.addEventListener('click', () => void doReopen(summary.id));

    const shareBtn = document.createElement('button');
    shareBtn.textContent = 'Share';
    shareBtn.addEventListener('click', () => void doShare(summary.id));

    actions.append(reopenBtn, shareBtn);
    r.append(left, actions, document.createElement('span'));
    return r;
  }

  async function doReopen(id: string): Promise<void> {
    const res = await reopenRun(id);
    setConnected(res.ok || res.kind !== 'offline');
    if (!res.ok) {
      opts.flashMessage(
        res.kind === 'offline' || res.kind === 'timeout'
          ? 'Could not reach the server — try again once it is back.'
          : 'That run could not be reopened.',
      );
      return;
    }
    close();
    opts.applyRun(res.data);
  }

  async function doShare(id: string): Promise<void> {
    const res = await shareRun(id);
    setConnected(res.ok || res.kind !== 'offline');
    if (!res.ok) {
      opts.flashMessage('Could not reach the server — sharing needs it online.');
      return;
    }
    const url = `${location.origin}${location.pathname}?shared=${res.data.token}`;
    try {
      await navigator.clipboard.writeText(url);
      opts.flashMessage('Share link copied — read-only, no login needed.');
    } catch {
      opts.flashMessage(url);
    }
  }

  async function render(): Promise<void> {
    body.replaceChildren();
    const loading = document.createElement('p');
    loading.className = 'runs-empty';
    loading.textContent = 'Loading…';
    body.append(loading);

    const res = await listRuns();
    setConnected(res.ok || res.kind !== 'offline');
    body.replaceChildren();

    const inner = document.createElement('div');
    inner.className = 'runs-inner';

    const note = document.createElement('p');
    note.className = 'runs-note';
    note.textContent =
      'Saved runs live on this machine only, tied to this browser — nothing here is an account. ' +
      'Nothing but the config and the final numbers is ever sent: weights are never uploaded, ' +
      'because reopening a run just re-trains it from the same seed.';
    inner.append(note);

    if (!res.ok) {
      const p = document.createElement('p');
      p.className = 'runs-empty';
      p.textContent =
        res.kind === 'offline' || res.kind === 'timeout'
          ? 'The server is not reachable right now — everything else in the app still works.'
          : 'Could not load your saved runs.';
      inner.append(p);
    } else if (res.data.length === 0) {
      const p = document.createElement('p');
      p.className = 'runs-empty';
      p.textContent = 'Nothing saved yet — press Save after a run finishes.';
      inner.append(p);
    } else {
      for (const s of res.data) inner.append(row(s));
    }

    body.append(inner);
  }

  $('runs-close').addEventListener('click', () => close());

  function close(): void {
    open = false;
    root.hidden = true;
  }

  return {
    open(): void {
      if (open) return;
      open = true;
      root.hidden = false;
      void render();
    },
    close,
    isOpen: () => open,

    async save(): Promise<void> {
      const input = opts.gatherSave();
      if (input === null) {
        opts.flashMessage('Nothing to save yet — train a step first.');
        return;
      }
      const res = await saveRun(input);
      setConnected(res.ok || res.kind !== 'offline');
      opts.flashMessage(
        res.ok
          ? 'Saved.'
          : res.kind === 'offline' || res.kind === 'timeout'
            ? 'Could not reach the server — nothing was saved.'
            : 'Save failed.',
      );
    },

    async openSharedFromUrl(token: string): Promise<void> {
      const res = await reopenShared(token);
      setConnected(res.ok || res.kind !== 'offline');
      if (!res.ok) {
        opts.flashMessage('That shared link could not be opened.');
        return;
      }
      opts.applyRun(res.data);
    },

    startHealthPoll(): void {
      const poll = (): void => {
        void checkHealth().then((res) => setConnected(res.ok));
      };
      poll();
      setInterval(poll, HEALTH_POLL_MS);
    },
  };
}
