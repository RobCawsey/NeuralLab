/**
 * Help — slice 14. A full-screen reference generated from the app's own data, never retyped
 * beside it: the shortcut list from `ui/keymap.ts`, the concept ladder from `run/challenges.ts`,
 * and the dataset glossary from `packages/data` and `run/somState.ts`. The same rule §6 has
 * applied to every piece of generated copy since slice 1's probe note was first wrong, one level
 * up — a reference screen that can go stale is worse than no reference screen.
 *
 * Built once, lazily, on first `open()` — every source it reads (the keymap, the challenge list,
 * the dataset dictionaries) is fixed by the time a reader could possibly press `?`, so there is
 * nothing here a second build would ever show differently.
 */

import { GENERATORS } from '@neurallab/data';
import { CHALLENGES } from '../run/challenges.ts';
import { SOM_DATASETS } from '../run/somState.ts';
import type { KeymapEntry } from './keymap.ts';

export interface HelpOptions {
  readonly getKeymap: () => readonly KeymapEntry[];
}

export interface HelpController {
  open(): void;
  close(): void;
  readonly isOpen: () => boolean;
}

/** Contextual keys the dispatch table deliberately does not carry — see `keymap.ts`'s own note. */
const CONTEXTUAL_KEYS: ReadonlyArray<readonly [string, string]> = [
  ['Esc', 'close whatever is open'],
  ['← →', 'step back or forward, inside the stepper'],
];

export function createHelp(opts: HelpOptions): HelpController {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from index.html`);
    return el as T;
  };

  const root = $('help');
  const body = $('help-body');
  let open = false;
  let built = false;

  function row(cls: string, left: string, right: string, leftTag: 'kbd' | 'span' | 'b' = 'span'): HTMLElement {
    const r = document.createElement('div');
    r.className = cls;
    const l = document.createElement(leftTag);
    l.textContent = left;
    const rt = document.createElement('span');
    rt.textContent = right;
    r.append(l, rt);
    return r;
  }

  function section(title: string): HTMLElement {
    const s = document.createElement('div');
    s.className = 'help-section';
    const h = document.createElement('h3');
    h.textContent = title;
    s.append(h);
    return s;
  }

  function build(): void {
    if (built) return;
    built = true;

    const inner = document.createElement('div');
    inner.className = 'help-inner';

    const keys = section('Keyboard');
    for (const entry of opts.getKeymap()) keys.append(row('help-key-row', entry.key, entry.does, 'kbd'));
    for (const [k, does] of CONTEXTUAL_KEYS) keys.append(row('help-key-row', k, does, 'kbd'));
    inner.append(keys);

    const ladder = section('The concept ladder');
    for (const c of CHALLENGES) {
      ladder.append(row('help-challenge-row', String(c.id), `${c.title} — ${c.concept}`, 'span'));
    }
    inner.append(ladder);

    const mlpBlurbs: Record<string, string> = Object.fromEntries(
      Object.entries(GENERATORS).map(([k, g]) => [k, g.blurb]),
    );
    const mlpSets = section('Datasets — perceptron');
    for (const g of Object.values(GENERATORS)) mlpSets.append(row('help-dataset-row', g.label, g.blurb, 'b'));
    inner.append(mlpSets);

    const somSets = section('Datasets — Kohonen map');
    for (const [k, d] of Object.entries(SOM_DATASETS)) {
      const blurb =
        mlpBlurbs[k] ??
        'Three weights, drawn as a colour — the one set built for the map, not borrowed from the perceptron side.';
      somSets.append(row('help-dataset-row', d.label, blurb, 'b'));
    }
    inner.append(somSets);

    body.append(inner);
  }

  $('help-close').addEventListener('click', () => close());

  function close(): void {
    open = false;
    root.hidden = true;
  }

  return {
    open(): void {
      build();
      open = true;
      root.hidden = false;
    },
    close,
    isOpen: () => open,
  };
}
