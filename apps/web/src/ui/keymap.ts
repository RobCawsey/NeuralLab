/**
 * The one array every global keyboard shortcut is defined in. `main.ts`'s keydown handler
 * dispatches from it, and `ui/help.ts` renders it — a shortcut cannot be documented without
 * existing, or exist without being documented, because both read the same array. §8/§14 of the
 * design document.
 *
 * Deliberately scoped to shortcuts that mean the same thing from anywhere in the app. Escape's
 * "close whatever is open" and the stepper's arrow-key paging are contextual — their effect
 * depends on which overlay happens to be on screen — and stay handled where that context already
 * lives in `main.ts`, rather than forced into a shape (one key, one fixed action) that does not
 * fit them. The help screen documents both anyway, by hand, once, in `ui/help.ts` — they are not
 * missing from the reference, only from this dispatch table.
 */

export interface KeymapEntry {
  /** As shown to a reader: `Space`, `.`, `2`, `?` — not the raw `KeyboardEvent.key`. */
  readonly key: string;
  /** One line, lower-case, no full stop — reads naturally next to the key. */
  readonly does: string;
  readonly match: (event: KeyboardEvent) => boolean;
  readonly run: () => void;
  /** Space needs this — its default is scrolling the page. Nothing else has needed it yet. */
  readonly preventDefault?: boolean;
}

/** A plain single-character shortcut, case-insensitive — the shape every entry but Space uses. */
export function key(k: string, does: string, run: () => void): KeymapEntry {
  return { key: k, does, match: (e) => e.key.toLowerCase() === k.toLowerCase(), run };
}

/** Runs the first matching entry's action and reports whether one matched. */
export function dispatchKeymap(entries: readonly KeymapEntry[], event: KeyboardEvent): boolean {
  for (const entry of entries) {
    if (entry.match(event)) {
      if (entry.preventDefault) event.preventDefault();
      entry.run();
      return true;
    }
  }
  return false;
}
