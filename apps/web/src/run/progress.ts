/**
 * Progress through the concept ladder — the one piece of app state that outlives a page reload,
 * because neither training run does and the ladder is supposed to. `localStorage`, parsed
 * defensively: this key is user-writable (devtools, an older schema, a hand-edited value) and
 * outlives the code reading it, the same rule every URL parameter in this project already
 * follows.
 *
 * Not unit-tested for the same reason `ui/guided.ts` is not: it touches a browser global this
 * project's test files do not have, and is exercised live instead.
 */

const KEY = 'neurallab.challenges.v1';

export function loadProgress(): Set<number> {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw === null) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    const ids = parsed.filter(
      (v): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 12,
    );
    return new Set(ids);
  } catch {
    return new Set();
  }
}

export function saveProgress(done: ReadonlySet<number>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(Array.from(done)));
  } catch {
    // Private browsing, storage disabled, or full — progress just does not persist this session.
  }
}
