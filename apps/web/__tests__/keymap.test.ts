import { describe, expect, it } from 'vitest';
import { dispatchKeymap, key, type KeymapEntry } from '../src/ui/keymap.ts';

/** No jsdom in this project's test environment — a plain object satisfies everything `keymap.ts`
 * actually reads off a `KeyboardEvent` (`key`, `code`, `preventDefault`), so a fake stands in
 * rather than pulling in a DOM. */
function fakeEvent(overrides: Partial<{ key: string; code: string }> = {}): KeyboardEvent {
  let prevented = false;
  const event = {
    key: '',
    code: '',
    ...overrides,
    preventDefault: () => {
      prevented = true;
    },
    get defaultPrevented() {
      return prevented;
    },
  };
  return event as unknown as KeyboardEvent;
}

describe('key', () => {
  it('matches case-insensitively', () => {
    let ran = 0;
    const entry = key('R', 'resample', () => ran++);
    expect(entry.match(fakeEvent({ key: 'r' }))).toBe(true);
    expect(entry.match(fakeEvent({ key: 'R' }))).toBe(true);
    expect(entry.match(fakeEvent({ key: 'x' }))).toBe(false);
  });

  it('does not run just by matching — only dispatchKeymap runs it', () => {
    let ran = 0;
    const entry = key('R', 'resample', () => ran++);
    entry.match(fakeEvent({ key: 'r' }));
    expect(ran).toBe(0);
  });
});

describe('dispatchKeymap', () => {
  it('runs the first matching entry and reports a match', () => {
    let ran = '';
    const entries: KeymapEntry[] = [
      key('R', 'resample', () => (ran = 'resample')),
      key('W', 'reinit', () => (ran = 'reinit')),
    ];
    const matched = dispatchKeymap(entries, fakeEvent({ key: 'w' }));
    expect(matched).toBe(true);
    expect(ran).toBe('reinit');
  });

  it('reports no match and runs nothing when no entry fits', () => {
    let ran = false;
    const entries: KeymapEntry[] = [key('R', 'resample', () => (ran = true))];
    const matched = dispatchKeymap(entries, fakeEvent({ key: 'z' }));
    expect(matched).toBe(false);
    expect(ran).toBe(false);
  });

  it('stops at the first match — a later entry never fires for the same event', () => {
    const order: string[] = [];
    const entries: KeymapEntry[] = [
      { key: 'X', does: 'first', match: () => true, run: () => order.push('first') },
      { key: 'X', does: 'second', match: () => true, run: () => order.push('second') },
    ];
    dispatchKeymap(entries, fakeEvent({ key: 'x' }));
    expect(order).toEqual(['first']);
  });

  it('calls preventDefault only when the entry asks for it', () => {
    const withPd: KeymapEntry = { key: 'Space', does: 'train', match: (e) => e.code === 'Space', run: () => {}, preventDefault: true };
    const withoutPd = key('.', 'step', () => {});

    const e1 = fakeEvent({ code: 'Space' });
    dispatchKeymap([withPd], e1);
    expect((e1 as unknown as { defaultPrevented: boolean }).defaultPrevented).toBe(true);

    const e2 = fakeEvent({ key: '.' });
    dispatchKeymap([withoutPd], e2);
    expect((e2 as unknown as { defaultPrevented: boolean }).defaultPrevented).toBe(false);
  });
});
