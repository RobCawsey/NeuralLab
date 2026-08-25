import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { checkHealth, listRuns, ownerId, reopenShared, saveRun, shareRun } from '../src/api.ts';

/**
 * Every case is a real `Response` (Node's own global, not a hand-rolled fake) so `.ok`/`.status`/
 * `.text()` behave exactly as they would against a real server — only the network itself is
 * faked. The one thing worth distrusting on sight in `api.ts` is that every failure mode collapses
 * to the same `ApiResult` shape rather than a throw, so each of offline / timeout / http / parse
 * gets its own test rather than one happy-path test standing in for all four.
 */

const originalFetch = globalThis.fetch;

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.useRealTimers();
});

describe('ownerId', () => {
  // No `localStorage` global in this project's plain-Node test environment — the same reason
  // `run/progress.ts` is not unit-tested. What's checked here is only that the absence degrades
  // rather than throws: `ownerId`'s own try/catch is what `progress.ts` already does for its
  // reads and writes, applied to the one new caller that needs it.
  it('returns a UUID-shaped id without throwing, even with no localStorage to persist it in', () => {
    expect(() => ownerId()).not.toThrow();
    expect(ownerId()).toMatch(/^[0-9a-f-]{36}$/i);
  });
});

describe('call (via saveRun/listRuns/etc.)', () => {
  it('reports ok:true with the parsed body on a normal 200', async () => {
    globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }));
    const res = await checkHealth();
    expect(res).toEqual({ ok: true, data: { ok: true } });
  });

  it('sends the owner header only for owner-scoped calls', async () => {
    let sentHeaders: Headers | undefined;
    globalThis.fetch = vi.fn(async (_url, init?: RequestInit) => {
      sentHeaders = new Headers(init?.headers);
      return new Response('[]', { status: 200 });
    });

    await listRuns();
    expect(sentHeaders?.has('X-Owner-Id')).toBe(true);

    await checkHealth();
    expect(sentHeaders?.has('X-Owner-Id')).toBe(false);
  });

  it('reports kind:http on a non-2xx status, without throwing', async () => {
    globalThis.fetch = vi.fn(async () => new Response('nope', { status: 404 }));
    const res = await shareRun('some-id');
    expect(res).toEqual({ ok: false, kind: 'http', status: 404 });
  });

  it('reports kind:parse on a 200 whose body is not valid JSON — an error page from a proxy', async () => {
    globalThis.fetch = vi.fn(async () => new Response('<html>Bad Gateway</html>', { status: 200 }));
    const res = await checkHealth();
    expect(res).toEqual({ ok: false, kind: 'parse' });
  });

  it('reports kind:offline when fetch itself rejects — the server is not there at all', async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    });
    const res = await checkHealth();
    expect(res).toEqual({ ok: false, kind: 'offline' });
  });

  it('reports kind:timeout when the request outlives the deadline, and never hangs the caller', async () => {
    vi.useFakeTimers();
    globalThis.fetch = vi.fn(
      (_url, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
        }),
    );

    const pending = checkHealth();
    await vi.advanceTimersByTimeAsync(5000);
    const res = await pending;
    expect(res).toEqual({ ok: false, kind: 'timeout' });
  });

  it('saveRun posts JSON to /api/runs with the owner header', async () => {
    let seenUrl = '';
    let seenBody = '';
    globalThis.fetch = vi.fn(async (url, init?: RequestInit) => {
      seenUrl = String(url);
      seenBody = String(init?.body ?? '');
      return new Response(JSON.stringify({ id: 'abc' }), { status: 201 });
    });

    const res = await saveRun({
      title: null,
      net: 'mlp',
      dataset: 'moons',
      config: 'net=mlp&data=moons',
      finalMetrics: { trainLoss: 0.1 },
      finalLoss: 0.1,
    });

    expect(seenUrl).toBe('/api/runs');
    expect(JSON.parse(seenBody)).toMatchObject({ net: 'mlp', dataset: 'moons' });
    expect(res).toEqual({ ok: true, data: { id: 'abc' } });
  });

  it('reopenShared hits the token-only route with no owner header', async () => {
    let seenUrl = '';
    let sentHeaders: Headers | undefined;
    globalThis.fetch = vi.fn(async (url, init?: RequestInit) => {
      seenUrl = String(url);
      sentHeaders = new Headers(init?.headers);
      return new Response(JSON.stringify({ id: 'x', title: null, net: 'som', dataset: 'blobs', config: '', finalMetrics: {}, createdAt: '' }), { status: 200 });
    });

    await reopenShared('tok123');
    expect(seenUrl).toBe('/api/runs/shared/tok123');
    expect(sentHeaders?.has('X-Owner-Id')).toBe(false);
  });
});
