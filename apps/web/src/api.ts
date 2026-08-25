/**
 * The client wrapper for the slice-15 server — §10 of the design document, followed exactly:
 * the app works with no server at all, and this file is the one place that has to be true. Every
 * function here returns an `ApiResult`, never throws, and never leaves the caller waiting more
 * than five seconds. Offline, a timeout, a 404, a 500, an HTML error page from a proxy, and a 200
 * with unparseable JSON all become the same union, so nothing outside this file needs a
 * `try`/`catch` to talk to the server.
 *
 * Nothing here is browser-specific-below-`apps/web` territory violated — `fetch`, `localStorage`
 * and `AbortController` are exactly the browser surface invariant 5 reserves for this layer.
 */

const TIMEOUT_MS = 5000;

export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly kind: 'offline' | 'timeout' | 'http' | 'parse'; readonly status?: number };

export interface RunSummary {
  readonly id: string;
  readonly title: string | null;
  readonly net: 'mlp' | 'som';
  readonly dataset: string;
  readonly finalLoss: number;
  readonly createdAt: string;
}

export interface RunDetail {
  readonly id: string;
  readonly title: string | null;
  readonly net: 'mlp' | 'som';
  readonly dataset: string;
  /** The same query string `writeUrl`/`writeSomUrl` already produce — §8's "what lives in the URL". */
  readonly config: string;
  readonly finalMetrics: unknown;
  readonly createdAt: string;
}

const OWNER_KEY = 'neurallab.owner.v1';

/**
 * The anonymous identity every saved run belongs to. There is no login anywhere in this project
 * — generated once with `crypto.randomUUID()`, kept in `localStorage` alongside the challenge
 * ladder's own progress, and sent as a bare header with no signature. It grants no real security;
 * it exists only so "list mine" means something on a server nobody else's browser can also poke
 * at the SQLite file of.
 */
export function ownerId(): string {
  try {
    const existing = localStorage.getItem(OWNER_KEY);
    if (existing !== null) return existing;
  } catch {
    // Storage disabled or unavailable — fall through to a fresh id that just won't persist.
  }
  const fresh = crypto.randomUUID();
  try {
    localStorage.setItem(OWNER_KEY, fresh);
  } catch {
    // Private browsing or a full quota — the id still works for this session, it just won't
    // survive a reload, same shape of degradation `run/progress.ts` already accepts.
  }
  return fresh;
}

async function call<T>(
  path: string,
  init: RequestInit & { readonly ownerScoped?: boolean } = {},
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const headers = new Headers(init.headers);
    if (init.ownerScoped) headers.set('X-Owner-Id', ownerId());
    if (init.body !== undefined) headers.set('Content-Type', 'application/json');

    const res = await fetch(path, { ...init, headers, signal: controller.signal });
    if (!res.ok) return { ok: false, kind: 'http', status: res.status };

    // A 200 with a body that isn't the JSON we expect — an error page from a proxy sitting in
    // front of a server that isn't there, most often — is a `parse` failure, not a crash.
    const text = await res.text();
    if (text.length === 0) return { ok: true, data: undefined as T };
    try {
      return { ok: true, data: JSON.parse(text) as T };
    } catch {
      return { ok: false, kind: 'parse' };
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return { ok: false, kind: 'timeout' };
    return { ok: false, kind: 'offline' };
  } finally {
    clearTimeout(timer);
  }
}

export interface SaveRunInput {
  readonly title: string | null;
  readonly net: 'mlp' | 'som';
  readonly dataset: string;
  readonly config: string;
  readonly finalMetrics: unknown;
  readonly finalLoss: number;
}

export function saveRun(input: SaveRunInput): Promise<ApiResult<{ id: string }>> {
  return call('/api/runs', { method: 'POST', body: JSON.stringify(input), ownerScoped: true });
}

export function listRuns(): Promise<ApiResult<readonly RunSummary[]>> {
  return call('/api/runs', { ownerScoped: true });
}

export function reopenRun(id: string): Promise<ApiResult<RunDetail>> {
  return call(`/api/runs/${encodeURIComponent(id)}`, { ownerScoped: true });
}

export function reopenShared(token: string): Promise<ApiResult<RunDetail>> {
  return call(`/api/runs/shared/${encodeURIComponent(token)}`);
}

export function shareRun(id: string): Promise<ApiResult<{ token: string }>> {
  return call(`/api/runs/${encodeURIComponent(id)}/share`, { method: 'POST', ownerScoped: true });
}

export function checkHealth(): Promise<ApiResult<{ ok: boolean }>> {
  return call('/api/health');
}
