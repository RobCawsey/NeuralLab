/**
 * The stepper's drawings: strips of cells for a vector, a matrix as a grid of tiles.
 *
 * Everything here takes numbers that are already in a `StepTrace` — nothing computes an
 * activation or a gradient. §6's rule is that the stepper shows the algorithm rather than an
 * illustration of it, and that rule lives in `packages/mlp/src/trace.ts`; this file only draws
 * what it is handed.
 */

import { heatColour } from './network.ts';

const CYAN = '#4ea8c4';
const AMBER = '#e9a13b';
const LINE = '#2c2a3a';
const PANEL = '#16151f';

/** A signed value to colour — the same diverging ramp as the network graph. §7. */
export function signedColour(v: number, max: number): string {
  const t = max > 0 ? Math.max(-1, Math.min(1, v / max)) : 0;
  const alpha = 0.1 + Math.abs(t) * 0.85;
  return t < 0 ? `rgba(78, 168, 196, ${alpha.toFixed(3)})` : `rgba(233, 161, 59, ${alpha.toFixed(3)})`;
}

export interface StripOptions {
  /** Draw as a diverging (signed) ramp rather than the monotone magnitude ramp. */
  readonly signed?: boolean;
  /** Highlight this cell — the sample's own unit, or the largest update. */
  readonly highlight?: number | null;
  readonly peak?: number;
}

/** One vector, as a row of coloured cells. Used for a, δ, and Δw viewed as a flat strip. */
export function drawStrip(
  ctx: CanvasRenderingContext2D,
  values: ArrayLike<number>,
  width: number,
  height: number,
  opts: StripOptions = {},
): void {
  ctx.clearRect(0, 0, width, height);
  const n = values.length;
  if (n === 0) return;

  let peak = opts.peak ?? 0;
  if (opts.peak === undefined) {
    for (let i = 0; i < n; i++) {
      const v = Math.abs(values[i] as number);
      if (Number.isFinite(v) && v > peak) peak = v;
    }
  }

  const gap = Math.min(2, width / n / 6);
  const cellW = width / n;

  for (let i = 0; i < n; i++) {
    const v = values[i] as number;
    const x = i * cellW;
    const w = Math.max(1, cellW - gap);

    if (!Number.isFinite(v)) {
      ctx.fillStyle = '#d9625c';
    } else if (opts.signed) {
      ctx.fillStyle = signedColour(v, peak);
    } else {
      ctx.fillStyle = heatColour(peak > 0 ? Math.abs(v) / peak : 0);
    }
    ctx.fillRect(x, 0, w, height);

    if (opts.highlight === i) {
      ctx.strokeStyle = '#e4e2ec';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.75, 0.75, w - 1.5, height - 1.5);
    }
  }
}

export interface TileOptions {
  readonly highlight?: { readonly row: number; readonly col: number } | null;
}

/**
 * A weight matrix, as a grid of tiles — `units × inputs`, row-major, the same layout `Dense.W`
 * already uses. No transpose, because the whole point of that layout (§3) is that this is a
 * straight blit.
 */
export function drawTiles(
  ctx: CanvasRenderingContext2D,
  W: ArrayLike<number>,
  units: number,
  inputs: number,
  width: number,
  height: number,
  opts: TileOptions = {},
): void {
  ctx.clearRect(0, 0, width, height);
  if (units === 0 || inputs === 0) return;

  let peak = 0;
  for (let i = 0; i < W.length; i++) {
    const v = Math.abs(W[i] as number);
    if (Number.isFinite(v) && v > peak) peak = v;
  }

  const cellW = width / inputs;
  const cellH = height / units;
  const gap = Math.min(1.5, Math.min(cellW, cellH) / 8);

  for (let u = 0; u < units; u++) {
    for (let i = 0; i < inputs; i++) {
      const v = W[u * inputs + i] as number;
      ctx.fillStyle = signedColour(v, peak);
      ctx.fillRect(i * cellW, u * cellH, Math.max(1, cellW - gap), Math.max(1, cellH - gap));
    }
  }

  if (opts.highlight) {
    const { row, col } = opts.highlight;
    ctx.strokeStyle = '#e4e2ec';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(col * cellW + 0.75, row * cellH + 0.75, cellW - 1.5, cellH - 1.5);
  }
}

/** A small legend swatch pair, drawn once per panel rather than computed per cell. */
export function legendColours(): { negative: string; positive: string; line: string; panel: string } {
  return { negative: CYAN, positive: AMBER, line: LINE, panel: PANEL };
}
