/**
 * The lattice, drawn: nodes filled by their own weight vector, the BMU ringed, hits shown by
 * radius. The SOM's answer to `network.ts` — same idea (a node's fill *is* its state), different
 * shape underneath.
 */

import type { Som } from '@neurallab/som';
import type { LatticeLayout } from './lattice-layout.ts';

const LINE = '#2c2a3a';
const AMBER = '#e9a13b';

/**
 * A node's weight vector as a colour.
 *
 * Literal for the colour cube — three weights *are* r, g, b, per §3 of the design document — and
 * a documented stand-in everywhere else: the first one or two weights mapped straight to
 * channels, a fixed mid-tone filling in whatever the data does not supply. A real projection for
 * dimension above 3 is deferred to slice 12, when a dataset that large first exists; nothing
 * today needs it.
 */
export function weightColour(som: Som, node: number): string {
  const base = node * som.dim;
  const c = (k: number): number => Math.round(Math.min(1, Math.max(0, (som.W[base + k] as number))) * 255);
  const r = c(0);
  const g = som.dim > 1 ? c(1) : 128;
  const b = som.dim > 2 ? c(2) : 128;
  return `rgb(${r},${g},${b})`;
}

export interface LatticeDrawOptions {
  readonly bmu?: number | null;
  readonly hover?: number | null;
  /** Node radius scales with `hits / maxHits` between these two fractions of the base radius. */
  readonly maxHits?: number;
}

export function drawLattice(
  ctx: CanvasRenderingContext2D,
  som: Som,
  layout: LatticeLayout,
  width: number,
  height: number,
  opts: LatticeDrawOptions = {},
): void {
  ctx.clearRect(0, 0, width, height);
  const n = som.cols * som.rows;
  const maxHits = opts.maxHits ?? Math.max(1, ...Array.from(som.hits));

  for (let i = 0; i < n; i++) {
    const x = layout.xy[i * 2] as number;
    const y = layout.xy[i * 2 + 1] as number;
    // Hit count widens a node a little rather than changing its colour — colour is already
    // spoken for (the weight vector), and §7's rule is one channel per fact.
    const hitFrac = maxHits > 0 ? (som.hits[i] as number) / maxHits : 0;
    const r = layout.nodeRadius * (0.62 + 0.38 * hitFrac);

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = weightColour(som, i);
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = LINE;
    ctx.stroke();
  }

  if (opts.hover !== null && opts.hover !== undefined && opts.hover >= 0) {
    const x = layout.xy[opts.hover * 2] as number;
    const y = layout.xy[opts.hover * 2 + 1] as number;
    ctx.beginPath();
    ctx.arc(x, y, layout.nodeRadius + 2, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  if (opts.bmu !== null && opts.bmu !== undefined && opts.bmu >= 0) {
    const x = layout.xy[opts.bmu * 2] as number;
    const y = layout.xy[opts.bmu * 2 + 1] as number;
    ctx.beginPath();
    ctx.arc(x, y, layout.nodeRadius + 4, 0, Math.PI * 2);
    ctx.strokeStyle = AMBER;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}
