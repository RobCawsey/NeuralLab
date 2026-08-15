/**
 * Where every node in the SOM lattice goes on screen — the same split `graph-layout.ts` makes
 * for the network graph, and for the same reason: this is arithmetic, it decides whether the
 * picture is readable, and it can be checked in Node without a canvas.
 *
 * Node positions here are a **drawing** decision, independent of `packages/som/src/lattice.ts`'s
 * axial coordinates, which exist to measure distance correctly. Pointy-top hex tiling happens to
 * use very similar geometry — that is a coincidence of both being "hex", not a shared
 * implementation, and the two are never imported from each other.
 */

import type { Topology } from '@neurallab/som';

export interface LatticeLayout {
  /** `cols * rows * 2`, node-major: node `i`'s pixel centre is `(xy[i*2], xy[i*2+1])`. */
  readonly xy: Float32Array;
  readonly nodeRadius: number;
}

/**
 * Pointy-top hex, "odd-r": odd rows shifted right by half a hex width. Adjacent centres — same
 * row or diagonal — all land `√3 · size` apart, which is what makes hex's six neighbours read as
 * equidistant rather than four close and two far.
 */
function hexPositions(cols: number, rows: number, innerW: number, innerH: number): { xy: Float32Array; size: number } {
  // Raw units where a hex's own "radius" (centre to vertex) is 1: width √3, height 2, row
  // pitch 1.5 — the standard pointy-top tiling. Margins add half a hex on every side so the
  // outermost row/column of circles is not clipped by the panel edge.
  const xSpan = Math.sqrt(3) * (cols + 0.5);
  const ySpan = 1.5 * rows + 0.5;
  const size = Math.max(0.5, Math.min(innerW / xSpan, innerH / ySpan));

  const xy = new Float32Array(cols * rows * 2);
  for (let row = 0; row < rows; row++) {
    // Half a column's worth of shift, not half a hex-width in pixels — it is added inside the
    // same `* √3 * size` step that turns `col` into pixels, so it has to be in the same units.
    const shift = (row & 1) === 1 ? 0.5 : 0;
    for (let col = 0; col < cols; col++) {
      const i = row * cols + col;
      xy[i * 2] = (col + 0.5 + shift) * Math.sqrt(3) * size;
      xy[i * 2 + 1] = (row + 0.75) * 1.5 * size;
    }
  }
  return { xy, size };
}

function rectPositions(cols: number, rows: number, innerW: number, innerH: number): { xy: Float32Array; size: number } {
  const size = Math.max(0.5, Math.min(innerW / (cols + 1), innerH / (rows + 1)));
  const xy = new Float32Array(cols * rows * 2);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const i = row * cols + col;
      xy[i * 2] = (col + 0.5) * size + size * 0.5;
      xy[i * 2 + 1] = (row + 0.5) * size + size * 0.5;
    }
  }
  return { xy, size };
}

export function layoutLattice(
  cols: number,
  rows: number,
  topology: Topology,
  width: number,
  height: number,
  pad = 14,
): LatticeLayout {
  const innerW = Math.max(1, width - pad * 2);
  const innerH = Math.max(1, height - pad * 2);
  if (cols < 1 || rows < 1) return { xy: new Float32Array(0), nodeRadius: 0 };

  const { xy, size } =
    topology === 'hex' ? hexPositions(cols, rows, innerW, innerH) : rectPositions(cols, rows, innerW, innerH);

  // Centre the whole grid in the panel: shift by however much span is left over after fitting.
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < cols * rows; i++) {
    maxX = Math.max(maxX, xy[i * 2] as number);
    maxY = Math.max(maxY, xy[i * 2 + 1] as number);
  }
  const offX = pad + (innerW - maxX) / 2;
  const offY = pad + (innerH - maxY) / 2;
  for (let i = 0; i < cols * rows; i++) {
    xy[i * 2] = (xy[i * 2] as number) + offX;
    xy[i * 2 + 1] = (xy[i * 2 + 1] as number) + offY;
  }

  const nearest = topology === 'hex' ? Math.sqrt(3) * size : size;
  return { xy, nodeRadius: Math.max(1.5, nearest * 0.42) };
}

/** Which node the pointer is over, by nearest centre within reach. */
export function hitLatticeNode(layout: LatticeLayout, px: number, py: number, slack = 3): number {
  const reach = layout.nodeRadius + slack;
  let best = -1;
  let bestSq = reach * reach;
  const n = layout.xy.length / 2;
  for (let i = 0; i < n; i++) {
    const dx = (layout.xy[i * 2] as number) - px;
    const dy = (layout.xy[i * 2 + 1] as number) - py;
    const sq = dx * dx + dy * dy;
    if (sq < bestSq) {
      bestSq = sq;
      best = i;
    }
  }
  return best;
}
