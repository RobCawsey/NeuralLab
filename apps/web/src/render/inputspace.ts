/**
 * The lattice, drawn where the data lives rather than where the nodes sit on screen — "a flat
 * sheet folded through the data", the same picture Fig 8.4's 3D view makes and the guided flow's
 * own words for its step 2. Two dimensions only, `dims`, because the panel is a flat canvas; for
 * the colour cube that is r and g, which is as arbitrary a pair as any other and named on screen
 * so it never reads as the whole story.
 *
 * Doubles as the stepper's "input space" panel: the same drawing, plus the current sample and,
 * on the update stage, the winning node's own before → after line.
 */

import { sample, type Dataset } from '@neurallab/core';
import type { Som } from '@neurallab/som';
import { NEIGHBOUR_SLOTS } from '@neurallab/som';
import { fitCamera, padBox, sx, sy, type Box } from './camera.ts';

const NODE = '#4ea8c4';
const EDGE = '#2f5a68';
const DATA_POINT = '#5c5871';
const AMBER = '#e9a13b';

function box(ds: Dataset, som: Som, dims: readonly [number, number]): Box {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  const grow = (x: number, y: number): void => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  };
  for (let i = 0; i < ds.n; i++) {
    const p = sample(ds, i);
    grow(p[dims[0]] as number, p[dims[1]] as number);
  }
  const n = som.cols * som.rows;
  for (let i = 0; i < n; i++) {
    const base = i * som.dim;
    grow(som.W[base + dims[0]] as number, som.W[base + dims[1]] as number);
  }
  if (!Number.isFinite(minX)) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

export interface InputSpaceOptions {
  /** Which two of the data's dimensions to plot — default the first two. */
  readonly dims?: readonly [number, number];
  /** The point the current step drew, ringed in amber. */
  readonly probe?: ArrayLike<number> | null;
  /** The winning node's weight vector *before* this step's update — draws a line to its `after`. */
  readonly bmuBefore?: { readonly node: number; readonly weights: ArrayLike<number> } | null;
  /** Fraction of the dataset's own points to draw as context — the rest are silently skipped. */
  readonly maxPoints?: number;
}

export function drawInputSpace(
  ctx: CanvasRenderingContext2D,
  som: Som,
  ds: Dataset,
  width: number,
  height: number,
  opts: InputSpaceOptions = {},
): void {
  const dims = opts.dims ?? [0, 1];
  const cam = fitCamera(padBox(box(ds, som, dims)), width, height);

  ctx.clearRect(0, 0, width, height);

  // Context: a thinned sample of the data, faint — this panel is about the lattice, not a
  // second scatter plot.
  const maxPoints = opts.maxPoints ?? 400;
  const stride = Math.max(1, Math.floor(ds.n / maxPoints));
  ctx.fillStyle = DATA_POINT;
  ctx.globalAlpha = 0.35;
  for (let i = 0; i < ds.n; i += stride) {
    const p = sample(ds, i);
    ctx.beginPath();
    ctx.arc(sx(cam, p[dims[0]] as number), sy(cam, p[dims[1]] as number), 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // The net: one line per lattice edge, each drawn once (only toward higher-indexed neighbours,
  // so a shared edge is not drawn twice on top of itself).
  const n = som.cols * som.rows;
  const nodeXY = (i: number): [number, number] => {
    const base = i * som.dim;
    return [
      sx(cam, som.W[base + dims[0]] as number),
      sy(cam, som.W[base + dims[1]] as number),
    ];
  };
  ctx.strokeStyle = EDGE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
      const j = som.neighbours[i * NEIGHBOUR_SLOTS + s] as number;
      if (j <= i) continue; // -1 fails this too, and the shared-edge dedupe falls out for free
      const [x1, y1] = nodeXY(i);
      const [x2, y2] = nodeXY(j);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
  }
  ctx.stroke();

  ctx.fillStyle = NODE;
  for (let i = 0; i < n; i++) {
    const [x, y] = nodeXY(i);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (opts.bmuBefore) {
    const { node, weights } = opts.bmuBefore;
    const bx = sx(cam, weights[dims[0]] as number);
    const by = sy(cam, weights[dims[1]] as number);
    const [ax, ay] = nodeXY(node);
    ctx.strokeStyle = AMBER;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(ax, ay);
    ctx.stroke();
    ctx.fillStyle = AMBER;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.arc(bx, by, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  if (opts.probe) {
    const px = sx(cam, opts.probe[dims[0]] as number);
    const py = sy(cam, opts.probe[dims[1]] as number);
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#0e0d15';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.strokeStyle = AMBER;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
