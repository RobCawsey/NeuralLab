/**
 * The confusion matrix — slice 16. A grid a reader reads two ways at once: which cells are dark
 * (rare) versus bright (common) via the same heatgrid blit the U-matrix already uses, and which
 * cell is *right* via a thin border traced along the diagonal — a distinction plain amber alone
 * cannot carry, since a wrong prediction made often is exactly as bright as a right one.
 */

import type { ConfusionMatrix } from '@neurallab/mlp';
import { drawHeatgrid } from './heatgrid.ts';

const AMBER: readonly [number, number, number] = [233, 161, 59];
const LABEL = '#5c5871';
const DIAG = 'rgba(233, 161, 59, .55)';

export function drawConfusion(
  ctx: CanvasRenderingContext2D,
  m: ConfusionMatrix,
  classNames: readonly string[],
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);
  const c = m.classes;
  if (c < 2 || width < 40 || height < 40) return;

  // Labels short enough to sit in a fixed-width gutter at any class count this project reaches
  // (2 for moons/circles/xor, 3 for blobs, 10 for digits) — measured against the widest one
  // rather than a guess, so a longer class name never clips against the grid.
  ctx.font = '9px "Cascadia Mono", Consolas, monospace';
  let gutter = 14;
  for (const name of classNames) gutter = Math.max(gutter, ctx.measureText(name.slice(0, 3)).width + 8);
  gutter = Math.min(gutter, width * 0.28, height * 0.28);

  const gridX = gutter;
  const gridY = gutter;
  const gridW = Math.max(1, width - gutter);
  const gridH = Math.max(1, height - gutter);

  // `drawHeatgrid` always draws at (0,0) — every other caller uses it full-panel, so a
  // translated context is the simplest correct way to place it in the gutter-offset box below.
  ctx.save();
  ctx.translate(gridX, gridY);
  drawHeatgrid(ctx, m.counts, c, c, gridW, gridH, AMBER);
  ctx.restore();

  // The diagonal — one cell per class, outlined rather than coloured differently, so "this cell
  // is the right answer" reads as a property of *position* a reader can check against any count.
  ctx.save();
  ctx.translate(gridX, gridY);
  ctx.strokeStyle = DIAG;
  ctx.lineWidth = 1.5;
  const cw = gridW / c;
  const ch = gridH / c;
  for (let i = 0; i < c; i++) ctx.strokeRect(i * cw + 0.75, i * ch + 0.75, cw - 1.5, ch - 1.5);
  ctx.restore();

  // Row labels (actual, left) and column labels (predicted, top).
  ctx.fillStyle = LABEL;
  ctx.font = '9px "Cascadia Mono", Consolas, monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let i = 0; i < c; i++) {
    const label = classNames[i] ?? String(i);
    ctx.fillText(label.slice(0, 4), gridX - 4, gridY + (i + 0.5) * ch);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  for (let i = 0; i < c; i++) {
    const label = classNames[i] ?? String(i);
    ctx.fillText(label.slice(0, 4), gridX + (i + 0.5) * cw, gridY - 3);
  }

  // "actual ↓ / predicted →" — the one thing a bare grid of labels does not say for itself.
  ctx.textAlign = 'left';
  ctx.font = '8px "Cascadia Mono", Consolas, monospace';
  ctx.save();
  ctx.translate(2, gridY - 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('actual', 0, 0);
  ctx.restore();
  ctx.fillText('predicted', gridX, 8);
}
