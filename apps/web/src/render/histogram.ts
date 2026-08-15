/**
 * One drawing: a histogram as bars. Weight distributions and activation distributions both use
 * it — §7 of the design document lists "Histograms" as one row, not two, because both are the
 * same picture of a different array.
 */

import type { Histogram } from '../run/diagnostics.ts';

const LINE = '#2c2a3a';

export function drawHistogram(
  ctx: CanvasRenderingContext2D,
  hist: Histogram,
  width: number,
  height: number,
  colour: string,
): void {
  ctx.clearRect(0, 0, width, height);
  const n = hist.counts.length;
  if (n === 0) return;

  let peak = 1;
  for (const c of hist.counts) if (c > peak) peak = c;

  const cellW = width / n;
  const gap = Math.min(1, cellW / 6);

  for (let i = 0; i < n; i++) {
    const count = hist.counts[i] as number;
    const h = (count / peak) * height;
    ctx.globalAlpha = count > 0 ? 0.35 + 0.6 * (count / peak) : 0;
    ctx.fillStyle = colour;
    ctx.fillRect(i * cellW, height - h, Math.max(1, cellW - gap), h);
  }
  ctx.globalAlpha = 1;

  // The zero line, when zero actually falls inside the range — signed distributions (weights,
  // tanh activations) read very differently depending on how much of them sits either side.
  if (hist.min < 0 && hist.max > 0) {
    const x = ((0 - hist.min) / (hist.max - hist.min)) * width;
    ctx.save();
    ctx.strokeStyle = LINE;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.restore();
  }
}
