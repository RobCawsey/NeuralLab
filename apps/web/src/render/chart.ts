/**
 * The loss chart: two lines and a band.
 *
 * The band is the point. Minibatch loss is very noisy — on two moons a single batch of 16 swings
 * between 0.02 and 0.7 while the network is learning steadily — so drawing one sampled value per
 * point as a line invents smoothness the run did not have. The band is the real spread of every
 * step the point covers; the line is their mean.
 *
 * Slice 2 kept two arrays sampled at different rates and reconciled them here. The worker now
 * produces both halves of a point at the same moment, so there is one array and nothing to
 * reconcile.
 */

import type { RunPoint } from '../workers/protocol.ts';

const GRID = '#1f1e2b';
const AXIS = '#2c2a3a';
const LABEL = '#5c5871';
const AMBER = '#e9a13b';
const CYAN = '#4ea8c4';
const BAD = '#d9625c';

export interface ChartOptions {
  readonly totalSteps?: number;
}

export function drawChart(
  ctx: CanvasRenderingContext2D,
  points: readonly RunPoint[],
  width: number,
  height: number,
  opts: ChartOptions = {},
): void {
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 30, right: 8, top: 8, bottom: 14 };
  const w = Math.max(1, width - pad.left - pad.right);
  const h = Math.max(1, height - pad.top - pad.bottom);
  const last = points[points.length - 1];

  const lastStep = Math.max(1, opts.totalSteps ?? 0, last?.step ?? 0);

  /*
   * The vertical range comes from the data, not a fixed 0–1.
   *
   * A fixed axis hides challenge 3 completely: a destroyed network sits at a loss of 13.8 and
   * would draw as a flat line pinned to the top, which reads as "no data" rather than
   * "catastrophe". The ceiling is printed on the axis so the rescale is visible.
   */
  let top = 0.001;
  for (const p of points) {
    if (Number.isFinite(p.lossMax)) top = Math.max(top, p.lossMax);
    if (Number.isFinite(p.trainLoss)) top = Math.max(top, p.trainLoss);
    if (Number.isFinite(p.valLoss)) top = Math.max(top, p.valLoss);
  }
  top *= 1.08;

  const X = (step: number): number => pad.left + (step / lastStep) * w;
  const Y = (v: number): number => pad.top + (1 - Math.max(0, Math.min(1, v / top))) * h;

  ctx.lineWidth = 1;
  ctx.font = '9px "Cascadia Mono", Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(pad.top + (i / 4) * h) + 0.5;
    ctx.strokeStyle = i === 4 ? AXIS : GRID;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + w, y);
    ctx.stroke();
    ctx.fillStyle = LABEL;
    ctx.fillText(fmt(top * (1 - i / 4)), pad.left - 4, y);
  }

  if (points.length === 0) {
    ctx.fillStyle = LABEL;
    ctx.textAlign = 'center';
    ctx.fillText('nothing trained yet', pad.left + w / 2, pad.top + h / 2);
    return;
  }

  if (points.length > 1) {
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const p = points[i] as RunPoint;
      const y = Y(clamp(p.lossMax));
      if (i === 0) ctx.moveTo(X(p.step), y);
      else ctx.lineTo(X(p.step), y);
    }
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i] as RunPoint;
      ctx.lineTo(X(p.step), Y(clamp(p.lossMin)));
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(233, 161, 59, 0.13)';
    ctx.fill();
  }

  series(ctx, points, X, Y, (p) => clamp(p.trainLoss), AMBER, 2);
  series(ctx, points, X, Y, (p) => clamp(p.valLoss), CYAN, 1.4);

  // Where the run is *now* is the thing read most often, so it gets a mark.
  if (last) {
    ctx.beginPath();
    ctx.arc(X(last.step), Y(clamp(last.trainLoss)), 3, 0, Math.PI * 2);
    ctx.fillStyle = AMBER;
    ctx.fill();

    // Validation above training is what challenge 7 is read from, so it gets a colour rather
    // than being left for the reader to notice.
    const overfitting = last.valLoss > last.trainLoss * 1.25;
    ctx.beginPath();
    ctx.arc(X(last.step), Y(clamp(last.valLoss)), 3, 0, Math.PI * 2);
    ctx.fillStyle = overfitting ? BAD : CYAN;
    ctx.fill();
  }
}

function series(
  ctx: CanvasRenderingContext2D,
  points: readonly RunPoint[],
  X: (step: number) => number,
  Y: (v: number) => number,
  read: (p: RunPoint) => number,
  colour: string,
  lineWidth: number,
): void {
  if (points.length < 2) return;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const p = points[i] as RunPoint;
    const v = read(p);
    if (!Number.isFinite(v)) continue;
    if (i === 0) ctx.moveTo(X(p.step), Y(v));
    else ctx.lineTo(X(p.step), Y(v));
  }
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

/** Keep a non-finite loss off the canvas rather than throwing the whole path away. */
function clamp(v: number): number {
  return Number.isFinite(v) ? v : 0;
}

function fmt(v: number): string {
  if (v >= 10) return v.toFixed(0);
  if (v >= 1) return v.toFixed(1);
  return v.toFixed(2);
}
