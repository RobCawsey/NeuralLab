/**
 * Quantisation and topographic error, charted together — the SOM's answer to the loss chart, and
 * the one panel §3 says has to show both at once: QE cannot see a twisted lattice, TE is the
 * number that catches it, and watching them diverge (QE falling, TE climbing) is the whole point
 * of drawing them on the same axes rather than two separate small panels.
 *
 * No band — unlike minibatch loss, QE and TE are already means over the whole dataset at the
 * point they are measured, so there is no per-step spread to show.
 */

export interface SomHistoryPoint {
  readonly step: number;
  readonly qe: number;
  readonly te: number;
}

const GRID = '#1f1e2b';
const AXIS = '#2c2a3a';
const LABEL = '#5c5871';
const AMBER = '#e9a13b';
const CYAN = '#4ea8c4';

export function drawSomChart(
  ctx: CanvasRenderingContext2D,
  points: readonly SomHistoryPoint[],
  width: number,
  height: number,
  totalSteps: number,
): void {
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 30, right: 8, top: 8, bottom: 14 };
  const w = Math.max(1, width - pad.left - pad.right);
  const h = Math.max(1, height - pad.top - pad.bottom);

  const maxStep = Math.max(1, totalSteps);
  let maxV = 0.05;
  for (const p of points) maxV = Math.max(maxV, p.qe, p.te);

  const px = (step: number): number => pad.left + (step / maxStep) * w;
  const py = (v: number): number => pad.top + h - (v / maxV) * h;

  // Grid + axis labels: two horizontal lines, top and bottom of range.
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.font = '9px "Cascadia Mono", Consolas, monospace';
  ctx.fillStyle = LABEL;
  ctx.textAlign = 'right';
  for (const frac of [0, 0.5, 1]) {
    const y = pad.top + h - frac * h;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + w, y);
    ctx.stroke();
    ctx.fillText((maxV * frac).toFixed(2), pad.left - 4, y + 3);
  }
  ctx.strokeStyle = AXIS;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + h);
  ctx.lineTo(pad.left + w, pad.top + h);
  ctx.stroke();

  if (points.length < 2) return;

  const line = (pick: (p: SomHistoryPoint) => number, colour: string): void => {
    ctx.beginPath();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.5;
    points.forEach((p, i) => {
      const x = px(p.step);
      const y = py(pick(p));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  line((p) => p.qe, AMBER);
  line((p) => p.te, CYAN);
}
