/**
 * The loss chart.
 *
 * Three series and one band, and the band is the point. Minibatch loss is very noisy — on two
 * moons a single batch of 16 swings between 0.02 and 0.7 while the network is learning steadily
 * — so drawing one sampled value per step as a line invents smoothness the run did not have.
 * The band shows the spread within each batch; the line is the mean.
 */

const GRID = '#1f1e2b';
const AXIS = '#2c2a3a';
const LABEL = '#5c5871';
const AMBER = '#e9a13b';
const CYAN = '#4ea8c4';
const BAD = '#d9625c';

export interface HistoryPoint {
  readonly step: number;
  /** Mean loss over the batch this step trained on. */
  readonly loss: number;
  readonly lossMin: number;
  readonly lossMax: number;
}

/** A full-dataset measurement, taken every `evalEvery` steps rather than every step. */
export interface EvalPoint {
  readonly step: number;
  readonly trainLoss: number;
  readonly valLoss: number;
  readonly trainAccuracy: number;
  readonly valAccuracy: number;
}

export interface ChartOptions {
  /** Draw accuracy (0–1, higher better) instead of loss. */
  readonly accuracy?: boolean;
  readonly totalSteps?: number;
}

export function drawChart(
  ctx: CanvasRenderingContext2D,
  history: readonly HistoryPoint[],
  evals: readonly EvalPoint[],
  width: number,
  height: number,
  opts: ChartOptions = {},
): void {
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 30, right: 8, top: 8, bottom: 14 };
  const w = Math.max(1, width - pad.left - pad.right);
  const h = Math.max(1, height - pad.top - pad.bottom);

  const lastStep = Math.max(
    1,
    opts.totalSteps ?? 0,
    history.length > 0 ? (history[history.length - 1] as HistoryPoint).step : 0,
  );

  /*
   * The vertical range is taken from the data, not fixed.
   *
   * A fixed 0–1 axis hides challenge 3 completely: a destroyed network sits at a loss of 13.8
   * and would draw as a flat line pinned to the top of the chart, which reads as "no data"
   * rather than "catastrophe". The ceiling is printed on the axis so the rescale is visible.
   */
  let top = opts.accuracy ? 1 : 0.001;
  if (!opts.accuracy) {
    for (const p of history) if (Number.isFinite(p.lossMax)) top = Math.max(top, p.lossMax);
    for (const e of evals) {
      if (Number.isFinite(e.trainLoss)) top = Math.max(top, e.trainLoss);
      if (Number.isFinite(e.valLoss)) top = Math.max(top, e.valLoss);
    }
    top *= 1.08;
  }

  const X = (step: number): number => pad.left + (step / lastStep) * w;
  const Y = (v: number): number => {
    const t = Math.max(0, Math.min(1, v / top));
    return pad.top + (1 - t) * h;
  };

  // grid
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

  if (history.length === 0 && evals.length === 0) {
    ctx.fillStyle = LABEL;
    ctx.textAlign = 'center';
    ctx.fillText('nothing trained yet', pad.left + w / 2, pad.top + h / 2);
    return;
  }

  if (!opts.accuracy && history.length > 1) {
    // The batch spread, as a filled band between min and max.
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const p = history[i] as HistoryPoint;
      const y = Y(clamp(p.lossMax));
      if (i === 0) ctx.moveTo(X(p.step), y);
      else ctx.lineTo(X(p.step), y);
    }
    for (let i = history.length - 1; i >= 0; i--) {
      const p = history[i] as HistoryPoint;
      ctx.lineTo(X(p.step), Y(clamp(p.lossMin)));
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(233, 161, 59, 0.13)';
    ctx.fill();
  }

  if (opts.accuracy) {
    series(ctx, evals, X, Y, (e) => e.trainAccuracy, AMBER, 2);
    series(ctx, evals, X, Y, (e) => e.valAccuracy, CYAN, 1.4);
  } else {
    series(ctx, evals, X, Y, (e) => clamp(e.trainLoss), AMBER, 2);
    series(ctx, evals, X, Y, (e) => clamp(e.valLoss), CYAN, 1.4);
  }

  // The last point, marked. Where the run is *now* is the thing being read most often.
  const last = evals[evals.length - 1];
  if (last) {
    const v = opts.accuracy ? last.trainAccuracy : clamp(last.trainLoss);
    ctx.beginPath();
    ctx.arc(X(last.step), Y(v), 3, 0, Math.PI * 2);
    ctx.fillStyle = AMBER;
    ctx.fill();

    // Validation above training is what challenge 7 is read from, so it gets a colour rather
    // than being left for the reader to notice.
    const overfitting = !opts.accuracy && last.valLoss > last.trainLoss * 1.25;
    ctx.beginPath();
    ctx.arc(X(last.step), Y(opts.accuracy ? last.valAccuracy : clamp(last.valLoss)), 3, 0, Math.PI * 2);
    ctx.fillStyle = overfitting ? BAD : CYAN;
    ctx.fill();
  }
}

function series<T>(
  ctx: CanvasRenderingContext2D,
  points: readonly T[],
  X: (step: number) => number,
  Y: (v: number) => number,
  read: (p: T) => number,
  colour: string,
  lineWidth: number,
): void {
  if (points.length < 2) return;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const p = points[i] as T & { step: number };
    const v = read(p);
    if (!Number.isFinite(v)) continue;
    const x = X(p.step);
    const y = Y(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
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
