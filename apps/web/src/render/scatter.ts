/**
 * The dataset, drawn. Slice 0's entire stage.
 *
 * Canvas rather than DOM, and that is not premature: slice 3 draws a decision field underneath
 * these same points at 64 × 64 and repaints the pair on every worker report. The points have to
 * live on the surface the field lands on.
 */

import { bounds2d, sample, type Dataset } from '@neurallab/core';
import { fitCamera, padBox, sx, sy, type Camera } from './camera.ts';

/** Class colours. Amber and cyan — §7 of the design document: the two ends of one scale. */
const CLASS_COLOURS = ['#E9A13B', '#4EA8C4', '#8B7BD8', '#4FB48C', '#D9625C'] as const;
const AXIS = '#2C2A3A';
const GRID = '#1F1E2B';
const LABEL = '#5C5871';

export function classColour(c: number): string {
  return CLASS_COLOURS[c % CLASS_COLOURS.length] as string;
}

export interface ScatterView {
  readonly camera: Camera;
}

/**
 * Size a canvas to its CSS box at device resolution.
 *
 * Returns CSS pixel dimensions, because every renderer works in CSS pixels and the transform
 * absorbs the ratio. Doing it the other way means every radius and font size in the project
 * has to be multiplied by dpr, and one of them eventually is not.
 */
export function resize(canvas: HTMLCanvasElement): { w: number; h: number; ctx: CanvasRenderingContext2D } | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(canvas.clientWidth));
  const h = Math.max(1, Math.round(canvas.clientHeight));
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx };
}

export function drawScatter(
  ctx: CanvasRenderingContext2D,
  ds: Dataset,
  width: number,
  height: number,
  hover: number | null = null,
): ScatterView {
  const cam = fitCamera(padBox(bounds2d(ds)), width, height);

  ctx.clearRect(0, 0, width, height);
  drawGrid(ctx, cam, width, height);

  // Points last and unsorted: with 240 samples the overlap is the information — a dense
  // overlap region is exactly where the boundary is hard, and sorting by class would
  // systematically hide one of them under the other.
  for (let i = 0; i < ds.n; i++) {
    const p = sample(ds, i);
    const px = sx(cam, p[0] as number);
    const py = sy(cam, p[1] as number);
    const cls = ds.y === null ? 0 : (ds.y[i] as number);

    ctx.beginPath();
    ctx.arc(px, py, i === hover ? 5.5 : 3.4, 0, Math.PI * 2);
    ctx.fillStyle = classColour(cls);
    ctx.globalAlpha = 0.92;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = i === hover ? '#E4E2EC' : '#0E0D15';
    ctx.stroke();
  }

  return { camera: cam };
}

function drawGrid(ctx: CanvasRenderingContext2D, cam: Camera, width: number, height: number): void {
  const step = niceStep(cam.plot.w / cam.scale);

  ctx.lineWidth = 1;
  ctx.font = '10px "Cascadia Mono", Consolas, monospace';
  ctx.textBaseline = 'top';

  const left = cam.centreX - cam.plot.w / 2 / cam.scale;
  const right = cam.centreX + cam.plot.w / 2 / cam.scale;
  const bottom = cam.centreY - cam.plot.h / 2 / cam.scale;
  const top = cam.centreY + cam.plot.h / 2 / cam.scale;

  for (let v = Math.ceil(left / step) * step; v <= right; v += step) {
    const px = Math.round(sx(cam, v)) + 0.5;
    ctx.strokeStyle = Math.abs(v) < step / 2 ? AXIS : GRID;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
    ctx.fillStyle = LABEL;
    ctx.textAlign = 'center';
    ctx.fillText(fmt(v, step), px, height - 14);
  }

  for (let v = Math.ceil(bottom / step) * step; v <= top; v += step) {
    const py = Math.round(sy(cam, v)) + 0.5;
    ctx.strokeStyle = Math.abs(v) < step / 2 ? AXIS : GRID;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(width, py);
    ctx.stroke();
    ctx.fillStyle = LABEL;
    ctx.textAlign = 'left';
    ctx.fillText(fmt(v, step), 5, py + 3);
  }
}

/** A 1 / 2 / 5 × 10^k step that puts roughly six gridlines across the span. */
function niceStep(span: number): number {
  const raw = span / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(1e-9, raw))));
  const norm = raw / mag;
  const mult = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
  return mult * mag;
}

function fmt(v: number, step: number): string {
  const dp = Math.max(0, Math.min(4, -Math.floor(Math.log10(step))));
  const s = v.toFixed(dp);
  // "-0.0" is a real output of toFixed and it looks like a bug on an axis.
  return s === (0).toFixed(dp) || Number(s) === 0 ? (0).toFixed(dp) : s;
}
