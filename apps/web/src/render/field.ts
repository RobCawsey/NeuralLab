/**
 * The decision field: what the network would answer at every point on the plane.
 *
 * **This is the expensive drawing in the project, and by a long way.** §5 budgets one training
 * step on 2-8-8-2 at roughly 15 k multiply–adds and a 128² field at 5.2 M — three hundred times
 * more. Everything about how this file is used follows from that ratio: it is throttled while
 * training, it drops to 64² when the network is moving, and it goes back up to 128² the moment
 * the run stops.
 *
 * `computeField` imports nothing from the DOM, so the arithmetic — which is where the bugs are —
 * is checked in Node. `drawField` is the only part that needs a canvas.
 */

import { forward, argmax, type Net, type Scratch } from '@neurallab/mlp';
import type { Standardiser } from '@neurallab/core';
import { sx, sy, type Box, type Camera } from './camera.ts';

export interface Field {
  readonly res: number;
  /** Winning class per cell, row-major from the **bottom-left**. */
  readonly cls: Uint8Array;
  /** The winner's probability, 1/classes … 1. */
  readonly conf: Float32Array;
  /** The world rectangle this field covers. */
  readonly box: Box;
  readonly classes: number;
}

/** Resolutions, and when each is used. Measured in slice 3 — see the notes in CLAUDE.md. */
export const FIELD_RES = { live: 64, paused: 128 } as const;

/** Milliseconds between recomputes while training. */
export const FIELD_THROTTLE_MS = 150;

/**
 * Evaluate the network across a grid.
 *
 * Row-major **from the bottom-left**, because the field is a thing in world space where y is up.
 * Flipping it here rather than at draw time would be a second y-flip, and the project has exactly
 * one of those (`camera.ts`). `drawField` walks the rows backwards instead.
 *
 * The point is standardised the same way training data was — the network has never seen a raw
 * coordinate, so a field computed on raw coordinates would be answering a different question
 * from the one the scatter beneath it is asking.
 */
export function computeField(
  net: Net,
  scratch: Scratch,
  standardiser: Standardiser,
  box: Box,
  res: number,
  classes: number,
): Field {
  const cls = new Uint8Array(res * res);
  const conf = new Float32Array(res * res);
  const point = new Float32Array(2);

  const mx = standardiser.mean[0] as number;
  const my = standardiser.mean[1] as number;
  const sdx = standardiser.sd[0] as number;
  const sdy = standardiser.sd[1] as number;

  for (let row = 0; row < res; row++) {
    // Cell centres, not corners: sampling the corner biases the whole field half a cell
    // down and left, which is invisible at 128² and obvious at 16².
    const wy = box.minY + ((row + 0.5) / res) * (box.maxY - box.minY);
    point[1] = (wy - my) / sdy;

    for (let col = 0; col < res; col++) {
      const wx = box.minX + ((col + 0.5) / res) * (box.maxX - box.minX);
      point[0] = (wx - mx) / sdx;

      const out = forward(net, point, scratch);
      const best = argmax(out);
      const i = row * res + col;
      cls[i] = best;
      conf[i] = out[best] as number;
    }
  }

  return { res, cls, conf, box, classes };
}

/** Forward passes one field costs — for the readout, so the price is visible rather than felt. */
export function fieldCost(res: number): number {
  return res * res;
}

const CLASS_RGB: readonly [number, number, number][] = [
  [233, 161, 59],
  [78, 168, 196],
  [139, 123, 216],
  [79, 180, 140],
  [217, 98, 92],
];

let scratchCanvas: HTMLCanvasElement | null = null;

/**
 * Blit the field under the scatter.
 *
 * One `putImageData` into an offscreen canvas at grid resolution, then one `drawImage` scaled up
 * with smoothing **off**. The obvious alternative — `fillRect` per cell — is 16 384 canvas calls
 * at 128², and it repaints several times a second. Evolab's behaviour map settled this exact
 * question and the trick is lifted intact.
 *
 * Smoothing off, not on, and it is a real choice: interpolation makes the boundary look smoother
 * than the network's actual resolution, which is a picture of the drawing rather than of the
 * network. Visible cells are honest about how much has been measured.
 */
export function drawField(
  ctx: CanvasRenderingContext2D,
  field: Field,
  camera: Camera,
  alpha = 1,
): void {
  const { res } = field;
  const canvas = (scratchCanvas ??= document.createElement('canvas'));
  if (canvas.width !== res || canvas.height !== res) {
    canvas.width = res;
    canvas.height = res;
  }
  const off = canvas.getContext('2d');
  if (!off) return;

  const image = off.createImageData(res, res);
  const data = image.data;
  const floor = 1 / Math.max(2, field.classes);

  for (let row = 0; row < res; row++) {
    // ImageData is top-down; the field is bottom-up. This is the flip, and it is a read-order
    // reversal rather than a second coordinate transform.
    const src = (res - 1 - row) * res;
    for (let col = 0; col < res; col++) {
      const i = src + col;
      const rgb = CLASS_RGB[(field.cls[i] as number) % CLASS_RGB.length] as [number, number, number];
      /*
       * Alpha carries confidence, and it is rescaled from the floor rather than from zero.
       * A three-class softmax cannot output less than 1/3 for its winner, so mapping [0,1]
       * straight to alpha would leave an undecided region at 33% opacity — visibly shaded,
       * and wrong. Rescaling makes "no idea" mean transparent for any number of classes.
       */
      const t = Math.max(0, ((field.conf[i] as number) - floor) / (1 - floor));
      const p = (row * res + col) * 4;
      data[p] = rgb[0];
      data[p + 1] = rgb[1];
      data[p + 2] = rgb[2];
      data[p + 3] = Math.round(Math.min(1, t) * 150 * alpha);
    }
  }
  off.putImageData(image, 0, 0);

  const x0 = sx(camera, field.box.minX);
  const x1 = sx(camera, field.box.maxX);
  const y0 = sy(camera, field.box.maxY);
  const y1 = sy(camera, field.box.minY);

  const smoothing = ctx.imageSmoothingEnabled;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(canvas, x0, y0, x1 - x0, y1 - y0);
  ctx.imageSmoothingEnabled = smoothing;
}
