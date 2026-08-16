/**
 * The loss surface — §7 of the design document, the MLP half of the 3D view. A height field of
 * loss over two directions in weight space, with the run's own path traced across it.
 *
 * **Two named weights produce a boring bowl.** 2-8-8-2 has 114 parameters; varying two of them
 * barely moves the loss, because the other 112 are still doing the work. The picture worth
 * looking at needs two **random, filter-normalised** directions (Li et al., 2018) — every
 * parameter perturbed at once, each layer's own slice of the direction rescaled to match that
 * layer's real weight norm, so a layer with large weights is not swamped by the same raw
 * perturbation a layer with small ones gets. `unitDirection` builds the boring version anyway,
 * because the difference between the two *is* the lesson §7 wants told.
 *
 * Everything here works in **flat weight space** — the same `Float32Array` shape
 * `flattenWeights`/`applyWeights` already use, invariant 3's own layout — so a direction is
 * exactly as transferable as a weight snapshot is, and projecting one onto the other is a plain
 * dot product.
 */

import { Rng, type Dataset } from '@neurallab/core';
import { sample } from '@neurallab/core';
import {
  applyWeights,
  createScratch,
  flattenWeights,
  forward,
  type Net,
  type Scratch,
} from './net.ts';
import { sampleLoss } from './loss.ts';

/** Start offset and length of each layer's `W` then `b` slice within the flattened buffer. */
function layerSegments(net: Net): { wStart: number; wLen: number; bStart: number; bLen: number }[] {
  let at = 0;
  return net.layers.map((l) => {
    const wStart = at;
    at += l.W.length;
    const bStart = at;
    at += l.b.length;
    return { wStart, wLen: l.W.length, bStart, bLen: l.b.length };
  });
}

function segmentNorm(v: ArrayLike<number>, start: number, len: number): number {
  let sum = 0;
  for (let i = start; i < start + len; i++) {
    const x = v[i] as number;
    sum += x * x;
  }
  return Math.sqrt(sum);
}

/** Rescale `dir[start..start+len)` so its norm matches `real`'s over the same range. */
function normaliseSegment(dir: Float32Array, real: Float32Array, start: number, len: number): void {
  if (len === 0) return;
  const realNorm = segmentNorm(real, start, len);
  const dirNorm = segmentNorm(dir, start, len);
  const scale = dirNorm > 1e-12 ? realNorm / dirNorm : 0;
  for (let i = start; i < start + len; i++) dir[i] = (dir[i] as number) * scale;
}

/**
 * A random direction in flat weight space, filter-normalised layer by layer — `W` and `b` each
 * their own slice, matching the two blocks `flattenWeights` already writes per layer.
 */
export function randomDirection(net: Net, rng: Rng): Float32Array {
  const real = flattenWeights(net);
  const dir = new Float32Array(real.length);
  for (let i = 0; i < dir.length; i++) dir[i] = rng.normal();
  for (const seg of layerSegments(net)) {
    normaliseSegment(dir, real, seg.wStart, seg.wLen);
    normaliseSegment(dir, real, seg.bStart, seg.bLen);
  }
  return dir;
}

/**
 * A single flat-index basis vector — "literal" mode. `unitDirection(net, 0)` and
 * `unitDirection(net, 1)` are the first layer's first two weights, moved independently of
 * everything else; the obvious implementation §7 warns produces a flat bowl.
 */
export function unitDirection(net: Net, index: number): Float32Array {
  const dir = new Float32Array(flattenWeights(net).length);
  if (index >= 0 && index < dir.length) dir[index] = 1;
  return dir;
}

export interface LossSurface {
  readonly res: number;
  /** The grid spans `[-range, range]` in each direction, in that direction's own units. */
  readonly range: number;
  /** `res * res`, row-major — `values[row * res + col]`. */
  readonly values: Float32Array;
}

function cloneShape(net: Net): Net {
  return {
    loss: net.loss,
    layers: net.layers.map((l) => ({
      inputs: l.inputs,
      units: l.units,
      act: l.act,
      W: new Float32Array(l.W.length),
      b: new Float32Array(l.b.length),
    })),
  };
}

function meanLoss(net: Net, ds: Dataset, rows: readonly number[], scratch: Scratch): number {
  let total = 0;
  for (const row of rows) {
    const y = ds.y === null ? 0 : (ds.y[row] as number);
    const out = forward(net, sample(ds, row), scratch);
    total += sampleLoss(net.loss, out, y);
  }
  return rows.length > 0 ? total / rows.length : 0;
}

/**
 * At most this many rows go into a surface evaluation. `res²` forward passes already multiplies
 * the cost of one; capping the row count keeps a 128-cell grid on a 1000-row dataset roughly the
 * same price as on a 100-row one, which matters because this is a picture, not a training run —
 * §8's "3D is a view, never the simulation" applies to its cost as much as its content.
 */
const SURFACE_ROWS = 64;

/** Every `stride`th row, up to `SURFACE_ROWS` — deterministic, not sampled, so the surface does
 * not flicker between renders of the same network. */
function subsampleRows(rows: Int32Array, cap: number): number[] {
  if (rows.length <= cap) return Array.from(rows);
  const stride = rows.length / cap;
  const out: number[] = [];
  for (let i = 0; i < cap; i++) out.push(rows[Math.floor(i * stride)] as number);
  return out;
}

/**
 * Evaluate the loss over a grid spanning two directions from the network's current weights.
 *
 * Runs on a **scratch network** with its own buffers — the real one is only read, via
 * `flattenWeights` once at the start, the same "own scratch" rule `evaluateRows` and the
 * stepper's trace already follow. Nothing here ever calls `applyWeights` on the network the
 * caller passed in.
 */
export function computeLossSurface(
  net: Net,
  ds: Dataset,
  rows: Int32Array,
  dir1: Float32Array,
  dir2: Float32Array,
  res: number,
  range: number,
): LossSurface {
  const base = flattenWeights(net);
  const scratchNet = cloneShape(net);
  const scratch = createScratch(scratchNet);
  const perturbed = new Float32Array(base.length);
  const sampleRows = subsampleRows(rows, SURFACE_ROWS);
  const values = new Float32Array(Math.max(1, res) * Math.max(1, res));

  for (let row = 0; row < res; row++) {
    const beta = res > 1 ? -range + (2 * range * row) / (res - 1) : 0;
    for (let col = 0; col < res; col++) {
      const alpha = res > 1 ? -range + (2 * range * col) / (res - 1) : 0;
      for (let i = 0; i < base.length; i++) {
        perturbed[i] = (base[i] as number) + alpha * (dir1[i] as number) + beta * (dir2[i] as number);
      }
      applyWeights(scratchNet, perturbed);
      values[row * res + col] = meanLoss(scratchNet, ds, sampleRows, scratch);
    }
  }

  return { res, range, values };
}

/**
 * Where a weight snapshot falls in the `(dir1, dir2)` plane — the run's own path, "replayed from
 * the snapshot ring, not a re-optimisation" per Fig 8.4's note. A plain per-axis projection
 * (`dot(delta, dir) / dot(dir, dir)`), not a least-squares fit against both at once: two
 * independently drawn random directions in a weight space this size are close enough to
 * orthogonal that the difference is not worth a bigger calculation for a picture.
 */
export function projectOntoDirections(
  base: Float32Array,
  weights: Float32Array,
  dir1: Float32Array,
  dir2: Float32Array,
): [number, number] {
  let dot1 = 0;
  let dot2 = 0;
  let n1 = 0;
  let n2 = 0;
  for (let i = 0; i < weights.length; i++) {
    const delta = (weights[i] as number) - (base[i] as number);
    const d1 = dir1[i] as number;
    const d2 = dir2[i] as number;
    dot1 += delta * d1;
    n1 += d1 * d1;
    dot2 += delta * d2;
    n2 += d2 * d2;
  }
  return [n1 > 1e-12 ? dot1 / n1 : 0, n2 > 1e-12 ? dot2 / n2 : 0];
}
