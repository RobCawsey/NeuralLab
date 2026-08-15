/**
 * Diagnostics computed from the mirror model — histograms and the dead-unit count.
 *
 * Everything here reads `state.model`/`state.z` after a report has synced them; nothing trains
 * and nothing needs the worker, because a histogram is just many forward passes over data the
 * page already has. `computeField` (slice 3) established the pattern this follows: run the
 * kernel's own `forward` straight on the main thread's mirror rather than inventing a second
 * implementation, and `ui/guided.ts` (slice 6) reused it the same way for its before/after
 * snapshots. This is the third reuse, not a new idea.
 *
 * Pure — no DOM, no canvas — so it sits in the same vitest glob as `camera.ts` and `guided.ts`.
 */

import { sample, type Dataset } from '@neurallab/core';
import { forward, type Net, type Scratch } from '@neurallab/mlp';

/** Fixed at 32, matching every histogram in the app — §7 of the design document. */
export const HISTOGRAM_BINS = 32;

export interface Histogram {
  readonly counts: readonly number[];
  readonly min: number;
  readonly max: number;
}

/**
 * Bin a set of values. The range comes from the data itself, not a fixed scale — a weight
 * distribution at initialisation spans ±0.5 and one after training might span ±8, and forcing
 * both onto one fixed axis would show the second as a single overfull bin.
 */
export function histogram(values: ArrayLike<number>, bins = HISTOGRAM_BINS): Histogram {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) {
    min = -1;
    max = 1;
  } else if (max - min < 1e-9) {
    // A constant layer (zero-init, or every activation saturated to the same value) still has
    // to draw *something* — one full bin rather than a divide-by-zero.
    min -= 0.5;
    max += 0.5;
  }

  const counts = new Array<number>(bins).fill(0);
  const span = max - min;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isFinite(v)) continue;
    let bin = Math.floor(((v - min) / span) * bins);
    if (bin >= bins) bin = bins - 1;
    if (bin < 0) bin = 0;
    counts[bin] = (counts[bin] as number) + 1;
  }
  return { counts, min, max };
}

export interface LayerActivationStats {
  readonly histogram: Histogram;
  /**
   * Units that never once fired positive across every row examined — the standard "dead ReLU"
   * definition. Meaningless for a layer that does not clamp at zero; `isRelu` says which is
   * which so a caller does not print "0 of 8 dead" under a tanh layer as if that were reassuring.
   */
  readonly deadUnits: number;
  readonly totalUnits: number;
  readonly isRelu: boolean;
}

/**
 * Activation statistics for every layer, over a set of rows.
 *
 * One pass per row, all layers read from the same forward call — not one pass per layer — so
 * this costs exactly what `evaluateRows` costs and nothing more. At 168 training rows on
 * 2-16-16-2 that is the same order of arithmetic as a single training step (§5's budget), so it
 * is cheap enough to recompute on every render without throttling, and it is not throttled.
 */
export function activationStats(
  net: Net,
  ds: Dataset,
  rows: Int32Array,
  scratch: Scratch,
): LayerActivationStats[] {
  const layers = net.layers;
  const values: number[][] = layers.map(() => []);
  const everFired: Uint8Array[] = layers.map((l) => new Uint8Array(l.units));

  for (let k = 0; k < rows.length; k++) {
    forward(net, sample(ds, rows[k] as number), scratch);
    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l]!;
      const a = scratch.a[l] as Float64Array;
      const bucket = values[l] as number[];
      const fired = everFired[l] as Uint8Array;
      for (let u = 0; u < layer.units; u++) {
        const v = a[u] as number;
        bucket.push(v);
        if (v > 0) fired[u] = 1;
      }
    }
  }

  return layers.map((layer, l) => {
    const fired = everFired[l] as Uint8Array;
    let dead = 0;
    for (let u = 0; u < layer.units; u++) if (fired[u] === 0) dead++;
    return {
      histogram: histogram(values[l] as number[]),
      deadUnits: dead,
      totalUnits: layer.units,
      isRelu: layer.act === 'relu',
    };
  });
}

/** One weight histogram per layer — cheap, since it reads buffers already in memory. */
export function weightStats(net: Net): readonly Histogram[] {
  return net.layers.map((layer) => histogram(layer.W));
}
