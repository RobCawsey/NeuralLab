/**
 * The confusion matrix — slice 16. `evaluateRows` already answers "how often is the network
 * right"; this answers "when it is wrong, wrong as what" — the question a single accuracy number
 * cannot, and the reason a ten-class problem needs a screen an accuracy readout alone cannot be.
 *
 * Its own scratch, for the same reason `evaluateRows`, `activationStats` and the stepper's trace
 * all already have one: a probe that runs `forward` over rows the reader is not currently looking
 * at must not leave `state.scratch` holding *this* probe's activations when the graph is next
 * drawn from it.
 */

import { sample, type Dataset } from '@neurallab/core';
import { argmax, forward, type Net, type Scratch } from './net.ts';

export interface ConfusionMatrix {
  /** `classes × classes`, row-major: `counts[actual * classes + predicted]`. */
  readonly counts: Int32Array;
  readonly classes: number;
  readonly total: number;
  readonly correct: number;
}

export function confusionMatrix(net: Net, ds: Dataset, rows: Int32Array, scratch: Scratch): ConfusionMatrix {
  const classes = Math.max(2, ds.classes);
  const counts = new Int32Array(classes * classes);
  let correct = 0;

  for (let k = 0; k < rows.length; k++) {
    const row = rows[k] as number;
    const actual = ds.y === null ? 0 : (ds.y[row] as number);
    const out = forward(net, sample(ds, row), scratch);
    const predicted = argmax(out);
    counts[actual * classes + predicted] = (counts[actual * classes + predicted] as number) + 1;
    if (predicted === actual) correct++;
  }

  return { counts, classes, total: rows.length, correct };
}

/** `counts[actual * classes + predicted]` — the one piece of index arithmetic worth a named
 * accessor rather than trusting every call site to get the row-major order right by eye. */
export function confusionAt(m: ConfusionMatrix, actual: number, predicted: number): number {
  return m.counts[actual * m.classes + predicted] as number;
}

/**
 * The `n` actual-class pairs the network confuses most often, off the diagonal, worst first —
 * what a reader looks for once the grid itself is on screen: not "how good is it" but "which two
 * digits does it mix up".
 */
export function topConfusions(
  m: ConfusionMatrix,
  n: number,
): ReadonlyArray<{ actual: number; predicted: number; count: number }> {
  const pairs: Array<{ actual: number; predicted: number; count: number }> = [];
  for (let a = 0; a < m.classes; a++) {
    for (let p = 0; p < m.classes; p++) {
      if (a === p) continue;
      const c = confusionAt(m, a, p);
      if (c > 0) pairs.push({ actual: a, predicted: p, count: c });
    }
  }
  pairs.sort((x, y) => y.count - x.count);
  return pairs.slice(0, n);
}
