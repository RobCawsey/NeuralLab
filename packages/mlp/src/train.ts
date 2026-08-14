/**
 * The training loop: shuffle, take a batch, forward, backward, step.
 *
 * A `Trainer` is state with a lifecycle, so it is a plain object with functions over it rather
 * than a class — the project convention, and it keeps the whole thing structured-cloneable for
 * when slice 4 moves it into a worker.
 */

import { Rng, sample, type Dataset } from '@neurallab/core';
import { argmax, createScratch, forward, type Net, type Scratch } from './net.ts';
import {
  backward,
  createGrads,
  scaleGrads,
  sgdStep,
  zeroGrads,
  hasDiverged,
  type Grads,
} from './backward.ts';
import { sampleLoss } from './loss.ts';

export interface TrainConfig {
  readonly learningRate: number;
  readonly batchSize: number;
}

export const DEFAULT_TRAIN: TrainConfig = { learningRate: 0.1, batchSize: 16 };

export interface Trainer {
  readonly net: Net;
  readonly scratch: Scratch;
  readonly grads: Grads;
  /** Training row indices, reshuffled at the end of each epoch. */
  readonly order: Int32Array;
  readonly rng: Rng;
  config: TrainConfig;
  cursor: number;
  step: number;
  epoch: number;
  diverged: boolean;
}

export function createTrainer(
  net: Net,
  rows: Int32Array,
  config: TrainConfig,
  rng: Rng,
): Trainer {
  const order = Int32Array.from(rows);
  rng.shuffle(order);
  return {
    net,
    scratch: createScratch(net),
    grads: createGrads(net),
    order,
    rng,
    config,
    cursor: 0,
    step: 0,
    epoch: 0,
    diverged: false,
  };
}

export interface StepMetrics {
  readonly step: number;
  readonly epoch: number;
  /** Mean loss over the batch — the number the chart's line is drawn from. */
  readonly loss: number;
  /** Per-sample spread within the batch, for the band. Minibatch loss is very noisy. */
  readonly lossMin: number;
  readonly lossMax: number;
  readonly samples: number;
}

/**
 * One training step: one minibatch, forward and backward, one weight update.
 *
 * This is the unit invariant 2 is written about. The loop that drives it may run one step or
 * five hundred, and the answer is identical either way — how steps are grouped into frames
 * changes nothing about the sequence, which is what keeps the golden test meaningful whatever
 * machine it runs on.
 */
export function trainStep(t: Trainer, ds: Dataset): StepMetrics {
  const { net, scratch, grads, order, config } = t;
  const size = Math.max(1, Math.min(config.batchSize, order.length));

  zeroGrads(grads);
  let total = 0;
  let min = Infinity;
  let max = -Infinity;

  for (let k = 0; k < size; k++) {
    if (t.cursor >= order.length) {
      // Epoch boundary. Reshuffling here rather than at a step count means the order changes
      // exactly once per pass over the data, whatever the batch size is.
      t.rng.shuffle(order);
      t.cursor = 0;
      t.epoch++;
    }
    const row = order[t.cursor++] as number;
    const x = sample(ds, row);
    const y = ds.y === null ? 0 : (ds.y[row] as number);

    forward(net, x, scratch);
    const loss = backward(net, x, y, scratch, grads);
    total += loss;
    if (loss < min) min = loss;
    if (loss > max) max = loss;
  }

  // Divide once, at the end, rather than scaling every sample's contribution as it arrives.
  scaleGrads(grads, 1 / size);
  sgdStep(net, grads, config.learningRate);
  t.step++;

  // Checked every step because challenge 3 is *about* reaching this state, and a diverged
  // network must stop rather than keep multiplying NaN through a chart.
  if (!t.diverged && hasDiverged(net)) t.diverged = true;

  return {
    step: t.step,
    epoch: t.epoch,
    loss: total / size,
    lossMin: min,
    lossMax: max,
    samples: size,
  };
}

export interface EvalResult {
  readonly loss: number;
  readonly accuracy: number;
  readonly correct: number;
  readonly total: number;
}

/**
 * Loss and accuracy over a set of rows, with no weight update.
 *
 * Takes its own scratch so it cannot disturb a trainer's — evaluating between training steps
 * would otherwise overwrite the activations the backward pass is about to read, which produces
 * gradients for the wrong sample and trains *almost* correctly.
 */
export function evaluateRows(
  net: Net,
  ds: Dataset,
  rows: Int32Array,
  scratch: Scratch,
): EvalResult {
  let total = 0;
  let correct = 0;
  for (let k = 0; k < rows.length; k++) {
    const row = rows[k] as number;
    const y = ds.y === null ? 0 : (ds.y[row] as number);
    const out = forward(net, sample(ds, row), scratch);
    total += sampleLoss(net.loss, out, y);
    if (argmax(out) === y) correct++;
  }
  const n = Math.max(1, rows.length);
  return { loss: total / n, accuracy: correct / n, correct, total: rows.length };
}
