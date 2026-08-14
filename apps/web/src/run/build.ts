/**
 * Building the data and the network from a configuration, in one place.
 *
 * From slice 4 there are **two** things that need to do this: the main thread, which draws the
 * scatter and the field, and the worker, which trains. They must agree exactly — a split that
 * differed by one row, or a standardiser fitted over slightly different data, would make every
 * number on screen quietly describe a different run from the one happening.
 *
 * So the worker is not sent the dataset. It is sent the *configuration*, and rebuilds it from
 * the same seed by calling the same function. That costs a few milliseconds at init and saves
 * cloning a buffer that both sides can derive; more importantly, it makes "they agree" a
 * property of there being one implementation rather than a thing to keep checking.
 *
 * This file imports nothing browser-specific, so both callers can use it and so can a test.
 */

import {
  Rng,
  fitStandardiser,
  split,
  standardise,
  type Dataset,
  type Split,
  type Standardiser,
} from '@neurallab/core';
import { GENERATORS, type GeneratorKey } from '@neurallab/data';
import {
  createNet,
  initialise,
  type Activation,
  type InitScheme,
  type Net,
} from '@neurallab/mlp';

export interface DataConfig {
  readonly dataset: GeneratorKey;
  readonly n: number;
  readonly noise: number;
  readonly seed: number;
  readonly trainFraction: number;
}

export interface NetConfig {
  readonly hidden: readonly number[];
  readonly hiddenAct: Activation;
  readonly init: InitScheme;
  readonly weightSeed: number;
}

export interface BuiltData {
  readonly data: Dataset;
  readonly parts: Split;
  readonly standardiser: Standardiser;
  /** The standardised copy — the only one the network ever sees. */
  readonly z: Dataset;
  /** 1 where the row is a validation sample. */
  readonly isVal: Uint8Array;
}

export function buildData(config: DataConfig): BuiltData {
  const data = GENERATORS[config.dataset].build({
    n: config.n,
    noise: config.noise,
    seed: config.seed,
  });

  /*
   * The split gets its own Rng, seeded from the same number.
   *
   * Sharing one generator with the dataset would make the split depend on how many draws the
   * generator happened to take, so moving the sample-count slider would silently reshuffle the
   * split too — two things moving when the reader moved one.
   */
  const parts = split(data, config.trainFraction, new Rng(config.seed ^ 0x5f3759df));
  const standardiser = fitStandardiser(data, parts.train);

  const isVal = new Uint8Array(data.n);
  for (let k = 0; k < parts.val.length; k++) isVal[parts.val[k] as number] = 1;

  return { data, parts, standardiser, z: standardise(data, standardiser), isVal };
}

/**
 * Input and output widths come from the data, not from a control.
 *
 * A network whose output count disagrees with the number of classes is not a configuration a
 * reader should be able to reach by dragging a slider.
 */
export function buildNet(config: NetConfig, dim: number, classes: number): Net {
  const net = createNet({
    shape: [dim, ...config.hidden, Math.max(2, classes)],
    hidden: config.hiddenAct,
    output: 'softmax',
    loss: 'crossEntropy',
  });
  initialise(net, config.init, new Rng(config.weightSeed));
  return net;
}
