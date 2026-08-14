/**
 * The shared builders, and the one property slice 4 depends on entirely.
 *
 * The worker is not sent the dataset. It is sent the configuration and rebuilds it from the same
 * seed, so if `buildData` were not deterministic the page would be drawing one dataset while the
 * worker trained on another — and both would look completely reasonable.
 *
 * Also here: that chunking the training loop does not change the run. The worker trains in 40 ms
 * bursts rather than a single pass, and invariant 2 says that must not matter.
 */

import { describe, expect, it } from 'vitest';
import { Rng } from '@neurallab/core';
import {
  applyWeights,
  createScratch,
  createTrainer,
  evaluateRows,
  flattenWeights,
  trainStep,
  weightCount,
} from '@neurallab/mlp';
import { buildData, buildNet, type DataConfig, type NetConfig } from '../src/run/build.ts';

const DATA: DataConfig = {
  dataset: 'moons',
  n: 240,
  noise: 0.15,
  seed: 4417,
  trainFraction: 0.7,
};

const NET: NetConfig = {
  hidden: [8, 8],
  hiddenAct: 'tanh',
  init: 'glorot',
  weightSeed: 1,
};

describe('buildData', () => {
  it('produces byte-identical results from the same configuration', () => {
    /*
     * The load-bearing property of slice 4. Two threads call this with the same config and must
     * agree on every sample, every split index and every standardiser coefficient.
     */
    const a = buildData(DATA);
    const b = buildData(DATA);
    expect(Array.from(a.z.x)).toEqual(Array.from(b.z.x));
    expect(Array.from(a.parts.train)).toEqual(Array.from(b.parts.train));
    expect(Array.from(a.parts.val)).toEqual(Array.from(b.parts.val));
    expect(Array.from(a.standardiser.mean)).toEqual(Array.from(b.standardiser.mean));
    expect(Array.from(a.standardiser.sd)).toEqual(Array.from(b.standardiser.sd));
  });

  it('marks exactly the validation rows', () => {
    const built = buildData(DATA);
    let flagged = 0;
    for (const v of built.isVal) if (v === 1) flagged++;
    expect(flagged).toBe(built.parts.val.length);
    for (const row of built.parts.val) expect(built.isVal[row]).toBe(1);
    for (const row of built.parts.train) expect(built.isVal[row]).toBe(0);
  });

  it('changes when any part of the configuration changes', () => {
    const base = buildData(DATA);
    for (const changed of [
      { ...DATA, seed: 4418 },
      { ...DATA, noise: 0.2 },
      { ...DATA, n: 260 },
      { ...DATA, dataset: 'circles' as const },
    ]) {
      expect(Array.from(buildData(changed).z.x)).not.toEqual(Array.from(base.z.x));
    }
  });

  it('takes the split fraction from the configuration', () => {
    expect(buildData({ ...DATA, trainFraction: 0.9 }).parts.train.length).toBeGreaterThan(
      buildData({ ...DATA, trainFraction: 0.5 }).parts.train.length,
    );
  });
});

describe('buildNet', () => {
  it('takes its input and output widths from the data, not the config', () => {
    // A network whose output count disagrees with the class count is not a configuration a
    // reader should be able to reach by dragging a slider.
    const net = buildNet(NET, 2, 3);
    expect(net.layers[0]!.inputs).toBe(2);
    expect(net.layers[net.layers.length - 1]!.units).toBe(3);
  });

  it('gives an unlabelled set at least two outputs', () => {
    expect(buildNet(NET, 2, 0).layers[2]!.units).toBe(2);
  });

  it('replays from the same weight seed', () => {
    expect(Array.from(flattenWeights(buildNet(NET, 2, 2)))).toEqual(
      Array.from(flattenWeights(buildNet(NET, 2, 2))),
    );
    expect(Array.from(flattenWeights(buildNet({ ...NET, weightSeed: 2 }, 2, 2)))).not.toEqual(
      Array.from(flattenWeights(buildNet(NET, 2, 2))),
    );
  });
});

describe('weight serialisation', () => {
  it('round-trips a whole network through one buffer', () => {
    // Invariant 3's promise, and the thing crossing the worker boundary 25 times a second.
    const source = buildNet(NET, 2, 2);
    const target = buildNet({ ...NET, weightSeed: 99 }, 2, 2);
    expect(Array.from(flattenWeights(target))).not.toEqual(Array.from(flattenWeights(source)));

    applyWeights(target, flattenWeights(source));
    expect(Array.from(flattenWeights(target))).toEqual(Array.from(flattenWeights(source)));
  });

  it('writes into a supplied buffer without allocating', () => {
    const net = buildNet(NET, 2, 2);
    const buffer = new Float32Array(weightCount(net));
    expect(flattenWeights(net, buffer)).toBe(buffer);
  });

  it('refuses a buffer of the wrong length', () => {
    /*
     * A mismatch means the two sides disagree about the architecture. Filling what fits would
     * leave a network answering as a mixture of two models — which still runs, and is not a
     * failure anybody would trace back to a message boundary.
     */
    const net = buildNet(NET, 2, 2);
    expect(() => applyWeights(net, new Float32Array(3))).toThrow(/expected/);
  });
});

describe('chunked training', () => {
  it('reaches the golden run however the steps are grouped', () => {
    /*
     * Invariant 2, restated for the worker. It trains in 40 ms bursts rather than one pass, and
     * the run has to be identical — otherwise the golden number pinned in slice 2 describes
     * something the app no longer does.
     *
     * 0.1007 / 0.9702 / 38 epochs, the same figures `golden.test.ts` pins and `npm run train`
     * prints.
     */
    const built = buildData(DATA);

    const runInChunks = (chunks: number[]): { loss: number; accuracy: number; epoch: number } => {
      const net = buildNet(NET, built.data.dim, built.data.classes);
      const trainer = createTrainer(
        net,
        built.parts.train,
        { learningRate: 0.1, batchSize: 16 },
        new Rng(NET.weightSeed),
      );
      for (const size of chunks) for (let i = 0; i < size; i++) trainStep(trainer, built.z);
      const result = evaluateRows(net, built.z, built.parts.train, createScratch(net));
      return { loss: result.loss, accuracy: result.accuracy, epoch: trainer.epoch };
    };

    const oneGo = runInChunks([400]);
    expect(oneGo.loss).toBeCloseTo(0.1007, 4);
    expect(oneGo.accuracy).toBeCloseTo(0.9702, 4);
    expect(oneGo.epoch).toBe(38);

    // Uneven chunks, as a worker's 40 ms bursts would be.
    const uneven = runInChunks([7, 113, 1, 96, 183]);
    expect(uneven).toEqual(oneGo);

    // And one step at a time, which is what the Step button does.
    const singles = runInChunks(new Array<number>(400).fill(1));
    expect(singles).toEqual(oneGo);
  });
});
