import { describe, expect, it } from 'vitest';
import { Rng, fitStandardiser, split, standardise, type Dataset } from '@neurallab/core';
import { moons, xor } from '@neurallab/data';
import { createNet, createScratch, initialise, type Net } from '../src/net.ts';
import { createTrainer, evaluateRows, trainStep } from '../src/train.ts';
import { hasDiverged } from '../src/backward.ts';

function prepared(data: Dataset, fraction = 0.7) {
  const parts = split(data, fraction, new Rng(data.n ^ 0x5f3759df));
  return { z: standardise(data, fitStandardiser(data, parts.train)), parts };
}

function build(z: Dataset, hidden: number[], weightSeed = 1): Net {
  const net = createNet({
    shape: [z.dim, ...hidden, Math.max(2, z.classes)],
    hidden: 'tanh',
    output: 'softmax',
    loss: 'crossEntropy',
  });
  initialise(net, 'glorot', new Rng(weightSeed));
  return net;
}

function train(z: Dataset, rows: Int32Array, net: Net, steps: number, lr = 0.1, batch = 16) {
  const trainer = createTrainer(net, rows, { learningRate: lr, batchSize: batch }, new Rng(1));
  let last = trainStep(trainer, z);
  for (let i = 1; i < steps; i++) last = trainStep(trainer, z);
  return { trainer, last };
}

describe('trainStep', () => {
  it('drives the loss down on a learnable problem', () => {
    const { z, parts } = prepared(moons({ n: 240, seed: 1 }));
    const net = build(z, [8, 8]);
    const scratch = createScratch(net);

    const before = evaluateRows(net, z, parts.train, scratch).loss;
    train(z, parts.train, net, 400);
    const after = evaluateRows(net, z, parts.train, scratch).loss;

    expect(after).toBeLessThan(before * 0.5);
  });

  it('counts steps and epochs consistently', () => {
    const { z, parts } = prepared(moons({ n: 240, seed: 2 }));
    const net = build(z, [4]);
    // 168 training rows, batch 16 → 10.5 batches per epoch.
    const { trainer } = train(z, parts.train, net, 100);
    expect(trainer.step).toBe(100);
    expect(trainer.epoch).toBe(Math.floor((100 * 16) / parts.train.length));
  });

  it('reshuffles at the epoch boundary, not at a step count', () => {
    // Reshuffling every N steps would make the order depend on the batch size, so changing the
    // batch size would silently change which samples are seen together.
    const { z, parts } = prepared(moons({ n: 240, seed: 3 }));
    const net = build(z, [4]);
    const trainer = createTrainer(net, parts.train, { learningRate: 0.1, batchSize: 16 }, new Rng(1));
    const first = Array.from(trainer.order);

    while (trainer.epoch === 0) trainStep(trainer, z);
    const second = Array.from(trainer.order);

    expect(second).not.toEqual(first);
    expect([...second].sort((a, b) => a - b)).toEqual([...first].sort((a, b) => a - b));
  });

  it('gives an identical result however the steps are grouped', () => {
    /*
     * Invariant 2, asserted. The driving loop may run one step or five hundred per frame, and
     * the answer has to be the same — otherwise the golden test means nothing on a machine with
     * a different frame rate.
     */
    const { z, parts } = prepared(moons({ n: 240, seed: 4 }));

    const a = build(z, [6, 6]);
    const ta = createTrainer(a, parts.train, { learningRate: 0.1, batchSize: 16 }, new Rng(1));
    for (let i = 0; i < 120; i++) trainStep(ta, z);

    const b = build(z, [6, 6]);
    const tb = createTrainer(b, parts.train, { learningRate: 0.1, batchSize: 16 }, new Rng(1));
    for (let block = 0; block < 4; block++) for (let i = 0; i < 30; i++) trainStep(tb, z);

    expect(Array.from(a.layers[0]!.W)).toEqual(Array.from(b.layers[0]!.W));
  });

  it('reports the batch spread, not just the mean', () => {
    // The chart draws a band because minibatch loss is very noisy; a sampled line would invent
    // smoothness the run did not have.
    const { z, parts } = prepared(moons({ n: 240, seed: 5 }));
    const net = build(z, [8]);
    const { last } = train(z, parts.train, net, 20);
    expect(last.lossMin).toBeLessThanOrEqual(last.loss);
    expect(last.lossMax).toBeGreaterThanOrEqual(last.loss);
    expect(last.lossMax).toBeGreaterThan(last.lossMin);
  });

  it('handles a batch larger than the dataset', () => {
    const { z, parts } = prepared(moons({ n: 40, seed: 6 }));
    const net = build(z, [4]);
    const { last } = train(z, parts.train, net, 5, 0.1, 1000);
    expect(last.samples).toBe(parts.train.length);
    expect(Number.isFinite(last.loss)).toBe(true);
  });
});

describe('challenge 1 — one line is not enough', () => {
  /*
   * The lesson, measured rather than asserted. If the XOR geometry or the trainer ever drifted
   * so that a flat network started succeeding, the card would quietly stop teaching anything and
   * nothing else would say so. `npm run train` prints both numbers.
   *
   * **The linear ceiling on XOR is 75%, not 50%**, and the first version of this test asserted
   * the wrong one. A line can isolate a single quadrant — say (+,+) — and answer "the other
   * class" everywhere else, which is right for three quadrants out of four. Measured across
   * seven split seeds the flat network lands anywhere between 0.29 and 0.77 depending on which
   * quadrant its boundary happens to catch, and it never gets the fourth.
   *
   * That is a better lesson than the one originally written down. "It cannot do better than a
   * coin toss" is false and a reader can see it is false; "it gets three quarters of them and
   * can never get the last quarter, however long you train it" is true and is exactly what a
   * missing hidden layer costs.
   */
  const SPLIT_SEEDS = [240, 4417, 1, 2, 3, 7, 99];

  it('never beats the linear ceiling, whatever the split', () => {
    const data = xor({ n: 240, noise: 0.15, seed: 4417 });
    for (const seed of SPLIT_SEEDS) {
      const parts = split(data, 0.7, new Rng(seed ^ 0x5f3759df));
      const z = standardise(data, fitStandardiser(data, parts.train));
      const net = build(z, []);
      train(z, parts.train, net, 1200);
      const result = evaluateRows(net, z, parts.train, createScratch(net));
      expect(result.accuracy, `split seed ${seed}`).toBeLessThan(0.8);
    }
  });

  it('is solved completely by one hidden layer', () => {
    const { z, parts } = prepared(xor({ n: 240, noise: 0.15, seed: 4417 }));
    const net = build(z, [8]);
    train(z, parts.train, net, 1200);
    const result = evaluateRows(net, z, parts.train, createScratch(net));
    expect(result.accuracy).toBeGreaterThan(0.95);
  });
});

describe('challenge 3 — too big a step', () => {
  /*
   * **This does not produce NaN, and the design document said it would.**
   *
   * With tanh hidden units, a softmax output and cross-entropy, every factor in the gradient is
   * bounded: |a − y| ≤ 1 at the output, tanh' ≤ 1, and tanh's own output is in [−1, 1]. So the
   * weight update is bounded by the learning rate, and the weights grow linearly rather than
   * exploding. Measured over 400 steps on two moons: lr 3 reaches 0.988 accuracy (better than
   * lr 0.1), lr 10 falls to 0.744, lr 500 collapses to 0.500 with a loss of 13.8 — and max|w|
   * reaches 1.4e4 without a single non-finite number anywhere.
   *
   * So challenge 3's real outcome is *the accuracy collapses to chance and the loss stops
   * falling*, not *everything becomes NaN*. The card and its afterword say that instead.
   */
  it('is destroyed by a large learning rate, against an identical sane run', () => {
    /*
     * Stated as a comparison rather than as an absolute threshold. Where exactly a destroyed
     * network lands is seed-dependent — 0.50 on one split, 0.65 on another — but it is always
     * far worse than the same network trained sanely on the same data, and its loss is two
     * orders of magnitude higher. That gap is the claim the challenge card makes.
     */
    const { z, parts } = prepared(moons({ n: 240, seed: 7 }));

    const sane = build(z, [8, 8]);
    train(z, parts.train, sane, 400, 0.1);
    const saneResult = evaluateRows(sane, z, parts.train, createScratch(sane));

    const wild = build(z, [8, 8]);
    train(z, parts.train, wild, 400, 500);
    const wildResult = evaluateRows(wild, z, parts.train, createScratch(wild));

    expect(saneResult.accuracy).toBeGreaterThan(0.9);
    expect(wildResult.accuracy).toBeLessThan(0.75);
    expect(wildResult.loss).toBeGreaterThan(saneResult.loss * 20);
  });

  it('stays finite while it collapses, because softmax shifts by its maximum', () => {
    /*
     * The max-shift is what makes challenge 3 *legible*. At lr 5000 the weights reach ~1e5, so
     * the logits do too — and `Math.exp(1e5)` is Infinity, which would make the whole output
     * NaN and every panel blank. Shifting by the maximum turns that into a clean one-hot.
     * Diverging is the lesson; NaN everywhere would be a bug.
     */
    const { z, parts } = prepared(moons({ n: 240, seed: 7 }));
    const net = build(z, [8, 8]);
    const { trainer } = train(z, parts.train, net, 400, 5000);
    expect(hasDiverged(net)).toBe(false);
    expect(trainer.diverged).toBe(false);

    let maxW = 0;
    for (const l of net.layers) for (const w of l.W) maxW = Math.max(maxW, Math.abs(w));
    expect(maxW).toBeGreaterThan(1e3);
  });

  it('trains normally at a sane learning rate', () => {
    const { z, parts } = prepared(moons({ n: 240, seed: 8 }));
    const net = build(z, [8, 8]);
    const { trainer } = train(z, parts.train, net, 400, 0.1);
    expect(trainer.diverged).toBe(false);
    expect(evaluateRows(net, z, parts.train, createScratch(net)).accuracy).toBeGreaterThan(0.9);
  });
});

describe('evaluateRows', () => {
  it('agrees with a hand count of correct predictions', () => {
    const { z, parts } = prepared(moons({ n: 240, seed: 9 }));
    const net = build(z, [8, 8]);
    train(z, parts.train, net, 300);
    const result = evaluateRows(net, z, parts.val, createScratch(net));
    expect(result.correct / result.total).toBeCloseTo(result.accuracy, 10);
    expect(result.total).toBe(parts.val.length);
  });

  it('does not disturb a trainer mid-step', () => {
    /*
     * It takes its own scratch. Sharing the trainer's would overwrite the activations the
     * backward pass is about to read, producing gradients for the wrong sample — which trains
     * *almost* correctly and is very hard to see.
     */
    const { z, parts } = prepared(moons({ n: 240, seed: 10 }));

    const a = build(z, [6]);
    const ta = createTrainer(a, parts.train, { learningRate: 0.1, batchSize: 16 }, new Rng(1));
    for (let i = 0; i < 50; i++) trainStep(ta, z);

    const b = build(z, [6]);
    const tb = createTrainer(b, parts.train, { learningRate: 0.1, batchSize: 16 }, new Rng(1));
    const spy = createScratch(b);
    for (let i = 0; i < 50; i++) {
      trainStep(tb, z);
      evaluateRows(b, z, parts.val, spy);
    }

    expect(Array.from(b.layers[0]!.W)).toEqual(Array.from(a.layers[0]!.W));
  });
});
