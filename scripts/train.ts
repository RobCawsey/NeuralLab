/**
 * Headless training. Prints the golden numbers, and proves challenge 1 on the way past.
 *
 *   node --experimental-strip-types scripts/train.ts
 *
 * Two runs, and the second is the interesting one:
 *
 *  1. **The golden run** — the protocol §2 of the design document pins. Its final loss is
 *     asserted here as well as in `golden.test.ts`, so that drift is caught by whichever of the
 *     two happens to be run first.
 *  2. **XOR with and without a hidden layer** — challenge 1, measured rather than asserted.
 *     A network with no hidden layer cannot beat chance on XOR; one hidden layer solves it.
 */

import { Rng, fitStandardiser, split, standardise } from '../packages/core/src/index.ts';
import { moons, xor } from '../packages/data/src/generators.ts';
import { createNet, createScratch, initialise } from '../packages/mlp/src/net.ts';
import { createTrainer, evaluateRows, trainStep } from '../packages/mlp/src/train.ts';
import type { Dataset } from '../packages/core/src/index.ts';

/* ---------------- the golden protocol ---------------- */

const GOLDEN = {
  seed: 4417,
  weightSeed: 1,
  n: 240,
  noise: 0.15,
  trainFraction: 0.7,
  hidden: [8, 8],
  learningRate: 0.1,
  batchSize: 16,
  steps: 400,
} as const;

interface RunResult {
  readonly trainLoss: number;
  readonly trainAccuracy: number;
  readonly valLoss: number;
  readonly valAccuracy: number;
  readonly checksum: number;
  readonly steps: number;
  readonly epochs: number;
}

function run(
  data: Dataset,
  hidden: number[],
  steps: number,
  learningRate: number,
  batchSize: number,
  seedSplit: number,
  weightSeed: number,
): RunResult {
  const parts = split(data, GOLDEN.trainFraction, new Rng(seedSplit ^ 0x5f3759df));
  const z = standardise(data, fitStandardiser(data, parts.train));

  const net = createNet({
    shape: [z.dim, ...hidden, Math.max(2, z.classes)],
    hidden: 'tanh',
    output: 'softmax',
    loss: 'crossEntropy',
  });
  initialise(net, 'glorot', new Rng(weightSeed));

  const trainer = createTrainer(
    net,
    parts.train,
    { learningRate, batchSize, optimiser: 'sgd' },
    new Rng(weightSeed),
  );
  for (let i = 0; i < steps; i++) trainStep(trainer, z);

  const scratch = createScratch(net);
  const tr = evaluateRows(net, z, parts.train, scratch);
  const va = evaluateRows(net, z, parts.val, scratch);

  /*
   * A checksum over every weight, as a single integer.
   *
   * The loss alone is a weak guard: two different weight vectors can reach the same loss to four
   * decimals, so a refactor that changed the *order* of the updates could leave it untouched.
   * Summing the raw float32 bits catches any change to any weight at all.
   */
  let checksum = 0;
  for (const layer of net.layers) {
    for (const view of [layer.W, layer.b]) {
      const bits = new Uint32Array(view.buffer, view.byteOffset, view.length);
      for (let i = 0; i < bits.length; i++) checksum = (checksum ^ (bits[i] as number)) >>> 0;
    }
  }

  return {
    trainLoss: tr.loss,
    trainAccuracy: tr.accuracy,
    valLoss: va.loss,
    valAccuracy: va.accuracy,
    checksum,
    steps: trainer.step,
    epochs: trainer.epoch,
  };
}

/* ---------------- 1. the golden run ---------------- */

const golden = run(
  moons({ n: GOLDEN.n, noise: GOLDEN.noise, seed: GOLDEN.seed }),
  [...GOLDEN.hidden],
  GOLDEN.steps,
  GOLDEN.learningRate,
  GOLDEN.batchSize,
  GOLDEN.seed,
  GOLDEN.weightSeed,
);

console.log('golden run — two moons, 2-8-8-2 tanh, SGD 0.1, batch 16, 400 steps, seed 4417');
console.log(`  train loss      ${golden.trainLoss.toFixed(4)}`);
console.log(`  train accuracy  ${golden.trainAccuracy.toFixed(4)}`);
console.log(`  val loss        ${golden.valLoss.toFixed(4)}`);
console.log(`  val accuracy    ${golden.valAccuracy.toFixed(4)}`);
console.log(`  weight checksum ${golden.checksum}`);
console.log(`  ${golden.steps} steps over ${golden.epochs} epochs`);

/* ---------------- 2. challenge 1, measured ---------------- */

const xorData = xor({ n: 240, noise: 0.15, seed: GOLDEN.seed });
const deep = run(xorData, [8], 1200, 0.1, 16, GOLDEN.seed, 1);

/*
 * The flat network is run across several splits, because its accuracy is strongly
 * seed-dependent — 0.29 to 0.77 — while its *ceiling* is not. A line can isolate one quadrant
 * of XOR and answer "the other class" everywhere else, which is right three times out of four.
 * One number here would look like luck; the spread is the point.
 */
const flatRuns = [240, 4417, 1, 2, 3, 7, 99].map((s) => run(xorData, [], 1200, 0.1, 16, s, 1));
const flatBest = Math.max(...flatRuns.map((r) => r.trainAccuracy));

console.log('\nchallenge 1 — XOR, 1200 steps each');
console.log(`  no hidden layer   accuracy ${flatRuns.map((r) => r.trainAccuracy.toFixed(2)).join(' ')}`);
console.log(`                    best ${flatBest.toFixed(4)} — the linear ceiling is 3 quadrants of 4`);
console.log(`  one hidden layer  accuracy ${deep.trainAccuracy.toFixed(4)}  loss ${deep.trainLoss.toFixed(4)}`);

/* ---------------- 3. challenge 3, measured ---------------- */

console.log('\nchallenge 3 — two moons, 2-8-8-2, 400 steps, by learning rate');
for (const lr of [0.1, 3, 10, 500]) {
  const r = run(moons({ n: GOLDEN.n, noise: GOLDEN.noise, seed: 7 }), [8, 8], 400, lr, 16, 7, 1);
  console.log(
    `  lr ${String(lr).padEnd(5)} loss ${r.trainLoss.toFixed(4).padEnd(9)} accuracy ${r.trainAccuracy.toFixed(3)}`,
  );
}
console.log('  nothing here becomes NaN — softmax shifts by its maximum, so a destroyed');
console.log('  network still reports a readable accuracy instead of a blank panel.');

/* ---------------- assertions ---------------- */

let failed = false;
function check(label: string, ok: boolean, detail: string): void {
  if (!ok) {
    console.error(`FAIL — ${label}: ${detail}`);
    failed = true;
  }
}

check('the golden run learns', golden.trainAccuracy > 0.9,
  `train accuracy ${golden.trainAccuracy.toFixed(4)}`);
check('replay is exact', (() => {
  const again = run(
    moons({ n: GOLDEN.n, noise: GOLDEN.noise, seed: GOLDEN.seed }),
    [...GOLDEN.hidden], GOLDEN.steps, GOLDEN.learningRate, GOLDEN.batchSize,
    GOLDEN.seed, GOLDEN.weightSeed,
  );
  return again.checksum === golden.checksum && again.trainLoss === golden.trainLoss;
})(), 'the same protocol produced different weights');
check('a flat network never beats the linear ceiling on XOR', flatBest < 0.8,
  `best accuracy ${flatBest.toFixed(4)} — challenge 1 has stopped failing`);
check('one hidden layer solves XOR', deep.trainAccuracy > 0.95,
  `accuracy ${deep.trainAccuracy.toFixed(4)} — challenge 1 has stopped succeeding`);

console.log(failed ? '\nFAILED' : '\nall checks passed');
if (failed) process.exit(1);
