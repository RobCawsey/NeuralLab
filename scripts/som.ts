/**
 * Headless: a 12×12 hex map ordering itself on the colour cube — slice 9's own version of
 * `scripts/train.ts`. Two grids are printed as real terminal colour, before and after training,
 * so the claim "the map orders itself" is something a reader can see without opening a browser,
 * the same reason `scripts/data.ts` prints an ASCII scatter.
 *
 *   node --experimental-strip-types scripts/som.ts
 *
 * **Quantisation error does not fall below its random-init reading here, and that is measured,
 * not a bug.** `createSom` draws weights uniform in [0, 1) specifically to match the colour
 * cube's own range, so 144 fresh random nodes are already 144 points drawn from the *exact*
 * distribution the data is — an unusually strong quantiser with no structure behind it at all
 * (topographic error at step 0 is 0.97: essentially every sample's best and second-best nodes are
 * unrelated). Training pulls the lattice into a coherent 2D sheet, which costs some of that raw
 * quantising power in exchange for the property an SOM actually promises. So this script tracks
 * quantisation error from shortly *after* the initial reorganisation settles, not from the random
 * start: that is where "goes down as the map fits the data" is the true, monotonic story, and the
 * checks below assert it there.
 */

import { Rng } from '../packages/core/src/index.ts';
import { colourCube } from '../packages/data/src/som.ts';
import {
  createSom,
  createSomTrainer,
  somStep,
  quantisationError,
  topographicError,
  type Schedule,
  type Som,
} from '../packages/som/src/index.ts';

const GOLDEN = {
  seed: 4417,
  weightSeed: 1,
  drawSeed: 2,
  n: 1500,
  cols: 12,
  rows: 12,
  steps: 3000,
  alpha0: 0.5,
  sigma0: 6,
  decay: 'exponential' as const,
};

const data = colourCube({ n: GOLDEN.n, seed: GOLDEN.seed });
const allRows = Int32Array.from({ length: data.n }, (_, i) => i);

const som = createSom(GOLDEN.cols, GOLDEN.rows, data.dim, 'hex', new Rng(GOLDEN.weightSeed));
const schedule: Schedule = {
  alpha0: GOLDEN.alpha0,
  sigma0: GOLDEN.sigma0,
  decay: GOLDEN.decay,
  steps: GOLDEN.steps,
};

/** A colour swatch per node, two characters wide, using 24-bit ANSI background colour. */
function printMap(som: Som, label: string): void {
  console.log(`  ${label}`);
  for (let row = 0; row < som.rows; row++) {
    let line = row % 2 === 1 ? ' ' : ''; // odd-r: nudge odd rows right, matching the lattice.
    for (let col = 0; col < som.cols; col++) {
      const base = (row * som.cols + col) * som.dim;
      const r = Math.round(Math.min(1, Math.max(0, som.W[base] as number)) * 255);
      const g = Math.round(Math.min(1, Math.max(0, som.W[base + 1] as number)) * 255);
      const b = Math.round(Math.min(1, Math.max(0, som.W[base + 2] as number)) * 255);
      line += `\x1b[48;2;${r};${g};${b}m  \x1b[0m`;
    }
    console.log('  ' + line);
  }
}

console.log('SOM — colour cube, 12×12 hex, 1 500 samples, 3 000 steps, seed 4417\n');
printMap(som, 'before training — random weights');
const qeRandom = quantisationError(som, data, allRows);
const teRandom = topographicError(som, data, allRows);
console.log(`\n  quantisation error, random weights  ${qeRandom.toFixed(4)}`);
console.log(`  topographic error, random weights   ${teRandom.toFixed(4)}  (no structure at all)`);

const trainer = createSomTrainer(som, allRows, schedule, new Rng(GOLDEN.drawSeed));
const earlyCheckpoint = Math.round(GOLDEN.steps * 0.1);
for (let i = 0; i < earlyCheckpoint; i++) somStep(trainer, data);
const qeEarly = quantisationError(som, data, allRows);
for (let i = earlyCheckpoint; i < GOLDEN.steps; i++) somStep(trainer, data);

console.log('');
printMap(som, 'after training — the map has organised itself');
const qeAfter = quantisationError(som, data, allRows);
const teAfter = topographicError(som, data, allRows);
console.log(`\n  quantisation error, step ${earlyCheckpoint}         ${qeEarly.toFixed(4)}`);
console.log(`  quantisation error, step ${GOLDEN.steps}         ${qeAfter.toFixed(4)}`);
console.log(`  topographic error, step ${GOLDEN.steps}          ${teAfter.toFixed(4)}`);

let checksum = 0;
{
  const bits = new Uint32Array(som.W.buffer, som.W.byteOffset, som.W.length);
  for (let i = 0; i < bits.length; i++) checksum = (checksum ^ (bits[i] as number)) >>> 0;
}
console.log(`  weight checksum                     ${checksum}`);

/* ---------------- assertions ---------------- */

let failed = false;
function check(label: string, ok: boolean, detail: string): void {
  if (!ok) {
    console.error(`FAIL — ${label}: ${detail}`);
    failed = true;
  }
}

check(
  'quantisation error falls once the lattice has finished its initial reorganisation',
  qeAfter < qeEarly * 0.5,
  `qe at step ${earlyCheckpoint} was ${qeEarly.toFixed(4)}, at step ${GOLDEN.steps} it is ${qeAfter.toFixed(4)}`,
);
check('the trained map stays topographically coherent', teAfter < 0.15, `te ${teAfter.toFixed(4)}`);
check(
  'the random baseline really is topologically incoherent, not secretly a good map',
  teRandom > 0.8,
  `te at random init ${teRandom.toFixed(4)} — expected near-total incoherence`,
);
check('replay is exact', (() => {
  const som2 = createSom(GOLDEN.cols, GOLDEN.rows, data.dim, 'hex', new Rng(GOLDEN.weightSeed));
  const trainer2 = createSomTrainer(som2, allRows, schedule, new Rng(GOLDEN.drawSeed));
  for (let i = 0; i < GOLDEN.steps; i++) somStep(trainer2, data);
  let checksum2 = 0;
  const bits = new Uint32Array(som2.W.buffer, som2.W.byteOffset, som2.W.length);
  for (let i = 0; i < bits.length; i++) checksum2 = (checksum2 ^ (bits[i] as number)) >>> 0;
  return checksum2 === checksum;
})(), 'the same protocol produced different weights');

console.log(failed ? '\nFAILED' : '\nall checks passed');
if (failed) process.exit(1);
