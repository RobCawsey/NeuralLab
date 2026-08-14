/**
 * Headless: build the default dataset, print what it is, and assert it replays.
 *
 * This exists because invariant 4 makes it possible, and because checking a generator without
 * a browser is much faster than checking it with one. It is also the first and smallest form of
 * the golden test — the same seed must produce the same bytes, or nothing later in this project
 * can be pinned to a number.
 *
 *   node --experimental-strip-types scripts/data.ts
 */

import { Rng, bounds2d, classCounts, fitStandardiser, split } from '../packages/core/src/index.ts';
import { moons } from '../packages/data/src/generators.ts';

const SEED = 4417;

const a = moons({ n: 240, noise: 0.15, seed: SEED });
const b = moons({ n: 240, noise: 0.15, seed: SEED });

let identical = a.x.length === b.x.length;
for (let i = 0; identical && i < a.x.length; i++) identical = a.x[i] === b.x[i];
if (!identical) {
  console.error('FAIL — the same seed produced different data.');
  process.exit(1);
}

const parts = split(a, 0.7, new Rng(SEED ^ 0x5f3759df));
const std = fitStandardiser(a, parts.train);
const box = bounds2d(a);
const counts = classCounts(a);

console.log(`${a.name} — ${a.n} samples, dim ${a.dim}, ${a.classes} classes, seed ${SEED}`);
console.log(`  x  ${box.minX.toFixed(4)} … ${box.maxX.toFixed(4)}`);
console.log(`  y  ${box.minY.toFixed(4)} … ${box.maxY.toFixed(4)}`);
console.log(`  classes  ${Array.from(counts).join(' / ')}`);
console.log(`  split    ${parts.train.length} train / ${parts.val.length} validation`);
console.log(
  `  standardiser (train only)  mean [${Array.from(std.mean).map((v) => v.toFixed(4)).join(', ')}]` +
    `  sd [${Array.from(std.sd).map((v) => v.toFixed(4)).join(', ')}]`,
);

// A 48 × 20 ASCII scatter. Crude on purpose: the point is to confirm the shape is two
// interleaved arcs without opening a browser, and two characters do that.
const W = 48;
const H = 20;
const grid: string[][] = Array.from({ length: H }, () => Array.from({ length: W }, () => ' '));
for (let i = 0; i < a.n; i++) {
  const px = Math.round(((a.x[i * 2] as number) - box.minX) / (box.maxX - box.minX) * (W - 1));
  const py = Math.round((1 - ((a.x[i * 2 + 1] as number) - box.minY) / (box.maxY - box.minY)) * (H - 1));
  const row = grid[py];
  if (row) row[px] = (a.y?.[i] ?? 0) === 0 ? 'o' : '+';
}
console.log('');
for (const row of grid) console.log('  ' + row.join(''));
console.log('');
console.log('  o upper   + lower');
console.log('');
console.log(`replay: identical for seed ${SEED}`);

// The Rng's own golden vector, printed so that pinning it in a test is a copy rather than a
// guess. If this line ever changes, every stored run in the project has been invalidated.
const rng = new Rng(SEED);
const vector = Array.from({ length: 6 }, () => rng.u32());
console.log(`rng u32 x6 @ ${SEED}: ${vector.join(', ')}`);
