/**
 * The one shape every dataset in the project takes, and the operations both networks need
 * from it.
 *
 * `mlp` and `som` never import each other; this type and `Rng` are the entire overlap between
 * them. That is deliberate — the moment something is shared because both happen to want it
 * rather than because it is genuinely one idea, the two kernels start growing together.
 */

import { Rng } from './rng.ts';

export interface Dataset {
  readonly name: string;
  /** `n * dim`, row-major: sample `i`, feature `k` is `x[i * dim + k]`. */
  readonly x: Float32Array;
  /**
   * Class index per sample, or null when the set is unlabelled.
   *
   * Nullable rather than absent because a self-organising map is trained on data it has no
   * labels for, but very often the labels *exist* and are held back so they can be drawn over
   * the finished map. "Unlabelled" is a fact about training, not about the file.
   */
  readonly y: Int32Array | null;
  readonly n: number;
  readonly dim: number;
  /** Number of distinct classes, or 0 when unlabelled. */
  readonly classes: number;
  /** One name per feature, for axis labels and component planes. */
  readonly featureNames: readonly string[];
  /** One name per class, for legends and the confusion matrix. Empty when unlabelled. */
  readonly classNames: readonly string[];
}

/** Read sample `i` into a subarray. A view, not a copy — do not retain it across a shuffle. */
export function sample(ds: Dataset, i: number): Float32Array {
  return ds.x.subarray(i * ds.dim, i * ds.dim + ds.dim);
}

/**
 * A train/validation split, as two index arrays.
 *
 * Indices rather than two new Datasets, because the alternative copies the feature buffer and
 * then every panel has to know which of the two copies it is holding. One buffer, two orderings
 * into it, and `sample(ds, idx[k])` reads from the only copy there is.
 *
 * Stratified when the set is labelled: shuffling 240 points and taking the first 70% can hand
 * a two-class problem a validation split that is 80% one class, and the resulting accuracy
 * number is then noise. Challenge 8 asks a reader to trust that curve, so it has to be sound.
 */
export interface Split {
  readonly train: Int32Array;
  readonly val: Int32Array;
}

export function split(ds: Dataset, trainFraction: number, rng: Rng): Split {
  if (!(trainFraction > 0 && trainFraction < 1)) {
    throw new Error(`trainFraction must be strictly between 0 and 1, got ${trainFraction}`);
  }

  const train: number[] = [];
  const val: number[] = [];

  // One bucket per class, or a single bucket holding everything when unlabelled.
  const buckets: number[][] = [];
  const bucketOf = (i: number): number => (ds.y === null ? 0 : (ds.y[i] as number));
  for (let i = 0; i < ds.n; i++) {
    const b = bucketOf(i);
    (buckets[b] ??= []).push(i);
  }

  for (const bucket of buckets) {
    if (bucket === undefined) continue;
    const idx = Int32Array.from(bucket);
    rng.shuffle(idx);
    // `round`, so a 4-sample class at 0.7 contributes 3 and not 2. Every class must reach both
    // sides: a class absent from validation makes its recall undefined rather than zero.
    const take = Math.min(idx.length - 1, Math.max(1, Math.round(idx.length * trainFraction)));
    for (let k = 0; k < idx.length; k++) {
      (k < take ? train : val).push(idx[k] as number);
    }
  }

  return { train: Int32Array.from(train), val: Int32Array.from(val) };
}

/** Per-feature mean and standard deviation, measured over a subset of rows. */
export interface Standardiser {
  readonly mean: Float32Array;
  readonly sd: Float32Array;
}

/**
 * Fit a standardiser over the given rows only.
 *
 * Taking the rows is not a convenience — it is the whole correctness argument. Fitting over the
 * full set and then splitting leaks the validation set's mean into the training set, and the
 * validation curve that challenge 8 is read from becomes quietly optimistic. Fit on train,
 * apply to both.
 */
export function fitStandardiser(ds: Dataset, rows: Int32Array): Standardiser {
  const { dim } = ds;
  const mean = new Float32Array(dim);
  const sd = new Float32Array(dim);
  if (rows.length === 0) {
    sd.fill(1);
    return { mean, sd };
  }

  for (let r = 0; r < rows.length; r++) {
    const base = (rows[r] as number) * dim;
    for (let k = 0; k < dim; k++) mean[k] = (mean[k] as number) + (ds.x[base + k] as number);
  }
  for (let k = 0; k < dim; k++) mean[k] = (mean[k] as number) / rows.length;

  for (let r = 0; r < rows.length; r++) {
    const base = (rows[r] as number) * dim;
    for (let k = 0; k < dim; k++) {
      const d = (ds.x[base + k] as number) - (mean[k] as number);
      sd[k] = (sd[k] as number) + d * d;
    }
  }
  for (let k = 0; k < dim; k++) {
    const v = Math.sqrt((sd[k] as number) / rows.length);
    // A constant feature has zero spread. Dividing by it produces NaN, which then spreads
    // through every weight in one forward pass and is very hard to trace back here.
    sd[k] = v > 1e-8 ? v : 1;
  }
  return { mean, sd };
}

/** Apply a standardiser to every row, returning a new Dataset. The original is untouched. */
export function standardise(ds: Dataset, s: Standardiser): Dataset {
  const x = new Float32Array(ds.x.length);
  for (let i = 0; i < ds.n; i++) {
    const base = i * ds.dim;
    for (let k = 0; k < ds.dim; k++) {
      x[base + k] = ((ds.x[base + k] as number) - (s.mean[k] as number)) / (s.sd[k] as number);
    }
  }
  return { ...ds, x };
}

/** Axis-aligned bounds of two chosen features, for a scatter plot or a decision field. */
export function bounds2d(
  ds: Dataset,
  fx = 0,
  fy = 1,
): { minX: number; maxX: number; minY: number; maxY: number } {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < ds.n; i++) {
    const a = ds.x[i * ds.dim + fx] as number;
    const b = ds.x[i * ds.dim + fy] as number;
    if (a < minX) minX = a;
    if (a > maxX) maxX = a;
    if (b < minY) minY = b;
    if (b > maxY) maxY = b;
  }
  // An empty set, or one with a degenerate axis, still has to produce a drawable box.
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  if (maxX - minX < 1e-6) {
    minX -= 0.5;
    maxX += 0.5;
  }
  if (maxY - minY < 1e-6) {
    minY -= 0.5;
    maxY += 0.5;
  }
  return { minX, maxX, minY, maxY };
}

/** How many samples fall in each class. Used by the split test and the dataset readout. */
export function classCounts(ds: Dataset): Int32Array {
  const counts = new Int32Array(Math.max(1, ds.classes));
  if (ds.y === null) return counts;
  for (let i = 0; i < ds.n; i++) {
    const c = ds.y[i] as number;
    counts[c] = (counts[c] as number) + 1;
  }
  return counts;
}
