/**
 * Measuring a map that nobody labelled — §3's three numbers for a network with no loss curve.
 */

import { sample, type Dataset } from '@neurallab/core';
import { NEIGHBOUR_SLOTS } from './lattice.ts';
import { bmu, bmu2, type Som } from './som.ts';

/** Euclidean distance between a node's weight vector and a sample — `sqrt`, unlike `bmu`'s search. */
function distance(som: Som, node: number, x: ArrayLike<number>): number {
  const base = node * som.dim;
  let sum = 0;
  for (let k = 0; k < som.dim; k++) {
    const d = (som.W[base + k] as number) - (x[k] as number);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Mean distance from each sample to its BMU — the SOM's answer to a loss curve, per §3. Goes
 * down as the map fits the data; unlike the MLP's loss it has no floor at zero the reader should
 * expect to reach, because the map has finitely many nodes and the data does not.
 */
export function quantisationError(som: Som, ds: Dataset, rows: Int32Array): number {
  if (rows.length === 0) return 0;
  let sum = 0;
  for (let r = 0; r < rows.length; r++) {
    const x = sample(ds, rows[r] as number);
    sum += distance(som, bmu(som, x), x);
  }
  return sum / rows.length;
}

/**
 * The fraction of samples whose best and second-best nodes are **not** lattice neighbours.
 *
 * This is the number a quantisation-error curve cannot see: QE keeps falling as long as nodes
 * keep getting closer to the data, whether or not the *map* stays coherent. A twisted lattice —
 * two folds crossed over each other — can still have every node sitting near real data and a QE
 * that looks fine, while TE climbs because a sample's two best nodes are now lattice neighbours
 * in name only, torn apart in input space. Watching QE fall while TE climbs is challenge 10.
 */
export function topographicError(som: Som, ds: Dataset, rows: Int32Array): number {
  if (rows.length === 0) return 0;
  const neighbours = som.neighbours;
  let bad = 0;
  for (let r = 0; r < rows.length; r++) {
    const x = sample(ds, rows[r] as number);
    const [best, second] = bmu2(som, x);
    let adjacent = false;
    for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
      if (neighbours[best * NEIGHBOUR_SLOTS + s] === second) {
        adjacent = true;
        break;
      }
    }
    if (!adjacent) bad++;
  }
  return bad / rows.length;
}

/**
 * One value per node: the mean distance, in weight space, to its own lattice neighbours (the
 * discrete `neighbours` table, not `latticeDistance` — see `lattice.ts`'s module comment). Drawn
 * as a heatmap it shows ridges where the data has gaps, which is cluster structure recovered
 * without a single label. A node with no neighbours at all (a 1×1 map) reads 0, not `NaN`.
 */
export function uMatrix(som: Som): Float32Array {
  const n = som.cols * som.rows;
  const out = new Float32Array(n);
  const neighbours = som.neighbours;
  for (let node = 0; node < n; node++) {
    let sum = 0;
    let count = 0;
    for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
      const other = neighbours[node * NEIGHBOUR_SLOTS + s] as number;
      if (other === -1) continue;
      sum += nodeDistance(som, node, other);
      count++;
    }
    out[node] = count > 0 ? sum / count : 0;
  }
  return out;
}

/**
 * One input dimension's weight, across every node — a component plane. §3's answer to reading a
 * map with more than three dimensions: the U-matrix shows where the data has gaps, and a plane
 * per feature shows how that one feature varies across the lattice, which together is how a
 * 13-dimensional map becomes readable without ever drawing 13 dimensions at once.
 */
export function componentPlane(som: Som, dim: number): Float32Array {
  const n = som.cols * som.rows;
  const out = new Float32Array(n);
  if (dim < 0 || dim >= som.dim) return out;
  for (let i = 0; i < n; i++) out[i] = som.W[i * som.dim + dim] as number;
  return out;
}

/**
 * A class per node — the map's own label, never trained on but recoverable afterward. Each row
 * votes for its BMU; a node's label is whichever class won it most, and a node nobody ever won
 * reads `-1` rather than `0`, so "unlabelled" and "labelled class zero" cannot be confused.
 *
 * This is deliberately a *reading* of a finished map, not a training signal — §3's rule that a
 * SOM is trained on data it is never told the answer for holds regardless of whether the answer
 * exists in the file. `ds.y === null` (the colour cube) returns every node as `-1`: there is
 * nothing to vote with, and this function says so rather than guessing.
 */
export function nodeLabels(som: Som, ds: Dataset, rows: Int32Array): Int32Array {
  const n = som.cols * som.rows;
  const labels = new Int32Array(n).fill(-1);
  if (ds.y === null || ds.classes === 0) return labels;

  const votes = new Int32Array(n * ds.classes);
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r] as number;
    const winner = bmu(som, sample(ds, row));
    const cls = ds.y[row] as number;
    votes[winner * ds.classes + cls] = (votes[winner * ds.classes + cls] as number) + 1;
  }

  for (let i = 0; i < n; i++) {
    let best = -1;
    let bestCount = 0;
    for (let c = 0; c < ds.classes; c++) {
      const v = votes[i * ds.classes + c] as number;
      if (v > bestCount) {
        bestCount = v;
        best = c;
      }
    }
    labels[i] = best;
  }
  return labels;
}

function nodeDistance(som: Som, a: number, b: number): number {
  const baseA = a * som.dim;
  const baseB = b * som.dim;
  let sum = 0;
  for (let k = 0; k < som.dim; k++) {
    const d = (som.W[baseA + k] as number) - (som.W[baseB + k] as number);
    sum += d * d;
  }
  return Math.sqrt(sum);
}
