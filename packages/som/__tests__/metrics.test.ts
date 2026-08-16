import { describe, expect, it } from 'vitest';
import { Rng, type Dataset } from '@neurallab/core';
import { createSom } from '../src/som.ts';
import { componentPlane, nodeLabels, quantisationError, topographicError, uMatrix } from '../src/metrics.ts';

function pointDataset(points: readonly number[]): Dataset {
  return {
    name: 'test points',
    x: Float32Array.from(points),
    y: null,
    n: points.length,
    dim: 1,
    classes: 0,
    featureNames: ['f0'],
    classNames: [],
  };
}

function labelledDataset(points: readonly number[], labels: readonly number[], classes: number): Dataset {
  return {
    name: 'test points',
    x: Float32Array.from(points),
    y: Int32Array.from(labels),
    n: points.length,
    dim: 1,
    classes,
    featureNames: ['f0'],
    classNames: Array.from({ length: classes }, (_, c) => `class ${c}`),
  };
}

describe('quantisationError', () => {
  it('is the hand-computed mean distance to each sample\'s BMU', () => {
    // 2×1 rect, node0 = 0, node1 = 1. Samples 0.1 and 0.9 each land 0.1 from their nearest node.
    const som = createSom(2, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1]);
    const ds = pointDataset([0.1, 0.9]);
    expect(quantisationError(som, ds, Int32Array.from([0, 1]))).toBeCloseTo(0.1, 6);
  });

  it('is 0 for an empty row set rather than NaN', () => {
    const som = createSom(2, 1, 1, 'rect', new Rng(1));
    const ds = pointDataset([0.1]);
    expect(quantisationError(som, ds, Int32Array.from([]))).toBe(0);
  });
});

describe('topographicError', () => {
  it('is the hand-worked case: one sample lands on adjacent nodes, one does not', () => {
    // 3×1 rect line: node0=0, node1=5, node2=0.5. Neighbours are 0-1 and 1-2 only; 0-2 are not.
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 5, 0.5]);
    const ds = pointDataset([0.3, 4.9]);
    // Sample 0.3: best=node2 (dist 0.2), second=node0 (dist 0.3) — 0 and 2 are not neighbours.
    // Sample 4.9: best=node1 (dist 0.1), second=node2 (dist 4.4) — 1 and 2 are neighbours.
    expect(topographicError(som, ds, Int32Array.from([0, 1]))).toBeCloseTo(0.5, 10);
  });

  it('is 0 when every sample is unambiguously best-matched by one node', () => {
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1, 2]);
    const ds = pointDataset([0.05, 0.95, 2.05]);
    expect(topographicError(som, ds, Int32Array.from([0, 1, 2]))).toBe(0);
  });
});

describe('uMatrix', () => {
  it('matches a hand-computed 3×1 line exactly', () => {
    // node0=0, node1=1, node2=3. node0's only neighbour is node1 (dist 1); node1's are node0
    // and node2 (mean of 1 and 2 = 1.5); node2's only neighbour is node1 (dist 2).
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1, 3]);
    const u = uMatrix(som);
    expect(Array.from(u)).toEqual([1, 1.5, 2]);
  });

  it('reads 0 for a node with no neighbours at all — a 1×1 map', () => {
    const som = createSom(1, 1, 2, 'hex', new Rng(1));
    expect(Array.from(uMatrix(som))).toEqual([0]);
  });
});

describe('componentPlane', () => {
  it('pulls out exactly one dimension across every node', () => {
    // 2×1 map, dim 2: node0 = (0.1, 0.9), node1 = (0.2, 0.8).
    const som = createSom(2, 1, 2, 'rect', new Rng(1));
    som.W.set([0.1, 0.9, 0.2, 0.8]);
    expect(Array.from(componentPlane(som, 0))).toEqual([
      Math.fround(0.1),
      Math.fround(0.2),
    ]);
    expect(Array.from(componentPlane(som, 1))).toEqual([
      Math.fround(0.9),
      Math.fround(0.8),
    ]);
  });

  it('is all zero for a dimension out of range, rather than throwing', () => {
    const som = createSom(2, 1, 2, 'rect', new Rng(1));
    expect(Array.from(componentPlane(som, 5))).toEqual([0, 0]);
  });
});

describe('nodeLabels', () => {
  it('matches the hand-worked vote: node0 and node1 unanimous, node2 a single voter', () => {
    // node0=0, node1=1, node2=3. Samples 0.1/0.2 (class 0) both nearest node0; 0.9/1.1 (class 1)
    // both nearest node1; 3.0 (class 0) nearest node2 — its only voter.
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1, 3]);
    const ds = labelledDataset([0.1, 0.2, 0.9, 1.1, 3.0], [0, 0, 1, 1, 0], 2);
    const labels = nodeLabels(som, ds, Int32Array.from([0, 1, 2, 3, 4]));
    expect(Array.from(labels)).toEqual([0, 1, 0]);
  });

  it('reads -1 for a node no row ever won', () => {
    // A fourth node at 100 is never nearest anything in this dataset.
    const som = createSom(4, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1, 3, 100]);
    const ds = labelledDataset([0.1, 0.9], [0, 1], 2);
    const labels = nodeLabels(som, ds, Int32Array.from([0, 1]));
    expect(labels[3]).toBe(-1);
  });

  it('reads -1 for every node on an unlabelled dataset, rather than guessing', () => {
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1, 3]);
    const ds = pointDataset([0.1, 0.9, 3.0]);
    const labels = nodeLabels(som, ds, Int32Array.from([0, 1, 2]));
    expect(Array.from(labels)).toEqual([-1, -1, -1]);
  });
});
