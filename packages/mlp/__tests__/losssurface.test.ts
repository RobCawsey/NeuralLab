import { describe, expect, it } from 'vitest';
import { Rng, type Dataset } from '@neurallab/core';
import { createNet, flattenWeights, type Net } from '../src/net.ts';
import {
  computeLossSurface,
  projectOntoDirections,
  randomDirection,
  unitDirection,
} from '../src/losssurface.ts';

function pointDataset(points: readonly (readonly number[])[]): Dataset {
  const dim = (points[0] as number[]).length;
  const x = new Float32Array(points.length * dim);
  points.forEach((p, i) => x.set(p, i * dim));
  return {
    name: 'test points',
    x,
    y: null,
    n: points.length,
    dim,
    classes: 0,
    featureNames: Array.from({ length: dim }, (_, k) => `f${k}`),
    classNames: [],
  };
}

function net2(shape: readonly number[]): Net {
  return createNet({ shape, hidden: 'tanh', output: 'softmax', loss: 'crossEntropy' });
}

describe('unitDirection', () => {
  it('is exactly one flat entry set to 1, everything else 0', () => {
    const net = net2([2, 4, 3]);
    const dir = unitDirection(net, 5);
    expect(dir[5]).toBe(1);
    let nonZero = 0;
    for (const v of dir) if (v !== 0) nonZero++;
    expect(nonZero).toBe(1);
  });

  it('is all zero for an index past the end, rather than throwing', () => {
    const net = net2([2, 2]);
    const dir = unitDirection(net, 999);
    expect(Array.from(dir).every((v) => v === 0)).toBe(true);
  });
});

describe('randomDirection', () => {
  it('filter-normalises each layer\'s W and b to match the real weights\' own norm', () => {
    const net = net2([2, 8, 8, 3]);
    const real = flattenWeights(net);
    const dir = randomDirection(net, new Rng(4417));
    expect(dir).toHaveLength(real.length);

    // Layer boundaries mirror flattenWeights' own layout: W then b, per layer, in order.
    let at = 0;
    const norm = (v: Float32Array, start: number, len: number): number => {
      let s = 0;
      for (let i = start; i < start + len; i++) s += (v[i] as number) ** 2;
      return Math.sqrt(s);
    };
    for (const layer of net.layers) {
      const wStart = at;
      at += layer.W.length;
      const bStart = at;
      at += layer.b.length;
      expect(norm(dir, wStart, layer.W.length)).toBeCloseTo(norm(real, wStart, layer.W.length), 4);
      expect(norm(dir, bStart, layer.b.length)).toBeCloseTo(norm(real, bStart, layer.b.length), 4);
    }
  });

  it('replays exactly from the same seed', () => {
    const net = net2([2, 4, 2]);
    const a = randomDirection(net, new Rng(1));
    const b = randomDirection(net, new Rng(1));
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe('computeLossSurface', () => {
  it('matches a hand-worked 1-1 linear network exactly', () => {
    // shape [1,1], linear output, mse loss: forward(x) = w*x + b.
    // W = [0.5], b = [0.5]; x = [2], unlabelled so target = 0 -> onehot = [1].
    // loss(w,b) = 0.5 * (w*2 + b - 1)^2. dir1 perturbs w, dir2 perturbs b (flat layout: [W, b]).
    const net = createNet({ shape: [1, 1], hidden: 'linear', output: 'linear', loss: 'mse' });
    net.layers[0]!.W.set([0.5]);
    net.layers[0]!.b.set([0.5]);
    const ds = pointDataset([[2]]);
    const dir1 = unitDirection(net, 0); // W
    const dir2 = unitDirection(net, 1); // b

    const surface = computeLossSurface(net, ds, Int32Array.from([0]), dir1, dir2, 3, 1);
    expect(surface.res).toBe(3);
    // row = beta (b offset), col = alpha (w offset), both in {-1, 0, 1}.
    const expected = [
      3.125, 0.125, 1.125,
      1.125, 0.125, 3.125,
      0.125, 1.125, 6.125,
    ];
    for (let i = 0; i < expected.length; i++) {
      expect(surface.values[i]).toBeCloseTo(expected[i] as number, 5);
    }
  });

  it('does not perturb the real network\'s weights', () => {
    const net = net2([2, 4, 2]);
    const before = Array.from(flattenWeights(net));
    const ds = pointDataset([[0.1, 0.2], [0.3, 0.4]]);
    computeLossSurface(net, ds, Int32Array.from([0, 1]), randomDirection(net, new Rng(1)), randomDirection(net, new Rng(2)), 5, 1);
    expect(Array.from(flattenWeights(net))).toEqual(before);
  });
});

describe('projectOntoDirections', () => {
  it('recovers the exact alpha, beta used to build a perturbation, for orthogonal directions', () => {
    const base = Float32Array.from([1, 2, 3, 4]);
    const dir1 = Float32Array.from([1, 0, 0, 0]);
    const dir2 = Float32Array.from([0, 1, 0, 0]);
    const weights = Float32Array.from([1 + 3 * 1, 2 + 3 * 0 + (-2) * 1, 3, 4]); // base + 3*dir1 - 2*dir2
    const [alpha, beta] = projectOntoDirections(base, weights, dir1, dir2);
    expect(alpha).toBeCloseTo(3, 5);
    expect(beta).toBeCloseTo(-2, 5);
  });

  it('is [0, 0] at the base weights themselves', () => {
    const base = Float32Array.from([1, 2, 3]);
    const dir1 = Float32Array.from([1, 1, 0]);
    const dir2 = Float32Array.from([0, 1, 1]);
    const [alpha, beta] = projectOntoDirections(base, base, dir1, dir2);
    expect(alpha).toBeCloseTo(0, 10);
    expect(beta).toBeCloseTo(0, 10);
  });
});
