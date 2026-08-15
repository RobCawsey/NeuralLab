import { describe, expect, it } from 'vitest';
import { Rng, fitStandardiser, split, standardise } from '@neurallab/core';
import { moons } from '@neurallab/data';
import { createNet, createScratch, initialise } from '@neurallab/mlp';
import { activationStats, histogram, weightStats, HISTOGRAM_BINS } from '../src/run/diagnostics.ts';

function prepared() {
  const data = moons({ n: 240, noise: 0.15, seed: 4417 });
  const parts = split(data, 0.7, new Rng(4417 ^ 0x5f3759df));
  return { z: standardise(data, fitStandardiser(data, parts.train)), rows: parts.train };
}

describe('histogram', () => {
  it('bins every value exactly once', () => {
    const h = histogram([0, 0.1, 0.5, 0.9, 1, -0.3]);
    const total = h.counts.reduce((a, b) => a + b, 0);
    expect(total).toBe(6);
  });

  it('uses the requested bin count', () => {
    expect(histogram([1, 2, 3]).counts).toHaveLength(HISTOGRAM_BINS);
    expect(histogram([1, 2, 3], 8).counts).toHaveLength(8);
  });

  it('takes its range from the data', () => {
    const tight = histogram([0, 0.01, -0.01]);
    const wide = histogram([0, 8, -8]);
    expect(tight.max - tight.min).toBeLessThan(wide.max - wide.min);
  });

  it('puts the minimum in the first bin and the maximum in the last', () => {
    const h = histogram([-5, 0, 5], 10);
    expect(h.counts[0]).toBeGreaterThan(0);
    expect(h.counts[9]).toBeGreaterThan(0);
  });

  it('survives a constant array instead of dividing by zero', () => {
    const h = histogram([3, 3, 3, 3]);
    expect(Number.isFinite(h.min)).toBe(true);
    expect(Number.isFinite(h.max)).toBe(true);
    expect(h.max).toBeGreaterThan(h.min);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(4);
  });

  it('survives an empty array', () => {
    const h = histogram([]);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(0);
    expect(Number.isFinite(h.min)).toBe(true);
  });

  it('ignores non-finite values rather than corrupting a bin', () => {
    const h = histogram([1, NaN, 2, Infinity, 3]);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(3);
  });
});

describe('activationStats', () => {
  it('reports one entry per layer, with the right unit counts', () => {
    const { z, rows } = prepared();
    const net = createNet({ shape: [2, 8, 8, 2], hidden: 'relu', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'he', new Rng(1));
    const stats = activationStats(net, z, rows, createScratch(net));
    expect(stats.map((s) => s.totalUnits)).toEqual([8, 8, 2]);
  });

  it('flags relu layers and only relu layers', () => {
    const { z, rows } = prepared();
    const net = createNet({ shape: [2, 6, 6, 2], hidden: 'relu', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'he', new Rng(1));
    const stats = activationStats(net, z, rows, createScratch(net));
    expect(stats[0]!.isRelu).toBe(true);
    expect(stats[1]!.isRelu).toBe(true);
    expect(stats[2]!.isRelu).toBe(false); // softmax
  });

  it('counts a unit as dead only if it never fires across every row examined', () => {
    /*
     * Hand-built: one relu unit wired to `-10x`, one to `10x`, over data that is entirely
     * positive. The first is negative for every row and never fires — genuinely dead. The
     * second is positive for every row and always fires — genuinely not.
     */
    const net = createNet({ shape: [1, 2], hidden: 'linear', output: 'relu', loss: 'mse' });
    net.layers[0]!.W.set([-10, 10]);
    net.layers[0]!.b.set([0, 0]);

    const ds = {
      name: 'x',
      x: Float32Array.from([1, 2, 0.5, 3]),
      y: null,
      n: 4,
      dim: 1,
      classes: 0,
      featureNames: ['x'],
      classNames: [],
    };
    const rows = Int32Array.from([0, 1, 2, 3]);
    const stats = activationStats(net, ds, rows, createScratch(net));
    expect(stats[0]!.deadUnits).toBe(1);
    expect(stats[0]!.totalUnits).toBe(2);
  });

  it('recognises the all-zero-init case as fully dead', () => {
    // Slice 1's challenge 5 network — a hidden relu layer under zero init produces zero
    // activation for every unit, on every sample, which is the purest possible dead layer.
    const { z, rows } = prepared();
    const net = createNet({ shape: [2, 8, 2], hidden: 'relu', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'zeros', new Rng(1));
    const stats = activationStats(net, z, rows, createScratch(net));
    expect(stats[0]!.deadUnits).toBe(8);
  });

  it('finds no dead units in a healthy trained-ish network', () => {
    const { z, rows } = prepared();
    const net = createNet({ shape: [2, 8, 8, 2], hidden: 'relu', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'he', new Rng(3));
    const stats = activationStats(net, z, rows, createScratch(net));
    // He init on a healthy network should not have every unit dead; a handful might be, but not all.
    expect(stats[0]!.deadUnits).toBeLessThan(stats[0]!.totalUnits);
  });

  it('does not disturb the network it examines', () => {
    const { z, rows } = prepared();
    const net = createNet({ shape: [2, 4, 2], hidden: 'tanh', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'glorot', new Rng(1));
    const before = Array.from(net.layers[0]!.W);
    activationStats(net, z, rows, createScratch(net));
    expect(Array.from(net.layers[0]!.W)).toEqual(before);
  });
});

describe('weightStats', () => {
  it('returns one histogram per layer', () => {
    const net = createNet({ shape: [2, 6, 3, 2], hidden: 'tanh', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'glorot', new Rng(1));
    expect(weightStats(net)).toHaveLength(3);
  });

  it('bins every weight in the layer, and only that layer', () => {
    const net = createNet({ shape: [2, 5, 2], hidden: 'tanh', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'glorot', new Rng(1));
    const stats = weightStats(net);
    expect(stats[0]!.counts.reduce((a, b) => a + b, 0)).toBe(net.layers[0]!.W.length);
    expect(stats[1]!.counts.reduce((a, b) => a + b, 0)).toBe(net.layers[1]!.W.length);
  });

  it('collapses to one bin under zero init, rather than a divide-by-zero spread', () => {
    const net = createNet({ shape: [2, 4, 2], hidden: 'tanh', output: 'softmax', loss: 'crossEntropy' });
    initialise(net, 'zeros', new Rng(1));
    const stats = weightStats(net);
    const nonEmpty = stats[0]!.counts.filter((c) => c > 0).length;
    expect(nonEmpty).toBe(1);
  });
});
