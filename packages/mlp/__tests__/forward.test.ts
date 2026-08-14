import { describe, expect, it } from 'vitest';
import { Rng } from '@neurallab/core';
import { activate, softmax } from '../src/activations.ts';
import {
  argmax,
  createNet,
  createScratch,
  describeShape,
  forward,
  initialise,
  maxAbsWeight,
  paramCount,
  parseHidden,
  shapeOf,
  type Net,
} from '../src/net.ts';

function net(shape: number[], hidden: 'relu' | 'tanh' | 'sigmoid' | 'linear' = 'relu'): Net {
  return createNet({ shape, hidden, output: 'softmax', loss: 'crossEntropy' });
}

describe('activations', () => {
  it('clamps negatives to zero for relu', () => {
    const z = Float64Array.from([-2, -0.001, 0, 0.5, 3]);
    const a = new Float64Array(5);
    activate('relu', z, a);
    expect(Array.from(a)).toEqual([0, 0, 0, 0.5, 3]);
  });

  it('keeps tanh in (-1, 1) and sigmoid in (0, 1)', () => {
    const z = Float64Array.from([-40, -1, 0, 1, 40]);
    const t = new Float64Array(5);
    const s = new Float64Array(5);
    activate('tanh', z, t);
    activate('sigmoid', z, s);
    for (let i = 0; i < 5; i++) {
      expect(t[i]).toBeGreaterThanOrEqual(-1);
      expect(t[i]).toBeLessThanOrEqual(1);
      expect(s[i]).toBeGreaterThanOrEqual(0);
      expect(s[i]).toBeLessThanOrEqual(1);
    }
    expect(t[2]).toBeCloseTo(0, 6);
    expect(s[2]).toBeCloseTo(0.5, 6);
  });

  it('can write in place', () => {
    // The forward pass passes the same buffer for z and a when it can. If an activation read
    // z after writing a, that would corrupt everything after the first element.
    const v = Float64Array.from([-1, 2, -3]);
    activate('relu', v, v);
    expect(Array.from(v)).toEqual([0, 2, 0]);
  });
});

describe('softmax', () => {
  it('produces a distribution', () => {
    const a = new Float64Array(4);
    softmax(Float64Array.from([1, 2, 3, 4]), a);
    let sum = 0;
    for (const v of a) {
      expect(v).toBeGreaterThan(0);
      sum += v;
    }
    expect(sum).toBeCloseTo(1, 5);
  });

  it('survives large logits instead of returning NaN', () => {
    /*
     * The reason softmax subtracts its maximum. `Math.exp(800)` is Infinity and
     * `Infinity / Infinity` is NaN, so an unshifted softmax turns the whole output into NaN the
     * first time a logit gets large. Slice 2 measured when that actually happens: at a
     * destructive learning rate the weights reach ~1e5 and so do the logits, and the shift is
     * what keeps a wrecked network readable. Diverging is the lesson; NaN everywhere is a bug.
     */
    const a = new Float64Array(3);
    softmax(Float64Array.from([1000, 900, 800]), a);
    for (const v of a) expect(Number.isFinite(v)).toBe(true);
    expect(a[0]).toBeCloseTo(1, 5);
  });

  it('is shift invariant', () => {
    const a = new Float64Array(3);
    const b = new Float64Array(3);
    softmax(Float64Array.from([1, 2, 3]), a);
    softmax(Float64Array.from([101, 102, 103]), b);
    for (let i = 0; i < 3; i++) expect(a[i]).toBeCloseTo(b[i] as number, 5);
  });

  it('degrades to uniform rather than propagating a non-finite maximum', () => {
    const a = new Float64Array(4);
    softmax(Float64Array.from([NaN, NaN, NaN, NaN]), a);
    for (const v of a) expect(v).toBeCloseTo(0.25, 6);
  });
});

describe('createNet', () => {
  it('sizes every buffer from the shape', () => {
    const n = net([2, 8, 8, 3]);
    expect(n.layers).toHaveLength(3);
    expect(n.layers[0]!.W).toHaveLength(8 * 2);
    expect(n.layers[1]!.W).toHaveLength(8 * 8);
    expect(n.layers[2]!.W).toHaveLength(3 * 8);
    expect(n.layers[2]!.b).toHaveLength(3);
    expect(paramCount(n)).toBe(16 + 8 + 64 + 8 + 24 + 3);
    expect(shapeOf(n)).toEqual([2, 8, 8, 3]);
    expect(describeShape(n)).toBe('2-8-8-3');
  });

  it('puts the output activation only on the last layer', () => {
    const n = net([2, 4, 4, 2]);
    expect(n.layers.map((l) => l.act)).toEqual(['relu', 'relu', 'softmax']);
  });

  it('supports no hidden layer at all', () => {
    // Challenge 1 — this configuration has to be reachable, and it has to be a real network.
    const n = net([2, 2]);
    expect(n.layers).toHaveLength(1);
    expect(describeShape(n)).toBe('2-2');
  });

  it('rejects a degenerate shape', () => {
    expect(() => net([2])).toThrow();
    expect(() => net([2, 0, 2])).toThrow();
  });
});

describe('forward', () => {
  it('computes a hand-checked network exactly', () => {
    /*
     * Two inputs, two linear units, no activation to hide an arithmetic slip.
     *   W = [[1, 2], [3, 4]]   b = [0.5, -0.5]   x = [1, 1]
     *   z = [1 + 2 + 0.5, 3 + 4 - 0.5] = [3.5, 6.5]
     *
     * This is the test that catches a transposed weight index: with W read column-major, z
     * comes out [4.5, 5.5], which is just as plausible-looking and completely wrong.
     */
    const n = createNet({ shape: [2, 2], hidden: 'linear', output: 'linear', loss: 'mse' });
    n.layers[0]!.W.set([1, 2, 3, 4]);
    n.layers[0]!.b.set([0.5, -0.5]);

    const out = forward(n, Float32Array.from([1, 1]), createScratch(n));
    expect(out[0]).toBeCloseTo(3.5, 5);
    expect(out[1]).toBeCloseTo(6.5, 5);
  });

  it('reads W row-major with the unit as the outer index', () => {
    // Same claim as above, isolated: a one-hot input must select one column, not one row.
    const n = createNet({ shape: [3, 2], hidden: 'linear', output: 'linear', loss: 'mse' });
    n.layers[0]!.W.set([1, 2, 3, 4, 5, 6]);
    const out = forward(n, Float32Array.from([0, 1, 0]), createScratch(n));
    expect(Array.from(out)).toEqual([2, 5]);
  });

  it('passes an identity network through unchanged', () => {
    const n = createNet({ shape: [3, 3], hidden: 'linear', output: 'linear', loss: 'mse' });
    n.layers[0]!.W.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const out = forward(n, Float32Array.from([0.25, -0.5, 2]), createScratch(n));
    expect(Array.from(out)).toEqual([0.25, -0.5, 2]);
  });

  it('fills z and a for every layer, so the graph can draw them', () => {
    const n = net([2, 5, 3]);
    initialise(n, 'he', new Rng(1));
    const s = createScratch(n);
    forward(n, Float32Array.from([0.3, -0.2]), s);

    expect(s.z).toHaveLength(2);
    expect(s.a[0]).toHaveLength(5);
    expect(s.a[1]).toHaveLength(3);
    // relu on the hidden layer: a is z clamped at zero, and the two must actually differ.
    for (let i = 0; i < 5; i++) expect(s.a[0]![i]).toBe(Math.max(0, s.z[0]![i] as number));
  });

  it('reuses its scratch rather than allocating', () => {
    const n = net([2, 4, 2]);
    const s = createScratch(n);
    const first = s.a[0];
    forward(n, Float32Array.from([1, 1]), s);
    forward(n, Float32Array.from([2, 2]), s);
    expect(s.a[0]).toBe(first);
  });

  it('is deterministic', () => {
    const n = net([2, 6, 6, 2], 'tanh');
    initialise(n, 'glorot', new Rng(7));
    const s = createScratch(n);
    const a = Array.from(forward(n, Float32Array.from([0.4, 0.9]), s));
    const b = Array.from(forward(n, Float32Array.from([0.4, 0.9]), s));
    expect(a).toEqual(b);
  });

  it('returns a probability distribution from a softmax output', () => {
    const n = net([2, 6, 3]);
    initialise(n, 'he', new Rng(2));
    const out = forward(n, Float32Array.from([0.1, 0.2]), createScratch(n));
    let sum = 0;
    for (const v of out) sum += v;
    expect(sum).toBeCloseTo(1, 5);
    expect(argmax(out)).toBeGreaterThanOrEqual(0);
    expect(argmax(out)).toBeLessThan(3);
  });
});

describe('initialise', () => {
  it('scales He by 2 / fan-in', () => {
    // A mis-scaled initialiser makes every network in the project hard to train with nothing
    // obviously wrong, so the variance is checked rather than assumed.
    const n = net([100, 100]);
    initialise(n, 'he', new Rng(3));
    const w = n.layers[0]!.W;
    let sumSq = 0;
    for (const v of w) sumSq += v * v;
    expect(Math.sqrt(sumSq / w.length)).toBeCloseTo(Math.sqrt(2 / 100), 2);
  });

  it('keeps Glorot inside its own limit', () => {
    const n = net([50, 30]);
    initialise(n, 'glorot', new Rng(4));
    const limit = Math.sqrt(6 / (50 + 30));
    for (const v of n.layers[0]!.W) expect(Math.abs(v)).toBeLessThanOrEqual(limit);
  });

  it('always zeroes the biases', () => {
    for (const scheme of ['he', 'glorot', 'small', 'zeros'] as const) {
      const n = net([4, 6, 2]);
      initialise(n, scheme, new Rng(5));
      for (const l of n.layers) for (const v of l.b) expect(v).toBe(0);
    }
  });

  it('makes every hidden unit identical under zero init — challenge 5', () => {
    /*
     * The lesson, asserted. With every weight zero, every hidden unit computes the same thing
     * for any input; and because they are identical they receive identical gradients forever, so
     * an 8-unit layer has the capacity of 1. The network graph shows this directly, which is why
     * `zeros` is an offered option rather than a mistake.
     */
    const n = net([2, 8, 2], 'tanh');
    initialise(n, 'zeros', new Rng(6));
    const s = createScratch(n);
    forward(n, Float32Array.from([0.7, -1.3]), s);
    const hidden = s.a[0] as Float64Array;
    for (let i = 1; i < hidden.length; i++) expect(hidden[i]).toBe(hidden[0]);
  });

  it('replays from the same seed and differs from another', () => {
    const a = net([3, 5, 2]);
    const b = net([3, 5, 2]);
    const c = net([3, 5, 2]);
    initialise(a, 'he', new Rng(11));
    initialise(b, 'he', new Rng(11));
    initialise(c, 'he', new Rng(12));
    expect(Array.from(a.layers[0]!.W)).toEqual(Array.from(b.layers[0]!.W));
    expect(Array.from(a.layers[0]!.W)).not.toEqual(Array.from(c.layers[0]!.W));
  });
});

describe('maxAbsWeight', () => {
  it('finds the largest magnitude, over the whole net or one layer', () => {
    const n = net([2, 2, 2]);
    n.layers[0]!.W.set([0.1, -0.9, 0.3, 0.2]);
    n.layers[1]!.W.set([2, 0, 0, 0]);
    expect(maxAbsWeight(n)).toBeCloseTo(2, 6);
    expect(maxAbsWeight(n, 0)).toBeCloseTo(0.9, 6);
  });

  it('returns zero for a zeroed network rather than dividing by it later', () => {
    const n = net([2, 2, 2]);
    initialise(n, 'zeros', new Rng(1));
    expect(maxAbsWeight(n)).toBe(0);
  });
});

describe('parseHidden', () => {
  it('reads the shapes a control or URL can produce', () => {
    expect(parseHidden('8-8')).toEqual([8, 8]);
    expect(parseHidden('4, 6 8')).toEqual([4, 6, 8]);
    expect(parseHidden('')).toEqual([]);
    expect(parseHidden('   ')).toEqual([]);
  });

  it('drops nonsense instead of producing a broken network', () => {
    expect(parseHidden('abc')).toEqual([]);
    expect(parseHidden('8-abc-4')).toEqual([8, 4]);
    expect(parseHidden('0-8')).toEqual([8]);
    expect(parseHidden('-5')).toEqual([5]);
  });

  it('caps width and depth', () => {
    expect(parseHidden('9999')).toEqual([64]);
    expect(parseHidden('2-2-2-2-2-2-2-2-2')).toHaveLength(6);
  });
});
