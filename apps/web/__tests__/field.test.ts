/**
 * The decision field's arithmetic, checked without a canvas.
 *
 * `computeField` imports nothing from the DOM, which is the whole reason it is a separate
 * function from `drawField`. Everything that can be wrong here is wrong quietly: an inverted
 * row order draws a mirrored boundary that still looks like a boundary, a missing standardiser
 * draws the network's answer to a question nobody asked, and a corner-sampled grid biases the
 * whole picture half a cell.
 */

import { describe, expect, it } from 'vitest';
import { Rng, type Standardiser } from '@neurallab/core';
import { createNet, createScratch, initialise, type Net } from '@neurallab/mlp';
import { computeField, fieldCost } from '../src/render/field.ts';

const UNIT: Standardiser = {
  mean: Float32Array.from([0, 0]),
  sd: Float32Array.from([1, 1]),
};
const BOX = { minX: -1, maxX: 1, minY: -1, maxY: 1 };

/** A hand-built network that answers purely on the sign of x: class 1 to the right. */
function signOfX(): Net {
  const net = createNet({ shape: [2, 2], hidden: 'linear', output: 'softmax', loss: 'crossEntropy' });
  // Unit 0 wins when x < 0, unit 1 when x > 0. Steep, so the boundary is crisp.
  net.layers[0]!.W.set([-8, 0, 8, 0]);
  return net;
}

function trained(shape: number[], seed = 1): Net {
  const net = createNet({ shape, hidden: 'tanh', output: 'softmax', loss: 'crossEntropy' });
  initialise(net, 'glorot', new Rng(seed));
  return net;
}

describe('computeField', () => {
  it('fills every cell of the grid', () => {
    const net = trained([2, 4, 2]);
    const f = computeField(net, createScratch(net), UNIT, BOX, 16, 2);
    expect(f.cls).toHaveLength(256);
    expect(f.conf).toHaveLength(256);
    expect(f.res).toBe(16);
    for (const c of f.conf) expect(c).toBeGreaterThan(0);
  });

  it('puts the origin of the grid at the bottom-left', () => {
    /*
     * The field is a thing in world space, where y is up; `drawField` reverses the row order
     * when it blits. If this ever became top-down, every boundary in the app would be drawn
     * mirrored — and a mirrored boundary still looks like a boundary, which is why it is
     * asserted rather than eyeballed.
     *
     * Network answers class 1 for x > 0, so within any row the left half is class 0.
     */
    const net = signOfX();
    const f = computeField(net, createScratch(net), UNIT, BOX, 8, 2);
    for (let row = 0; row < 8; row++) {
      expect(f.cls[row * 8 + 0], `row ${row} left`).toBe(0);
      expect(f.cls[row * 8 + 7], `row ${row} right`).toBe(1);
    }
  });

  it('samples cell centres, not corners', () => {
    /*
     * A corner-sampled grid biases the whole field half a cell down and left. Invisible at
     * 128², obvious at 4² — and it is the sort of thing that gets "fixed" by nudging the
     * drawing rather than the sampling.
     *
     * With a boundary at x = 0 and 4 columns over [-1, 1], centres fall at -0.75, -0.25, 0.25,
     * 0.75 — a clean 2/2 split. Corners would fall at -1, -0.5, 0, 0.5, putting the boundary
     * cell itself on the wrong side.
     */
    const net = signOfX();
    const f = computeField(net, createScratch(net), UNIT, BOX, 4, 2);
    expect(Array.from(f.cls.slice(0, 4))).toEqual([0, 0, 1, 1]);
  });

  it('applies the standardiser', () => {
    /*
     * The network has never seen a raw coordinate. A field computed without standardising is
     * the network's answer to a different question from the one the scatter is asking, and the
     * two would disagree while both looking entirely reasonable.
     */
    const net = signOfX();
    const shifted: Standardiser = {
      mean: Float32Array.from([0.5, 0]),
      sd: Float32Array.from([1, 1]),
    };
    const plain = computeField(net, createScratch(net), UNIT, BOX, 16, 2);
    const moved = computeField(net, createScratch(net), shifted, BOX, 16, 2);

    const boundary = (f: typeof plain): number => {
      for (let col = 0; col < 16; col++) if (f.cls[col] === 1) return col;
      return -1;
    };
    // Subtracting a positive mean shifts the boundary to the right in world space.
    expect(boundary(moved)).toBeGreaterThan(boundary(plain));
  });

  it('reports a real winner for three classes', () => {
    // Two-class problems hide a family of bugs — a field that assumes p and 1 − p, an argmax
    // that always returns the last index. Blobs is a three-class set, so this path is live.
    const net = trained([2, 6, 3], 4);
    const f = computeField(net, createScratch(net), UNIT, BOX, 24, 3);
    const seen = new Set(Array.from(f.cls));
    for (const c of seen) expect(c).toBeLessThan(3);
    // A softmax winner over three classes cannot be below 1/3.
    for (const c of f.conf) expect(c).toBeGreaterThanOrEqual(1 / 3 - 1e-6);
  });

  it('never reports a confidence outside [1/classes, 1]', () => {
    const net = trained([2, 8, 2], 7);
    const f = computeField(net, createScratch(net), UNIT, BOX, 32, 2);
    for (const c of f.conf) {
      expect(c).toBeGreaterThanOrEqual(0.5 - 1e-6);
      expect(c).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  it('is deterministic', () => {
    const net = trained([2, 6, 6, 2], 9);
    const a = computeField(net, createScratch(net), UNIT, BOX, 32, 2);
    const b = computeField(net, createScratch(net), UNIT, BOX, 32, 2);
    expect(Array.from(a.cls)).toEqual(Array.from(b.cls));
    expect(Array.from(a.conf)).toEqual(Array.from(b.conf));
  });

  it('covers exactly the box it was given', () => {
    const box = { minX: -3, maxX: 5, minY: 2, maxY: 4 };
    const net = trained([2, 4, 2]);
    expect(computeField(net, createScratch(net), UNIT, box, 8, 2).box).toEqual(box);
  });

  it('prices itself in forward passes', () => {
    // The number printed on the badge. §5's whole argument is this against ~15 k per step.
    expect(fieldCost(64)).toBe(4096);
    expect(fieldCost(128)).toBe(16384);
  });
});
