/**
 * The graph's arithmetic, checked without a canvas — the same bargain `camera.ts` keeps.
 * Whether the picture is readable is decided here, not in the drawing code.
 */

import { describe, expect, it } from 'vitest';
import { hitNode, layoutNetwork, UNIT_CAP } from '../src/render/graph-layout.ts';

describe('layoutNetwork', () => {
  it('makes one column per layer, spanning the width', () => {
    const l = layoutNetwork([2, 8, 8, 2], 640, 300, 40);
    expect(l.cols).toHaveLength(4);
    expect(l.cols[0]!.x).toBeCloseTo(40, 6);
    expect(l.cols[3]!.x).toBeCloseTo(600, 6);
    // Evenly spaced, so no layer looks closer to its neighbour than another.
    const gaps = [1, 2, 3].map((i) => (l.cols[i]!.x as number) - (l.cols[i - 1]!.x as number));
    for (const g of gaps) expect(g).toBeCloseTo(gaps[0] as number, 6);
  });

  it('gives every column the same vertical pitch', () => {
    /*
     * One gap for the whole graph, not one per column. Per-column spacing would draw a 2-unit
     * input layer at the same pitch as a 16-unit hidden layer, and the eye reads that as the
     * two layers being differently *scaled* rather than differently sized.
     */
    const l = layoutNetwork([2, 16, 16, 3], 640, 300);
    const pitch = (col: number) => (l.cols[col]!.ys[1] as number) - (l.cols[col]!.ys[0] as number);
    /*
     * Four decimals, not more. `ys` is a Float32Array, so a pitch of ~14.13 is held to about
     * seven significant digits and two columns centred at different absolute offsets round
     * differently in the last one — 14.133331 against 14.133335. That is the storage, not the
     * arithmetic. A tighter tolerance than the type can represent is a test that fails for a
     * reason unrelated to what it is checking.
     */
    expect(pitch(0)).toBeCloseTo(pitch(1), 4);
    expect(pitch(1)).toBeCloseTo(pitch(3), 4);
  });

  it('centres every column on the same axis', () => {
    const l = layoutNetwork([2, 7, 4], 640, 300);
    const centre = (col: number) => {
      const ys = l.cols[col]!.ys;
      return ((ys[0] as number) + (ys[ys.length - 1] as number)) / 2;
    };
    expect(centre(0)).toBeCloseTo(centre(1), 5);
    expect(centre(1)).toBeCloseTo(centre(2), 5);
  });

  it('keeps every node inside the canvas', () => {
    for (const [w, h] of [[640, 300], [320, 200], [900, 500]] as const) {
      const l = layoutNetwork([2, 20, 20, 2], w, h, 40);
      for (const col of l.cols) {
        for (const y of col.ys) {
          expect(y).toBeGreaterThanOrEqual(0);
          expect(y).toBeLessThanOrEqual(h);
        }
        expect(col.x).toBeGreaterThanOrEqual(0);
        expect(col.x).toBeLessThanOrEqual(w);
      }
    }
  });

  it('shrinks the node radius as the layer gets crowded', () => {
    const roomy = layoutNetwork([2, 4, 2], 640, 300);
    const packed = layoutNetwork([2, 22, 2], 640, 300);
    expect(packed.nodeRadius).toBeLessThan(roomy.nodeRadius);
    // Nodes must not overlap: diameter has to fit inside the pitch.
    expect(packed.nodeRadius * 2).toBeLessThanOrEqual(packed.gap + 0.001);
  });

  it('flags a layer too wide to draw as a graph', () => {
    // §7's stated limit. The trigger is asserted so it cannot drift silently.
    expect(layoutNetwork([2, UNIT_CAP, 2], 640, 300).overCap).toBe(false);
    expect(layoutNetwork([2, UNIT_CAP + 1, 2], 640, 300).overCap).toBe(true);
    expect(layoutNetwork([64, 8, 2], 640, 300).overCap).toBe(true);
  });

  it('handles a network with no hidden layer', () => {
    const l = layoutNetwork([2, 2], 640, 300);
    expect(l.cols).toHaveLength(2);
    expect(l.cols[0]!.x).toBeLessThan(l.cols[1]!.x as number);
  });

  it('does not divide by zero on a single column or a single unit', () => {
    for (const shape of [[3], [1, 1], [1]]) {
      const l = layoutNetwork(shape, 640, 300);
      for (const col of l.cols) {
        expect(Number.isFinite(col.x)).toBe(true);
        for (const y of col.ys) expect(Number.isFinite(y)).toBe(true);
      }
    }
  });

  it('returns an empty layout for an empty shape', () => {
    expect(layoutNetwork([], 640, 300).cols).toHaveLength(0);
  });
});

describe('hitNode', () => {
  it('finds the node under the pointer', () => {
    const l = layoutNetwork([2, 4, 2], 640, 300);
    const col = l.cols[1]!;
    expect(hitNode(l, col.x, col.ys[2] as number)).toEqual([1, 2]);
  });

  it('misses when the pointer is between nodes', () => {
    const l = layoutNetwork([2, 4, 2], 640, 300);
    const col = l.cols[1]!;
    const between = ((col.ys[0] as number) + (col.ys[1] as number)) / 2;
    // Only meaningful while the nodes do not touch — which the radius rule guarantees.
    if (l.gap > (l.nodeRadius + 4) * 2) expect(hitNode(l, col.x, between)).toBeNull();
  });

  it('misses when the pointer is between columns', () => {
    const l = layoutNetwork([2, 4, 2], 640, 300);
    const mid = ((l.cols[0]!.x as number) + (l.cols[1]!.x as number)) / 2;
    expect(hitNode(l, mid, l.cols[1]!.ys[0] as number)).toBeNull();
  });

  it('reports the input column as layer 0', () => {
    // The renderer dims edges by `focus[0] === l + 1`, so an off-by-one here would highlight
    // the wrong layer's wiring and look almost right.
    const l = layoutNetwork([2, 4, 2], 640, 300);
    expect(hitNode(l, l.cols[0]!.x, l.cols[0]!.ys[1] as number)).toEqual([0, 1]);
  });
});
