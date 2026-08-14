/**
 * The stepper's drawing primitives — pixel-level checks that the right values land in the right
 * cells. `jsdom`'s canvas has no real 2D context, so these run against a small hand-rolled one
 * that records fillRect calls, which is enough to check layout and colour selection without a
 * browser.
 */

import { describe, expect, it } from 'vitest';
import { drawStrip, drawTiles, signedColour } from '../src/render/stepper.ts';

interface Call {
  x: number;
  y: number;
  w: number;
  h: number;
  colour: string;
}

/** Just enough of CanvasRenderingContext2D for these functions to run and be inspected. */
function fakeCtx(): { ctx: CanvasRenderingContext2D; calls: Call[] } {
  const calls: Call[] = [];
  let fillStyle = '#000';
  const ctx = {
    clearRect() {},
    fillRect(x: number, y: number, w: number, h: number) {
      calls.push({ x, y, w, h, colour: fillStyle });
    },
    strokeRect() {},
    set fillStyle(v: string) {
      fillStyle = v;
    },
    get fillStyle() {
      return fillStyle;
    },
    strokeStyle: '#000',
    lineWidth: 1,
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

describe('signedColour', () => {
  it('is cyan below zero and amber above', () => {
    expect(signedColour(-1, 1)).toContain('78, 168, 196');
    expect(signedColour(1, 1)).toContain('233, 161, 59');
  });

  it('is near-transparent at zero and opaque at the peak', () => {
    const zero = signedColour(0, 1);
    const full = signedColour(1, 1);
    const alphaOf = (s: string) => Number(s.split(',')[3]!.replace(')', ''));
    expect(alphaOf(zero)).toBeLessThan(0.15);
    expect(alphaOf(full)).toBeGreaterThan(0.9);
  });

  it('does not divide by zero when every value is zero', () => {
    expect(() => signedColour(0, 0)).not.toThrow();
    expect(signedColour(5, 0)).toBeTruthy();
  });
});

describe('drawStrip', () => {
  it('draws one cell per value, left to right', () => {
    const { ctx, calls } = fakeCtx();
    drawStrip(ctx, [0.1, 0.5, -0.3], 300, 20);
    expect(calls).toHaveLength(3);
    expect(calls[0]!.x).toBeCloseTo(0, 6);
    expect(calls[1]!.x).toBeCloseTo(100, 6);
    expect(calls[2]!.x).toBeCloseTo(200, 6);
  });

  it('colours the magnitude ramp when unsigned', () => {
    const { ctx, calls } = fakeCtx();
    drawStrip(ctx, [0, 1], 200, 20);
    // heatColour(0) and heatColour(1) are the two ends of the monotone ramp — different, and
    // neither of them the signed cyan/amber.
    expect(calls[0]!.colour).not.toBe(calls[1]!.colour);
    expect(calls[0]!.colour).not.toContain('78, 168, 196');
  });

  it('colours cyan/amber when signed', () => {
    const { ctx, calls } = fakeCtx();
    drawStrip(ctx, [-1, 1], 200, 20, { signed: true });
    expect(calls[0]!.colour).toContain('78, 168, 196');
    expect(calls[1]!.colour).toContain('233, 161, 59');
  });

  it('marks a non-finite value distinctly rather than silently drawing nothing', () => {
    const { ctx, calls } = fakeCtx();
    drawStrip(ctx, [1, NaN, -1], 300, 20, { signed: true });
    expect(calls[1]!.colour).toBe('#d9625c');
  });

  it('accepts an external peak so two strips can share one scale', () => {
    const { ctx, calls } = fakeCtx();
    drawStrip(ctx, [0.1], 100, 20, { peak: 10 });
    const { ctx: ctx2, calls: calls2 } = fakeCtx();
    drawStrip(ctx2, [0.1], 100, 20); // self-scaled: 0.1 is the peak, so full colour
    expect(calls[0]!.colour).not.toBe(calls2[0]!.colour);
  });

  it('survives an empty vector', () => {
    const { ctx, calls } = fakeCtx();
    expect(() => drawStrip(ctx, [], 100, 20)).not.toThrow();
    expect(calls).toHaveLength(0);
  });
});

describe('drawTiles', () => {
  it('lays out units as rows and inputs as columns — no transpose', () => {
    /*
     * The same claim as the hand-checked forward-pass test in forward.test.ts, one layer up:
     * W is units × inputs, row-major, and the weight-matrix view has to walk it exactly that
     * way or it draws a transposed picture that still looks like a heatmap.
     */
    const { ctx, calls } = fakeCtx();
    // 2 units, 3 inputs
    drawTiles(ctx, [1, 2, 3, 4, 5, 6], 2, 3, 300, 200);
    expect(calls).toHaveLength(6);
    // Row 0 (unit 0) occupies y ≈ 0; row 1 (unit 1) occupies y ≈ 100.
    const rowYs = calls.map((c) => Math.round(c.y));
    expect(new Set(rowYs.slice(0, 3))).toEqual(new Set([0]));
    expect(new Set(rowYs.slice(3, 6))).toEqual(new Set([100]));
    // Three distinct columns.
    const colXs = new Set(calls.slice(0, 3).map((c) => Math.round(c.x)));
    expect(colXs.size).toBe(3);
  });

  it('normalises against the largest magnitude in the whole matrix, not per row', () => {
    const { ctx, calls } = fakeCtx();
    // Row 0 has a tiny weight; row 1 has the largest. If normalised per row, cell 0 would be
    // drawn at full saturation for a weight of 0.01 — misleadingly "large".
    drawTiles(ctx, [0.01, 0, 10, 0], 2, 2, 200, 200);
    const alphaOf = (s: string) => Number(s.split(',')[3]!.replace(')', ''));
    expect(alphaOf(calls[0]!.colour)).toBeLessThan(0.2);
  });

  it('survives a degenerate shape', () => {
    const { ctx, calls } = fakeCtx();
    expect(() => drawTiles(ctx, [], 0, 0, 100, 100)).not.toThrow();
    expect(calls).toHaveLength(0);
  });
});
