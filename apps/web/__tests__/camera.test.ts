/**
 * The only app-level test in slice 0, and it exists because `render/camera.ts` deliberately
 * imports nothing — no DOM, no canvas context. If this file ever becomes impossible to write,
 * the render layer has stopped being separable from the drawing surface.
 */

import { describe, expect, it } from 'vitest';
import { fitCamera, padBox, sx, sy, wx, wy, type Box } from '../src/render/camera.ts';

const unit: Box = { minX: -1, maxX: 1, minY: -1, maxY: 1 };

describe('camera', () => {
  it('round-trips a point back to itself', () => {
    const cam = fitCamera(unit, 400, 300);
    for (const [x, y] of [[0, 0], [0.5, -0.25], [-1, 1], [0.99, 0.01]] as const) {
      expect(wx(cam, sx(cam, x))).toBeCloseTo(x, 9);
      expect(wy(cam, sy(cam, y))).toBeCloseTo(y, 9);
    }
  });

  it('flips y and only y', () => {
    // World is y-up, canvas is y-down. This is the project's single sign flip; a second one
    // anywhere cancels this and nobody notices until a boundary is drawn upside down.
    const cam = fitCamera(unit, 400, 300);
    expect(sy(cam, 1)).toBeLessThan(sy(cam, -1));
    expect(sx(cam, 1)).toBeGreaterThan(sx(cam, -1));
  });

  it('uses one scale for both axes', () => {
    /*
     * Two moons stretched to fill a 3:2 panel is not two moons — distances stop being
     * comparable between the axes, and the decision boundary slice 3 draws over it would be a
     * boundary in a space the network never saw. Letterboxing is correct and costs some pixels.
     */
    const cam = fitCamera(unit, 900, 300);
    const spanX = sx(cam, 1) - sx(cam, -1);
    const spanY = sy(cam, -1) - sy(cam, 1);
    expect(spanX).toBeCloseTo(spanY, 6);
  });

  it('fits inside the padded plot on either aspect', () => {
    for (const [w, h] of [[900, 300], [300, 900], [500, 500]] as const) {
      const cam = fitCamera(unit, w, h, 22);
      for (const [x, y] of [[-1, -1], [1, 1], [-1, 1], [1, -1]] as const) {
        expect(sx(cam, x)).toBeGreaterThanOrEqual(21);
        expect(sx(cam, x)).toBeLessThanOrEqual(w - 21);
        expect(sy(cam, y)).toBeGreaterThanOrEqual(21);
        expect(sy(cam, y)).toBeLessThanOrEqual(h - 21);
      }
    }
  });

  it('centres the world box in the canvas', () => {
    const cam = fitCamera({ minX: 4, maxX: 6, minY: 10, maxY: 12 }, 400, 300);
    expect(sx(cam, 5)).toBeCloseTo(200, 6);
    expect(sy(cam, 11)).toBeCloseTo(150, 6);
  });

  it('survives a degenerate box rather than dividing by zero', () => {
    const cam = fitCamera({ minX: 2, maxX: 2, minY: 2, maxY: 2 }, 400, 300);
    expect(Number.isFinite(sx(cam, 2))).toBe(true);
    expect(Number.isFinite(sy(cam, 2))).toBe(true);
  });

  it('pads a box by a fraction of its own span', () => {
    const padded = padBox({ minX: 0, maxX: 10, minY: 0, maxY: 2 }, 0.1);
    expect(padded.minX).toBeCloseTo(-1, 9);
    expect(padded.maxX).toBeCloseTo(11, 9);
    // Per-axis, so a wide flat box does not get a square margin.
    expect(padded.minY).toBeCloseTo(-0.2, 9);
    expect(padded.maxY).toBeCloseTo(2.2, 9);
  });
});
