/**
 * World coordinates to canvas pixels, and back.
 *
 * **The y-flip happens here and nowhere else.** Data space is y-up, canvas space is y-down, and
 * a project that flips in two places eventually flips in one of them twice. Every renderer takes
 * a Camera and calls `sx`/`sy`; nothing else multiplies by a scale.
 *
 * This file imports nothing — not from the DOM, not from a canvas context — so the mapping is
 * checked in Node. The moment that stops being possible, the render layer has stopped being
 * separable from the drawing surface.
 */

export interface Box {
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
}

export interface Camera {
  /** Pixel rect the world box is drawn into. */
  readonly plot: { readonly x: number; readonly y: number; readonly w: number; readonly h: number };
  /** World units per pixel — identical on both axes, see `fitCamera`. */
  readonly scale: number;
  readonly centreX: number;
  readonly centreY: number;
}

/**
 * Fit a world box into a canvas, preserving aspect ratio.
 *
 * One scale for both axes rather than one each. Two moons stretched to fill a 3:2 panel is not
 * two moons: distances stop being comparable between the axes, and a reader who then looks at a
 * decision boundary is being shown a boundary in a space the network never saw. Letterboxing is
 * the correct answer and it costs some pixels.
 */
export function fitCamera(box: Box, width: number, height: number, pad = 22): Camera {
  const w = Math.max(1, width - pad * 2);
  const h = Math.max(1, height - pad * 2);
  const spanX = Math.max(1e-6, box.maxX - box.minX);
  const spanY = Math.max(1e-6, box.maxY - box.minY);
  const scale = Math.min(w / spanX, h / spanY);

  return {
    plot: { x: pad, y: pad, w, h },
    scale,
    centreX: (box.minX + box.maxX) / 2,
    centreY: (box.minY + box.maxY) / 2,
  };
}

/** World x to canvas x. */
export function sx(cam: Camera, x: number): number {
  return cam.plot.x + cam.plot.w / 2 + (x - cam.centreX) * cam.scale;
}

/** World y to canvas y. The only sign flip in the project. */
export function sy(cam: Camera, y: number): number {
  return cam.plot.y + cam.plot.h / 2 - (y - cam.centreY) * cam.scale;
}

/** Canvas x back to world x — for hit-testing a pointer. */
export function wx(cam: Camera, px: number): number {
  return cam.centreX + (px - cam.plot.x - cam.plot.w / 2) / cam.scale;
}

/** Canvas y back to world y. */
export function wy(cam: Camera, py: number): number {
  return cam.centreY - (py - cam.plot.y - cam.plot.h / 2) / cam.scale;
}

/** Expand a box by a fraction of its own span, so points do not sit on the frame. */
export function padBox(box: Box, fraction = 0.08): Box {
  const dx = (box.maxX - box.minX) * fraction;
  const dy = (box.maxY - box.minY) * fraction;
  return {
    minX: box.minX - dx,
    maxX: box.maxX + dx,
    minY: box.minY - dy,
    maxY: box.maxY + dy,
  };
}
