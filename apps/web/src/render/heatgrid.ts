/**
 * One value per lattice node, as a heatmap — the U-matrix and the component planes both use this.
 * §7's drawing table lists them as one technique for a reason: an `ImageData` blit at grid
 * resolution scaled up with smoothing off, the same trick `field.ts` uses for the decision field.
 *
 * Unsigned quantities — a U-matrix distance, one weight of a component plane — use a single
 * monotone ramp from the panel background up to one accent colour, normalised to the grid's own
 * maximum. §7 of the design document.
 */

const PANEL: readonly [number, number, number] = [22, 21, 31];

let scratchCanvas: HTMLCanvasElement | null = null;

export function drawHeatgrid(
  ctx: CanvasRenderingContext2D,
  values: ArrayLike<number>,
  cols: number,
  rows: number,
  width: number,
  height: number,
  accent: readonly [number, number, number],
): number {
  ctx.clearRect(0, 0, width, height);
  if (cols < 1 || rows < 1) return 0;

  let max = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (Number.isFinite(v) && v > max) max = v;
  }

  const canvas = (scratchCanvas ??= document.createElement('canvas'));
  if (canvas.width !== cols || canvas.height !== rows) {
    canvas.width = cols;
    canvas.height = rows;
  }
  const off = canvas.getContext('2d');
  if (!off) return max;

  const image = off.createImageData(cols, rows);
  const data = image.data;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const src = row * cols + col;
      const t = max > 0 ? Math.min(1, Math.max(0, (values[src] as number) / max)) : 0;
      const p = src * 4;
      data[p] = PANEL[0] + (accent[0] - PANEL[0]) * t;
      data[p + 1] = PANEL[1] + (accent[1] - PANEL[1]) * t;
      data[p + 2] = PANEL[2] + (accent[2] - PANEL[2]) * t;
      data[p + 3] = 255;
    }
  }
  off.putImageData(image, 0, 0);

  const smoothing = ctx.imageSmoothingEnabled;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(canvas, 0, 0, width, height);
  ctx.imageSmoothingEnabled = smoothing;
  return max;
}
