/**
 * Where every node in the network graph goes.
 *
 * Split from the drawing for the same reason `camera.ts` is: it is arithmetic, it decides
 * whether the picture is readable, and it can be checked in Node. The renderer beside this file
 * imports a canvas context; this one imports nothing.
 */

export interface GraphLayout {
  /** One column per layer, including the input layer. */
  readonly cols: readonly { readonly x: number; readonly ys: Float32Array }[];
  readonly nodeRadius: number;
  /** Vertical spacing actually used, after clamping to fit. */
  readonly gap: number;
  /**
   * True when a layer is too wide for the graph to mean anything — §7 of the design document.
   *
   * The limit is stated rather than hoped for. 2-16-16-2 is 320 edges and draws in well under a
   * millisecond; 64-128-128-10 is 25 856 and is not slow so much as *meaningless*, a grey
   * rectangle of overlapping hairlines. Above the cap the centre panel is supposed to switch to
   * weight matrices, and the switch is announced rather than silent.
   */
  readonly overCap: boolean;
}

/** Above this many units in any one layer, the graph stops being the right drawing. */
export const UNIT_CAP = 24;

export function layoutNetwork(
  shape: readonly number[],
  width: number,
  height: number,
  pad = 44,
): GraphLayout {
  const cols: { x: number; ys: Float32Array }[] = [];
  if (shape.length === 0) {
    return { cols, nodeRadius: 0, gap: 0, overCap: false };
  }

  const innerW = Math.max(1, width - pad * 2);
  const innerH = Math.max(1, height - pad * 2);
  const widest = Math.max(...shape);

  // One gap for the whole graph, not one per column. Per-column spacing would draw a 2-unit
  // input layer at the same pitch as a 16-unit hidden layer, and the eye reads that as the two
  // layers being differently scaled rather than differently sized.
  const gap = widest > 1 ? Math.min(30, innerH / (widest - 1)) : 0;
  const nodeRadius = Math.max(2.5, Math.min(8, gap * 0.34));

  const centreY = pad + innerH / 2;
  for (let l = 0; l < shape.length; l++) {
    // A single-layer graph has no span to divide; put it in the middle rather than at pad.
    const x = shape.length === 1 ? pad + innerW / 2 : pad + (l * innerW) / (shape.length - 1);
    const n = shape[l] as number;
    const ys = new Float32Array(n);
    for (let i = 0; i < n; i++) ys[i] = centreY + (i - (n - 1) / 2) * gap;
    cols.push({ x, ys });
  }

  return { cols, nodeRadius, gap, overCap: widest > UNIT_CAP };
}

/** Which node the pointer is over, as `[layer, unit]`, or null. */
export function hitNode(
  layout: GraphLayout,
  px: number,
  py: number,
  slack = 4,
): [number, number] | null {
  const reach = layout.nodeRadius + slack;
  for (let l = 0; l < layout.cols.length; l++) {
    const col = layout.cols[l];
    if (!col) continue;
    if (Math.abs(col.x - px) > reach) continue;
    for (let i = 0; i < col.ys.length; i++) {
      const dy = (col.ys[i] as number) - py;
      if (Math.abs(dy) <= reach) return [l, i];
    }
  }
  return null;
}
