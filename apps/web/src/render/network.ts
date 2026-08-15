/**
 * The network, drawn: edges coloured by signed weight, nodes filled by activation.
 *
 * Canvas rather than SVG, and not prematurely — from slice 4 this repaints on every worker
 * report, several times a second, and 320 SVG nodes restyled at that rate spend more time in
 * style recalculation than the network spends training.
 */

import { maxAbsWeight, type Dense, type Net, type Scratch } from '@neurallab/mlp';
import { layoutNetwork, type GraphLayout } from './graph-layout.ts';

const PANEL = '#16151f';
const LINE = '#2c2a3a';
const FAINT = '#5c5871';
const TEXT = '#e4e2ec';
const AMBER = '#e9a13b';

/**
 * Signed weight to colour — the diverging ramp from §7.
 *
 * Cyan for negative, amber for positive, and the panel background at zero, so a dead weight is
 * invisible. That is the correct impression: a weight of zero contributes nothing, and drawing
 * it as a visible grey line implies a connection that is not doing anything.
 */
function edgeColour(w: number, max: number): string {
  const t = max > 0 ? Math.max(-1, Math.min(1, w / max)) : 0;
  const m = Math.abs(t);
  const alpha = 0.06 + m * 0.84;
  return t < 0 ? `rgba(78, 168, 196, ${alpha.toFixed(3)})` : `rgba(233, 161, 59, ${alpha.toFixed(3)})`;
}

/** Unsigned magnitude to colour — the monotone-luminance ramp, indigo → cyan → amber. */
export function heatColour(t: number): string {
  const u = Math.max(0, Math.min(1, t));
  if (u < 0.5) {
    const v = u / 0.5;
    return `rgb(${Math.round(40 + v * 38)}, ${Math.round(38 + v * 130)}, ${Math.round(90 + v * 106)})`;
  }
  const v = (u - 0.5) / 0.5;
  return `rgb(${Math.round(78 + v * 155)}, ${Math.round(168 - v * 7)}, ${Math.round(196 - v * 137)})`;
}

export interface NetworkView {
  readonly layout: GraphLayout;
  readonly maxWeight: number;
  readonly edges: number;
}

export interface DrawNetworkOptions {
  /** Highlight every edge into this node, as `[layer, unit]`. Layer 0 is the input. */
  readonly focus?: [number, number] | null;
  readonly labels?: readonly string[];
}

export function drawNetwork(
  ctx: CanvasRenderingContext2D,
  net: Net,
  input: ArrayLike<number>,
  scratch: Scratch,
  width: number,
  height: number,
  opts: DrawNetworkOptions = {},
): NetworkView {
  const shape = [net.layers[0]?.inputs ?? 0, ...net.layers.map((l) => l.units)];
  const layout = layoutNetwork(shape, width, height);
  const maxWeight = maxAbsWeight(net);
  const focus = opts.focus ?? null;

  ctx.clearRect(0, 0, width, height);

  // Edges first, so nodes sit on top of them rather than being crossed by them.
  let edges = 0;
  for (let l = 0; l < net.layers.length; l++) {
    const layer = net.layers[l] as Dense;
    const from = layout.cols[l];
    const to = layout.cols[l + 1];
    if (!from || !to) continue;

    for (let u = 0; u < layer.units; u++) {
      const row = u * layer.inputs;
      // A focused node dims everything that is not wired into it. Alpha rather than hiding,
      // because the shape of the rest of the network is still the context for the one edge.
      const dim = focus !== null && !(focus[0] === l + 1 && focus[1] === u);
      for (let i = 0; i < layer.inputs; i++) {
        const w = layer.W[row + i] as number;
        ctx.beginPath();
        ctx.moveTo(from.x, from.ys[i] as number);
        ctx.lineTo(to.x, to.ys[u] as number);
        ctx.strokeStyle = edgeColour(w, maxWeight);
        ctx.lineWidth = 0.4 + (maxWeight > 0 ? Math.abs(w) / maxWeight : 0) * 1.6;
        ctx.globalAlpha = dim ? 0.16 : 1;
        ctx.stroke();
        edges++;
      }
    }
  }
  ctx.globalAlpha = 1;

  // Nodes. The input column is drawn from `input`; every other column from the forward pass.
  for (let l = 0; l < layout.cols.length; l++) {
    const col = layout.cols[l];
    if (!col) continue;
    const values = l === 0 ? input : (scratch.a[l - 1] as Float64Array);
    const scale = l === 0 ? magnitude(input) : magnitude(values);

    for (let i = 0; i < col.ys.length; i++) {
      const v = (values[i] as number) ?? 0;
      const t = scale > 0 ? Math.abs(v) / scale : 0;
      const isFocus = focus !== null && focus[0] === l && focus[1] === i;

      ctx.beginPath();
      ctx.arc(col.x, col.ys[i] as number, layout.nodeRadius + (isFocus ? 1.6 : 0), 0, Math.PI * 2);
      ctx.fillStyle = Number.isFinite(v) ? heatColour(t * 0.92) : '#d9625c';
      ctx.fill();
      ctx.lineWidth = isFocus ? 2 : 1;
      ctx.strokeStyle = isFocus ? TEXT : t > 0.72 ? AMBER : LINE;
      ctx.stroke();
    }
  }

  drawLabels(ctx, layout, shape, net, height, opts.labels);
  return { layout, maxWeight, edges };
}

function drawLabels(
  ctx: CanvasRenderingContext2D,
  layout: GraphLayout,
  shape: readonly number[],
  net: Net,
  height: number,
  labels: readonly string[] | undefined,
): void {
  ctx.font = '9px "Cascadia Mono", Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';

  for (let l = 0; l < layout.cols.length; l++) {
    const col = layout.cols[l];
    if (!col) continue;
    const name =
      labels?.[l] ??
      (l === 0
        ? 'input'
        : l === layout.cols.length - 1
          ? 'output'
          : `hidden ${l}`);
    const act = l === 0 ? '' : ` · ${(net.layers[l - 1] as Dense).act}`;

    ctx.fillStyle = FAINT;
    ctx.fillText(`${name.toUpperCase()} ${shape[l]}${act.toUpperCase()}`, col.x, height - 12);
  }
}

/** Largest absolute value, so node fill normalises against the layer's own range. */
function magnitude(v: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < v.length; i++) {
    const a = Math.abs(v[i] as number);
    if (Number.isFinite(a) && a > max) max = a;
  }
  return max;
}

/**
 * The "graph is the wrong drawing now" notice — §7's stated limit, announced not silent.
 *
 * This used to promise "weight matrices arrive in slice 7" — true when §1 was written, wrong
 * since slice 7 shipped histograms instead and left the graph's own replacement-at-scale still
 * unbuilt. §6's rule about fixed strings going stale applies to this one too; it now says only
 * what is true today.
 */
export function drawOverCapNotice(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, height / 2 - 26, width, 52);
  ctx.font = '12px Bahnschrift, "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = TEXT;
  ctx.textAlign = 'center';
  ctx.fillText('Too many units to draw as a graph.', width / 2, height / 2 - 4);
  ctx.font = '10px "Cascadia Mono", Consolas, monospace';
  ctx.fillStyle = FAINT;
  ctx.fillText('Keep every layer at 24 units or fewer — the weight panels below still show this one.',
    width / 2, height / 2 + 14);
}
