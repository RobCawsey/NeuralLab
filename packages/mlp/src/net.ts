/**
 * A network, and the forward pass through it.
 *
 * Slice 1 builds and evaluates; nothing here learns. `backward` arrives in slice 2 and lands in
 * its own file beside this one, so that the two halves can be read against each other.
 */

import { Rng } from '@neurallab/core';
import { activate, type Activation } from './activations.ts';
// Owned by loss.ts, which is where the two kinds are actually distinguished.
import type { LossKind } from './loss.ts';

export type { LossKind };

export interface Dense {
  readonly inputs: number;
  readonly units: number;
  readonly act: Activation;
  /**
   * `units * inputs`, row-major with the **unit** as the outer index, so one unit's weights are
   * contiguous. Two things follow, and both were wanted: the forward pass walks `W` linearly,
   * and the weight-matrix heatmap in slice 7 is a straight `ImageData` blit with no transpose.
   */
  W: Float32Array;
  b: Float32Array;
}

export interface Net {
  readonly layers: Dense[];
  readonly loss: LossKind;
}

export type InitScheme = 'he' | 'glorot' | 'small' | 'zeros';

export const INIT_SCHEMES: readonly InitScheme[] = ['he', 'glorot', 'small', 'zeros'];

export function isInitScheme(v: string): v is InitScheme {
  return (INIT_SCHEMES as readonly string[]).includes(v);
}

export interface NetSpec {
  /** Every layer width including input and output: `[2, 8, 8, 2]`. */
  readonly shape: readonly number[];
  /** Activation for the hidden layers. */
  readonly hidden: Activation;
  /** Activation for the output layer. */
  readonly output: Activation;
  readonly loss: LossKind;
}

/** Build a network with zeroed buffers. Call `initialise` to fill them. */
export function createNet(spec: NetSpec): Net {
  if (spec.shape.length < 2) throw new Error('a network needs at least an input and an output');
  const layers: Dense[] = [];
  for (let l = 1; l < spec.shape.length; l++) {
    const inputs = spec.shape[l - 1] as number;
    const units = spec.shape[l] as number;
    if (inputs < 1 || units < 1) throw new Error(`layer ${l} has a zero dimension`);
    layers.push({
      inputs,
      units,
      act: l === spec.shape.length - 1 ? spec.output : spec.hidden,
      W: new Float32Array(units * inputs),
      b: new Float32Array(units),
    });
  }
  return { layers, loss: spec.loss };
}

/**
 * Fill the weights. Biases are always zero, which is the standard choice and one fewer thing
 * for a reader to wonder about.
 *
 * The scheme table is §4 of the design document. `zeros` is offered on purpose: with every
 * weight identical, every hidden unit computes the same thing and receives the same gradient
 * forever, and the network graph shows it directly — every edge into row 2 is the same colour as
 * every edge into row 5, and stays that way. It is the fastest way to make "symmetry breaking"
 * mean something, and it is challenge 5.
 */
export function initialise(net: Net, scheme: InitScheme, rng: Rng): void {
  for (const layer of net.layers) {
    layer.b.fill(0);
    switch (scheme) {
      case 'he': {
        // Keeps activation variance stable through depth for relu, which discards half the
        // distribution — hence 2/n rather than 1/n.
        const sd = Math.sqrt(2 / layer.inputs);
        for (let i = 0; i < layer.W.length; i++) layer.W[i] = rng.normal() * sd;
        break;
      }
      case 'glorot': {
        const limit = Math.sqrt(6 / (layer.inputs + layer.units));
        for (let i = 0; i < layer.W.length; i++) layer.W[i] = rng.range(-limit, limit);
        break;
      }
      case 'small': {
        for (let i = 0; i < layer.W.length; i++) layer.W[i] = rng.range(-0.05, 0.05);
        break;
      }
      case 'zeros': {
        layer.W.fill(0);
        break;
      }
    }
  }
}

/**
 * Reusable buffers for one forward pass.
 *
 * Allocated once and reused because this runs on every pointer move now and tens of thousands of
 * times per second from slice 4. Allocating two arrays per layer per call would put the garbage
 * collector in the inner loop of the whole project.
 *
 * **Float64, not Float32, and the distinction is exactly invariant 3's.** Weights are Float32
 * because they are *state* — they get posted to the main thread and serialised. Activations are
 * *intermediates*; nothing transfers them, and §4 of the design document already says every
 * intermediate is a double. Storing them as Float32 rounds twice per layer for no benefit, and
 * it costs more than it looks: it put ~1e-7 of relative noise on the loss, which made the
 * slice-2 gradient check bottom out at a relative error of 2.5e-3 — plausible enough to be
 * mistaken for a wrong gradient. As doubles the same check agrees to 1e-9.
 */
export interface Scratch {
  /** Pre-activation, per layer. */
  readonly z: Float64Array[];
  /** Post-activation, per layer. */
  readonly a: Float64Array[];
}

export function createScratch(net: Net): Scratch {
  return {
    z: net.layers.map((l) => new Float64Array(l.units)),
    a: net.layers.map((l) => new Float64Array(l.units)),
  };
}

/**
 * One forward pass. Returns the output layer's activations — a view into `scratch`, not a copy.
 *
 * The accumulator is a plain JavaScript number and therefore a double, so each dot product
 * accumulates at full precision and rounds once on store. That is deliberate and it is what the
 * mainstream frameworks do; rounding after every multiply–add would be both slower and less
 * accurate. §4 of the design document.
 */
export function forward(net: Net, x: ArrayLike<number>, scratch: Scratch): Float64Array {
  let input: ArrayLike<number> = x;

  for (let l = 0; l < net.layers.length; l++) {
    const layer = net.layers[l] as Dense;
    const z = scratch.z[l] as Float64Array;
    const a = scratch.a[l] as Float64Array;

    for (let u = 0; u < layer.units; u++) {
      const row = u * layer.inputs;
      let sum = layer.b[u] as number;
      for (let i = 0; i < layer.inputs; i++) {
        sum += (layer.W[row + i] as number) * (input[i] as number);
      }
      z[u] = sum;
    }

    activate(layer.act, z, a);
    input = a;
  }

  return scratch.a[net.layers.length - 1] as Float64Array;
}

/** Index of the largest output — the predicted class. */
export function argmax(v: ArrayLike<number>): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if ((v[i] as number) > bestValue) {
      bestValue = v[i] as number;
      best = i;
    }
  }
  return best;
}

/** Total trainable parameters — weights and biases. Printed next to the sample count. */
export function paramCount(net: Net): number {
  let total = 0;
  for (const l of net.layers) total += l.W.length + l.b.length;
  return total;
}

/** `[2, 8, 8, 2]` — input width followed by every layer's unit count. */
export function shapeOf(net: Net): number[] {
  const first = net.layers[0];
  if (!first) return [];
  return [first.inputs, ...net.layers.map((l) => l.units)];
}

/** `2-8-8-2`, for a panel header and the URL. */
export function describeShape(net: Net): string {
  return shapeOf(net).join('-');
}

/**
 * Largest `|w|` in a layer, or in the whole network.
 *
 * The renderer normalises edge colour against this rather than against a fixed range: at
 * initialisation the weights are all small and a fixed scale would draw a uniformly blank graph,
 * which is exactly the moment the graph has the most to say about symmetry. The consequence —
 * that colour is not comparable between two screenshots — is why the value is printed.
 */
export function maxAbsWeight(net: Net, layer?: number): number {
  let max = 0;
  const layers = layer === undefined ? net.layers : [net.layers[layer] as Dense];
  for (const l of layers) {
    for (let i = 0; i < l.W.length; i++) {
      const v = Math.abs(l.W[i] as number);
      if (v > max) max = v;
    }
  }
  return max;
}

/** Parse `8-8` (hidden widths only) from a control or the URL. Empty means no hidden layer. */
export function parseHidden(text: string, maxUnits = 64, maxLayers = 6): number[] {
  if (text.trim() === '') return [];
  const parts = text.split(/[-,\s]+/).filter((p) => p !== '');
  const out: number[] = [];
  for (const p of parts.slice(0, maxLayers)) {
    const v = Number.parseInt(p, 10);
    if (Number.isFinite(v) && v >= 1) out.push(Math.min(maxUnits, v));
  }
  return out;
}
