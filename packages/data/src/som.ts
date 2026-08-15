/**
 * The SOM half's own generators. Kept apart from `generators.ts`, whose header comment describes
 * six sets ordered by what they demand of an MLP — a 3-dimensional, unlabelled set answering to
 * neither claim would only clutter that story. `Dataset` is still the one shape both networks
 * share; this file just produces one nobody standardises before the map sees it.
 */

import { Rng, type Dataset } from '@neurallab/core';

export interface ColourCubeOptions {
  readonly n?: number;
  readonly seed?: number;
}

/**
 * `n` points drawn uniformly from the RGB unit cube — the classic Kohonen demonstration.
 *
 * It earns that place twice over: it is the demo everyone has seen, and it is the only dataset
 * in the project where a node's weight vector can be drawn *as itself* rather than through a
 * projection — three weights are literally a colour. Every other SOM visualisation is a way of
 * recovering that same legibility for data that is not already a colour. Unlabelled throughout:
 * a self-organising map is trained on data it is never told the answer for, and there is no
 * answer here to hold back.
 */
export function colourCube(opts: ColourCubeOptions = {}): Dataset {
  const n = Math.max(1, Math.floor(opts.n ?? 1500));
  const rng = new Rng(opts.seed ?? 4417);

  const x = new Float32Array(n * 3);
  for (let i = 0; i < x.length; i++) x[i] = rng.float();

  return {
    name: 'Colour cube',
    x,
    y: null,
    n,
    dim: 3,
    classes: 0,
    featureNames: ['r', 'g', 'b'],
    classNames: [],
  };
}
