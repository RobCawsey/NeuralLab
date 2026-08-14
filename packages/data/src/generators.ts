/**
 * Dataset generators. Everything is produced from a seed; nothing is fetched.
 *
 * Slice 0 ships **two moons** and nothing else, on purpose. It is the default set for the whole
 * project, it is the one the golden test in slice 2 will be pinned against, and one generator
 * is enough to build the chassis around. The other five arrive in slice 3, where the decision
 * field gives them something to be different *for*.
 */

import { Rng, type Dataset } from '@neurallab/core';

export interface MoonsOptions {
  /** Total samples, split as evenly as possible between the two classes. */
  readonly n?: number;
  /** Standard deviation of the gaussian jitter added to each point. */
  readonly noise?: number;
  readonly seed?: number;
}

/**
 * Two interleaved half-circles — the project's default problem.
 *
 * It earns that place by being the smallest dataset that is obviously not linearly separable
 * while still being obviously *separable*: a reader can see the boundary that ought to exist
 * before the network finds it, which is what makes watching it get found worth doing.
 *
 * Geometry, fixed rather than parameterised: an upper arc of radius 1 centred at the origin,
 * and a lower arc shifted right by 1 and down by 0.5. That offset is what interleaves them;
 * with no offset the two arcs are separable by a straight line and the set teaches nothing.
 */
export function moons(opts: MoonsOptions = {}): Dataset {
  const n = Math.max(4, Math.floor(opts.n ?? 240));
  const noise = opts.noise ?? 0.15;
  const rng = new Rng(opts.seed ?? 4417);

  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);
  const upper = Math.ceil(n / 2);

  for (let i = 0; i < n; i++) {
    const isUpper = i < upper;
    // Deterministic angle plus jitter, rather than a random angle: an evenly swept arc looks
    // like an arc at n = 40, where 40 uniform draws look like a scatter with a hole in it.
    const t = isUpper ? i / Math.max(1, upper - 1) : (i - upper) / Math.max(1, n - upper - 1);
    const angle = t * Math.PI;

    let px: number;
    let py: number;
    if (isUpper) {
      px = Math.cos(angle);
      py = Math.sin(angle);
    } else {
      px = 1 - Math.cos(angle);
      py = 0.5 - Math.sin(angle);
    }

    x[i * 2] = px + rng.normal() * noise;
    x[i * 2 + 1] = py + rng.normal() * noise;
    y[i] = isUpper ? 0 : 1;
  }

  return {
    name: 'Two moons',
    x,
    y,
    n,
    dim: 2,
    classes: 2,
    featureNames: ['x', 'y'],
    classNames: ['upper', 'lower'],
  };
}

/**
 * Four clusters, one per quadrant, labelled by the sign of `x · y`.
 *
 * Arrives in slice 2 rather than slice 3 because it is the dataset challenge 1 is built on:
 * it is the smallest problem where a network with no hidden layer provably cannot do better
 * than chance, and where adding one layer fixes it completely. Two moons is *harder* but its
 * failure is a matter of degree; XOR's is absolute, and a reader watching accuracy sit at 0.50
 * while the loss refuses to fall learns something a 0.83 does not teach.
 */
export function xor(opts: MoonsOptions = {}): Dataset {
  const n = Math.max(4, Math.floor(opts.n ?? 240));
  const noise = opts.noise ?? 0.15;
  const rng = new Rng(opts.seed ?? 4417);

  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    // Cycled rather than drawn at random, so the four quadrants stay balanced at any n. A
    // random quadrant would leave a 40-sample set visibly lopsided and the lesson arguable.
    const quadrant = i % 4;
    const cx = quadrant === 0 || quadrant === 3 ? 1 : -1;
    const cy = quadrant === 0 || quadrant === 1 ? 1 : -1;
    x[i * 2] = cx + rng.normal() * noise;
    x[i * 2 + 1] = cy + rng.normal() * noise;
    y[i] = cx * cy > 0 ? 0 : 1;
  }

  return {
    name: 'XOR',
    x,
    y,
    n,
    dim: 2,
    classes: 2,
    featureNames: ['x', 'y'],
    classNames: ['same sign', 'opposite'],
  };
}

/** Every generator the app can offer, by key. Slice 3 fills this out. */
export const GENERATORS = {
  moons: { label: 'Two moons', build: moons },
  xor: { label: 'XOR', build: xor },
} as const;

export type GeneratorKey = keyof typeof GENERATORS;

export function isGeneratorKey(k: string): k is GeneratorKey {
  return Object.prototype.hasOwnProperty.call(GENERATORS, k);
}
