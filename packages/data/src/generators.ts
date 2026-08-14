/**
 * Dataset generators. Everything is produced from a seed; nothing is fetched.
 *
 * Slice 0 shipped **two moons** alone, and slice 2 added **XOR** because challenge 1 needs it.
 * Slice 3 completes the set, now that the decision field gives them something to be different
 * *for*: until there was a boundary to draw, six datasets were six scatter plots.
 *
 * They are ordered by what they demand of the network, and the order is the point:
 *
 *   xor          four clusters; one hidden layer, and a hard ceiling without one
 *   moons        one smooth curve
 *   circles      a closed boundary — no half-plane can do it at all
 *   blobs        three classes, so softmax has more than two things to weigh
 *   checkerboard sixteen alternating cells; capacity, not cleverness
 *   spirals      the one that needs both capacity and patience
 */

import { Rng, type Dataset } from '@neurallab/core';

export interface GeneratorOptions {
  /** Total samples, split as evenly as possible between the classes. */
  readonly n?: number;
  /** Standard deviation of the gaussian jitter added to each point. */
  readonly noise?: number;
  readonly seed?: number;
}

/** Retained under its old name, because slice 0 and slice 2 both import it. */
export type MoonsOptions = GeneratorOptions;

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

/**
 * Two concentric rings.
 *
 * The one where the *shape* of the answer is obviously not a half-plane. Moons can be separated
 * by a sufficiently bent line and a reader may not notice the bend is essential; a ring encloses,
 * and no amount of tilting a straight boundary encloses anything. It is also the cheapest way to
 * see a decision field do something a linear model provably cannot.
 */
export function circles(opts: GeneratorOptions = {}): Dataset {
  const n = Math.max(4, Math.floor(opts.n ?? 240));
  const noise = opts.noise ?? 0.15;
  const rng = new Rng(opts.seed ?? 4417);

  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);
  const inner = Math.ceil(n / 2);

  for (let i = 0; i < n; i++) {
    const isInner = i < inner;
    // The angle is swept rather than drawn, for the same reason as moons: 40 uniform draws
    // look like a scatter with a hole in it, and a swept arc looks like an arc.
    const k = isInner ? i : i - inner;
    const count = isInner ? inner : n - inner;
    const angle = (k / Math.max(1, count)) * Math.PI * 2;
    const radius = isInner ? 0.4 : 1;

    x[i * 2] = Math.cos(angle) * radius + rng.normal() * noise;
    x[i * 2 + 1] = Math.sin(angle) * radius + rng.normal() * noise;
    y[i] = isInner ? 0 : 1;
  }

  return {
    name: 'Concentric circles',
    x, y, n, dim: 2, classes: 2,
    featureNames: ['x', 'y'],
    classNames: ['inner', 'outer'],
  };
}

/**
 * Three gaussian clusters — the only default set with more than two classes.
 *
 * It exists to exercise the multi-class path end to end: softmax over three outputs, three
 * colours in the legend, three bars in the output panel, and a decision field that has to pick a
 * winner rather than shade one number. Two-class problems hide a whole family of bugs — an
 * `argmax` that returns the last index on a tie, a field that assumes `p` and `1 − p`.
 */
export function blobs(opts: GeneratorOptions = {}): Dataset {
  const n = Math.max(3, Math.floor(opts.n ?? 240));
  const noise = opts.noise ?? 0.15;
  const rng = new Rng(opts.seed ?? 4417);
  const classes = 3;

  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const c = i % classes;
    const angle = (c / classes) * Math.PI * 2;
    // Spread 1.0 against a jitter of 0.15σ: far enough apart to be three clusters, close
    // enough that the boundaries between them are somewhere a network has to decide.
    x[i * 2] = Math.cos(angle) + rng.normal() * noise * 2.2;
    x[i * 2 + 1] = Math.sin(angle) + rng.normal() * noise * 2.2;
    y[i] = c;
  }

  return {
    name: 'Three blobs',
    x, y, n, dim: 2, classes,
    featureNames: ['x', 'y'],
    classNames: ['one', 'two', 'three'],
  };
}

/**
 * A 4 × 4 checkerboard.
 *
 * Sixteen alternating cells, which is capacity rather than cleverness: no clever architecture
 * gets this with four hidden units, and a wide enough one gets it without any insight at all.
 * That is the lesson, and it is why this sits next to spirals rather than next to moons.
 *
 * The class comes from the **pre-jitter** cell, then the point moves. Computing it after the
 * jitter would make the boundaries perfectly sharp however high the noise went, and the noise
 * slider would do nothing visible — which is the wrong lesson about noise.
 */
export function checkerboard(opts: GeneratorOptions = {}): Dataset {
  const n = Math.max(4, Math.floor(opts.n ?? 240));
  const noise = opts.noise ?? 0.15;
  const rng = new Rng(opts.seed ?? 4417);

  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const px = rng.range(-2, 2);
    const py = rng.range(-2, 2);
    y[i] = (((Math.floor(px) + Math.floor(py)) % 2) + 2) % 2;
    x[i * 2] = px + rng.normal() * noise;
    x[i * 2 + 1] = py + rng.normal() * noise;
  }

  return {
    name: 'Checkerboard',
    x, y, n, dim: 2, classes: 2,
    featureNames: ['x', 'y'],
    classNames: ['light', 'dark'],
  };
}

/**
 * Two interleaved Archimedean spirals — the hardest set here, and deliberately so.
 *
 * It needs capacity *and* patience: the boundary between the arms is thin everywhere and gets
 * thinner toward the centre, so a network that is large enough still takes thousands of steps.
 * It is the set that makes the learning-rate and step-count sliders mean something, because it
 * is the only one where the default 400 steps visibly is not enough.
 */
export function spirals(opts: GeneratorOptions = {}): Dataset {
  const n = Math.max(4, Math.floor(opts.n ?? 400));
  const noise = opts.noise ?? 0.15;
  const rng = new Rng(opts.seed ?? 4417);

  const x = new Float32Array(n * 2);
  const y = new Int32Array(n);
  const perArm = Math.ceil(n / 2);

  for (let i = 0; i < n; i++) {
    const isFirst = i < perArm;
    const k = isFirst ? i : i - perArm;
    const count = isFirst ? perArm : n - perArm;
    // 1.75 turns. More and the arms crowd past what 240 samples can describe; fewer and it
    // stops being harder than moons.
    const t = (k / Math.max(1, count - 1)) * Math.PI * 3.5;
    const radius = t / (Math.PI * 3.5);
    const angle = t + (isFirst ? 0 : Math.PI);

    // Noise scaled by radius: a fixed jitter obliterates the centre, where the arms are
    // closest together, and leaves the outside untouched.
    const jitter = noise * (0.25 + radius * 0.75);
    x[i * 2] = Math.cos(angle) * radius + rng.normal() * jitter;
    x[i * 2 + 1] = Math.sin(angle) * radius + rng.normal() * jitter;
    y[i] = isFirst ? 0 : 1;
  }

  return {
    name: 'Two spirals',
    x, y, n, dim: 2, classes: 2,
    featureNames: ['x', 'y'],
    classNames: ['arm A', 'arm B'],
  };
}

/**
 * Every generator the app can offer, in the order they get harder.
 *
 * **`steps` is what this set needs, not a preference**, and it was measured rather than guessed.
 * Two moons is done in 400; the checkerboard is at 0.66 after 4 000 and 0.88 after 20 000, and
 * spirals behaves the same way. Without this the two hardest sets would open at a step count that
 * cannot solve them, and a reader would read that as the app being broken rather than the problem
 * being hard. Selecting a dataset adopts its figure; the slider then overrides it freely, and a
 * challenge card that wants to *demonstrate* too few steps sets a low one on purpose.
 */
export const GENERATORS = {
  xor: { label: 'XOR', build: xor, steps: 1200 },
  moons: { label: 'Two moons', build: moons, steps: 400 },
  circles: { label: 'Concentric circles', build: circles, steps: 800 },
  blobs: { label: 'Three blobs', build: blobs, steps: 600 },
  checkerboard: { label: 'Checkerboard', build: checkerboard, steps: 20000 },
  spirals: { label: 'Two spirals', build: spirals, steps: 20000 },
} as const;

export type GeneratorKey = keyof typeof GENERATORS;

export function isGeneratorKey(k: string): k is GeneratorKey {
  return Object.prototype.hasOwnProperty.call(GENERATORS, k);
}
