/**
 * The activation functions, and — from slice 2 — their derivatives immediately below them.
 *
 * There is no `Activation` class with a virtual `apply()`. The whole point of this project is
 * that a reader can put a finger on `1 - a * a` and see it is the derivative of `tanh`, so the
 * forward and backward halves of each function live next to each other in one file and are
 * dispatched by a `switch` that the compiler checks is exhaustive.
 */

export type Activation = 'relu' | 'tanh' | 'sigmoid' | 'linear' | 'softmax';

export const ACTIVATIONS: readonly Activation[] = ['relu', 'tanh', 'sigmoid', 'linear', 'softmax'];

export function isActivation(v: string): v is Activation {
  return (ACTIVATIONS as readonly string[]).includes(v);
}

/**
 * Apply an activation to `z`, writing into `a`. May be the same buffer.
 *
 * `softmax` is the one that reads all of `z` rather than working element by element, which is
 * why this takes vectors rather than scalars.
 */
export function activate(kind: Activation, z: Float64Array, a: Float64Array): void {
  switch (kind) {
    case 'relu':
      for (let i = 0; i < z.length; i++) a[i] = Math.max(0, z[i] as number);
      return;
    case 'tanh':
      for (let i = 0; i < z.length; i++) a[i] = Math.tanh(z[i] as number);
      return;
    case 'sigmoid':
      for (let i = 0; i < z.length; i++) a[i] = 1 / (1 + Math.exp(-(z[i] as number)));
      return;
    case 'linear':
      for (let i = 0; i < z.length; i++) a[i] = z[i] as number;
      return;
    case 'softmax':
      softmax(z, a);
      return;
  }
}

/**
 * Softmax, shifted by the maximum before exponentiating.
 *
 * The shift is not an optimisation and it is not optional. `Math.exp(800)` is `Infinity`, and
 * `Infinity / Infinity` is `NaN` — so an unshifted softmax turns the whole network's output into
 * NaN the first time a logit gets large. Subtracting the maximum is algebraically a no-op,
 * because the shift cancels between the numerator and the denominator.
 *
 * This is what makes challenge 3 *legible* rather than merely broken. At a destructive learning
 * rate the weights reach ~1e5 within a few hundred steps (measured), so the logits do too. With
 * the shift the output collapses to a clean one-hot and the reader watches accuracy fall to
 * chance; without it, every panel would fill with NaN and there would be nothing to read.
 * Diverging is the lesson; NaN everywhere is a bug.
 */
export function softmax(z: Float64Array, a: Float64Array): void {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) if ((z[i] as number) > max) max = z[i] as number;

  // An all-Infinity or all-NaN input has no usable maximum. Fall back to a uniform
  // distribution rather than propagating NaN into every downstream panel.
  if (!Number.isFinite(max)) {
    a.fill(1 / Math.max(1, a.length));
    return;
  }

  let sum = 0;
  for (let i = 0; i < z.length; i++) {
    const e = Math.exp((z[i] as number) - max);
    a[i] = e;
    sum += e;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < a.length; i++) a[i] = (a[i] as number) * inv;
}

/** Human label, for panel headers and the stepper. */
export function activationLabel(kind: Activation): string {
  return kind;
}
