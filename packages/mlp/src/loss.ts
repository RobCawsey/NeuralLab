/**
 * How wrong the network was, and the first derivative of that answer.
 *
 * Targets are a **class index**, not a one-hot vector. Every problem in this project is
 * classification, an index is what the Dataset already holds, and building a one-hot vector per
 * sample would allocate in the inner loop to store a number we already have.
 */

import type { Activation } from './activations.ts';

export type LossKind = 'mse' | 'crossEntropy';

/**
 * The loss for one sample.
 *
 * Cross-entropy is `-log(p)` of the probability assigned to the correct class, and nothing
 * else — the other outputs enter only through softmax having normalised them.
 */
export function sampleLoss(kind: LossKind, output: Float64Array, target: number): number {
  if (kind === 'crossEntropy') {
    // A confident wrong answer drives p to 0 and the loss to Infinity, which then poisons every
    // mean it is averaged into. The clamp caps one sample's contribution at ~27.6 instead.
    const p = Math.max(1e-12, output[target] as number);
    return -Math.log(p);
  }
  // 0.5 * sum of squares. The half is there so the derivative is exactly (a − t) with no
  // stray factor of two for a reader to wonder about.
  let total = 0;
  for (let i = 0; i < output.length; i++) {
    const t = i === target ? 1 : 0;
    const d = (output[i] as number) - t;
    total += d * d;
  }
  return 0.5 * total;
}

/**
 * `dL/dz` at the **output layer**, written into `dz`.
 *
 * This is the one place the loss and the activation are considered together, and it is worth
 * saying why rather than leaving it as a coincidence.
 *
 * For softmax with cross-entropy the general chain rule would need the full softmax Jacobian —
 * an n × n matrix per sample, because every output depends on every logit. Multiply it by
 * `dL/da` and almost everything cancels, leaving `dz = a − onehot(t)`. So the fused form is not
 * a shortcut around the maths; it *is* the maths, already simplified. A test asserts it matches
 * a numerical gradient, because "it cancels" is exactly the kind of claim that is easy to
 * believe and easy to get slightly wrong.
 *
 * For an element-wise output activation with MSE there is no cancellation and the two factors
 * are applied separately, which is what makes the softmax case look as special as it is.
 */
export function outputDelta(
  kind: LossKind,
  act: Activation,
  output: Float64Array,
  target: number,
  dz: Float64Array,
): void {
  if (act === 'softmax') {
    if (kind !== 'crossEntropy') {
      // Reachable only by editing code, not by using the app. Better a loud error than a
      // silently wrong gradient that still trains.
      throw new Error('softmax is only paired with cross-entropy');
    }
    for (let i = 0; i < output.length; i++) {
      dz[i] = (output[i] as number) - (i === target ? 1 : 0);
    }
    return;
  }

  // dL/da for MSE, then × the activation's own derivative.
  for (let i = 0; i < output.length; i++) {
    const a = output[i] as number;
    const dA = a - (i === target ? 1 : 0);
    dz[i] = dA * derivativeFromOutput(act, a);
  }
}

/**
 * An activation's derivative, expressed in terms of its **output** rather than its input.
 *
 * Every activation here has that property, and it is why the backward pass needs only `a` and
 * never re-reads `z`: `tanh' = 1 − a²`, `sigmoid' = a(1 − a)`, `relu' = a > 0`. One buffer less
 * to keep, and the formula a reader recognises from the textbook.
 */
export function derivativeFromOutput(act: Activation, a: number): number {
  switch (act) {
    case 'relu':
      // Undefined at exactly zero; the convention is 0, and it is what makes a dead relu dead.
      return a > 0 ? 1 : 0;
    case 'tanh':
      return 1 - a * a;
    case 'sigmoid':
      return a * (1 - a);
    case 'linear':
      return 1;
    case 'softmax':
      throw new Error('softmax has no element-wise derivative — see outputDelta');
  }
}
