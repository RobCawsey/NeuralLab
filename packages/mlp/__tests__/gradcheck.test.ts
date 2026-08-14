/**
 * Gradient check — the test this slice exists to be trusted by.
 *
 * It exists because **a wrong gradient still trains.** A sign error in one term, a missing
 * transpose, or an activation derivative applied one layer too early typically *slows* learning
 * rather than stopping it, so the loss curve still descends, the decision boundary still forms,
 * and nothing on screen looks wrong. Every other test in this file could pass with backprop
 * subtly broken.
 *
 * Finite differences do not care how plausible the curve looks. For each weight independently:
 * nudge it by ±h, measure the loss both times, and compare (L₊ − L₋) / 2h against what the
 * analytic backward pass claimed. They must agree.
 */

import { describe, expect, it } from 'vitest';
import { Rng } from '@neurallab/core';
import {
  createNet,
  createScratch,
  forward,
  initialise,
  paramCount,
  type Net,
} from '../src/net.ts';
import type { Activation } from '../src/activations.ts';
import { backward, createGrads, zeroGrads } from '../src/backward.ts';
import { sampleLoss } from '../src/loss.ts';

/**
 * Central difference, with the **actual stored perturbation** as the denominator.
 *
 * Two decisions, and the second one was learned the hard way.
 *
 * Central rather than one-sided: `(L(w+h) − L(w−h)) / 2h` has truncation error O(h²) where
 * `(L(w+h) − L(w)) / h` has O(h). That is the difference between agreeing to eight digits and
 * to four, and four is not enough to distinguish a correct gradient from a subtly wrong one.
 *
 * Dividing by `(wUp − wDown)` rather than by `2h`: the weights are a `Float32Array` — invariant
 * 3 — so `w + h` is *rounded on store*. At w ≈ 0.5 the spacing between representable float32
 * values is about 6e-8, so the perturbation that actually happened is not the one requested, and
 * the requested one is the wrong denominator. Reading the value back costs nothing and removes
 * the error entirely. This is the same float32-ULP effect §4 of the design document warns about
 * and challenge 4 puts a reader in front of; here it shows up as a test that fails at 2.5e-3
 * while the gradient is perfectly correct.
 */
function numericalGradient(
  net: Net,
  buffer: Float32Array,
  index: number,
  input: Float32Array,
  target: number,
  h: number,
): number {
  const scratch = createScratch(net);
  const original = buffer[index] as number;

  buffer[index] = original + h;
  const wUp = buffer[index] as number;
  const up = sampleLoss(net.loss, forward(net, input, scratch), target);

  buffer[index] = original - h;
  const wDown = buffer[index] as number;
  const down = sampleLoss(net.loss, forward(net, input, scratch), target);

  buffer[index] = original;
  return (up - down) / (wUp - wDown);
}

/** Relative error, with an absolute floor so two near-zero gradients do not divide badly. */
function relativeError(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1e-6, Math.abs(a) + Math.abs(b));
}

interface Case {
  readonly name: string;
  readonly net: Net;
  readonly input: Float32Array;
  readonly target: number;
}

function build(
  shape: number[],
  hidden: Activation,
  output: Activation,
  loss: 'mse' | 'crossEntropy',
  seed: number,
): Net {
  const net = createNet({ shape, hidden, output, loss });
  // Glorot rather than He across the board: relu at He scale puts many units exactly at zero,
  // where the derivative is a convention rather than a limit and finite differences legitimately
  // disagree. That case is tested separately and deliberately below.
  initialise(net, 'glorot', new Rng(seed));
  return net;
}

function cases(): Case[] {
  const out: Case[] = [];
  const hiddens: Activation[] = ['relu', 'tanh', 'sigmoid', 'linear'];

  for (const h of hiddens) {
    out.push({
      name: `${h} hidden → softmax + cross-entropy`,
      net: build([3, 5, 4, 3], h, 'softmax', 'crossEntropy', 11),
      input: Float32Array.from([0.6, -0.4, 0.2]),
      target: 1,
    });
    out.push({
      name: `${h} hidden → ${h} output + mse`,
      net: build([3, 5, 3], h, h, 'mse', 12),
      input: Float32Array.from([-0.3, 0.8, 0.1]),
      target: 2,
    });
  }

  // No hidden layer at all — challenge 1's network. Its gradient has to be right too.
  out.push({
    name: 'no hidden layer → softmax + cross-entropy',
    net: build([2, 2], 'linear', 'softmax', 'crossEntropy', 13),
    input: Float32Array.from([0.4, -0.9]),
    target: 0,
  });

  // Deep enough that an error in the recursion compounds instead of cancelling.
  out.push({
    name: 'five layers → softmax + cross-entropy',
    net: build([2, 4, 4, 4, 4, 2], 'tanh', 'softmax', 'crossEntropy', 14),
    input: Float32Array.from([0.7, 0.3]),
    target: 1,
  });

  return out;
}

describe('gradient check', () => {
  /*
   * h = 1e-4, tolerance 1e-7 — and both numbers are measured rather than chosen.
   *
   * Swept: the worst relative error over this suite is 1e-5 at h = 1e-2, 9.5e-8 at 1e-3,
   * 9.5e-10 at 1e-4, and back up to 1.5e-9 at 1e-5 as cancellation takes over. So 1e-4 sits at
   * the floor, and 1e-7 is three orders above it — tight enough that no real gradient bug can
   * hide under it, loose enough not to fail on arithmetic noise.
   */
  const h = 1e-4;
  const TOLERANCE = 1e-7;

  for (const c of cases()) {
    it(`matches finite differences — ${c.name}`, () => {
      const scratch = createScratch(c.net);
      const grads = createGrads(c.net);
      zeroGrads(grads);
      forward(c.net, c.input, scratch);
      backward(c.net, c.input, c.target, scratch, grads);

      let checked = 0;
      let worst = 0;

      for (let l = 0; l < c.net.layers.length; l++) {
        const layer = c.net.layers[l]!;
        const dW = grads.dW[l]!;
        const db = grads.db[l]!;

        for (let i = 0; i < layer.W.length; i++) {
          const numeric = numericalGradient(c.net, layer.W, i, c.input, c.target, h);
          const error = relativeError(dW[i] as number, numeric);
          worst = Math.max(worst, error);
          checked++;
          expect(error, `layer ${l} W[${i}]: analytic ${dW[i]} vs numeric ${numeric}`)
            .toBeLessThan(TOLERANCE);
        }
        for (let i = 0; i < layer.b.length; i++) {
          const numeric = numericalGradient(c.net, layer.b, i, c.input, c.target, h);
          const error = relativeError(db[i] as number, numeric);
          worst = Math.max(worst, error);
          checked++;
          expect(error, `layer ${l} b[${i}]: analytic ${db[i]} vs numeric ${numeric}`)
            .toBeLessThan(TOLERANCE);
        }
      }

      // A loop that examined nothing would pass every assertion above. Asserting the exact
      // parameter count also catches a layer being skipped — challenge 1's network has 6.
      expect(checked).toBe(paramCount(c.net));
      expect(worst).toBeLessThan(TOLERANCE);
    });
  }

  it('would fail if a sign were flipped', () => {
    /*
     * The gradient check checking itself.
     *
     * A test that never fails proves nothing, and this one is load-bearing enough to be worth
     * demonstrating. Negating the analytic gradient is the smallest possible bug of the kind
     * this is meant to catch — and one that would still train, just uphill.
     */
    const net = build([2, 4, 2], 'tanh', 'softmax', 'crossEntropy', 15);
    const input = Float32Array.from([0.5, -0.2]);
    const scratch = createScratch(net);
    const grads = createGrads(net);
    zeroGrads(grads);
    forward(net, input, scratch);
    backward(net, input, 1, scratch, grads);

    const layer = net.layers[0]!;
    const numeric = numericalGradient(net, layer.W, 0, input, 1, h);
    const analytic = grads.dW[0]![0] as number;

    expect(relativeError(analytic, numeric)).toBeLessThan(TOLERANCE);
    expect(relativeError(-analytic, numeric)).toBeGreaterThan(0.5);
  });

  it('agrees for relu even where units are clamped off', () => {
    /*
     * Relu's derivative at exactly z = 0 is a convention, not a limit, so a weight nudge that
     * moves a unit across zero makes the finite difference legitimately disagree. This uses an
     * input far from any boundary so every unit is decisively on or off, which is the regime the
     * network actually trains in — and confirms the `a > 0` test is on the right side.
     */
    const net = build([2, 6, 2], 'relu', 'softmax', 'crossEntropy', 16);
    const input = Float32Array.from([2.5, -1.8]);
    const scratch = createScratch(net);
    const grads = createGrads(net);
    zeroGrads(grads);
    forward(net, input, scratch);
    backward(net, input, 0, scratch, grads);

    // There must actually be some dead units, or this is testing the same thing as the case above.
    let dead = 0;
    for (const v of scratch.a[0]!) if (v === 0) dead++;
    expect(dead).toBeGreaterThan(0);

    const layer = net.layers[0]!;
    for (let i = 0; i < layer.W.length; i++) {
      const numeric = numericalGradient(net, layer.W, i, input, 0, 1e-5);
      expect(relativeError(grads.dW[0]![i] as number, numeric)).toBeLessThan(1e-6);
    }
  });

  it('accumulates a batch as the sum of its samples', () => {
    // trainStep relies on this: it backpropagates every sample into one Grads and divides once.
    const net = build([2, 4, 3], 'tanh', 'softmax', 'crossEntropy', 17);
    const scratch = createScratch(net);
    const inputs = [Float32Array.from([0.2, 0.9]), Float32Array.from([-0.7, 0.1])];
    const targets = [0, 2];

    const separate = new Float64Array(net.layers[0]!.W.length);
    for (let s = 0; s < inputs.length; s++) {
      const g = createGrads(net);
      zeroGrads(g);
      forward(net, inputs[s]!, scratch);
      backward(net, inputs[s]!, targets[s]!, scratch, g);
      for (let i = 0; i < separate.length; i++) {
        separate[i] = (separate[i] as number) + (g.dW[0]![i] as number);
      }
    }

    const combined = createGrads(net);
    zeroGrads(combined);
    for (let s = 0; s < inputs.length; s++) {
      forward(net, inputs[s]!, scratch);
      backward(net, inputs[s]!, targets[s]!, scratch, combined);
    }

    for (let i = 0; i < separate.length; i++) {
      expect(combined.dW[0]![i]).toBeCloseTo(separate[i] as number, 6);
    }
  });
});
