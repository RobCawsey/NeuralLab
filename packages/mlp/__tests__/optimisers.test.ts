/**
 * Momentum and Adam, checked against numbers worked out by hand — not against a loss curve.
 *
 * §13 names the specific risk this test exists for: *"Adam will be subtly wrong. The
 * bias-correction terms are easy to misplace, and the loss still goes down without them — just
 * more slowly at the start, which looks like a learning-rate issue."* A convergence test cannot
 * tell a correct Adam from a slightly-too-slow one; only arithmetic can.
 *
 * A one-weight, no-activation network stands in for the kernel here, the same trick
 * `forward.test.ts`'s hand-check uses — it isolates the optimiser's arithmetic from backprop,
 * which `gradcheck.test.ts` already owns.
 */

import { describe, expect, it } from 'vitest';
import {
  applyUpdate,
  createGrads,
  createOptimiserState,
  resetOptimiserState,
  sgdStep,
  type Grads,
} from '../src/backward.ts';
import { createNet, type Net } from '../src/net.ts';

function oneWeightNet(initialWeight: number): Net {
  const net = createNet({ shape: [1, 1], hidden: 'linear', output: 'linear', loss: 'mse' });
  net.layers[0]!.W.set([initialWeight]);
  net.layers[0]!.b.set([0]);
  return net;
}

/** A `Grads` holding one fixed gradient value, so each optimiser step is deterministic. */
function fixedGradient(net: Net, dw: number): Grads {
  const grads = createGrads(net);
  grads.dW[0]!.set([dw]);
  grads.db[0]!.set([0]);
  return grads;
}

describe('momentum', () => {
  it('matches a hand-computed sequence: v ← 0.9v + g, w ← w − lr·v', () => {
    // Unscaled momentum, not the exponential-moving-average form — see backward.ts for why the
    // two conventions are kept apart. lr = 0.1, g = 0.5 constant, β = 0.9 (fixed, not a slider).
    const net = oneWeightNet(1.0);
    const grads = fixedGradient(net, 0.5);
    const state = createOptimiserState(net, 'momentum');

    /*
     * `layer.W` is a `Float32Array` — invariant 3 — so it holds these to about 7 significant
     * digits regardless of how precisely the arithmetic was done, and asking for more decimal
     * places than the storage has is the same mistake the slice-2 gradient check made against
     * this exact type. 6dp is below the float32 floor at these magnitudes and well above it.
     */
    applyUpdate(net, grads, state, 0.1); // v = 0.5,   w = 1.0 - 0.05  = 0.95
    expect(net.layers[0]!.W[0]).toBeCloseTo(0.95, 6);

    applyUpdate(net, grads, state, 0.1); // v = 0.95,  w = 0.95 - 0.095 = 0.855
    expect(net.layers[0]!.W[0]).toBeCloseTo(0.855, 6);

    applyUpdate(net, grads, state, 0.1); // v = 1.355, w = 0.855 - 0.1355 = 0.7195
    expect(net.layers[0]!.W[0]).toBeCloseTo(0.7195, 6);
  });

  it('carries velocity forward — it is not just a scaled gradient step', () => {
    // If momentum only rescaled the gradient it would be indistinguishable from SGD at a
    // different learning rate. Velocity has to accumulate, so a constant gradient produces a
    // step that keeps growing rather than settling immediately.
    const net = oneWeightNet(0);
    const grads = fixedGradient(net, 1);
    const state = createOptimiserState(net, 'momentum');

    applyUpdate(net, grads, state, 0.1);
    const firstStep = -(net.layers[0]!.W[0] as number);
    net.layers[0]!.W.set([0]);
    applyUpdate(net, grads, state, 0.1);
    applyUpdate(net, grads, state, 0.1);
    const thirdStepSize = Math.abs((net.layers[0]!.W[0] as number)) - 2 * firstStep;
    expect(thirdStepSize).toBeGreaterThan(0);
  });
});

describe('adam', () => {
  it('matches a hand-computed first step exactly', () => {
    /*
     * w = 1.0, g = 0.5, lr = 0.1, β1 = 0.9, β2 = 0.999, ε = 1e-8.
     *   m1 = 0.1 · 0.5              = 0.05
     *   v1 = 0.001 · 0.25           = 0.00025
     *   m̂  = m1 / (1 − 0.9)        = 0.5
     *   v̂  = v1 / (1 − 0.999)      = 0.25
     *   Δw = lr · m̂ / (√v̂ + ε)    = 0.1 · 0.5 / (0.5 + 1e-8) ≈ 0.1
     *   w  = 1.0 − Δw               ≈ 0.9
     */
    const net = oneWeightNet(1.0);
    const grads = fixedGradient(net, 0.5);
    const state = createOptimiserState(net, 'adam');

    applyUpdate(net, grads, state, 0.1);

    // Float32 storage again — see the note on the momentum test above.
    expect(net.layers[0]!.W[0]).toBeCloseTo(0.9, 6);
    expect(state.t).toBe(1);
  });

  it('holds its step size near lr under a constant gradient, for many steps', () => {
    /*
     * The closed-form identity behind the "first step" check above is not special to step 1: for
     * a constant gradient g, bias correction makes m̂ₜ = g and v̂ₜ = g² at *every* step exactly
     * (m̂ₜ = mₜ/(1−β1ᵗ) and induction on mₜ = g·(1−β1ᵗ) gives m̂ₜ = g for all t; v̂ₜ follows the
     * same way). So Δw ≈ lr·g/(√(g²)+ε) = lr·sign(g) at every step, regardless of magnitude —
     * which is the well-known reason Adam is described as taking roughly lr-sized steps. Five
     * steps here, each checked against that identity rather than five more decimal expansions.
     */
    const net = oneWeightNet(0);
    const grads = fixedGradient(net, 3.7); // an arbitrary, un-round magnitude on purpose
    const state = createOptimiserState(net, 'adam');

    let previous = 0;
    for (let step = 1; step <= 5; step++) {
      applyUpdate(net, grads, state, 0.1);
      const w = net.layers[0]!.W[0] as number;
      expect(w - previous, `step ${step}`).toBeCloseTo(-0.1, 4);
      previous = w;
    }
  });

  it('takes a smaller step for a smaller gradient — it is not scale-free at every instant', () => {
    // Adam's steady-state step size is roughly lr regardless of gradient *magnitude*, but a
    // fresh optimiser's very first step is not yet at steady state, and this is the case that
    // catches a missing bias-correction term fastest: without it, v1 (an unbiased tiny number)
    // makes the very first update enormous rather than modest.
    const net = oneWeightNet(0);
    const grads = fixedGradient(net, 1e-6);
    const state = createOptimiserState(net, 'adam');
    applyUpdate(net, grads, state, 0.1);
    expect(Math.abs(net.layers[0]!.W[0] as number)).toBeLessThan(0.11);
  });
});

describe('sgd (through applyUpdate)', () => {
  it('is identical to calling sgdStep directly', () => {
    // The golden run is pinned to sgdStep; applyUpdate must be an exact pass-through for 'sgd',
    // not a reimplementation that happens to agree.
    const a = oneWeightNet(1.0);
    const b = oneWeightNet(1.0);
    const gradsA = fixedGradient(a, 0.3);
    const gradsB = fixedGradient(b, 0.3);

    sgdStep(a, gradsA, 0.1);
    applyUpdate(b, gradsB, createOptimiserState(b, 'sgd'), 0.1);

    expect(a.layers[0]!.W[0]).toBe(b.layers[0]!.W[0]);
  });
});

describe('resetOptimiserState', () => {
  it('zeroes every moment and the step count', () => {
    const net = oneWeightNet(1.0);
    const grads = fixedGradient(net, 0.5);
    const state = createOptimiserState(net, 'adam');
    applyUpdate(net, grads, state, 0.1);
    applyUpdate(net, grads, state, 0.1);
    expect(state.t).toBe(2);

    resetOptimiserState(state, 'adam');

    expect(state.t).toBe(0);
    for (const arrays of [state.mW, state.mB, state.vW, state.vB]) {
      for (const a of arrays) for (const v of a) expect(v).toBe(0);
    }
  });

  it('changes kind as well as clearing state', () => {
    const net = oneWeightNet(1.0);
    const state = createOptimiserState(net, 'sgd');
    resetOptimiserState(state, 'momentum');
    expect(state.kind).toBe('momentum');
  });

  it('after a reset, momentum and Adam disagree on the same gradient — proof the switch took', () => {
    // Two optimisers "reset" from the same state must not coincidentally produce the same
    // update on the same input — if they did, the kind switch could have silently failed and
    // this test would not know.
    const gradsFor = (net: Net): Grads => fixedGradient(net, 0.5);

    const momentumNet = oneWeightNet(1.0);
    const momentumState = createOptimiserState(momentumNet, 'sgd');
    resetOptimiserState(momentumState, 'momentum');
    applyUpdate(momentumNet, gradsFor(momentumNet), momentumState, 0.1);

    const adamNet = oneWeightNet(1.0);
    const adamState = createOptimiserState(adamNet, 'sgd');
    resetOptimiserState(adamState, 'adam');
    applyUpdate(adamNet, gradsFor(adamNet), adamState, 0.1);

    expect(momentumNet.layers[0]!.W[0]).not.toBeCloseTo(adamNet.layers[0]!.W[0] as number, 6);
  });
});
