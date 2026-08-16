import { describe, expect, it } from 'vitest';
import { Rng, type Dataset } from '@neurallab/core';
import { bmu, createSom, type Schedule } from '../src/som.ts';
import { createSomTrainer, somStep } from '../src/train.ts';

function pointDataset(points: readonly (readonly number[])[]): Dataset {
  const dim = (points[0] as number[]).length;
  const x = new Float32Array(points.length * dim);
  points.forEach((p, i) => x.set(p, i * dim));
  return {
    name: 'test points',
    x,
    y: null,
    n: points.length,
    dim,
    classes: 0,
    featureNames: Array.from({ length: dim }, (_, k) => `f${k}`),
    classNames: [],
  };
}

describe('somStep', () => {
  it('moves the winning node toward the sample and increments its hit count', () => {
    const som = createSom(3, 3, 2, 'rect', new Rng(1));
    const ds = pointDataset([[0.9, 0.9]]);
    const schedule: Schedule = { alpha0: 0.5, sigma0: 3, decay: 'exponential', steps: 100 };
    const trainer = createSomTrainer(som, Int32Array.from([0]), schedule, new Rng(1));

    const winnerBefore = bmu(som, [0.9, 0.9]);
    const distBefore = Math.hypot(
      (som.W[winnerBefore * 2] as number) - 0.9,
      (som.W[winnerBefore * 2 + 1] as number) - 0.9,
    );

    const result = somStep(trainer, ds);
    expect(result.bmuIndex).toBe(winnerBefore);
    expect(som.hits[winnerBefore]).toBe(1);

    const distAfter = Math.hypot(
      (som.W[winnerBefore * 2] as number) - 0.9,
      (som.W[winnerBefore * 2 + 1] as number) - 0.9,
    );
    expect(distAfter).toBeLessThan(distBefore);
  });

  it('advances the trainer step by exactly one', () => {
    const som = createSom(2, 2, 2, 'rect', new Rng(1));
    const ds = pointDataset([[0.1, 0.1]]);
    const schedule: Schedule = { alpha0: 0.5, sigma0: 3, decay: 'exponential', steps: 100 };
    const trainer = createSomTrainer(som, Int32Array.from([0]), schedule, new Rng(1));
    expect(trainer.step).toBe(0);
    somStep(trainer, ds);
    expect(trainer.step).toBe(1);
    somStep(trainer, ds);
    expect(trainer.step).toBe(2);
  });

  it('leaves a node far outside a small σ untouched — the h < 1e-7 shortcut', () => {
    // A wide rect map, tiny σ: node (9,0) is 9 lattice units from a winner forced to (0,0),
    // which puts h at exp(-81/(2σ²)) — far below the 1e-7 cutoff at σ well under 3.
    const som = createSom(10, 1, 1, 'rect', new Rng(1));
    som.W.set([0], 0); // node 0 sits exactly on the sample, so it wins deterministically
    const farBefore = som.W[9] as number;
    const ds = pointDataset([[0]]);
    const schedule: Schedule = { alpha0: 0.9, sigma0: 1, decay: 'exponential', steps: 100 };
    const trainer = createSomTrainer(som, Int32Array.from([0]), schedule, new Rng(1));
    somStep(trainer, ds);
    expect(som.W[9]).toBe(farBefore);
  });

  it('draws rows from the given index array, not from every dataset row', () => {
    // Two points far apart; the trainer only ever sees row 1, so the map should end up nearer
    // point 1 than point 0 after enough steps, regardless of which row Rng.int would otherwise
    // have picked.
    const som = createSom(1, 1, 1, 'rect', new Rng(1));
    som.W[0] = 0.5;
    const ds = pointDataset([[0], [1]]);
    const schedule: Schedule = { alpha0: 0.9, sigma0: 1, decay: 'exponential', steps: 50 };
    const trainer = createSomTrainer(som, Int32Array.from([1]), schedule, new Rng(9));
    for (let i = 0; i < 20; i++) somStep(trainer, ds);
    expect(som.W[0] as number).toBeGreaterThan(0.9);
  });
});

describe('somStep tracing', () => {
  it('records the hand-worked case: a 3×1 line, sample 0.9, σ₀ 3', () => {
    // node0=0, node1=1, node2=3. Squared distances to 0.9: 0.81, 0.01, 4.41 — BMU is node1.
    // At step 0, σ(0) = σ₀ = 3 exactly. h(d,3) = exp(-d²/18): h(0)=1, h(1)=exp(-1/18)≈0.94596.
    const som = createSom(3, 1, 1, 'rect', new Rng(1));
    som.W.set([0, 1, 3]);
    const ds = pointDataset([[0.9]]);
    const schedule: Schedule = { alpha0: 0.5, sigma0: 3, decay: 'exponential', steps: 100 };
    const trainer = createSomTrainer(som, Int32Array.from([0]), schedule, new Rng(1));

    const { trace } = somStep(trainer, ds, { trace: true });
    expect(trace).toBeDefined();
    expect(trace!.bmu).toBe(1);
    expect(trace!.sigma).toBeCloseTo(3, 10);
    expect(Array.from(trace!.distances)).toEqual([
      expect.closeTo(0.9, 6),
      expect.closeTo(0.1, 6),
      expect.closeTo(2.1, 6),
    ]);
    expect(Array.from(trace!.before)).toEqual([0, 1, 3]);
    // strength is stored Float32, same reason as every other precision note in this project —
    // 6dp, not 10.
    expect(trace!.strength[1]).toBeCloseTo(1, 6);
    expect(trace!.strength[0]).toBeCloseTo(Math.exp(-1 / 18), 6);
    expect(trace!.strength[2]).toBeCloseTo(Math.exp(-1 / 18), 6);
  });

  it('leaves every node covered, including the ones the h < 1e-7 shortcut skips updating', () => {
    const som = createSom(10, 1, 1, 'rect', new Rng(1));
    som.W.set([0], 0);
    const ds = pointDataset([[0]]);
    const schedule: Schedule = { alpha0: 0.9, sigma0: 1, decay: 'exponential', steps: 100 };
    const trainer = createSomTrainer(som, Int32Array.from([0]), schedule, new Rng(1));
    const { trace } = somStep(trainer, ds, { trace: true });
    expect(trace!.distances).toHaveLength(10);
    expect(trace!.strength).toHaveLength(10);
    // Node 9 is far enough that the update loop skips it, but the trace still reports its
    // (vanishingly small, not zeroed) strength — the heatmap draws the whole lattice.
    expect(trace!.strength[9] as number).toBeGreaterThan(0);
    expect(trace!.strength[9] as number).toBeLessThan(1e-7);
  });

  it('does not change the weights tracing produces — bit-identical to an untraced run', () => {
    const a = createSom(6, 6, 3, 'hex', new Rng(4417));
    const b = createSom(6, 6, 3, 'hex', new Rng(4417));
    const ds = pointDataset(
      Array.from({ length: 20 }, (_, i) => [((i * 37) % 100) / 100, ((i * 53) % 100) / 100, ((i * 71) % 100) / 100]),
    );
    const rows = Int32Array.from({ length: 20 }, (_, i) => i);
    const schedule: Schedule = { alpha0: 0.5, sigma0: 4, decay: 'exponential', steps: 200 };
    const trainerA = createSomTrainer(a, rows, schedule, new Rng(2));
    const trainerB = createSomTrainer(b, rows, schedule, new Rng(2));

    for (let i = 0; i < 200; i++) {
      somStep(trainerA, ds, { trace: true }); // traced every step
      somStep(trainerB, ds); // never traced
    }

    expect(Array.from(a.W)).toEqual(Array.from(b.W));
  });
});
