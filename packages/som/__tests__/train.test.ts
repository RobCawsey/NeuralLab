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
