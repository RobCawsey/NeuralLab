import { describe, expect, it } from 'vitest';
import { MLP_FLOW, SHAPES, compareAfterword, stepStatus } from '../src/run/guided.ts';

describe('MLP_FLOW', () => {
  it('has the four steps in order, no hyperparameters implied by any of them', () => {
    expect(MLP_FLOW.map((s) => s.id)).toEqual(['data', 'shape', 'watch', 'compare']);
    expect(MLP_FLOW).toHaveLength(4);
  });

  it('gives every step a real title', () => {
    for (const step of MLP_FLOW) expect(step.title.length).toBeGreaterThan(0);
  });
});

describe('stepStatus', () => {
  it('marks everything before the current step done', () => {
    expect(stepStatus(0, 2)).toBe('done');
    expect(stepStatus(1, 2)).toBe('done');
  });

  it('marks the current step on', () => {
    expect(stepStatus(2, 2)).toBe('on');
  });

  it('marks everything after later', () => {
    expect(stepStatus(3, 2)).toBe('later');
  });
});

describe('SHAPES', () => {
  it('maps three plain-language choices to hidden-layer widths', () => {
    expect(SHAPES.map((s) => s.id)).toEqual(['line', 'curve', 'anything']);
  });

  it('is a straight line for no hidden layer — challenge 1 reachable from guided too', () => {
    expect(SHAPES.find((s) => s.id === 'line')!.hidden).toEqual([]);
  });

  it('is a curve for one hidden layer', () => {
    expect(SHAPES.find((s) => s.id === 'curve')!.hidden).toEqual([8]);
  });

  it('is anything for two hidden layers, matching the app default', () => {
    expect(SHAPES.find((s) => s.id === 'anything')!.hidden).toEqual([8, 8]);
  });

  it('never says "hidden layer" in its own labels', () => {
    // The whole point of the step: the vocabulary comes before the term.
    for (const s of SHAPES) expect(s.label.toLowerCase()).not.toContain('hidden');
  });
});

describe('compareAfterword', () => {
  it('quotes the real before and after percentages', () => {
    const text = compareAfterword(0.512, 0.947, 400);
    expect(text).toContain('51.2%');
    expect(text).toContain('94.7%');
    expect(text).toContain('400');
  });

  it('formats a large step count with a thousands separator', () => {
    expect(compareAfterword(0.5, 0.9, 20000)).toContain('20,000');
  });

  it('takes the improvement branch when the run clearly got better', () => {
    const text = compareAfterword(0.5, 0.95, 400);
    expect(text).toContain('Nobody told it where the boundary was');
  });

  it('takes the other branch when the run barely moved', () => {
    /*
     * §6's rule, exercised: a run that did not improve must not be told it did. This is the
     * branch a fixed string could not produce honestly.
     */
    const text = compareAfterword(0.5, 0.51, 400);
    expect(text).not.toContain('Nobody told it where the boundary was');
    expect(text).toContain('not the dramatic jump');
  });

  it('renders every branch — both are real strings, and they differ', () => {
    const improved = compareAfterword(0.5, 0.95, 400);
    const flat = compareAfterword(0.5, 0.51, 400);
    expect(improved).not.toBe(flat);
    expect(improved.length).toBeGreaterThan(0);
    expect(flat.length).toBeGreaterThan(0);
  });

  it('sits right at the boundary sensibly', () => {
    // Exactly the threshold is the "did not clearly improve" branch, not a coin flip.
    expect(compareAfterword(0.5, 0.52, 400)).toContain('not the dramatic jump');
    expect(compareAfterword(0.5, 0.521, 400)).toContain('Nobody told it');
  });
});
