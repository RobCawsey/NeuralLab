import { describe, expect, it } from 'vitest';
import {
  MLP_FLOW,
  SHAPES,
  SOM_FLOW,
  compareAfterword,
  labelAfterword,
  regionsAfterword,
  stepStatus,
} from '../src/run/guided.ts';

describe('MLP_FLOW', () => {
  it('has the four steps in order, no hyperparameters implied by any of them', () => {
    expect(MLP_FLOW.map((s) => s.id)).toEqual(['data', 'shape', 'watch', 'compare']);
    expect(MLP_FLOW).toHaveLength(4);
  });

  it('gives every step a real title', () => {
    for (const step of MLP_FLOW) expect(step.title.length).toBeGreaterThan(0);
  });
});

describe('SOM_FLOW', () => {
  it('has four steps, and none of them is the MLP\'s "shape"', () => {
    expect(SOM_FLOW.map((s) => s.id)).toEqual(['data', 'watch', 'regions', 'label']);
    expect(SOM_FLOW).toHaveLength(4);
    expect(SOM_FLOW.some((s) => s.id === 'shape')).toBe(false);
  });

  it('gives every step a real title', () => {
    for (const step of SOM_FLOW) expect(step.title.length).toBeGreaterThan(0);
  });

  it('shares its type with MLP_FLOW rather than a parallel one', () => {
    // Both are arrays of the same GuidedStep shape — checked structurally rather than by
    // `typeof`, since TS erases the type and there is nothing else to assert against at runtime.
    for (const step of [...MLP_FLOW, ...SOM_FLOW]) {
      expect(typeof step.id).toBe('string');
      expect(typeof step.title).toBe('string');
    }
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

describe('regionsAfterword', () => {
  it('quotes the real U-matrix maximum and topographic error', () => {
    const text = regionsAfterword(0.234, 0.087);
    expect(text).toContain('0.234');
    expect(text).toContain('8.7%');
  });

  it('takes the "barely any ridges" branch when the map is nearly flat', () => {
    expect(regionsAfterword(0.005, 0.02)).toContain('Barely any ridges yet');
  });

  it('takes the "ridges" branch once the U-matrix has real structure', () => {
    const text = regionsAfterword(0.3, 0.05);
    expect(text).not.toContain('Barely any ridges yet');
    expect(text).toContain('brightest ridges');
  });
});

describe('labelAfterword', () => {
  it('says plainly that an unlabelled dataset has nothing to recover', () => {
    const text = labelAfterword([-1, -1, -1], []);
    expect(text).toContain('no labels to recover');
  });

  it('counts labelled nodes against the total, on a labelled dataset', () => {
    const text = labelAfterword([0, 1, -1, 0], ['upper', 'lower']);
    expect(text).toContain('3');
    expect(text).toContain('4');
    expect(text).toContain('75%');
    expect(text).toContain('upper');
    expect(text).toContain('lower');
  });

  it('handles a single-class dataset without indexing past the end', () => {
    expect(() => labelAfterword([0, 0], ['only'])).not.toThrow();
  });
});
