/**
 * The guided flow's data — §13's answered question, taken as data from the first version.
 *
 * One `GuidedFlow` type and one array of steps for the perceptron now; the map's array arrives
 * in slice 11 and shares this exact type and the renderer in `ui/guided.ts`. Building the frame
 * generically from the start is the point — retrofitting a second client onto a hardcoded first
 * one is how the copy ends up duplicated, which is the risk that made this an open question.
 *
 * Step *metadata* lives here, deliberately apart from the DOM in `ui/guided.ts`: this file is
 * pure, so it is checked by vitest under Node, matching the rule `vitest.config.ts` already
 * states for `apps/web` — modules that import nothing from the DOM belong in its test glob.
 */

export type GuidedStepId = 'data' | 'shape' | 'watch' | 'compare';

export interface GuidedStep {
  readonly id: GuidedStepId;
  readonly title: string;
}

export type GuidedFlow = readonly GuidedStep[];

export const MLP_FLOW: GuidedFlow = [
  { id: 'data', title: 'Pick some data' },
  { id: 'shape', title: 'Pick a shape it can draw' },
  { id: 'watch', title: 'Watch it learn' },
  { id: 'compare', title: 'See what changed' },
];

export type StepStatus = 'done' | 'on' | 'later';

export function stepStatus(index: number, current: number): StepStatus {
  if (index < current) return 'done';
  if (index === current) return 'on';
  return 'later';
}

/**
 * The vocabulary a beginner has before they know the words "hidden layer".
 *
 * Three shapes, three hidden-layer widths. The mapping is the whole point of the step: a reader
 * chooses a shape and only later, in Explorer, discovers that "a curve" was one hidden layer of
 * eight all along.
 */
export interface ShapeChoice {
  readonly id: 'line' | 'curve' | 'anything';
  readonly label: string;
  readonly hidden: readonly number[];
}

export const SHAPES: readonly ShapeChoice[] = [
  { id: 'line', label: 'A straight line', hidden: [] },
  { id: 'curve', label: 'A curve', hidden: [8] },
  { id: 'anything', label: 'Anything at all', hidden: [8, 8] },
];

/**
 * The afterword for step 4, written against the run's own numbers — §6's rule that explanations
 * are never fixed strings that can be wrong.
 *
 * Two branches, not one. Naive training on an easy default dataset improves almost every time,
 * but "almost every time" is not "every time", and a reader who lands on the one run that barely
 * moved deserves a sentence that is still true rather than one asserting progress that did not
 * happen. `guided.test.ts` renders both.
 */
export function compareAfterword(beforeAccuracy: number, afterAccuracy: number, steps: number): string {
  const before = (beforeAccuracy * 100).toFixed(1);
  const after = (afterAccuracy * 100).toFixed(1);
  const n = steps.toLocaleString();

  if (afterAccuracy > beforeAccuracy + 0.02) {
    return (
      `It started by guessing right <b>${before}%</b> of the time. Nobody told it where the ` +
      `boundary was — it found one by being wrong, over and over, for <b>${n}</b> steps, and ` +
      `adjusting a little each time. Now it is right <b>${after}%</b> of the time, on points it ` +
      `never trained on.`
    );
  }
  return (
    `It started at <b>${before}%</b> and after <b>${n}</b> steps it reached <b>${after}%</b> — ` +
    `not the dramatic jump most runs show here. Try <em>Anything at all</em> for more capacity, ` +
    `or open the loss chart in Explorer to see whether it is still falling.`
  );
}
