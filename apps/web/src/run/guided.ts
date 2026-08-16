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

export type GuidedStepId = 'data' | 'shape' | 'watch' | 'compare' | 'regions' | 'label';

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

/**
 * The map's own flow — slice 11, sharing this type and `ui/guided.ts`'s visual vocabulary with
 * `MLP_FLOW` rather than inventing a second one. Four steps, not the MLP's four-with-a-shape:
 * there is no SOM equivalent of "pick a hidden layer" worth a step of its own — the lattice's
 * 12×12 hex default is already the sensible choice — so step 2 starts training the moment data is
 * picked. Step 3 *is* the U-matrix slice 10 built, which is why this flow waits for slice 11
 * rather than arriving with slice 6's.
 */
export const SOM_FLOW: GuidedFlow = [
  { id: 'data', title: 'Pick some data' },
  { id: 'watch', title: 'Watch a flat sheet fold into it' },
  { id: 'regions', title: 'See what it kept apart' },
  { id: 'label', title: 'Label it' },
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

/**
 * The afterword for the map's own step 3 — written against the trained map's own U-matrix and
 * topographic error, the same rule `compareAfterword` follows: a fixed sentence about "ridges"
 * would be wrong the one time training leaves the map nearly flat.
 */
export function regionsAfterword(uMax: number, te: number): string {
  if (uMax < 0.02) {
    return (
      `This map's neighbours are all still close to each other — <b>${uMax.toFixed(3)}</b> is the ` +
      `largest distance between any two touching nodes. Barely any ridges yet; a longer run gives ` +
      `the lattice more room to spread into the data's own gaps.`
    );
  }
  return (
    `The brightest ridges are where neighbouring nodes sit <b>${uMax.toFixed(3)}</b> apart in the ` +
    `data — gaps nobody labelled, found only because two nodes that touch on the lattice ended up ` +
    `far apart in the data. Topographic error is <b>${(te * 100).toFixed(1)}%</b>: the fraction of ` +
    `points whose two best nodes are strangers on the lattice, which is the number that would ` +
    `climb if the map had torn.`
  );
}

/**
 * The afterword for the map's own step 4 — labelling a map that was never shown a label during
 * training. Branches on whether the dataset had any to recover: the colour cube does not, and
 * saying so plainly is the point of §3's "unlabelled is a fact about training, not the file"
 * rather than a gap to paper over.
 */
export function labelAfterword(labels: ArrayLike<number>, classNames: readonly string[]): string {
  if (classNames.length === 0) {
    return (
      `This dataset has no labels to recover — the colour cube never had any. Pick <em>Two moons</em> ` +
      `or <em>Three blobs</em> above to see the map label itself from data it was never trained on ` +
      `the answer for.`
    );
  }
  let labelled = 0;
  for (let i = 0; i < labels.length; i++) if ((labels[i] as number) >= 0) labelled++;
  const pct = labels.length > 0 ? (100 * labelled) / labels.length : 0;
  return (
    `Every node just voted: whichever class most often won it becomes its label, decided entirely ` +
    `after training finished. <b>${labelled}</b> of <b>${labels.length}</b> nodes ` +
    `(<b>${pct.toFixed(0)}%</b>) were won by at least one point and got a label at all — the rest ` +
    `never won a single sample. The map was never told <em>${classNames[0]}</em> from ` +
    `<em>${classNames[1] ?? classNames[0]}</em>; it only ever chased the shape of the data.`
  );
}
