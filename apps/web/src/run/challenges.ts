/**
 * The concept ladder — twelve cards, four phases, one dot per concept. Every "challenge N"
 * reference elsewhere in this project has been pointing at this file since slice 0.
 *
 * Pure, like `run/guided.ts` beside it: a card names an idea, configures the app in one click,
 * and its afterword is a function of the run's own numbers, never a fixed string. §6's rule —
 * "explanations are written against live values" — applies to twelve cards exactly as it applies
 * to one probe note.
 */

import type { GeneratorKey } from '@neurallab/data';
import type { Activation, InitScheme } from '@neurallab/mlp';
import type { Decay, Topology } from '@neurallab/som';
import type { SomDatasetKey } from './somState.ts';

export interface ChallengeConfig {
  readonly net: 'mlp' | 'som';
  readonly dataset?: GeneratorKey;
  readonly hidden?: readonly number[];
  readonly hiddenAct?: Activation;
  readonly init?: InitScheme;
  readonly learningRate?: number;
  readonly targetSteps?: number;
  readonly n?: number;
  readonly trainFraction?: number;

  readonly somDataset?: SomDatasetKey;
  readonly cols?: number;
  readonly rows?: number;
  readonly topology?: Topology;
  readonly decay?: Decay;
  readonly somTargetSteps?: number;
  readonly scheduleSteps?: number;
}

/**
 * Whatever a card's afterword might need, gathered once by the caller and read selectively —
 * one shape rather than twelve bespoke ones, because the alternative is `ui/challenges.ts`
 * needing to know exactly which fields each of twelve functions wants.
 */
export interface ChallengeOutcome {
  readonly finished: boolean;
  readonly trainAccuracy?: number;
  readonly valAccuracy?: number;
  readonly trainLoss?: number;
  readonly gradNorms?: readonly number[];
  readonly paramCount?: number;
  readonly trainRows?: number;
  readonly overBudget?: boolean;
  readonly qeStart?: number;
  readonly qeEnd?: number;
  readonly teStart?: number;
  readonly teEnd?: number;
  readonly uMax?: number;
  readonly nodes?: number;
  readonly mlpFinished?: boolean;
  readonly somFinished?: boolean;
}

export interface Challenge {
  readonly id: number;
  readonly phase: string;
  readonly title: string;
  readonly concept: string;
  readonly setup: string;
  readonly doneWhen: string;
  readonly config: ChallengeConfig;
  readonly afterword: (o: ChallengeOutcome) => string;
}

const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

export const CHALLENGES: readonly Challenge[] = [
  {
    id: 1,
    phase: 'the shape of a network',
    title: 'One line is not enough',
    concept: 'linear separability',
    setup: 'no hidden layer · XOR',
    doneWhen: 'you can say why the accuracy stalls where it does, not just that it does.',
    config: { net: 'mlp', dataset: 'xor', hidden: [] },
    afterword: (o) =>
      o.finished && o.trainAccuracy !== undefined
        ? `Finished at <b>${pct(o.trainAccuracy)}</b>. A single straight line can isolate one ` +
          `quadrant of XOR and call everything else the other class — right three times out of ` +
          `four, and no amount of training moves a flat network past that.`
        : 'Press Train and watch a straight line try, and fail, to separate XOR.',
  },
  {
    id: 2,
    phase: 'the shape of a network',
    title: 'Give it a middle',
    concept: 'non-linearity',
    setup: 'one hidden layer · same data',
    doneWhen: 'you can point at the one thing that changed between this card and the last.',
    config: { net: 'mlp', dataset: 'xor', hidden: [8] },
    afterword: (o) =>
      o.finished && o.trainAccuracy !== undefined
        ? `One hidden layer of 8 units reaches <b>${pct(o.trainAccuracy)}</b> — the same data a ` +
          `straight line could only ever get three quarters of.`
        : 'Press Train — the same data as the last card, one hidden layer added.',
  },
  {
    id: 3,
    phase: 'how it learns',
    title: 'Too big a step',
    concept: 'the accuracy collapses',
    setup: 'lr 500 · two moons',
    doneWhen: 'you can explain why nothing here goes NaN.',
    config: { net: 'mlp', dataset: 'moons', hidden: [8, 8], learningRate: 500 },
    afterword: (o) =>
      o.finished && o.trainAccuracy !== undefined && o.trainLoss !== undefined
        ? o.trainAccuracy < 0.75
          ? `Accuracy collapsed to <b>${pct(o.trainAccuracy)}</b>, loss <b>${o.trainLoss.toFixed(2)}</b> ` +
            `— and nothing went NaN. Softmax shifts by its maximum before it exponentiates, so even ` +
            `a wrecked network stays readable instead of blank.`
          : `This run landed at <b>${pct(o.trainAccuracy)}</b> — try Resample for a seed where lr 500 ` +
            `collapses harder; it usually does.`
        : 'Press Train and watch a learning rate two orders of magnitude too large.',
  },
  {
    id: 4,
    phase: 'how it learns',
    title: 'Too small a step',
    concept: 'step size, and float32 ULP',
    setup: 'lr 1e−6 · 2 000 steps',
    doneWhen: 'you can say what "too small to represent" means for a Float32 weight.',
    config: { net: 'mlp', dataset: 'moons', hidden: [8, 8], learningRate: 0.000001, targetSteps: 2000 },
    afterword: (o) =>
      o.finished && o.trainLoss !== undefined
        ? `After 2 000 steps the loss is still <b>${o.trainLoss.toFixed(4)}</b> — barely moved from ` +
          `where random weights started. Each step's own update is smaller than the gap between one ` +
          `Float32 value and the next, so most of them round away to nothing at all.`
        : 'Press Train — 2 000 steps at a learning rate small enough to lose updates to rounding.',
  },
  {
    id: 5,
    phase: 'how it learns',
    title: 'All the same to begin with',
    concept: 'symmetry breaking',
    setup: 'zero init · 8 hidden units',
    doneWhen: 'you can say why 8 hidden units behave as one.',
    config: { net: 'mlp', dataset: 'xor', hidden: [8], init: 'zeros' },
    afterword: (o) =>
      o.finished && o.trainAccuracy !== undefined
        ? `Stuck at <b>${pct(o.trainAccuracy)}</b>. Every hidden unit started identical and has ` +
          `received an identical gradient every step since — nothing in gradient descent can break ` +
          `a tie it did not start with.`
        : "Press Train and watch the network graph: every edge into one hidden unit stays the " +
          'same colour as every edge into another.',
  },
  {
    id: 6,
    phase: 'how it learns',
    title: 'The signal fades',
    concept: 'vanishing gradients',
    setup: '6 sigmoid layers',
    doneWhen: 'you can point at the gradient-flow bar that is smallest, and say why.',
    config: { net: 'mlp', dataset: 'moons', hidden: [8, 8, 8, 8, 8, 8], hiddenAct: 'sigmoid' },
    afterword: (o) =>
      o.finished && o.gradNorms !== undefined && o.gradNorms.length >= 2
        ? `Layer 1's gradient norm is <b>${(o.gradNorms[0] as number).toExponential(1)}</b> against ` +
          `layer ${o.gradNorms.length}'s <b>${(o.gradNorms[o.gradNorms.length - 1] as number).toExponential(1)}</b> ` +
          `— six sigmoid derivatives, each at most 0.25, multiply together on the way back and shrink ` +
          `the error to almost nothing by the time it reaches the first layer.`
        : "Press Train and watch the gradient-flow bars in Explorer's diagnostics column.",
  },
  {
    id: 7,
    phase: 'how it learns',
    title: 'Learning the noise',
    concept: 'overfitting',
    setup: '20 samples · 64 units · barely any held back',
    doneWhen: 'you can read the parameter-budget readout and say what it means.',
    config: { net: 'mlp', dataset: 'moons', n: 20, hidden: [64], trainFraction: 0.9 },
    afterword: (o) =>
      o.paramCount !== undefined && o.trainRows !== undefined
        ? o.overBudget
          ? `<b>${o.paramCount}</b> parameters for <b>${o.trainRows}</b> training rows — more free ` +
            `numbers than data points. A network this size can memorise the training set outright ` +
            `rather than learn the shape of it.`
          : `<b>${o.paramCount}</b> parameters for <b>${o.trainRows}</b> training rows — under ` +
            'budget this run; widen the hidden layer or drop the sample count further to push it over.'
        : 'Press Train and watch the parameter-budget readout in the Network panel.',
  },
  {
    id: 8,
    phase: 'how it learns',
    title: 'Knowing when to stop',
    concept: 'generalisation',
    setup: 'same run · a real validation split',
    doneWhen: 'you can say what a gap between train and validation accuracy means.',
    config: { net: 'mlp', dataset: 'moons', n: 20, hidden: [64], trainFraction: 0.7 },
    afterword: (o) =>
      o.trainAccuracy !== undefined && o.valAccuracy !== undefined
        ? o.valAccuracy < o.trainAccuracy - 0.05
          ? `Train accuracy <b>${pct(o.trainAccuracy)}</b>, validation <b>${pct(o.valAccuracy)}</b> — ` +
            'a real gap. The network is fitting this sample better than it fits the problem.'
          : `Train accuracy <b>${pct(o.trainAccuracy)}</b>, validation <b>${pct(o.valAccuracy)}</b> — ` +
            'close together this run. The gap challenge 7 predicts is not guaranteed on every seed; try Resample.'
        : 'Press Train — the same network as the last card, with real validation rows held back this time.',
  },
  {
    id: 9,
    phase: 'a map that teaches itself',
    title: 'A grid that finds its own shape',
    concept: 'competitive learning',
    setup: 'SOM 12×12 · concentric circles',
    doneWhen: 'you can say what "competitive" means without using the word "gradient".',
    config: { net: 'som', somDataset: 'circles', cols: 12, rows: 12 },
    afterword: (o) =>
      o.qeEnd !== undefined && o.teEnd !== undefined && o.nodes !== undefined
        ? `Quantisation error settled at <b>${o.qeEnd.toFixed(4)}</b>, topographic error at ` +
          `<b>${pct(o.teEnd)}</b> — a grid of <b>${o.nodes}</b> nodes found the ring's own shape with ` +
          'no label ever telling it where the ring was.'
        : 'Press Train and watch the lattice organise itself around a shape it was never described.',
  },
  {
    id: 10,
    phase: 'a map that teaches itself',
    title: 'Cooling too fast',
    concept: 'the schedule — QE down, TE up',
    setup: 'σ decays in 50 steps, the run keeps going',
    doneWhen: 'you can say why QE and TE moving in opposite directions is possible at all.',
    config: {
      net: 'som',
      somDataset: 'circles',
      cols: 12,
      rows: 12,
      somTargetSteps: 3000,
      scheduleSteps: 50,
    },
    afterword: (o) =>
      o.qeStart !== undefined && o.qeEnd !== undefined && o.teStart !== undefined && o.teEnd !== undefined
        ? `Quantisation error fell from <b>${o.qeStart.toFixed(3)}</b> to <b>${o.qeEnd.toFixed(3)}</b> ` +
          `while topographic error climbed from <b>${pct(o.teStart)}</b> to <b>${pct(o.teEnd)}</b> — ` +
          'the lattice kept fitting individual points more closely long after it had stopped moving ' +
          'as one connected sheet, which is exactly what a schedule that finished cooling at step 50 predicts.'
        : 'Press Train — the schedule finishes cooling at step 50; the run keeps going to 3 000.',
  },
  {
    id: 11,
    phase: 'a map that teaches itself',
    title: 'The map has edges',
    concept: 'structure without labels',
    setup: 'U-matrix · three blobs, unlabelled',
    doneWhen: 'you can point at a U-matrix ridge and say what it marks.',
    config: { net: 'som', somDataset: 'blobs', cols: 12, rows: 12 },
    afterword: (o) =>
      o.uMax !== undefined
        ? `The U-matrix's brightest ridge sits at <b>${o.uMax.toFixed(3)}</b> — two lattice neighbours ` +
          'that far apart in the data mark a gap nobody labelled: the boundary between two of the ' +
          'three blobs, found without a single class name ever entering training.'
        : 'Press Train, then look at the U-matrix panel — the ridges are the boundaries the map found on its own.',
  },
  {
    id: 12,
    phase: 'seeing it both ways',
    title: 'Two ways to see one dataset',
    concept: 'unsupervised vs supervised',
    setup: 'the same three blobs · MLP and SOM',
    doneWhen: 'you can say what each network was told that the other was not.',
    config: { net: 'mlp', dataset: 'blobs', hidden: [8, 8], somDataset: 'blobs', cols: 12, rows: 12 },
    afterword: (o) => {
      if (o.mlpFinished && o.somFinished) {
        return (
          'Both finished on the same data. The perceptron found a boundary because it was told ' +
          'the right answer for every point; the map found structure because it never needed one.'
        );
      }
      if (o.mlpFinished) return 'The perceptron side is done — switch to Kohonen above and train the map on the same blobs.';
      if (o.somFinished) return 'The map is done — switch to Perceptron above and train it on the same blobs.';
      return 'Train each network on the same data, then compare what "understanding" turned out to mean for each.';
    },
  },
];

export function challengeById(id: number): Challenge | undefined {
  return CHALLENGES.find((c) => c.id === id);
}

/** Phases in ladder order, each exactly once — the concept-dot row's four groups. */
export function phases(): readonly string[] {
  const seen: string[] = [];
  for (const c of CHALLENGES) if (!seen.includes(c.phase)) seen.push(c.phase);
  return seen;
}
