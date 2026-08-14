export {
  ACTIVATIONS,
  activate,
  activationLabel,
  isActivation,
  softmax,
  type Activation,
} from './activations.ts';
export {
  INIT_SCHEMES,
  argmax,
  createNet,
  createScratch,
  describeShape,
  forward,
  initialise,
  isInitScheme,
  maxAbsWeight,
  paramCount,
  parseHidden,
  shapeOf,
  type Dense,
  type InitScheme,
  type LossKind,
  type Net,
  type NetSpec,
  type Scratch,
} from './net.ts';
export {
  derivativeFromOutput,
  outputDelta,
  sampleLoss,
} from './loss.ts';
export {
  backward,
  createGrads,
  gradNorm,
  hasDiverged,
  scaleGrads,
  sgdStep,
  zeroGrads,
  type Grads,
  type OptimiserKind,
} from './backward.ts';
export {
  DEFAULT_TRAIN,
  createTrainer,
  evaluateRows,
  trainStep,
  type EvalResult,
  type StepMetrics,
  type TrainConfig,
  type Trainer,
} from './train.ts';
