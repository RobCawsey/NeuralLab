export {
  NEIGHBOUR_SLOTS,
  buildNeighbours,
  latticeDistance,
  nodeIndex,
  type Topology,
} from './lattice.ts';
export {
  alphaAt,
  bmu,
  bmu2,
  createSom,
  neighbourhoodStrength,
  sigmaAt,
  type Decay,
  type Schedule,
  type Som,
} from './som.ts';
export { createSomTrainer, somStep, type SomTrainer, type StepResult } from './train.ts';
export { componentPlane, quantisationError, topographicError, uMatrix } from './metrics.ts';
