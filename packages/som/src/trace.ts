/**
 * A recording of one step, for the stepper — the SOM side's answer to `packages/mlp/src/trace.ts`.
 *
 * Five stages, not seven: **sample → distances → BMU → neighbourhood → update**. There is no
 * forward/backward split to trace between, so `somStep` builds this directly rather than calling
 * out to a sibling function — the per-node loop it already runs to apply the update is the same
 * loop that has to visit every node to report a distance and a strength, and splitting that into
 * two loops would either duplicate it or force an awkward second pass.
 *
 * Every array covers every node, not just the ones whose weights actually moved: the `h < 1e-7`
 * shortcut `somStep` takes for the update itself does not apply here, because the "distances" and
 * "neighbourhood" stages are heatmaps of the *whole* lattice.
 */

export interface SomStepTrace {
  /** The step this trace belongs to — the one that has just been applied. */
  readonly step: number;
  readonly row: number;
  readonly input: Float32Array;
  readonly bmu: number;
  readonly alpha: number;
  readonly sigma: number;
  /** `‖x − w‖` per node, before the update — the "distances" stage. */
  readonly distances: Float32Array;
  /** `h(d, σ)` per node — the "neighbourhood" stage. */
  readonly strength: Float32Array;
  /** Every node's weight vector before this step's update, `cols * rows * dim`, node-major. */
  readonly before: Float32Array;
}
