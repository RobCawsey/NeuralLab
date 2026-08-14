/**
 * The message protocol between the page and the training worker.
 *
 * Raw `postMessage` with a discriminated union, not an RPC wrapper. There are six message types
 * each way, and the one thing that genuinely needs care — which buffers are copied and which are
 * transferred — is exactly what an RPC wrapper hides. §12 of the design document.
 *
 * **The worker is not sent the dataset.** It is sent the configuration and rebuilds it from the
 * same seed with the same function, so the two sides agree by construction rather than by
 * bookkeeping. See `run/build.ts`.
 */

import type { DataConfig, NetConfig } from '../run/build.ts';
import type { StepTrace, TrainConfig } from '@neurallab/mlp';

export interface TrainSetup {
  /**
   * Which rebuild this is. Echoed back on every message the session produces.
   *
   * Without it a report already in flight when the architecture changes lands on a mirror that
   * has been rebuilt to a different shape. `applyWeights` catches the size mismatch and throws —
   * which is how this was found — but a rebuild that only changed the *data* keeps the same
   * shape, and there the stale weights would apply cleanly and silently be wrong.
   */
  readonly generation: number;
  readonly data: DataConfig;
  readonly net: NetConfig;
  readonly train: TrainConfig;
  /** How often the full train/validation sets are measured, in steps. */
  readonly evalEvery: number;
}

/**
 * One point on the chart, and there is only one kind.
 *
 * Slice 2 kept two arrays — a per-step loss for the band and a periodic evaluation for the lines
 * — which meant the chart drew two series sampled at different rates and had to reconcile them.
 * The worker produces both at the same moment now: the band aggregates every step since the last
 * point, so nothing is dropped and nothing is invented.
 */
export interface RunPoint {
  readonly step: number;
  readonly epoch: number;
  /** Mean minibatch loss across the steps this point covers. */
  readonly loss: number;
  /** The spread of individual sample losses within those steps. Minibatch loss is very noisy. */
  readonly lossMin: number;
  readonly lossMax: number;
  readonly trainLoss: number;
  readonly valLoss: number;
  readonly trainAccuracy: number;
  readonly valAccuracy: number;
}

export type ToWorker =
  | { readonly type: 'init'; readonly setup: TrainSetup }
  /** Train until this step, reporting as it goes. */
  | { readonly type: 'run'; readonly untilStep: number }
  | { readonly type: 'pause' }
  /** Rebuild from a new configuration without tearing the worker down. */
  | { readonly type: 'reset'; readonly setup: TrainSetup }
  | { readonly type: 'config'; readonly train: TrainConfig }
  /**
   * Evaluate the network across a grid — the decision field.
   *
   * Requested by the page rather than pushed, because only the page knows how big the panel is
   * and whether the last one is still good enough.
   */
  | {
      readonly type: 'probe';
      readonly requestId: number;
      readonly res: number;
      readonly box: { readonly minX: number; readonly maxX: number; readonly minY: number; readonly maxY: number };
    }
  /**
   * Run one step and record a sample of it — the stepper.
   *
   * It advances the run by one step, because a trace of a step that did not happen would be a
   * prediction rather than a recording. The stepper *is* stepping.
   */
  | { readonly type: 'trace'; readonly requestId: number; readonly indexInBatch: number };

export type FromWorker =
  | {
      readonly type: 'ready';
      readonly generation: number;
      readonly initMs: number;
      readonly weights: Float32Array;
    }
  | {
      readonly type: 'report';
      readonly generation: number;
      readonly step: number;
      readonly epoch: number;
      readonly points: readonly RunPoint[];
      /** A **copy**, not a transfer — see below. */
      readonly weights: Float32Array;
      readonly stepsPerSecond: number;
      readonly diverged: boolean;
      readonly running: boolean;
    }
  | {
      readonly type: 'probe';
      readonly generation: number;
      readonly requestId: number;
      readonly res: number;
      readonly cls: Uint8Array;
      readonly conf: Float32Array;
      readonly ms: number;
    }
  | {
      readonly type: 'trace';
      readonly generation: number;
      readonly requestId: number;
      readonly trace: StepTrace;
      readonly step: number;
      readonly epoch: number;
      readonly weights: Float32Array;
    }
  | { readonly type: 'error'; readonly message: string };

/**
 * Buffers to hand to `postMessage`'s transfer list for a probe result.
 *
 * The field is the one payload worth transferring: at 128² it is 16 384 cells across two arrays,
 * regenerated from scratch every time, so the worker has no use for them once sent.
 *
 * The weights go the other way and are **copied**. A 2-8-8-2 network is 354 floats — 1.4 kB —
 * and even 64-128-128-10 is 106 kB. Transferring would detach the worker's own buffer, so it
 * would have to allocate a fresh one every report anyway; the copy is the cheaper of the two and
 * far the simpler. §5.
 */
export function probeTransfer(cls: Uint8Array, conf: Float32Array): Transferable[] {
  return [cls.buffer as ArrayBuffer, conf.buffer as ArrayBuffer];
}
