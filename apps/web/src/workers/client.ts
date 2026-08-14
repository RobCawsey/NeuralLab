/**
 * The page's side of the worker.
 *
 * Thin on purpose: it owns the `Worker`, turns callbacks into events, and keeps the one piece of
 * bookkeeping that cannot live on either side alone — matching a probe reply to the request that
 * asked for it, so a stale field from a smaller resolution cannot overwrite a fresher one.
 */

import type { Field } from '../render/field.ts';
import type { Box } from '../render/camera.ts';
import type { StepTrace, TrainConfig } from '@neurallab/mlp';
import type { FromWorker, RunPoint, ToWorker, TrainSetup } from './protocol.ts';

export interface TrainerEvents {
  onReady?: (weights: Float32Array, initMs: number) => void;
  onReport?: (report: {
    step: number;
    epoch: number;
    points: readonly RunPoint[];
    weights: Float32Array;
    stepsPerSecond: number;
    diverged: boolean;
    running: boolean;
  }) => void;
  onField?: (field: Field, ms: number) => void;
  onTrace?: (trace: StepTrace, weights: Float32Array, step: number) => void;
  onError?: (message: string) => void;
}

export class TrainerClient {
  private readonly worker: Worker;
  private readonly events: TrainerEvents;
  private nextRequest = 1;
  /** The only probe reply worth accepting. Anything older is a field for weights that moved on. */
  private awaitingProbe = 0;
  private awaitingTrace = 0;
  private probeBox: Box | null = null;

  constructor(events: TrainerEvents) {
    this.events = events;
    this.worker = new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<FromWorker>) => this.receive(event.data);
    this.worker.onerror = (event) => this.events.onError?.(event.message || 'worker failed');
  }

  private send(message: ToWorker): void {
    this.worker.postMessage(message);
  }

  private receive(message: FromWorker): void {
    /*
     * Anything from a session the page has already replaced is dropped.
     *
     * A rebuild does not cancel the messages already in flight, and a report that arrives after
     * one carries the *old* network's weights. When the architecture changed too, `applyWeights`
     * throws on the length — which is how this was found, as `weight buffer is 114, expected
     * 354`. The dangerous case is the one that does not throw: changing only the dataset keeps
     * the shape, so stale weights would apply cleanly and the graph would show a network that no
     * longer exists.
     */
    if (message.type !== 'error' && message.generation !== this.generation) return;

    switch (message.type) {
      case 'ready':
        this.events.onReady?.(message.weights, message.initMs);
        return;
      case 'report':
        this.events.onReport?.(message);
        return;
      case 'probe': {
        /*
         * Drop everything but the newest request.
         *
         * Probes are asked for while training, so two can be outstanding when a resize or a
         * pause changes the resolution. Replies do not necessarily arrive in order of
         * usefulness, and a 64² field landing after a 128² one would visibly coarsen the
         * boundary for no reason the reader could account for.
         */
        if (message.requestId !== this.awaitingProbe || this.probeBox === null) return;
        this.events.onField?.(
          {
            res: message.res,
            cls: message.cls,
            conf: message.conf,
            box: this.probeBox,
            classes: this.classes,
          },
          message.ms,
        );
        return;
      }
      case 'trace':
        if (message.requestId !== this.awaitingTrace) return;
        this.events.onTrace?.(message.trace, message.weights, message.step);
        return;
      case 'error':
        this.events.onError?.(message.message);
        return;
    }
  }

  /** Set by `init`/`reset`, because the field's alpha floor depends on it. */
  private classes = 2;

  /** Which rebuild the page is currently interested in. Anything older is discarded. */
  private generation = 0;

  /** The generation to stamp on the next setup — the caller reads it into `trainSetup`. */
  nextGeneration(): number {
    return this.generation + 1;
  }

  init(setup: TrainSetup, classes: number): void {
    this.classes = classes;
    this.generation = setup.generation;
    this.send({ type: 'init', setup });
  }

  reset(setup: TrainSetup, classes: number): void {
    this.classes = classes;
    this.generation = setup.generation;
    this.awaitingProbe = 0;
    this.awaitingTrace = 0;
    this.send({ type: 'reset', setup });
  }

  run(untilStep: number): void {
    this.send({ type: 'run', untilStep });
  }

  pause(): void {
    this.send({ type: 'pause' });
  }

  configure(train: TrainConfig): void {
    this.send({ type: 'config', train });
  }

  /** Run one traced step. Advances the run — see the protocol note on `trace`. */
  requestTrace(indexInBatch: number): void {
    this.awaitingTrace = this.nextRequest++;
    this.send({ type: 'trace', requestId: this.awaitingTrace, indexInBatch });
  }

  requestField(res: number, box: Box): void {
    this.awaitingProbe = this.nextRequest++;
    this.probeBox = box;
    this.send({ type: 'probe', requestId: this.awaitingProbe, res, box });
  }

  dispose(): void {
    this.worker.terminate();
  }
}
