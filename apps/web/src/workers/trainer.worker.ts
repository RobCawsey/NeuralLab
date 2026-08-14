/**
 * The training worker. Owns the network and the data; the page owns a copy of the weights.
 *
 * Slice 2 trained on the main thread inside a frame budget, which worked and had two costs the
 * design document predicted: the decision field competed with training for the same thread
 * (measured at 19% in slice 3), and `requestAnimationFrame` does not fire in a background tab, so
 * a run stopped the moment the reader looked away. Neither is true here.
 *
 * **The step sequence is unchanged.** `trainStep` is the same function, called in the same order,
 * on data rebuilt from the same seed — so the golden run reproduces exactly, and a test asserts
 * it against the number slice 2 pinned. How steps are grouped into chunks changes nothing,
 * which is invariant 2 stated one more time.
 */

import { evaluateRows, flattenWeights, trainStep, type Net, type Trainer } from '@neurallab/mlp';
import { createScratch, createTrainer, createTraceScratch, type Scratch, type TraceScratch } from '@neurallab/mlp';
import { Rng, type Dataset, type Split, type Standardiser } from '@neurallab/core';
import { buildData, buildNet } from '../run/build.ts';
import { computeField } from '../render/field.ts';
import { probeTransfer, type FromWorker, type RunPoint, type ToWorker, type TrainSetup } from './protocol.ts';

/**
 * How long the worker trains before yielding.
 *
 * It has to yield: `onmessage` cannot run while a loop is running, so a worker that trained to
 * completion in one go could not be paused. 40 ms gives roughly 25 reports a second — faster
 * than anything on screen can be read, and slow enough that the yield overhead is negligible.
 */
const CHUNK_MS = 40;

interface Session {
  readonly setup: TrainSetup;
  readonly net: Net;
  readonly trainer: Trainer;
  readonly z: Dataset;
  readonly parts: Split;
  readonly standardiser: Standardiser;
  readonly evalScratch: Scratch;
  readonly probeScratch: Scratch;
  /** Its own buffers again, so a trace cannot touch the gradients the update is about to use. */
  readonly traceScratch: TraceScratch;
  readonly weights: Float32Array;
  /** Loss spread accumulated since the last chart point. */
  lossSum: number;
  lossCount: number;
  lossMin: number;
  lossMax: number;
  pending: RunPoint[];
  untilStep: number;
  running: boolean;
  startedAt: number;
  stepsAtStart: number;
}

let session: Session | null = null;

/*
 * `self` is a `Window` as far as the compiler is concerned, because tsconfig carries the DOM lib
 * and not WebWorker — the app needs DOM everywhere else, and adding both would give every file in
 * `apps/web` a `postMessage` that means two different things.
 *
 * So the two members this file actually uses are declared, rather than pulling in a second global
 * lib or reaching for `any`. It is four lines, and it keeps the transfer list type-checked, which
 * is the part worth checking.
 */
interface WorkerScope {
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ToWorker>) => void) | null;
}

const ctx = self as unknown as WorkerScope;

function post(message: FromWorker, transfer?: Transferable[]): void {
  if (transfer) ctx.postMessage(message, transfer);
  else ctx.postMessage(message);
}

function open(setup: TrainSetup): Session {
  const built = buildData(setup.data);
  const net = buildNet(setup.net, built.data.dim, built.data.classes);
  return {
    setup,
    net,
    // The trainer's shuffle Rng is seeded from the weight seed, so reinitialising and training
    // again replays exactly: the same weights *and* the same batch order.
    trainer: createTrainer(net, built.parts.train, setup.train, new Rng(setup.net.weightSeed)),
    z: built.z,
    parts: built.parts,
    standardiser: built.standardiser,
    // Three separate scratches. Sharing one would let an evaluation overwrite the activations
    // the backward pass is about to read — gradients for the wrong sample, which trains almost
    // correctly and is very hard to see.
    evalScratch: createScratch(net),
    probeScratch: createScratch(net),
    traceScratch: createTraceScratch(net),
    weights: new Float32Array(flattenWeights(net).length),
    lossSum: 0,
    lossCount: 0,
    lossMin: Infinity,
    lossMax: -Infinity,
    pending: [],
    untilStep: 0,
    running: false,
    startedAt: 0,
    stepsAtStart: 0,
  };
}

/** Close off a chart point: the loss band since the last one, plus a full evaluation. */
function takePoint(s: Session): void {
  const tr = evaluateRows(s.net, s.z, s.parts.train, s.evalScratch);
  const va = evaluateRows(s.net, s.z, s.parts.val, s.evalScratch);
  s.pending.push({
    step: s.trainer.step,
    epoch: s.trainer.epoch,
    loss: s.lossCount > 0 ? s.lossSum / s.lossCount : tr.loss,
    lossMin: Number.isFinite(s.lossMin) ? s.lossMin : tr.loss,
    lossMax: Number.isFinite(s.lossMax) ? s.lossMax : tr.loss,
    trainLoss: tr.loss,
    valLoss: va.loss,
    trainAccuracy: tr.accuracy,
    valAccuracy: va.accuracy,
  });
  s.lossSum = 0;
  s.lossCount = 0;
  s.lossMin = Infinity;
  s.lossMax = -Infinity;
}

function report(s: Session): void {
  flattenWeights(s.net, s.weights);
  const elapsed = (performance.now() - s.startedAt) / 1000;
  post({
    type: 'report',
    generation: s.setup.generation,
    step: s.trainer.step,
    epoch: s.trainer.epoch,
    points: s.pending,
    // A copy. The worker keeps training into `s.weights` the moment this returns, so handing
    // over the buffer itself would either detach it or race.
    weights: Float32Array.from(s.weights),
    stepsPerSecond: elapsed > 0.05 ? (s.trainer.step - s.stepsAtStart) / elapsed : 0,
    diverged: s.trainer.diverged,
    running: s.running,
  });
  s.pending = [];
}

function pump(): void {
  const s = session;
  if (!s || !s.running) return;

  const started = performance.now();
  while (s.running && s.trainer.step < s.untilStep && performance.now() - started < CHUNK_MS) {
    const m = trainStep(s.trainer, s.z);
    s.lossSum += m.loss;
    s.lossCount++;
    if (m.lossMin < s.lossMin) s.lossMin = m.lossMin;
    if (m.lossMax > s.lossMax) s.lossMax = m.lossMax;

    if (m.step % s.setup.evalEvery === 0 || m.step === s.untilStep) takePoint(s);

    if (s.trainer.diverged) {
      s.running = false;
      break;
    }
  }

  if (s.trainer.step >= s.untilStep) s.running = false;
  report(s);

  // setTimeout rather than a tight loop: it is the yield that lets `pause` be received at all.
  if (s.running) setTimeout(pump, 0);
}

/**
 * Evaluate the field, using the same `computeField` the main thread used in slice 3.
 *
 * Imported rather than reimplemented. The first version of this worker had its own copy of the
 * grid loop, which would have meant slice 3's tests — cell centres, bottom-left origin, the
 * standardiser — covering a function that no longer ran anywhere.
 */
function probe(
  s: Session,
  requestId: number,
  res: number,
  box: { minX: number; maxX: number; minY: number; maxY: number },
): void {
  const started = performance.now();
  const field = computeField(
    s.net,
    s.probeScratch,
    s.standardiser,
    box,
    res,
    Math.max(2, s.z.classes),
  );
  post(
    {
      type: 'probe',
      generation: s.setup.generation,
      requestId,
      res,
      cls: field.cls,
      conf: field.conf,
      ms: performance.now() - started,
    },
    probeTransfer(field.cls, field.conf),
  );
}

ctx.onmessage = (event: MessageEvent<ToWorker>): void => {
  const message = event.data;
  try {
    switch (message.type) {
      case 'init':
      case 'reset': {
        const started = performance.now();
        session = open(message.setup);
        post({
          type: 'ready',
          generation: message.setup.generation,
          initMs: performance.now() - started,
          weights: flattenWeights(session.net),
        });
        return;
      }
      case 'run': {
        if (!session) return;
        session.untilStep = message.untilStep;
        if (session.trainer.step >= session.untilStep) return;
        session.running = true;
        session.startedAt = performance.now();
        session.stepsAtStart = session.trainer.step;
        pump();
        return;
      }
      case 'pause': {
        if (!session) return;
        session.running = false;
        // Report immediately so the page's step counter matches where the worker actually is,
        // rather than wherever the last chunk happened to end.
        report(session);
        return;
      }
      case 'config': {
        if (!session) return;
        session.trainer.config = message.train;
        return;
      }
      case 'probe': {
        if (!session) return;
        probe(session, message.requestId, message.res, message.box);
        return;
      }
      case 'trace': {
        const s = session;
        if (!s) return;
        // Stop first: a trace is a deliberate single step, and letting the run continue
        // underneath would make the recording describe a network that had already moved on.
        s.running = false;
        const m = trainStep(s.trainer, s.z, {
          trace: { indexInBatch: message.indexInBatch, into: s.traceScratch },
        });
        if (m.step % s.setup.evalEvery === 0) takePoint(s);
        if (m.trace !== undefined) {
          post({
            type: 'trace',
            generation: s.setup.generation,
            requestId: message.requestId,
            trace: m.trace,
            step: s.trainer.step,
            epoch: s.trainer.epoch,
            weights: flattenWeights(s.net),
          });
        }
        report(s);
        return;
      }
    }
  } catch (error) {
    // Never let a worker die silently. A dead worker looks exactly like a paused one.
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
