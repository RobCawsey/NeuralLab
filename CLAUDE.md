# NeuralLab

A browser workbench for building a Kohonen map and a multilayer perceptron, training them on
data you can see, and learning how both work by watching them happen. Personal project, built in
slices. Teaching tool first, framework never.

Sibling project to **Evolab** (`F:\Evolab`) — same chassis, same rules, same slice discipline.
Where a decision here has no reason of its own, the reason is "Evolab does it this way and it
worked", and that is a good enough reason.

One document does the architectural job:

- **[docs/technical-design.html](docs/technical-design.html)** — the architecture and UI
  specification. Open it in a browser. 13 sections, 8 interface mockups, every decision and
  *why* it was made. Stable; changes only when a decision changes.

When this file and the design document disagree, the design document wins.

## Current state

**Slice 12 — "3D".** A `2D / 3D` toggle in the toolbar, or the `2`/`3` keys. The MLP gets an
orbitable loss surface — two random, filter-normalised directions through weight space by
default, with a `literal` mode that varies two named weights instead, and the run's own path
traced across whichever is showing. The SOM gets its lattice folding through input space in three
dimensions: nodes at their own weight vectors, connected by their lattice edges, floating in a
thinned sample of the data. Both are dynamically imported — `three` is not in the graph any
request reaches before a reader presses `3` for the first time. 296 tests.

**The snapshot ring was an open question, and it is answered now: capped at 60, in count, not
megabytes.** §13 asked whether an epoch scrubber was worth keeping weights every 20th report and
flagged the number needed stating in steps rather than size. `AppState.snapshots` settles it:
60 entries regardless of run length, the same "pace by a fixed count, not by how long the run
happens to be" rule `SOM_TICKS` already used for the training pump. At 60 snapshots the largest
network this project will reach (64-128-128-10, slice 16) is under 6.4 MB; the default 2-8-8-2 is
under 7 kB. One snapshot is taken per worker report that closed a chart point, not per report —
reports arrive around 25 times a second, points at most 200 times a run.

**The loss surface and the path share one flat-weight-space kernel with the run's own weight
buffers, not a parallel notion of "direction".** A `Direction` is exactly a `Float32Array` the
same shape `flattenWeights`/`applyWeights` already use, so projecting a snapshot onto it is a
plain dot product and evaluating the surface is `applyWeights` into a scratch network followed by
`forward`. Nothing here perturbs the real network — `computeLossSurface` reads `flattenWeights`
once and runs every grid cell through its own scratch copy, the same "own scratch" rule
`evaluateRows` and the stepper's trace already follow, checked by a test that asserts the real
network's weights are byte-identical before and after a surface is computed.

**Filter-normalisation is checked against the real network's own layer norms, and the hand-worked
grid test is what actually proves the arithmetic.** A 1-1 linear network with `mse` loss has a
closed form, `loss(w, b) = 0.5·(wx + b − 1)²`, worked out on paper and checked against a 3×3 grid
exactly — the same discipline the SOM's lattice table used in slice 9, applied here to a
continuous surface instead of a discrete one.

**Literal and representative modes produce measurably different pictures, and the difference
is the lesson the design document asked for — verified live, not merely quoted.** On the trained
golden run, representative mode's path (projected onto two random directions) stays visibly short
near the current-position marker; switching to literal mode (two named weights) on the same
trained network shows a path several times longer, because an individual weight's own trajectory
correlates with itself in a way a random projection of 114 dimensions mostly does not. Both
pictures are real; only their honesty about what two dimensions out of many can show differs.

**A custom orbit camera, not `three`'s own `OrbitControls` addon.** Spherical coordinates around a
fixed target, updated from three pointer listeners and one wheel listener, reset on double-click —
about eighty lines, and it keeps the dependency surface at the one `three` package itself rather
than also importing from `three/examples`. Shared by both 3D scenes rather than written twice.

**No bug needed fixing to get this working, which is itself worth recording plainly rather than
inventing one.** The one place a mistake was designed around rather than found live:
`render/scatter.ts`'s `resize()` calls `canvas.getContext('2d')`, which permanently locks a canvas
out of ever getting a WebGL context afterward — so the 3D canvases get their own `size3d()`, which
only ever reads `clientWidth`/`clientHeight`. Known going in, from how the 2D field's own canvas
helper works, rather than discovered by a crash.

**Slice 11 — "The SOM stepper, and the second guided flow".** Both networks now get the two
teaching screens the project exists for. The stepper pages through five stages — sample →
distances → BMU → neighbourhood → update — with the lattice heatmapped per stage beside a second
view of the same map folded through input space, node-to-node edges and all. The guided flow adds
`SOM_FLOW` beside `MLP_FLOW`: pick data, watch it fold, see the U-matrix's ridges, then label the
map from data it was never trained on the answer for. 288 tests.

**No worker round trip, because there is nothing on the other side of one.** The MLP stepper
requests a trace and waits for `onTrace`, since training runs on a different thread. SOM training
runs on this one (slice 10's own finding), so "request a trace" is `somStep(trainer, ds, {trace:
true})`, called directly and returned synchronously. `SomStepTrace` is built *inside* `somStep`
rather than by a sibling function — the per-node loop the update already runs is the same loop
that has to visit every node to report a distance and a strength, and splitting that into two
loops would either duplicate it or force an awkward second pass. A test proves tracing changes
nothing: 200 steps traced every time land on bit-identical weights to 200 steps never traced.

**Two distinct pictures share the stepper screen, and neither one alone is the algorithm.** The
lattice view shows what the update looks like in the map's own coordinates — a heatmap of
distance, then of neighbourhood strength, with the BMU ringed. The input-space view shows the same
step where the *data* lives: every node's weight vector as a point, connected to its lattice
neighbours by a line — literally the net the map is, folded through the space it is learning. The
BMU's own before → after line is the one thing drawn only on the update stage. Built once as
`render/inputspace.ts` rather than as a stepper-only view, because the guided flow's own step 2
("watch a flat sheet fold into it") is the same drawing at a different moment.

**Labelling a map is a vote taken after the fact, not a training signal — `nodeLabels` says so in
its own return value.** Every training row's BMU gets one vote for that row's class; a node's
label is whichever class won it most, and a node no row ever won reads `-1` rather than `0`, so
"unlabelled" and "labelled class zero" cannot be confused by an off-by-one. Datasets with no `y`
at all — the colour cube — return every node as `-1`, and the guided flow says so in plain words
rather than drawing an empty grid and letting a reader wonder why.

**The "Reveal labels" button was clicked and nothing happened, and that is the second render-order
bug this slice found live.** Unlike picking a dataset, revealing labels changes no state a rebuild
would render as a side effect — there is no `regenerateSomData()` for the guided controller's own
click handler to piggyback a render on. Fixed by giving `SomGuidedOptions` a `requestRender`
callback and calling it explicitly, the same shape of fix slice 6 needed for the MLP flow's own
step-advance ordering, found the same way: clicking the button and watching the screen not change.

**`ui/guided.ts` is not literally one function for both networks, and §13's "one renderer" turned
out to mean something narrower once there were two state shapes to read.** `AppState` and
`SomState` share nothing; a single polymorphic controller would need as many branches as two
controllers have lines. What *is* shared, because it is genuinely one idea, is the step-card and
choice-button vocabulary — extracted into `ui/guidedShared.ts` once `ui/somGuided.ts` needed it a
second time, rather than copied.

**Slice 10 — "Reading a map".** The Kohonen switch has something behind it now: a full second
Explorer, the lattice drawn hex or rect with nodes filled by their own weight vector, a QE/TE
chart, the U-matrix and component planes as `ImageData` blits, and controls for dataset, lattice
size, topology and schedule. Two parallel DOM trees gated by `body[data-net]`, the same pattern
`data-stage` already used for Guided/Explorer, so switching networks destroys neither side's
state. 273 tests.

**Trained on the main thread, and measured before deciding that, not assumed.** A SOM step has no
backward pass — a nearest-node search and a linear pull over at most a few hundred prototypes —
and benchmarks at roughly 240 000 steps/s on a 12×12 map, about twenty times the MLP's own
pre-worker throughput. A full 20 000-step run finishes in under 100ms of raw compute, so nothing
here would be bought back by a worker. The pacing a reader actually sees — a run visibly organising
over about two seconds regardless of its step count — is a fixed 120-tick schedule, not a
performance constraint.

**That decision immediately re-triggered slice 3's exact bug, and running it live is what caught
it.** The first version paced those ticks with `requestAnimationFrame`, which does not fire in a
hidden tab — precisely the limitation slice 4's worker was built to escape for the MLP, reintroduced
by choosing not to build one here. Verified live: `document.hidden` was `true` in the very
environment used to test this, and training sat at step 0 indefinitely. Fixed by switching to
`setTimeout`, which browsers throttle in the background rather than suspending outright — a run
paces slower while nobody is watching and still finishes, rather than never finishing at all.

**A second bug in the same loop: the "finished" render ran one step too early.** `s.running` was
set to `false` *after* the pump's own `render()` call rather than before it, so the final frame of
a completed run painted with the Train button still reading "Pause" and the badge still reading
"training" — correct only once some *other* render happened to fire afterwards, which nothing did.
Moving the finished-check before that last render fixed it; the general lesson, restated from the
MLP side's own ordering bugs, is that anything a render reads has to be settled before the render
runs, not after.

**A third bug had nothing to do with training at all: `#som-lattice` was invisible at its actual
size.** `resize()` measures a canvas's CSS box, and the project's existing rule that gives `#stage`
and `#graph` `width/height: 100%` is scoped to those two ids specifically — a new canvas id needs
adding to it explicitly, or it falls back to the browser default of 300×150 and every draw call
still succeeds, quietly, into a box a tenth the size of its panel. Caught by screenshot, not by any
test, because nothing about a wrong CSS rule throws.

**SOM datasets are mostly the MLP's own generators, reused unsupervised.** `colourCube` is the one
built for this half specifically, but moons/circles/blobs/spirals/XOR all already produce a
`Dataset`, and a SOM simply never reads `y` — §3's rule that "unlabelled" describes training, not
the file, made this free rather than five new generators to write and verify.

**Weight-to-colour is literal for the colour cube and a documented stand-in otherwise, with real
projection deferred on purpose.** Three weights map straight to r/g/b when `dim` is 3; below that
a fixed mid-tone fills the missing channels. Nothing in the current dataset roster exceeds three
dimensions — the one that will (digits, dim 64) arrives in slice 16 — so a real PCA projection has
no data to be wrong against yet and stays out until it does.

**Slice 9 — "Kohonen kernel".** `packages/som`: the hex lattice, axial coordinates, BMU, the
neighbourhood function, three decay schedules, quantisation error, topographic error, the
U-matrix, and a golden run — a 12×12 hex map ordering itself on the colour cube, printed as real
terminal colour by `npm run som`. Pure, headless, no app wiring; the disabled Kohonen button's
tooltip is corrected rather than left promising a slice number this one still doesn't deliver.
265 tests.

**Two distinct notions of "neighbour" live in `packages/som/src/lattice.ts`, and conflating them
is the second-easiest way to get this package wrong.** `latticeDistance` is continuous and
answers `h(d, t)`'s question — every node in the lattice has one, however large. `neighbours` is
discrete and answers "which nodes does this one touch" — used for topographic error and the
U-matrix. For hex the two agree (distance 1 in the metric is exactly the six neighbours), but for
rect they deliberately don't: the design document is explicit that a rect node's four diagonal
cells at distance √2 are *not* lattice neighbours for TE or the U-matrix, even though
`latticeDistance` sees them as close. Getting this backwards would still train something that
looks plausible, which is exactly the failure mode the design document warns about.

**The neighbour table is tested against a 3×3 hand count, worked out on paper before any code
ran — not trusted after the fact.** Odd rows are shoved right by half a hex ("odd-r"), which means
a shifted row's diagonal neighbours land at the *same* column and the *next* column over in the
row above and below, not one column either side as Euclidean intuition suggests. The derivation is
in `lattice.test.ts`'s comments, not just its assertions, so a reader can check the geometry
without re-deriving it: centre `(1,1)` touches all six neighbours of a 3×3 grid, corner `(0,0)`
touches exactly two. All eleven tests passed on the first run against numbers worked out by hand,
which is the point of doing it that way rather than asserting whatever the code happened to
produce.

**Quantisation error does not fall below its random-init reading on the colour cube, and that
took a probe script to understand rather than a guess.** `createSom` draws weights uniform in
`[0, 1)` specifically to match the colour cube's own range — which means a fresh random map is
144 points drawn from the *exact* data distribution, an unusually strong quantiser with zero
structure behind it (topographic error 0.97 at step 0: essentially every sample's best and
second-best nodes are unrelated). Training pulls the lattice into a coherent sheet, which costs
some of that raw quantising power in exchange for the property a SOM actually promises. Checked
by sampling QE at nine checkpoints across a 3 000-step run: it spikes to 0.36 as soon as topology
starts to matter, then falls monotonically to 0.12 — the real, honest "goes down as the map fits
the data" story, just measured from *after* the initial reorganisation rather than from the random
start. The golden test pins QE from step 300, not step 0, and asserts the random baseline's own
topographic error separately so the exception is checked rather than merely asserted away.

**The exponential schedule was retuned after the first run, live, not guessed in advance.**
`v0 · e⁻¹` remaining at the horizon (the textbook constant) measured visibly under-converged: a
12×12 map still had a *higher* QE after 3 000 steps than its random init, because σ was still 2.2
hex-units wide at the very last step — plenty to keep the lattice smoothed into a manifold rather
than letting individual nodes settle. Retuned to `v0 · e⁻³` (~5% remaining), which is what the
golden numbers above are pinned against.

**A step is one sample, not a minibatch — a deliberate divergence from the MLP side's shape of
invariant 2, not a violation of it.** The MLP batches because a gradient is an average over rows;
a SOM update has no such average, each sample drags the lattice on its own. Samples are drawn
*with replacement*, the classical Kohonen loop, rather than a shuffled epoch — tying "how many
times has this row been seen" to `rows.length` would fight the schedule's own `steps` horizon.
Invariant 2's actual rule — training advances in whole, fixed units of work, and the render loop
drains only what has completed — holds regardless of what the unit is.

**`npm run som` mirrors `npm run data`'s ASCII scatter with real terminal colour.** Two 12×12
grids of 24-bit ANSI background swatches, before and after training, so "the map orders itself"
is something a reader can see without opening a browser — before is scattered noise, after is a
smooth gradient across the lattice. The golden test and the script assert the same numbers, so
drift is caught by whichever runs first, the same pattern `scripts/train.ts` set for the MLP side.

**Slice 8 — "The architecture editor's neighbour".** A parameter-budget readout beside the
hidden-layer editor: parameter count against the training split's own row count, flagged the
moment the network has at least as many free numbers as data points. 222 tests.

**The threshold is `params >= samples`, not a margin chosen to feel cautious.** §8's own mockup
note draws the line at exactly this — "a network with more parameters than samples is the
definition of challenge 7" — so `paramBudget` in `packages/mlp/src/net.ts` tests `>=` and a test
pins the boundary case itself: a 2-2 network (6 parameters) against exactly 6 training rows reads
over budget, against 7 it does not.

**The readout is a live comparison, not just a warning.** Both branches of the note are written
out — "room to generalise" under budget, the memorisation warning over it — because the point
isn't to stay quiet until something is wrong; it's to make the relationship between parameter
count and training-row count legible on every edit, the same way `renderNetPanels` already showed
raw parameter and connection counts without saying what they meant. Verified live: dropping
samples to 20 (14 training rows after the 70/30 split) against the default 2-8-8-2 reads
**114 / 14 — over budget**, quoting challenge 7 by name; putting samples back to 240 reads
**114 / 168** and the colour and class flip from `--bad` to `--ok` exactly, confirmed against
`getComputedStyle` rather than eyeballed.

**A stale string got fixed in passing.** `drawOverCapNotice`'s second line read "Weight matrices
arrive in slice 7" — true when §1 was written, wrong since slice 7 shipped diagnostics histograms
instead and left the graph's own replacement-at-scale unbuilt. §6's rule about fixed strings
rotting isn't only about runtime values; a string that names a future slice number is exactly as
liable to go stale as one that names a training result, and this is the third time in the project
a string like that has needed correcting. It now says only what is true today, and points at the
weight-histogram panels slice 7 actually built as the thing to look at instead.

**Lab is retired. Two stages now, not three.** The design document's original three-card split
gave Lab the architecture editor, optimiser internals, gradient statistics, weight matrices and
throughput counters — everything that sounded like it needed its own workbench before any of it
existed. By this slice all of it had landed in Explorer instead, on the same reasoning each time:
nothing about the feature actually needed to differ from Explorer, so gating a separate stage for
it would have meant inventing a distinction that did not exist. `AppStage` drops `'lab'`, the
toolbar button is gone, and `?stage=lab` in the URL is downgraded to `explorer` rather than
rejected — the same defensive-read rule every other query parameter already gets, because an old
link should still open rather than silently fall back to the default guided flow. The design
document's Fig 8.6 mockup and the Guided/Explorer/Lab card grid are both amended to match.

**Slice 7 — "Diagnostics".** Momentum and Adam join SGD in the training panel, and Explorer gained
three panels that had been empty since slice 1: gradient flow, weight and activation histograms,
and a dead-ReLU-unit count. 218 tests.

**Optimiser state has to be reset on a kind change, not carried across it, and that is the whole
design of `setTrainConfig`.** A plain `trainer.config = newConfig` was safe when the only thing a
config held was numbers. It stops being safe the moment an optimiser carries its own memory —
switching Adam → Momentum mid-run would leave Adam's second moment sitting under Momentum's `v`,
silently wrong rather than reset, and the loss would still go down because a mislabelled velocity
is still a velocity. `setTrainConfig` compares `config.optimiser !== state.kind` before deciding
whether to call `resetOptimiserState`, and it is the only path either side of the worker boundary
uses to change training config — `trainer.worker.ts`'s `case 'config'` calls it too, replacing a
plain assignment that had exactly this bug.

**Momentum is deliberately the unscaled form, `v ← β·v + g`, not the exponential moving average
some textbooks use.** The two conventions differ by a factor of `(1 − β)`, and writing momentum
right next to Adam's actual EMA (`m ← β₁·m + (1 − β₁)·g`) in the same file made picking the
mismatched convention on purpose feel wrong — so both are commented with which they are and why,
rather than left to look like the same idea twice.

**Adam is checked two ways, because §13 flagged that a subtly wrong Adam still trains.** A
hand-computed first step against the closed-form update, and a stronger identity: for a *constant*
gradient, Adam's bias-corrected moments converge exactly to `g` and `g²` at every step, so the
effective step size is exactly the learning rate regardless of the gradient's magnitude — a
property a convergence-only test would never catch a violation of.

**Diagnostics get their own scratch buffer, for the same reason evaluation and the stepper's trace
already do.** `activationStats` runs `forward` once per training row to build the histograms;
sharing `state.scratch` would mean the network graph repaints with the *last diagnostics sample's*
activations instead of the probe's, right after the graph was told to read them from there. A
`diagScratch`, rebuilt alongside the network in `rebuildEverything`, keeps the two apart —
the pattern is now three-for-three (`evaluateRows`, `captureTrace`, `activationStats`).

**Dead units are counted only under ReLU, and only from real data.** A unit is dead if it never
fires positive across every training row — checked by running the whole training set through
`diagScratch`, not by inspecting one probe point, since a unit dead at the probe might fire
elsewhere. Verified live: forcing zero-initialisation with a ReLU hidden stack (challenge 5) reads
**16 of 16 dead** before a single step runs, and the same network after 400 steps of Adam training
reads 0 dead — the count moves with the run rather than being fixed at either extreme.

**Gradient-flow bars read a snapshot, not an average.** `Trainer.lastGradNorms` is overwritten
every `trainStep`, one Euclidean norm per layer, computed from the same `gradNorm()` helper that
had sat unused since slice 2's gradient check. `RunPoint.gradNorms` carries whichever step closed
the reporting window — consistent with `trainLoss`/`valLoss` already being end-of-window snapshots
rather than window averages, so the chart's numbers don't disagree with each other about what
"this point" means.

**Slice 6 — "Guided first run".** The app opens in a guided flow: pick data, pick a shape, watch
it learn, see what changed. No hyperparameters, four steps, ending with the network's first guess
replayed against its last. 190 tests.

**React is still not here, and neither is a second render path.** `pickDataset`/`pickShape` call
straight into the same `regenerateData`/`regenerateNet` Explorer's own controls use, so a choice
made in Guided is not a parallel way of changing the run — it is the same one. Reusing them is
also what makes "Skip to the full app" free: the run underneath is already exactly the one
Explorer would show.

**Two controls went stale, and the bug is the same shape both times.** Guided sets `state.dataset`
and `state.hidden` directly rather than driving the `<select>` or the preset buttons, so neither
control's DOM was told about the change — picking XOR in Guided left Explorer's dropdown reading
"Two moons". Fixed once, centrally: `renderDataPanels`/`renderNetPanels` now write the controls
from state on every call, so a display element can no longer disagree with the state it is
supposed to be showing. Three redundant call sites (the ones that used to remember to do this by
hand) were deleted along with the fix.

**Progress lives in the flow's own closure, not in `AppState`.** Switching to Explorer and back
must not restart anything, and it doesn't: `current`, `before` and `after` persist across the
stage switch because nothing about switching stage touches them. Verified live — trained a run,
skipped to Explorer, switched back to Guided, and step 4's numbers were exactly where they were
left.

**A found-by-running ordering bug: progress advanced *after* the rebuild it was supposed to
precede.** `pickDataset`/`pickShape` call straight into `regenerateData`/`regenerateNet`, which
end in a synchronous `render()` — and that render calls back into the guided panel's own render
before either function returns. Advancing `current` afterwards left the panel repainting once
with the *previous* step still marked "on", with nothing left to prompt a second repaint. Fixed
by moving the advance before the call, not after — the general lesson is that anything a
synchronous rebuild renders mid-call has already missed its chance to see state set afterwards.

**The afterword has to survive an unlucky run, and one was on hand to prove it.** Picking XOR with
no hidden layer from Guided is challenge 1, reachable now from the flow that never mentions
hidden layers — and it landed at 47.2% → 44.4%, an accuracy that went *down*. The branching
afterword said so honestly rather than claiming the improvement a fixed string would have.

**The "before" snapshot is the network's only unrepeatable moment, so it is taken exactly once.**
Random weights, step zero — a network is only ever this untouched right after `regenerateNet`
returns and before training starts. `captureSnapshot` runs synchronously in that gap, off the
same mirror model and the same `computeField` slice 3 already built; no protocol round trip and
no second implementation of the field.

**Slice 5 — "The stepper".** The teaching screen — the point of the whole project. `S` or the
toolbar button pauses training and opens a full-screen view of one real step: 7 stages for
2-8-8-2 (sample → forward × 2 → output+loss → backward × 2 → update), each drawn from one
`StepTrace` produced by the same `trainStep` the worker drains at full speed. 173 tests.

**Tracing cannot change the run, and that is proven rather than careful.** A trace is captured
between averaging the batch's gradients and applying them — the one moment both are true at
once — using its own scratch and its own gradient buffers; the real ones are only read. A test
runs 200 steps with tracing on and 200 with it off, from the same seed, and asserts every weight
in the network agrees exactly. Verified live too: the golden run reached its pinned 0.1007 /
0.9702 / 38 epochs after two stepper steps had run at the start of the same session.

**One trace covers a whole step, not one layer.** "Stepping" through the stages is a client-side
cursor into a single recording — forward through every layer, backward through every layer, the
update already applied — not a re-run of the network one layer at a time. A new trace is only
requested from the worker when the reader pages past the *last* stage of the one on screen, which
is also why `next` never allocates and `run to end of step` is instant.

**The output layer's backward step is folded into "output and loss", not given its own stage.**
Under softmax + cross-entropy the Jacobian cancels and `dz = a − onehot(target)` directly — the
one place in backprop where the chain rule does not appear as two visible factors. The stepper
says so rather than leaving a reader hunting for a δ × a′ that was never computed. `trace.fused`
carries the flag; `trace.test.ts` asserts it is true for exactly one layer.

**Δw shown is the change applied to the weight, not the raw gradient.** Negated and scaled by the
learning rate at capture time, so a reader comparing the strip to the weight before and after sees
them agree — the raw gradient would have the sign backwards.

**The trace protocol follows slice 4's generation rule exactly**, because it is the same shape of
bug: a `trace` request advances the run by one real step, so the reply has to be checked against
the session it was asked of, the same as every other worker message.

**Slice 4 — "Off the main thread".** Training lives in a Web Worker. `apps/web/src/main.ts` no
longer calls `trainStep` at all — it sends a configuration, receives weights and chart points
about 25 times a second, and draws. 145 tests.

**Measured, on the identical spirals run: 6 377 → 11 960 steps/s.** Slice 3's own numbers are the
baseline, so this is 88% on the same work. On moons at 2-16-16-2 it reaches 28 500.

**Training now survives a background tab, which was slice 3's stated limitation.** Verified by
running 20 000 steps to completion with `document.hidden === true` — slice 3 would have stalled
at step 0, because `requestAnimationFrame` does not fire there. Workers are not
visibility-throttled, so the `visibilitychange` pause slice 3 needed is deleted.

**The worker is not sent the dataset.** It is sent the *configuration* and rebuilds from the same
seed through the same `run/build.ts`. The two sides agree because there is one implementation, not
because anything is kept in sync — and a test asserts `buildData` is byte-identical across calls,
since a split differing by one row would have the page drawing one dataset while the worker
trained on another, both looking entirely reasonable.

**The golden run reproduces through the worker: 0.1007 / 0.9702 / 38 epochs.** `build.test.ts`
also asserts that chunking the loop into uneven bursts — 7, 113, 1, 96, 183 — and into 400 single
steps gives bit-identical results. That is invariant 2, and it is what makes the worker's 40 ms
chunking safe.

### Two races, both found by running it

**Pressing Train during a rebuild sent `run` against a session the page had discarded.** The run
happened and the step counter advanced to 20 000, but not one chart point arrived — so accuracy
read `—` under a completed run. The intent is now held and honoured on `ready` rather than
dropped.

**A report already in flight when the architecture changed was applied to the rebuilt mirror.**
`applyWeights` threw `weight buffer is 114, expected 354` — *the guard written in this slice
caught the bug in this slice*. The dangerous case is the one that does **not** throw: a rebuild
that changes only the dataset keeps the same shape, so stale weights would apply cleanly and the
graph would show a network that no longer exists. Every session now carries a **generation**,
echoed on every message, and the client drops anything older.

> The general shape, worth remembering for slice 5: **a rebuild does not cancel messages already
> in flight.** Anything crossing the worker boundary needs to say which session it belongs to.

**`fieldPending` exists for the same reason.** A probe request and its reply are separated by a
round trip now; without the flag every render in that gap would queue another, and the worker
would spend its time drawing fields for weights it had already left behind.

**One `RunPoint`, not two arrays.** Slice 2 kept a per-step loss for the band and a periodic
evaluation for the lines, sampled at different rates and reconciled in the chart. The worker
produces both halves at the same moment, so nothing is dropped and nothing is invented.

**The worker imports `computeField` rather than reimplementing it.** The first version had its own
copy of the grid loop, which would have left slice 3's tests — cell centres, bottom-left origin,
the standardiser — covering a function that no longer ran anywhere.

**Slice 3 — "Data and boundaries".** Six generators, the decision field as an `ImageData` blit,
and the train/validation split made visible. 134 tests.

**The field is four to six times cheaper than §5 budgeted, and that section said "budget, not
measurement" for exactly this reason.** Measured: 64² is 2.6 ms and 128² is 5.6 ms, against
estimates of 9 ms and 35 ms. The 150 M MAC/s assumption was too pessimistic — V8 does much better
on this loop. Throughput with the field on is **12 274 steps/s against 15 083 without**, so it
costs about 19% and the §13 risk ("the field will fight the training loop") is real but small.
256² is now viable on pause rather than export-only; the ladder is unchanged because nothing yet
needs it.

**Two of the new sets could not be solved at any setting the app could reach.** Checkerboard sat
at 0.66 and spirals at 0.69 — which reads as a broken app, not a hard problem. Measured
headlessly, both reach ~0.88 at **20 000 steps**, and the step slider stopped at 4 000. At
12 000 steps/s that is under two seconds, so it was never a performance limit, only a UI one.

- The steps slider is now **logarithmic, 100 → 20 000**, snapped to 1/2/5 × 10ⁿ.
- **Each generator declares the step count it needs** and selecting it adopts that. Measured, not
  preferred. A challenge card that wants to *demonstrate* too few steps sets a low one on purpose.

**`evalEvery` scales with the run instead of being fixed at 10.** A fixed interval was fine for
400 steps and badly wrong for 20 000: 2 000 measurements for a 300 px chart — six per pixel —
costing about 45% of the run. Roughly 200 samples across whatever the target is took spirals from
**2 546 to 6 377 steps/s** with no visible change to the chart.

**Validation points are drawn hollow, training points filled.** Shape rather than colour, because
colour already means class (§7). With a field underneath, "the boundary misses that point" reads
very differently depending on whether the network was ever shown it.

**The field is computed over the camera's *visible* box, not the data's bounds.** One scale on
both axes means one axis is letterboxed, and leaving those margins blank would imply the boundary
stops where the samples do. It does not.

**Training now stops when the tab is hidden, and says so.** `requestAnimationFrame` does not fire
in a background tab, so training stopped whether the app agreed or not — while the button still
read *Pause* and `elapsedMs` kept accruing wall-clock against a step count that was not moving,
quietly corrupting steps/s for the rest of the run. Slice 4's worker is not visibility-throttled
and removes the limitation.

**A slice-1 note had rotted and was saying something false.** The Output panel read "the weights
are random and nothing has been trained" underneath a network at 97% accuracy — true when it was
written, wrong from the first step of slice 2. It branches on `trainer.step` now and quotes the
live validation accuracy. *This is the second time a fixed string in this project has gone stale;
§6's rule applies to static copy, not only to challenge afterwords.*

**Slice 2 — "It learns".** Backpropagation, SGD, the loss chart, Train / Step / Reset. The golden
run reproduces **bit-identically in the browser and in Node**: 0.1007 train loss, 0.9702 accuracy,
38 epochs. 15 000 steps/s on 2-8-8-2, main thread; the worker is slice 4.

**The gradient check is the reason to trust any of it.** Analytic against a central finite
difference, every activation, every layer type, `2-2` through `2-4-4-4-4-2`, weights and biases —
because *a wrong gradient still trains*. A sign error or a missing transpose just slows learning,
so the loss still descends and nothing looks broken. It agrees to **9.5e-10**, and one test
asserts the check itself fails when the gradient is negated.

**It did not agree at first, and the cause was not the gradient.** It bottomed out at 2.5e-3 —
small enough to look like a subtle backprop bug. Two things were wrong with the *measurement*,
and finding them is what makes the number trustworthy now:

1. **Scratch buffers were `Float32Array`.** §4 always said intermediates are doubles; invariant 3
   is specifically about *weights*, which are state and get transferred. Rounding activations twice
   per layer put ~1e-7 of relative noise on the loss and a floor of 2.5e-3 under the check.
   `Scratch` and `Grads` are `Float64Array` now; `W` and `b` stay `Float32Array`.
2. **The finite difference divided by `2h` rather than the perturbation that happened.** `w + h`
   on a float32 buffer is rounded on store. Reading the value back costs nothing.

The diagnostic that settled it: the error *grew as h shrank* (6.4e-4 at 1e-2 up to 1.0 at 1e-5).
That is cancellation noise. A wrong gradient gives an error that does not move with h at all.

**Two challenge cards described outcomes the build does not produce**, and both are amended in
the design document against measurements:

- **A flat network's ceiling on XOR is 75%, not 50%.** A line isolates one quadrant and answers
  "the other class" everywhere else — right three times out of four. Measured 0.29–0.77 across
  seven split seeds, never higher; one hidden layer reaches 1.0000. The better lesson: *it gets
  three quarters and can never get the last quarter.*
- **The loss never goes NaN, and lr 3.0 is not too big — it is better than 0.1.** tanh + softmax +
  cross-entropy bounds every gradient factor, so weights grow linearly and never overflow. lr 3 →
  0.988 accuracy; lr 10 → 0.744; lr 500 → 0.500 at loss 13.8 with max|w| 1.4e4 and nothing
  non-finite. The learning-rate slider runs to 500 and is logarithmic.

**Softmax's max-shift is what makes challenge 3 legible rather than blank.** At lr 500 the logits
reach ~1e5; unshifted `exp` would fill every panel with NaN. Shifted, a destroyed network still
reports a readable accuracy. *Diverging is the lesson; NaN everywhere is a bug.*

**Challenge tests assert comparisons, not absolute thresholds, where the value is seed-dependent.**
A destroyed network lands at 0.50 on one split and 0.65 on another; what is always true is that it
is far worse than the same network trained sanely on the same data. Two of my first thresholds
failed for that reason.

**The golden run is pinned twice: the loss to four decimals and a checksum over every weight.**
Two different weight vectors can agree on loss to 4dp, so a change that reordered updates could
leave it untouched. XOR-ing the raw float32 bits catches any change to any weight.

**Slice 1 — "Forward, drawn".** A network you set by hand — hidden widths, activation, init
scheme — with the graph above the scatter and a draggable probe. Edges are coloured by signed
weight, nodes filled by activation, and both follow the probe live. **Nothing learns**; the
weights are random and stay random.

**The forward pass has a hand-checked test, and it is the one that matters.** A 2×2 linear layer
with `W = [[1,2],[3,4]]`, `b = [0.5,−0.5]`, `x = [1,1]` must give `[3.5, 6.5]`. Read column-major
instead of row-major and it gives `[4.5, 5.5]` — just as plausible-looking and completely wrong.
A transposed index is the single most likely bug in this file and it does not announce itself.

**Softmax subtracts its maximum, and that is not an optimisation.** `Math.exp(800)` is `Infinity`
and `Infinity / Infinity` is `NaN`, so an unshifted softmax turns the whole output into NaN the
first time a logit gets large. Tested at logits of 1000. (Slice 2 measured *why this matters*: at
a destructive learning rate the logits reach ~1e5, and the shift is what keeps a wrecked network
readable instead of blank.)

**Zero init makes every hidden unit identical, and a test asserts it.** Challenge 5's mechanism
exists from the slice that first draws activations, because the graph is the evidence: every edge
into unit 2 is the same colour as every edge into unit 7. Verified live — all eight hidden
activations read `0.0000`.

**The graph has a stated limit of 24 units per layer**, not a hope. Above it the centre panel
draws a notice instead, and the trigger is asserted in `graph-layout.test.ts` so it cannot drift.
2-16-16-2 is 320 edges; 64-128-128-10 is 25 856, at which point the graph is not slow so much as
*meaningless*. Weight matrices replace it in slice 7.

**One gap for the whole graph, not one per column.** Per-column spacing draws a 2-unit input
layer at the same pitch as a 16-unit hidden layer, and the eye reads that as the two layers being
differently *scaled* rather than differently sized.

**The probe is held in data coordinates and standardised at the last moment.** The network only
ever sees standardised inputs, so a forward pass on raw coordinates would be answering a
different question from the one the scatter is asking.

**A panel note was rewritten because it asserted a value it could not know.** It said the outputs
would be "near-even" with random weights; a live probe read 0.766. §6's rule is that explanations
are written against live values and never fixed strings that can be wrong — that applies to
static copy too, and the fix was to state what is true regardless (the weights are random, so a
confident answer is as arbitrary as an even one).

**Slice 0 — "Chassis".** The three-column shell, dark, with a seeded dataset in the stage and
nothing learning. `packages/core` (Rng, Dataset, split, standardiser), `packages/data` (two
moons), `apps/web` (scatter, panels, narrow-width drawers), and `npm run data` headless.

**190 tests in 3.7 s.** The pinned ones are the `Rng` golden vector for seed 4417 and the
noiseless-moons geometry. Both are load-bearing: every reproducibility claim the project makes
descends from that vector, and the moons geometry is what makes challenge 1 fail on purpose.

**Two moons is asserted to be non-linearly-separable, not assumed.** A brute force over 720
projection directions checks that no line splits the classes. Slice 2's opening lesson is that a
network with no hidden layer visibly *cannot* do this; if the geometry ever drifted — a smaller
offset, a different radius — the arcs would come apart, the challenge would quietly start
succeeding, and nothing would fail to say so.

**The split is stratified and the standardiser is fitted on the training rows only.** Both are
tested, and both exist for the same downstream reason: challenge 8 asks a reader to trust the
validation curve. A 12/4 class split at 70% can hand validation zero samples of the rare class,
and fitting the mean over everything leaks the validation set into training. Either one makes
that curve quietly wrong while nothing looks broken.

**`packages/core/dataset.ts` returns index arrays, not new Datasets.** One feature buffer, two
orderings into it. Copying would mean every panel has to know which of the two copies it holds.

**The split gets its own `Rng`, seeded from the same number.** Sharing one generator with the
dataset would make the split depend on how many draws the generator happened to take, so moving
the sample-count slider would silently reshuffle the split too — two things moving when the
reader moved one.

**`.pb { flex: none }` is not cosmetic.** As a shrinkable flex item a panel body compresses below
its own content and draws straight over the next panel's header, while `overflow: hidden` on the
column clips the evidence so `scrollHeight` still reports a clean fit. Found while building the
design document's mockups, fixed in the app's CSS before the app could have it.

## Two open questions are now decided

Both were listed in §13 as open. They are answered, and the design document records the reasoning
and the cost.

**The SOM lattice is hexagonal, and hex is the default.** Rectangular stays selectable because two
topologies side by side is itself a lesson. The reason is not aesthetic: on a rectangular lattice a
node has four neighbours at distance 1 and four more at √2, so `h(d,t)` pulls harder along the axes
than the diagonals and the map contracts anisotropically. Hex gives six neighbours all at distance
1. Since §3's whole claim is that the neighbourhood is a distance *in the lattice*, the lattice
should be one where that distance behaves.

> **The bug to design around:** lattice distance is **not** Euclidean on `(col, row)`. Offset rows
> must convert to axial coordinates before measuring. Get it wrong and every neighbourhood is
> subtly the wrong shape on alternate rows — a map that still trains, still looks plausible, and is
> quietly not a SOM. `Som.neighbours` is therefore a table built **once per topology** and tested
> against a hand-counted 3×3, never a distance computed inline.

**Both networks get a guided flow, and the map's arrives at slice 11, not slice 6.** A reader who
came for Kohonen maps and is handed a perceptron tutorial has been told their question is the less
important one — the opposite of what §1 claims. But a flow can only be built once there is
something to guide somebody *through*, and the map's third step *is* the U-matrix (slice 10).

So slice 6 builds the frame plus the perceptron's flow; slice 11 adds the map's beside the SOM
stepper. **The frame takes the flow as data from the first version**, not generalised later —
retrofitting a second client onto a hardcoded first one is how the copy ends up duplicated, and
duplicated copy was the exact risk that made this an open question. One `GuidedFlow` type, two
arrays of steps, one renderer, and a test that renders every branch of both.

### Next: slice 13 — "Challenge track"

Twelve cards, four phases, one dot per concept — the ladder every "challenge N" reference in this
file has been pointing at since slice 0. A collapsed card is its title alone; twelve briefs would
not fit on screen but twelve titles do. Cards past the frontier are dimmed as guidance rather than
locked, because a reader who already knows the material should not have to replay the ladder to
reach it, and a completed card is never dimmed. Progress lives in `localStorage`, parsed
defensively like every other piece of state that outlives the code reading it. The afterword
branches on the run's own outcome and quotes its numbers — the same rule §6 has applied to every
piece of generated copy in the project since slice 1's probe note was first wrong.

## Invariants

These are the rules that must survive between sessions. Breaking one is a bug even when the app
still works.

1. **Seeded RNG only.** No `Math.random()` anywhere in `packages/`. Weight initialisation, batch
   shuffling, dropout masks and dataset generation all draw from an `Rng` threaded through as a
   parameter. This is what makes runs reproducible and the golden test possible.

2. **Fixed step granularity.** Training advances in whole minibatch steps. The render loop drains
   steps that have completed; it never trains "for a frame". Evolab's fixed-timestep rule, one
   domain over.

3. **Weights are flat `Float32Array`, row-major, owned by their layer.** No arrays of neuron
   objects, no nested arrays. They post to a worker without a structured clone, the renderer
   indexes them directly, and a whole network serialises as one buffer.

4. **Packages stay pure.** No DOM, no `window`, no timers, no I/O, no `console`. They must run
   under Node in a test. If something needs to report progress it takes a callback.

5. **Nothing browser-specific below `apps/web`.** Workers, canvas, Three.js, storage, URL parsing
   — all live in the app.

6. **No autodiff, and no library that provides one.** The hand-written backward pass is the
   artefact being taught. A framework would delete exactly the code a reader came to see, and the
   stepper would then need a second implementation of backprop to display — one of which would be
   a lie. §12 of the design document.

7. **Tests are never deleted or loosened to make a change pass.** If one fails, either the change
   is wrong or the change is deliberate — say which, in the commit message, and update the
   expected value in the same commit.

## Two tests that are not like the others

- **`Rng` golden vector** (now). Six integers for seed 4417. If it fails, every stored run and
  every pinned number in the project has been invalidated. That is a decision to make
  deliberately, not a number to update until the test goes green.
- **Gradient check** (slice 2). Analytic gradient against a central finite difference, every
  layer type and activation. It exists because *a wrong gradient still trains* — a sign error or
  a missing transpose usually just slows learning, so the loss still descends and nothing looks
  broken. Finite differences do not care how plausible the curve looks.
- **Golden run** (slice 2). Gradient check proves the maths is *right*; this proves it is
  *unchanged*. Refactoring the inner loop is safe exactly when this number does not move.

## Determinism is scoped to one engine

IEEE-754 `+ − × ÷` and `Math.sqrt` are correctly rounded and identical everywhere, so a dot
product reproduces bit-for-bit given a fixed loop order — which invariant 3 guarantees. But
`Math.exp`, `Math.tanh`, `Math.log` and `Math.pow` are **implementation-defined** in ECMAScript.

So a run replays bit-identically on the same browser and version, and not across engines. The
golden tests run under Node in CI where the engine is pinned. `Rng.normal()` is the one method in
`core` affected, and it says so where it is defined. §4 of the design document has the full
argument, including why shipping our own `exp` was rejected.

## Layout

```
packages/core/   pure TS — Rng, Dataset, split, standardiser, bounds
packages/data/   pure TS — six 2D generators, each declaring the steps it needs
packages/mlp/    pure TS — activations, layers, forward, backward, SGD, training loop
packages/som/    slice 9 — hex lattice (axial coords), bmu, neighbourhood, schedules, u-matrix
apps/web/        Vite app — canvas render, the training worker, later Three.js
server/          slice 15 — ASP.NET Core + SQLite
docs/            the technical design document
```

Packages are consumed as source via Vite aliases (`@neurallab/core`, `@neurallab/data`). There is
no build step for packages and there should not be one.

## Commands

```bash
npm install          # once, from the repo root
npm run dev          # http://localhost:5173
npm test             # 190 tests in ~3.7 s — run these before every commit
npm run check        # typecheck everything
npm run data         # headless: build the default set, print it, assert it replays
npm run train        # headless: the golden run, challenge 1 and challenge 3, all asserted
```

`npm run data` prints an ASCII scatter, so the shape of a generator can be confirmed without
opening a browser. It also asserts the same seed produces the same bytes — the first and smallest
form of the golden test — and prints the `Rng` vector, so pinning it in a test is a copy rather
than a guess.

The dev page takes `?data=moons&n=240&noise=0.15&split=0.70&seed=4417&net=mlp&stage=explorer`.
Every parameter is read defensively: junk degrades to the default rather than throwing on boot,
because these values are user-writable and outlive the code reading them.

### Node runs TypeScript directly

`npm run data` uses `node --experimental-strip-types`, which **erases** types rather than
compiling them. Syntax that emits code is therefore rejected: no parameter properties
(`constructor(private x: number)`), no `enum`, no namespaces, no decorators. Use explicit fields,
`const` objects with `as const`, and plain modules. Keeping every file strip-compatible is what
avoids a build step for packages and scripts.

## Conventions

- TypeScript strict, `noUncheckedIndexedAccess` on. The compiler is the code reviewer, and this
  project is mostly index arithmetic.
- **World coordinates are y-up. Canvas is y-down. The flip happens in exactly one place**
  (`render/camera.ts`) and nowhere else. A second flip cancels the first and nobody notices until
  a decision boundary is drawn upside down.
- **One scale for both axes.** Two moons stretched to fill a 3:2 panel is not two moons —
  distances stop being comparable between the axes, and a boundary drawn over it is a boundary in
  a space the network never saw. Letterboxing is correct and costs some pixels.
- **Colour is data** — §7. Amber is "look here now" (best value, active node, positive weight);
  cyan is the other end of a signed scale (negative weight, validation series, the SOM half);
  violet is a secondary series and the MLP's own hue; green is a threshold met, red one broken.
  Signed quantities use a diverging cyan ↔ panel ↔ amber ramp with the background at zero, so a
  dead weight is invisible and that is the correct impression.
- Prefer plain functions and plain objects. No classes unless there is state with a lifecycle
  (`Rng` is the current exception).
- Commit per slice. Put the observable result in the message — from slice 2, that result is a
  number.

## Things not to do

- **Do not add a machine-learning library.** Invariant 6. If the project ever outgrows hand-written
  gradients, the honest move is a second project, not a dependency.
- **Do not build the SOM before slice 9.** Convolutions, dropout, batch norm and learning-rate
  schedules are each one afternoon and all more familiar than Kohonen maps, which is exactly why
  the slice order is the defence. Slice 9 is the line.
- **Do not make the 3D view the default**, and do not let it grow features the 2D view lacks. 2D
  is the teaching surface. Evolab had to amend three sections of its design document to learn this.
- **Do not start the .NET server before slice 15.** It is the familiar, comfortable part, and it
  would serve a client that does not exist yet.
- **Do not "tidy up" the kernels without the golden test passing before and after** — from slice 2,
  when there is one.
