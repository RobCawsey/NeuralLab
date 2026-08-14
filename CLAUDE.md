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

**109 tests in 3.4 s.** The pinned ones are the `Rng` golden vector for seed 4417 and the
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

### Next: slice 3 — "Data and boundaries"

Six 2D generators, the decision field as an `ImageData` blit, and the train/validation split made
visible. The field is the expensive drawing, not the training — §5 budgets it at ninety times a
training step, so it arrives with a throttle and a resolution that changes on pause.

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
packages/data/   pure TS — dataset generators (moons, xor; the rest in slice 3)
packages/mlp/    pure TS — activations, layers, forward, backward, SGD, training loop
packages/som/    slice 9 — hex lattice (axial coords), bmu, neighbourhood, schedules, u-matrix
apps/web/        Vite app — canvas render, later workers and Three.js
server/          slice 15 — ASP.NET Core + SQLite
docs/            the technical design document
```

Packages are consumed as source via Vite aliases (`@neurallab/core`, `@neurallab/data`). There is
no build step for packages and there should not be one.

## Commands

```bash
npm install          # once, from the repo root
npm run dev          # http://localhost:5173
npm test             # 109 tests in ~3.4 s — run these before every commit
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
