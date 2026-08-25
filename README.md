# NeuralLab

A browser workbench for building a **Kohonen self-organising map** and a **multilayer
perceptron**, training them on data you can see, and learning how both work by watching them
happen.

Most neural-network teaching material shows you a diagram of backpropagation and then hands you a
library that does it for you. The diagram and the library never meet. NeuralLab closes that gap:
the arrows in the diagram are the numbers on screen, and the numbers on screen come from the code
that is actually training.

No TensorFlow, no autodiff. Every gradient is written out by hand next to the forward pass it
belongs to, because a reader has to be able to follow it.

## Status

**Slice 16 of 16 — Model scorecard.** The last slice on the roadmap. Real handwritten digits (the
UCI/scikit-learn set, not a synthetic stand-in), a confusion matrix, a Kohonen map that draws its
own lattice as recognisable handwriting, and a scorecard that retrains the reader's own
configuration five times and hands out a badge it can fail to earn — graded on the worst of five
seeds, not the average, so one unlucky initialisation cannot hide behind four good ones.

## Getting started

```bash
npm install
npm run dev
```

Then open <http://localhost:5173>. That's the whole app — the server below is optional and adds
only Save/Runs/Share; everything else works with it never having been started.

```bash
npm test       # 339 tests, ~3.6 s
npm run check  # typecheck
npm run data   # headless: print the default dataset as ASCII, assert it replays
npm run train  # headless: the MLP golden run, plus challenges 1 and 3, all asserted
npm run som    # headless: the SOM golden run, printed as real terminal colour
```

The server is a separate ASP.NET Core project — `npm run dev` and `npm test` above need no .NET,
and the commands below need no Node:

```bash
npm run server   # dotnet run --project server/NeuralLab.Server, on :5150
dotnet test server  # 12 tests — the SQL layer and the HTTP contract in front of it
```

## The design

**[docs/technical-design.html](docs/technical-design.html)** — open it in a browser. The full
architecture and UI specification: 13 sections, 8 interface mockups, and the reasoning behind
every decision, including the ones that were rejected.

Short version:

- **Two algorithms, because they fail in opposite ways.** The perceptron is supervised and fails
  numerically — vanishing gradients, steps that overshoot, a model that memorises. The Kohonen map
  is unsupervised and fails structurally — a map that stays twisted, a schedule that cools too
  fast. The last lesson is the same dataset seen through a boundary it was told to find and a
  structure it found on its own.
- **The stepper is the point.** A full-screen view that pauses training between operators and
  shows each one acting on real values — driven by the same function the worker drains at full
  speed, with a test asserting the two are bit-identical.
- **Twelve challenge cards**, each configuring the app in one click to make something go wrong on
  purpose: XOR without a hidden layer, a learning rate of 500, zero initialisation, six sigmoid
  layers, a neighbourhood that cools in fifty steps.
- **The server never has to be there.** Weights are never uploaded — a run is deterministic in its
  seed and step count, so reopening one re-trains it rather than fetching it back. Kill the server
  mid-session and nothing about training notices.
- **Digits is real, not generated.** Every other dataset is procedural; this one is the actual UCI
  handwritten-digits set, specifically because the confusion matrix and the map of handwriting it
  buys are not lessons a synthetic stand-in could honestly teach.

Built as a sibling to [Evolab](../Evolab) — same chassis, same rules, same slice discipline.

## Licence

MIT.
