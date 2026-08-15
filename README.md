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

**Slice 9 of 16 — the Kohonen kernel.** `packages/som` exists: the hex lattice, best-matching
unit, neighbourhood function, decay schedules, quantisation and topographic error, the U-matrix,
and a golden run — a 12×12 map ordering itself on the colour cube, printed as real terminal colour
by `npm run som`. Pure and headless, same as the MLP kernel was through slice 2; nothing in the
app uses it yet.

## Getting started

```bash
npm install
npm run dev
```

Then open <http://localhost:5173>.

```bash
npm test       # 265 tests, ~3.7 s
npm run check  # typecheck
npm run data   # headless: print the default dataset as ASCII, assert it replays
npm run train  # headless: the MLP golden run, plus challenges 1 and 3, all asserted
npm run som    # headless: the SOM golden run, printed as real terminal colour
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
  purpose: XOR without a hidden layer, a learning rate of 3.0, zero initialisation, six sigmoid
  layers, a neighbourhood that cools in fifty steps.

Built as a sibling to [Evolab](../Evolab) — same chassis, same rules, same slice discipline.

## Licence

MIT.
