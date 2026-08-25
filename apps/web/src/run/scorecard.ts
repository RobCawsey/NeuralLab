/**
 * The model scorecard — slice 16. "A trained network across five held-out seeds, with a badge it
 * can fail to earn" (§11's own words for this slice). A task suite that grades the run rather
 * than a chart that merely describes it: the same architecture and hyperparameters the reader
 * already has configured in Explorer, retrained five times on Digits — a different train/val
 * split and a different weight initialisation each time — so the badge answers "is this
 * configuration robustly good" rather than "did this one lucky run turn out well".
 *
 * Pure data and the one pure decision (`badgeEarned`); the actual five-run sequencing lives in
 * `main.ts` next to `applyChallenge` and `applyRunDetail`, which need the same worker and render
 * loop this does.
 */

/** Both seed and weight-seed for each of the five runs — one shared number, so each of the five
 * is a genuinely different (split, initialisation) pair rather than five draws of the same one. */
export const SCORECARD_SEEDS: readonly number[] = [1, 2, 3, 4, 5];

/**
 * Measured, not guessed: adam at 0.005 on a [128,128] hidden network reaches ~97.5% validation
 * accuracy by step 100 and is flat by step 600 across every seed tried — see CLAUDE.md's slice
 * 16 entry for the run that pinned it. Long enough for a reasonable configuration to visibly
 * settle, short enough that five runs finish in a few seconds even on a slower machine.
 */
export const SCORECARD_STEPS = 600;

/**
 * The badge's bar — measured against five configurations tried by hand on Digits at
 * `SCORECARD_STEPS`: a sane one clears 0.92+ on every seed, a tiny-but-workable one clears 0.90+,
 * a destructive learning rate lands in the 0.70s, and zero initialisation never leaves 0.10 at
 * all (symmetry never breaks — challenge 5's own lesson, replayed on a harder problem). 0.85 sits
 * cleanly below every sane configuration's worst seed and above every broken one's best, so the
 * badge is earnable by a reasonable choice and not by accident.
 */
export const SCORECARD_BAR = 0.85;

export interface ScorecardSeedResult {
  readonly seed: number;
  /** null until that seed's run has finished. */
  readonly valAccuracy: number | null;
}

/** null while any seed is still pending — there is no partial verdict, only a finished one. */
export function badgeEarned(results: readonly ScorecardSeedResult[]): boolean | null {
  if (results.length === 0 || results.some((r) => r.valAccuracy === null)) return null;
  const worst = Math.min(...results.map((r) => r.valAccuracy as number));
  return worst >= SCORECARD_BAR;
}

export function freshResults(): ScorecardSeedResult[] {
  return SCORECARD_SEEDS.map((seed) => ({ seed, valAccuracy: null }));
}
