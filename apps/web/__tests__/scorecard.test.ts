import { describe, expect, it } from 'vitest';
import { SCORECARD_BAR, badgeEarned, freshResults, type ScorecardSeedResult } from '../src/run/scorecard.ts';

describe('badgeEarned', () => {
  it('is null — no verdict yet — while any seed is still pending', () => {
    const results: ScorecardSeedResult[] = [
      { seed: 1, valAccuracy: 0.97 },
      { seed: 2, valAccuracy: null },
    ];
    expect(badgeEarned(results)).toBeNull();
    expect(badgeEarned(freshResults())).toBeNull();
  });

  it('is true only when every seed clears the bar, judged by the worst one', () => {
    const allAbove: ScorecardSeedResult[] = [1, 2, 3, 4, 5].map((seed) => ({
      seed,
      valAccuracy: SCORECARD_BAR + 0.05,
    }));
    expect(badgeEarned(allAbove)).toBe(true);

    const oneBelow: ScorecardSeedResult[] = [1, 2, 3, 4, 5].map((seed) => ({
      seed,
      valAccuracy: seed === 3 ? SCORECARD_BAR - 0.01 : 0.99,
    }));
    expect(badgeEarned(oneBelow)).toBe(false);
  });

  it('is exactly at the boundary correctly — the bar itself passes', () => {
    const atBar: ScorecardSeedResult[] = [1, 2, 3, 4, 5].map((seed) => ({ seed, valAccuracy: SCORECARD_BAR }));
    expect(badgeEarned(atBar)).toBe(true);
  });

  it('is judged by the worst seed, not the mean — a high mean cannot rescue one bad seed', () => {
    // Mean here is (1.0*4 + 0.5) / 5 = 0.90, comfortably above SCORECARD_BAR — but one seed is
    // far below it, and that is the one that has to decide the badge.
    const results: ScorecardSeedResult[] = [1, 2, 3, 4, 5].map((seed) => ({
      seed,
      valAccuracy: seed === 5 ? 0.5 : 1.0,
    }));
    expect(badgeEarned(results)).toBe(false);
  });
});

describe('freshResults', () => {
  it('has one entry per seed, all pending', () => {
    const results = freshResults();
    expect(results).toHaveLength(5);
    expect(results.every((r) => r.valAccuracy === null)).toBe(true);
    expect(new Set(results.map((r) => r.seed)).size).toBe(5); // five distinct seeds
  });
});
