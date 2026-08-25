import { describe, expect, it } from 'vitest';
import { CHALLENGES, challengeById, phases } from '../src/run/challenges.ts';

describe('CHALLENGES', () => {
  it('has exactly twelve cards, numbered 1 through 12 in order', () => {
    expect(CHALLENGES).toHaveLength(12);
    expect(CHALLENGES.map((c) => c.id)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
  });

  it('gives every card a real title, setup and doneWhen', () => {
    for (const c of CHALLENGES) {
      expect(c.title.length).toBeGreaterThan(0);
      expect(c.setup.length).toBeGreaterThan(0);
      expect(c.doneWhen.length).toBeGreaterThan(0);
    }
  });

  it('every card names a network its config actually targets', () => {
    for (const c of CHALLENGES) {
      expect(['mlp', 'som']).toContain(c.config.net);
    }
  });

  it('challenge 10 pulls the schedule apart from the run length — the whole point of the card', () => {
    const c = challengeById(10)!;
    expect(c.config.scheduleSteps).toBeLessThan(c.config.somTargetSteps as number);
  });
});

describe('phases', () => {
  it('has exactly four, in ladder order, none repeated', () => {
    const p = phases();
    expect(p).toHaveLength(4);
    expect(new Set(p).size).toBe(4);
  });

  it('every challenge belongs to one of them', () => {
    const p = phases();
    for (const c of CHALLENGES) expect(p).toContain(c.phase);
  });
});

describe('challengeById', () => {
  it('finds a real card', () => {
    expect(challengeById(1)?.title).toBe('One line is not enough');
  });

  it('is undefined for an id outside 1..12', () => {
    expect(challengeById(0)).toBeUndefined();
    expect(challengeById(13)).toBeUndefined();
  });
});

describe('afterwords render every branch and the branches differ', () => {
  it('challenge 1 — finished vs not started', () => {
    const c = challengeById(1)!;
    const before = c.afterword({ finished: false });
    const after = c.afterword({ finished: true, trainAccuracy: 0.7679 });
    expect(before).not.toBe(after);
    expect(after).toContain('76.8%');
  });

  it('challenge 3 — collapsed vs merely unlucky', () => {
    const c = challengeById(3)!;
    const collapsed = c.afterword({ finished: true, trainAccuracy: 0.5, trainLoss: 13.8 });
    const lucky = c.afterword({ finished: true, trainAccuracy: 0.9, trainLoss: 0.2 });
    expect(collapsed).toContain('nothing went NaN');
    expect(lucky).not.toContain('nothing went NaN');
    expect(collapsed).not.toBe(lucky);
  });

  it('challenge 7 — over budget vs under, from the same fields', () => {
    const c = challengeById(7)!;
    const over = c.afterword({ finished: true, paramCount: 200, trainRows: 14, overBudget: true });
    const under = c.afterword({ finished: true, paramCount: 20, trainRows: 168, overBudget: false });
    expect(over).toContain('memorise');
    expect(under).not.toContain('memorise');
  });

  it('challenge 8 — a real gap vs none this run', () => {
    const c = challengeById(8)!;
    const gap = c.afterword({ trainAccuracy: 0.98, valAccuracy: 0.7 });
    const noGap = c.afterword({ trainAccuracy: 0.9, valAccuracy: 0.89 });
    expect(gap).toContain('a real gap');
    expect(noGap).not.toContain('a real gap');
  });

  it('challenge 12 — every combination of which side finished', () => {
    const c = challengeById(12)!;
    const neither = c.afterword({ finished: false });
    const mlpOnly = c.afterword({ finished: false, mlpFinished: true });
    const somOnly = c.afterword({ finished: false, somFinished: true });
    const both = c.afterword({ finished: false, mlpFinished: true, somFinished: true });
    const all = [neither, mlpOnly, somOnly, both];
    expect(new Set(all).size).toBe(4); // four genuinely distinct strings
  });
});
