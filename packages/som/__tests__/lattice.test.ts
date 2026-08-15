import { describe, expect, it } from 'vitest';
import { buildNeighbours, latticeDistance, nodeIndex, NEIGHBOUR_SLOTS } from '../src/lattice.ts';

/*
 * The hand count this file exists to check against, worked out against the module's own
 * "odd-r" convention (odd rows shoved right by half a hex) before any code ran:
 *
 *   row 0:  (0,0) (1,0) (2,0)
 *   row 1:    (0,1) (1,1) (2,1)      <- shifted right
 *   row 2:  (0,2) (1,2) (2,2)
 *
 * Row 1 sitting to the right of rows 0 and 2 means its "up" and "down" diagonal neighbours land
 * at the *same* column and the *next* column over in the unshifted rows — not one column either
 * side, which is what a reader would guess from Euclidean intuition and exactly the mistake the
 * design document warns this file is for.
 */
describe('buildNeighbours — hex, hand-counted 3×3', () => {
  const n = buildNeighbours(3, 3, 'hex');
  const row = (node: number): number[] => Array.from(n.subarray(node * NEIGHBOUR_SLOTS, node * NEIGHBOUR_SLOTS + NEIGHBOUR_SLOTS));

  it('centre (1,1) touches all six neighbours — none of them fall off a 3×3 lattice', () => {
    // Direction order is [+1,0] [+1,-1] [0,-1] [-1,0] [-1,+1] [0,+1] (see AXIAL_DIRS).
    // (1,1) -> axial(1,1); walking each direction and converting back to offset gives, in order:
    // (2,1) (2,0) (1,0) (0,1) (1,2) (2,2) — indices 5, 2, 1, 3, 7, 8.
    expect(row(nodeIndex(3, 1, 1))).toEqual([5, 2, 1, 3, 7, 8]);
  });

  it('corner (0,0) touches only its right and below-right neighbours', () => {
    // (0,0) is in an unshifted row with nothing above or to the left. Its two in-bounds
    // neighbours are (1,0) [right, index 1] and (0,1) [row 1 shifted right, so directly below,
    // index 3]. Four of the six slots fall off the lattice.
    expect(row(nodeIndex(3, 0, 0))).toEqual([1, -1, -1, -1, -1, 3]);
  });

  it('a bottom-row corner (0,2) mirrors (0,0) upward instead of downward', () => {
    // (0,2) is also unshifted. Its neighbours are (1,2) [right, index 7] and (0,1) [row 1
    // shifted right, directly above, index 3].
    expect(row(nodeIndex(3, 0, 2))).toEqual([7, 3, -1, -1, -1, -1]);
  });

  it('every listed neighbour lists the node back — adjacency is symmetric', () => {
    for (let a = 0; a < 9; a++) {
      for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
        const b = n[a * NEIGHBOUR_SLOTS + s] as number;
        if (b === -1) continue;
        const back = row(b);
        expect(back).toContain(a);
      }
    }
  });
});

describe('buildNeighbours — rect, hand-counted 3×3', () => {
  const n = buildNeighbours(3, 3, 'rect');
  const row = (node: number): number[] => Array.from(n.subarray(node * NEIGHBOUR_SLOTS, node * NEIGHBOUR_SLOTS + NEIGHBOUR_SLOTS));

  it('slots 0-3 are N, S, W, E and slots 4-5 are always -1', () => {
    // (1,1) is the one fully-interior cell of a 3×3 grid: N=(1,0)=1, S=(1,2)=7, W=(0,1)=3, E=(2,1)=5.
    expect(row(nodeIndex(3, 1, 1))).toEqual([1, 7, 3, 5, -1, -1]);
  });

  it('a corner has exactly two neighbours, diagonals excluded', () => {
    // (0,0): N and W fall off; S=(0,1)=3, E=(1,0)=1. The diagonal (1,1) is √2 away and does not
    // appear here even though `latticeDistance` sees it as close — see the module comment.
    expect(row(nodeIndex(3, 0, 0))).toEqual([-1, 3, -1, 1, -1, -1]);
  });

  it('the opposite corner mirrors it', () => {
    // (2,2): N=(2,1)=5, S falls off, W=(1,2)=7, E falls off.
    expect(row(nodeIndex(3, 2, 2))).toEqual([5, -1, 7, -1, -1, -1]);
  });
});

describe('latticeDistance', () => {
  it('is zero for a node against itself, on both topologies', () => {
    expect(latticeDistance(5, 'hex', 12, 12)).toBe(0);
    expect(latticeDistance(5, 'rect', 12, 12)).toBe(0);
  });

  it('agrees with the neighbour table on hex: every listed neighbour is distance 1', () => {
    const n = buildNeighbours(4, 4, 'hex');
    for (let a = 0; a < 16; a++) {
      for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
        const b = n[a * NEIGHBOUR_SLOTS + s] as number;
        if (b === -1) continue;
        expect(latticeDistance(4, 'hex', a, b)).toBeCloseTo(1, 10);
      }
    }
  });

  it('is the plain Euclidean distance on rect, including the √2 diagonal', () => {
    // (0,0) to (1,0): distance 1. (0,0) to (1,1): distance √2 — the anisotropy the design
    // document names as the reason hex is the default.
    expect(latticeDistance(3, 'rect', nodeIndex(3, 0, 0), nodeIndex(3, 1, 0))).toBeCloseTo(1, 10);
    expect(latticeDistance(3, 'rect', nodeIndex(3, 0, 0), nodeIndex(3, 1, 1))).toBeCloseTo(
      Math.SQRT2,
      10,
    );
  });

  it('rect neighbours (N/S/W/E) are exactly the nodes at distance 1', () => {
    const n = buildNeighbours(4, 4, 'rect');
    for (let a = 0; a < 16; a++) {
      for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
        const b = n[a * NEIGHBOUR_SLOTS + s] as number;
        if (b === -1) continue;
        expect(latticeDistance(4, 'rect', a, b)).toBeCloseTo(1, 10);
      }
    }
  });
});
