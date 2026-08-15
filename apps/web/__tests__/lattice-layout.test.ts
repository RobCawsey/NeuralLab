import { describe, expect, it } from 'vitest';
import { hitLatticeNode, layoutLattice } from '../src/render/lattice-layout.ts';

describe('layoutLattice', () => {
  it('places every node inside the panel, for both topologies', () => {
    for (const topology of ['hex', 'rect'] as const) {
      const layout = layoutLattice(12, 12, topology, 300, 300);
      for (let i = 0; i < 144; i++) {
        const x = layout.xy[i * 2] as number;
        const y = layout.xy[i * 2 + 1] as number;
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(300);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThanOrEqual(300);
      }
    }
  });

  it('gives hex neighbours (same row) and diagonal neighbours the same spacing', () => {
    // The property that makes hex the default — §3 of the design document — checked in pixels:
    // a same-row pair and a diagonal pair, both lattice-adjacent, land the same distance apart.
    const layout = layoutLattice(4, 4, 'hex', 400, 400);
    const at = (col: number, row: number): [number, number] => {
      const i = row * 4 + col;
      return [layout.xy[i * 2] as number, layout.xy[i * 2 + 1] as number];
    };
    const dist = (a: [number, number], b: [number, number]): number => Math.hypot(a[0] - b[0], a[1] - b[1]);

    const sameRow = dist(at(1, 1), at(2, 1));
    const diagonal = dist(at(1, 1), at(1, 2)); // adjacent row, hex-neighbour per odd-r geometry
    expect(sameRow).toBeCloseTo(diagonal, 3);
  });

  it('rect neighbours are axis-aligned at one spacing and diagonal at √2 that spacing', () => {
    const layout = layoutLattice(4, 4, 'rect', 400, 400);
    const at = (col: number, row: number): [number, number] => {
      const i = row * 4 + col;
      return [layout.xy[i * 2] as number, layout.xy[i * 2 + 1] as number];
    };
    const dist = (a: [number, number], b: [number, number]): number => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const axisAligned = dist(at(1, 1), at(2, 1));
    const diagonal = dist(at(1, 1), at(2, 2));
    expect(diagonal / axisAligned).toBeCloseTo(Math.SQRT2, 2);
  });

  it('does not produce NaN or Infinity for a degenerate 1×1 map', () => {
    const layout = layoutLattice(1, 1, 'hex', 100, 100);
    expect(Number.isFinite(layout.xy[0])).toBe(true);
    expect(Number.isFinite(layout.xy[1])).toBe(true);
    expect(Number.isFinite(layout.nodeRadius)).toBe(true);
  });
});

describe('hitLatticeNode', () => {
  it('finds the node under the pointer', () => {
    const layout = layoutLattice(3, 3, 'rect', 90, 90);
    const target = 4; // (1,1), roughly centre
    const x = layout.xy[target * 2] as number;
    const y = layout.xy[target * 2 + 1] as number;
    expect(hitLatticeNode(layout, x, y)).toBe(target);
  });

  it('returns -1 well outside the lattice', () => {
    const layout = layoutLattice(3, 3, 'rect', 90, 90);
    expect(hitLatticeNode(layout, -500, -500)).toBe(-1);
  });
});
