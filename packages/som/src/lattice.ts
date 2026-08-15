/**
 * Lattice geometry — the bug to design around, per the design document's own warning.
 *
 * Lattice distance is **not** Euclidean on `(col, row)`. That is true for hex and false for rect,
 * and the difference is exactly why this file exists on its own rather than being three lines
 * inlined wherever a distance is needed: getting the hex case wrong produces a map that still
 * trains, still looks plausible on the colour cube, and is quietly not a SOM — every neighbourhood
 * subtly the wrong shape on alternate rows. `buildNeighbours` is tested against a hand-counted
 * 3×3 for exactly that reason; nothing here is derived inline and trusted.
 *
 * Two distinct notions of "neighbour" live in this file, and conflating them is the second
 * easiest way to get this wrong:
 *
 *  - `latticeDistance` is continuous and answers "how far, for the neighbourhood function
 *    `h(d, t)`" — every node in the lattice has one, however large.
 *  - `buildNeighbours` is discrete and answers "which nodes does this one touch" — used for
 *    topographic error and the U-matrix, where the design document is explicit that a rect node's
 *    four diagonal cells at distance √2 do **not** count, even though `latticeDistance` sees them
 *    as close. Six slots either way, `-1` where the lattice or the topology has nothing there.
 */

export type Topology = 'hex' | 'rect';

/** `neighbours` is `cols * rows * 6` regardless of topology — see the module comment. */
export const NEIGHBOUR_SLOTS = 6;

export function nodeIndex(cols: number, col: number, row: number): number {
  return row * cols + col;
}

/**
 * Offset `(col, row)` to axial `(q, r)`, "odd-r" — odd rows are shoved right by half a hex.
 *
 * This is the one conversion in the file with a choice to make (odd-r vs. even-r vs. axial
 * storage from the start), and odd-r is picked and fixed here, once, because every neighbour
 * offset and every hand-counted test below is derived against it. Changing it later means
 * re-deriving all of them, not just this function.
 */
function toAxial(col: number, row: number): { q: number; r: number } {
  return { q: col - (row - (row & 1)) / 2, r: row };
}

/**
 * The six axial step directions, in a fixed order.
 *
 * The order is part of the contract: `buildNeighbours` writes each direction to the same slot
 * index for every node, so slot 0 always means "the same direction" across the whole lattice, and
 * the hand-counted test below can assert exact arrays rather than sets.
 */
const AXIAL_DIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
];

/** Hex distance between two axial points — the third cube coordinate falls out as `-q - r`. */
function hexDistance(q1: number, r1: number, q2: number, r2: number): number {
  const dq = q1 - q2;
  const dr = r1 - r2;
  const s1 = -q1 - r1;
  const s2 = -q2 - r2;
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(s1 - s2)) / 2;
}

/**
 * The continuous distance `h(d, t)` is evaluated against — every node in the lattice has one,
 * however far. Hex converts to axial first, per the module comment; rect is plain Euclidean on
 * `(col, row)`, which is exactly the design document's own description of rect's anisotropy
 * (four neighbours at distance 1, four more at √2) — nothing to convert, that √2 *is* the point.
 */
export function latticeDistance(
  cols: number,
  topology: Topology,
  a: number,
  b: number,
): number {
  const colA = a % cols;
  const rowA = Math.floor(a / cols);
  const colB = b % cols;
  const rowB = Math.floor(b / cols);
  if (topology === 'rect') {
    const dc = colA - colB;
    const dr = rowA - rowB;
    return Math.sqrt(dc * dc + dr * dr);
  }
  const A = toAxial(colA, rowA);
  const B = toAxial(colB, rowB);
  return hexDistance(A.q, A.r, B.q, B.r);
}

/**
 * The discrete adjacency table — six slots per node, `-1` past the lattice edge or where the
 * topology simply has nothing (rect's slots 4 and 5 are always `-1`; it only has four immediate
 * neighbours). Built once per `Som`, never derived inline — see the module comment for why.
 */
export function buildNeighbours(cols: number, rows: number, topology: Topology): Int32Array {
  const out = new Int32Array(cols * rows * NEIGHBOUR_SLOTS).fill(-1);

  if (topology === 'rect') {
    // N, S, W, E — the four at distance 1. Slots 4 and 5 stay -1: rect's diagonal cells are
    // distance √2 in `latticeDistance` but are not lattice *neighbours* for TE or the U-matrix.
    const dirs: ReadonlyArray<readonly [number, number]> = [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const node = nodeIndex(cols, col, row);
        for (let d = 0; d < dirs.length; d++) {
          const [dc, dr] = dirs[d] as [number, number];
          const nc = col + dc;
          const nr = row + dr;
          if (nc >= 0 && nc < cols && nr >= 0 && nr < rows) {
            out[node * NEIGHBOUR_SLOTS + d] = nodeIndex(cols, nc, nr);
          }
        }
      }
    }
    return out;
  }

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const node = nodeIndex(cols, col, row);
      const { q, r } = toAxial(col, row);
      for (let d = 0; d < AXIAL_DIRS.length; d++) {
        const [dq, dr] = AXIAL_DIRS[d] as [number, number];
        const nq = q + dq;
        const nr = r + dr;
        // Axial back to offset, odd-r: inverse of `toAxial`.
        const nRow = nr;
        const nCol = nq + (nRow - (nRow & 1)) / 2;
        if (nCol >= 0 && nCol < cols && nRow >= 0 && nRow < rows) {
          out[node * NEIGHBOUR_SLOTS + d] = nodeIndex(cols, nCol, nRow);
        }
      }
    }
  }
  return out;
}
