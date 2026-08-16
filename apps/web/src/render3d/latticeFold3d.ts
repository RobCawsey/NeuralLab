/**
 * The SOM lattice, folding through input space in three dimensions — §7/§8's "the best thirty
 * seconds in the whole project". `render/inputspace.ts`'s flat two dimensions, extended by one:
 * nodes at their own weight vectors, connected by their lattice edges, floating in the data cloud.
 *
 * Exact at three dimensions — every dataset this project has today, colour cube included. Above
 * three (digits, slice 16) the design document's own answer is PCA on the data, named on screen
 * rather than left unstated; nothing here needs it yet, so it is not built yet either.
 */

import * as THREE from 'three';
import { sample, type Dataset } from '@neurallab/core';
import { NEIGHBOUR_SLOTS, type Som } from '@neurallab/som';
import { createOrbitCamera, type OrbitCamera } from './orbit.ts';

const EDGE = 0x2f5a68;
const DATA_POINT = 0x5c5871;
const LINE = 0x2c2a3a;
const BG = 0x0e0d15;

export interface LatticeFoldHandle {
  update(som: Som, ds: Dataset, dims: readonly [number, number, number]): void;
  resize(width: number, height: number): void;
  dispose(): void;
}

function weightColour(som: Som, node: number): THREE.Color {
  const base = node * som.dim;
  const c = (k: number): number => Math.min(1, Math.max(0, som.W[base + k] as number));
  return new THREE.Color(c(0), som.dim > 1 ? c(1) : 0.5, som.dim > 2 ? c(2) : 0.5);
}

export function createLatticeFold3dScene(canvas: HTMLCanvasElement): LatticeFoldHandle {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(BG, 1);

  const scene = new THREE.Scene();
  const orbit: OrbitCamera = createOrbitCamera(canvas, new THREE.Vector3(0.5, 0.5, 0.5), {
    radius: 2.6,
    theta: 0.8,
    phi: 1.1,
  });
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: LINE })).translateX(0.5).translateY(0.5).translateZ(0.5));

  let dataPoints: THREE.Points | null = null;
  let edges: THREE.LineSegments | null = null;
  let nodes: THREE.Points | null = null;
  let raf: number | null = null;

  function loop(): void {
    renderer.render(scene, orbit.camera);
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);

  return {
    update(som, ds, dims) {
      if (dataPoints) {
        scene.remove(dataPoints);
        dataPoints.geometry.dispose();
        (dataPoints.material as THREE.Material).dispose();
      }
      if (edges) {
        scene.remove(edges);
        edges.geometry.dispose();
        (edges.material as THREE.Material).dispose();
      }
      if (nodes) {
        scene.remove(nodes);
        nodes.geometry.dispose();
        (nodes.material as THREE.Material).dispose();
      }

      // Context: a thinned sample of the data, the same reasoning `inputspace.ts` uses — this
      // scene is about the lattice, not a second scatter plot.
      const maxPoints = 400;
      const stride = Math.max(1, Math.floor(ds.n / maxPoints));
      const dataPos: number[] = [];
      for (let i = 0; i < ds.n; i += stride) {
        const p = sample(ds, i);
        dataPos.push(p[dims[0]] as number, p[dims[1]] as number, p[dims[2]] as number);
      }
      const dataGeo = new THREE.BufferGeometry();
      dataGeo.setAttribute('position', new THREE.Float32BufferAttribute(dataPos, 3));
      dataPoints = new THREE.Points(dataGeo, new THREE.PointsMaterial({ color: DATA_POINT, size: 0.02, transparent: true, opacity: 0.5 }));
      scene.add(dataPoints);

      const n = som.cols * som.rows;
      const nodeAt = (i: number): [number, number, number] => {
        const base = i * som.dim;
        return [som.W[base + dims[0]] as number, som.W[base + dims[1]] as number, som.W[base + dims[2]] as number];
      };

      const edgePos: number[] = [];
      for (let i = 0; i < n; i++) {
        for (let s = 0; s < NEIGHBOUR_SLOTS; s++) {
          const j = som.neighbours[i * NEIGHBOUR_SLOTS + s] as number;
          if (j <= i) continue;
          edgePos.push(...nodeAt(i), ...nodeAt(j));
        }
      }
      const edgeGeo = new THREE.BufferGeometry();
      edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
      edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: EDGE }));
      scene.add(edges);

      const nodePos: number[] = [];
      const nodeColour: number[] = [];
      for (let i = 0; i < n; i++) {
        nodePos.push(...nodeAt(i));
        const c = weightColour(som, i);
        nodeColour.push(c.r, c.g, c.b);
      }
      const nodeGeo = new THREE.BufferGeometry();
      nodeGeo.setAttribute('position', new THREE.Float32BufferAttribute(nodePos, 3));
      nodeGeo.setAttribute('color', new THREE.Float32BufferAttribute(nodeColour, 3));
      nodes = new THREE.Points(nodeGeo, new THREE.PointsMaterial({ size: 0.045, vertexColors: true }));
      scene.add(nodes);
    },
    resize(width, height) {
      renderer.setSize(width, height, false);
      orbit.camera.aspect = width / Math.max(1, height);
      orbit.camera.updateProjectionMatrix();
    },
    dispose() {
      if (raf !== null) cancelAnimationFrame(raf);
      orbit.dispose();
      renderer.dispose();
    },
  };
}
