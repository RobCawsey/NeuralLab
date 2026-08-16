/**
 * The MLP's loss surface, in three dimensions — §7/§8 of the design document. A height field over
 * two directions in weight space, lit rather than colour-ramped: `packages/mlp/src/losssurface.ts`
 * already does the arithmetic, so everything here is geometry and a camera.
 */

import * as THREE from 'three';
import type { LossSurface } from '@neurallab/mlp';
import { createOrbitCamera, type OrbitCamera } from './orbit.ts';

const VIOLET = 0x8b7bd8;
const AMBER = 0xe9a13b;
const LINE = 0x2c2a3a;
const BG = 0x0e0d15;

export interface PathPoint {
  readonly alpha: number;
  readonly beta: number;
}

export interface LossSurfaceHandle {
  update(surface: LossSurface, path: readonly PathPoint[], current: PathPoint): void;
  resize(width: number, height: number): void;
  dispose(): void;
}

/** Loss values span whatever a network happens to produce; the mesh's own height never should. */
const HEIGHT_SCALE_TARGET = 1.4;

export function createLossSurfaceScene(canvas: HTMLCanvasElement): LossSurfaceHandle {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(BG, 1);

  const scene = new THREE.Scene();
  const orbit: OrbitCamera = createOrbitCamera(canvas, new THREE.Vector3(0, 0.4, 0), {
    radius: 3.2,
    theta: 0.7,
    phi: 1.05,
  });

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(2, 3, 2);
  scene.add(key);

  let mesh: THREE.Mesh | null = null;
  let wire: THREE.LineSegments | null = null;
  let pathLine: THREE.Line | null = null;
  let marker: THREE.Mesh | null = null;
  let raf: number | null = null;

  function loop(): void {
    renderer.render(scene, orbit.camera);
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);

  function heightAt(surface: LossSurface, scale: number, alpha: number, beta: number): number {
    // Nearest cell — the surface is a picture, not something worth interpolating precisely.
    const t = (v: number): number => Math.min(1, Math.max(0, (v + surface.range) / (2 * surface.range)));
    const col = Math.round(t(alpha) * (surface.res - 1));
    const row = Math.round(t(beta) * (surface.res - 1));
    return (surface.values[row * surface.res + col] as number) * scale;
  }

  return {
    update(surface, path, current) {
      if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
      if (wire) {
        scene.remove(wire);
        wire.geometry.dispose();
        (wire.material as THREE.Material).dispose();
      }
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
        (pathLine.material as THREE.Material).dispose();
      }
      if (marker) {
        scene.remove(marker);
        marker.geometry.dispose();
        (marker.material as THREE.Material).dispose();
      }

      const { res, range, values } = surface;
      let maxV = 1e-9;
      for (const v of values) if (v > maxV) maxV = v;
      const scale = HEIGHT_SCALE_TARGET / maxV;

      const geo = new THREE.PlaneGeometry(range * 2, range * 2, res - 1, res - 1);
      geo.rotateX(-Math.PI / 2); // plane's local Y becomes world-up
      const pos = geo.attributes['position'] as THREE.BufferAttribute;
      for (let row = 0; row < res; row++) {
        for (let col = 0; col < res; col++) {
          pos.setY(row * res + col, (values[row * res + col] as number) * scale);
        }
      }
      geo.computeVertexNormals();

      mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: VIOLET, flatShading: true, side: THREE.DoubleSide }));
      scene.add(mesh);
      wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: LINE, transparent: true, opacity: 0.5 }));
      scene.add(wire);

      if (path.length > 1) {
        const pts = path.map((p) => new THREE.Vector3(p.alpha, heightAt(surface, scale, p.alpha, p.beta) + 0.02, p.beta));
        pathLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: AMBER }));
        scene.add(pathLine);
      }

      marker = new THREE.Mesh(
        new THREE.SphereGeometry(range * 0.03, 16, 16),
        new THREE.MeshBasicMaterial({ color: AMBER }),
      );
      marker.position.set(current.alpha, heightAt(surface, scale, current.alpha, current.beta) + 0.03, current.beta);
      scene.add(marker);
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
