/**
 * A minimal orbit camera — drag to orbit, wheel to zoom, double-click to reset. Both 3D scenes
 * use this rather than three's own `OrbitControls` addon: the whole interaction is a camera held
 * in spherical coordinates around a fixed target, updated from three pointer events, and writing
 * it here keeps the dependency surface at the one `three` package itself.
 */

import * as THREE from 'three';

export interface OrbitSpherical {
  readonly radius: number;
  readonly theta: number; // azimuth, radians
  readonly phi: number; // polar, radians — clamped away from the poles
}

export interface OrbitCamera {
  readonly camera: THREE.PerspectiveCamera;
  dispose(): void;
}

const MIN_PHI = 0.06;
const MAX_PHI = Math.PI - 0.06;

export function createOrbitCamera(
  canvas: HTMLCanvasElement,
  target: THREE.Vector3,
  initial: OrbitSpherical,
): OrbitCamera {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
  let state: OrbitSpherical = { ...initial };
  const home = { ...initial };

  function place(): void {
    const { radius, theta, phi } = state;
    camera.position.set(
      target.x + radius * Math.sin(phi) * Math.sin(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.cos(theta),
    );
    camera.lookAt(target);
  }
  place();

  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const onDown = (e: PointerEvent): void => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onMove = (e: PointerEvent): void => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    state = {
      radius: state.radius,
      theta: state.theta - dx * 0.008,
      phi: Math.min(MAX_PHI, Math.max(MIN_PHI, state.phi - dy * 0.008)),
    };
    place();
  };
  const onUp = (e: PointerEvent): void => {
    dragging = false;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    state = { ...state, radius: Math.min(40, Math.max(0.3, state.radius * (1 + e.deltaY * 0.001))) };
    place();
  };
  const onDblClick = (): void => {
    state = { ...home };
    place();
  };

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointerleave', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDblClick);

  return {
    camera,
    dispose(): void {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointerleave', onUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDblClick);
    },
  };
}
