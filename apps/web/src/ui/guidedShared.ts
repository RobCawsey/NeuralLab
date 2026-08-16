/**
 * The guided flow's visual vocabulary, shared between the MLP's controller and the SOM's.
 *
 * Not "one renderer" in the literal sense §13 first imagined — the two controllers read
 * different-shaped state (`AppState` vs `SomState`) and show different steps, so a single
 * polymorphic function would need as many branches as two functions have lines. What *is* shared,
 * because it is genuinely the same idea both times, is how a step card and a choice button look
 * and behave — pure DOM construction with no state of its own, extracted here rather than copied.
 */

import { type StepStatus } from '../run/guided.ts';

export function stepEl(
  index: number,
  status: StepStatus,
  title: string,
  body: (() => HTMLElement) | null,
): HTMLElement {
  const step = document.createElement('div');
  step.className = 'gd-step' + (status === 'later' ? ' later' : status === 'on' ? ' on' : status === 'done' ? ' done' : '');

  const head = document.createElement('div');
  head.className = 'gd-head';
  const n = document.createElement('span');
  n.className = 'gd-n';
  n.textContent = String(index + 1);
  const b = document.createElement('b');
  b.textContent = title;
  head.append(n, b);
  if (status === 'done') {
    const em = document.createElement('em');
    em.className = 'ok';
    em.textContent = 'done';
    head.append(em);
  } else if (status === 'on') {
    const em = document.createElement('em');
    em.className = 'am';
    em.textContent = 'now';
    head.append(em);
  }
  step.append(head);

  if (status !== 'later' && body) {
    const bodyEl = document.createElement('div');
    bodyEl.className = 'gd-body';
    bodyEl.append(body());
    step.append(bodyEl);
  }
  return step;
}

export function choiceButton(label: string, sub: string, on: boolean, onClick: () => void): HTMLElement {
  const btn = document.createElement('button');
  btn.className = 'gd-choice' + (on ? ' on' : '');
  const b = document.createElement('b');
  b.textContent = label;
  const span = document.createElement('span');
  span.textContent = sub;
  btn.append(b, span);
  btn.addEventListener('click', onClick);
  return btn;
}

export function skipButton(onClick: () => void): HTMLElement {
  const skip = document.createElement('button');
  skip.className = 'ghost wide';
  skip.style.margin = '4px 12px 0';
  skip.style.width = 'calc(100% - 24px)';
  skip.textContent = 'Skip to the full app';
  skip.addEventListener('click', onClick);
  return skip;
}
