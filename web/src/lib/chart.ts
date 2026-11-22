// The score chart's geometry, kept pure so it can be tested: a panel per metric, steps on x, the metric on y.

export interface Panel {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export const PANEL: Panel = { width: 300, height: 176, left: 34, right: 12, top: 16, bottom: 26 };

/**
 * A y-range that holds every value and the baseline, padded and snapped to 0.05. Clamped to [0, 1] when the
 * values lie inside it (clamping data outside would invert or collapse the range); [0, 1] for a panel with no
 * values yet.
 */
export function yRange(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return [0, 1];
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const pad = Math.max((hi - lo) * 0.12, 0.02);
  const snap = (x: number, f: (n: number) => number) => f(x / 0.05) * 0.05;
  const padded: [number, number] = [snap(lo - pad, Math.floor), snap(hi + pad, Math.ceil)];
  if (lo < 0 || hi > 1) return padded;
  return [Math.max(0, padded[0]), Math.min(1, padded[1])];
}

/** Gridline values: every 0.05, 0.1 or 0.2, whichever gives at most five lines. */
export function ticks([lo, hi]: [number, number]): number[] {
  const step = [0.05, 0.1, 0.2, 0.25].find((s) => (hi - lo) / s <= 4) ?? 0.5;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t * 100) / 100);
  return out;
}

export function scales(steps: number[], range: [number, number], p: Panel = PANEL) {
  const x0 = p.left + 14;
  const x1 = p.width - p.right - 14;
  const x = (i: number) => (steps.length === 1 ? (x0 + x1) / 2 : x0 + ((x1 - x0) * i) / (steps.length - 1));
  const y = (v: number) => p.top + (p.height - p.top - p.bottom) * (1 - (v - range[0]) / (range[1] - range[0]));
  return { x, y };
}
