import { describe, expect, it } from "vitest";

import { PANEL, scales, ticks, yRange } from "./chart";

describe("score chart geometry", () => {
  it("holds every value and the baseline, padded, inside [0, 1]", () => {
    const [lo, hi] = yRange([0.47, 0.41, 0.11]);
    expect(lo).toBeLessThan(0.11);
    expect(hi).toBeGreaterThan(0.47);
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(yRange([0.99, 1.0])[1]).toBe(1);
  });

  it("draws at most five gridlines, inside the range", () => {
    for (const r of [yRange([0.11, 0.47]), yRange([0.76, 0.86]), yRange([0.21, 0.88])]) {
      const t = ticks(r);
      expect(t.length).toBeGreaterThan(1);
      expect(t.length).toBeLessThanOrEqual(5);
      for (const v of t) expect(v >= r[0] - 1e-9 && v <= r[1] + 1e-9).toBe(true);
    }
  });

  it("maps the first and last step inside the plot, higher values higher up", () => {
    const { x, y } = scales([320, 640, 960, 1280, 1600], [0, 1]);
    expect(x(0)).toBeGreaterThan(PANEL.left);
    expect(x(4)).toBeLessThan(PANEL.width - PANEL.right);
    expect(y(1)).toBeLessThan(y(0));
  });
});
