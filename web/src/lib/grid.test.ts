import { describe, expect, it } from "vitest";

// Imported per test, so a tree without the layout module fails these tests rather than failing to load the file.
const grid = () => import("./grid");

describe("evaluation grid layout", () => {
  it("has one column per prompt, not a fixed eight", async () => {
    const { gridCols } = await grid();
    expect(gridCols(8)).toBe(8);
    expect(gridCols(5)).toBe(5);
    expect(gridCols(12)).toBe(12);
    expect(gridCols(0)).toBe(1);
  });

  it("fills every refs line to exactly the prompt columns, whatever the ref count", async () => {
    const { gridCols, refLines } = await grid();
    for (const [nRefs, cols] of [[8, 8], [3, 8], [11, 8], [8, 5], [0, 4]]) {
      const refs = Array.from({ length: nRefs }, (_, i) => `r${i}`);
      const lines = refLines(refs, gridCols(cols));
      for (const line of lines) expect(line.items.length + line.pad).toBe(cols);
      expect(lines.flatMap((l) => l.items.map((it) => it.ref))).toEqual(refs);
      expect(lines.flatMap((l) => l.items.map((it) => it.index))).toEqual(refs.map((_, i) => i));
    }
  });
});
