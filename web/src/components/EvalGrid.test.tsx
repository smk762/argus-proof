import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { EvalGrid, lightboxPrompt, reasonKey, type GridRow } from "./EvalGrid";

const row: GridRow = {
  id: "baseline",
  label: "Baseline",
  recommended: false,
  cells: [
    [
      {
        tile: "/t.webp",
        full: "/f.webp",
        metrics: { identity: 0.1, clip_score: 0.2, aesthetic: 0.3, safety: 1 },
        passed: false,
        reasons: ["same threshold", "same threshold"],
      },
    ],
  ],
  runs: [{ reasons: ["same threshold", "same threshold"], passed: false, pending: false }],
};

describe("EvalGrid lightbox", () => {
  it("shows the prompt the tape recorded", () => {
    expect(lightboxPrompt(["sks person, studio"], 0)).toBe("sks person, studio");
    const html = renderToStaticMarkup(
      <EvalGrid
        prompts={["studio"]}
        fullPrompts={["sks person, studio"]}
        rows={[row]}
        refs={[]}
        seeds={[1]}
        initialOpen={{ kind: "eval", row: 0, prompt: 0 }}
      />,
    );
    expect(html).toContain("sks person, studio");
    expect(html).not.toContain("ohwx woman");
  });

  it("keys duplicate reason strings apart", () => {
    expect(reasonKey(0, "same threshold")).toBe("0-same threshold");
    expect(reasonKey(1, "same threshold")).not.toBe(reasonKey(0, "same threshold"));
  });
});
