// The committed tape against the story the page tells: if a re-record changes the numbers, these fail before the
// page says something the data no longer shows.
import { describe, expect, it } from "vitest";

import { baseline, cell, checkpoints, RECOMMENDED, shortPrompt, tape, value, verdictKey } from "./tape";

const byId = (id: string) => tape.variants.find((v) => v.id === id)!;

describe("tape", () => {
  it("has the baseline and five checkpoints, each with every prompt and seed", () => {
    expect(tape.variants.map((v) => v.id)).toEqual(["baseline", "320", "640", "960", "1280", "1600"]);
    expect(tape.prompts).toHaveLength(8);
    for (const v of tape.variants) {
      expect(v.runs.map((r) => r.prompt)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      for (let p = 0; p < 8; p++) for (const s of [1, 2, 3]) expect(cell(v, p, s).seed).toBe(s);
      expect(v.summary.n_images).toBe(24);
    }
  });

  it("keys runs by prompt text: a retried run sits under its prompt", () => {
    expect(byId("960").runs[7].run_id).toBe("synth-s00000960-retry-p007-l00-w00-p000");
    expect(byId("baseline").runs[3].run_id).toBe("baseline-retry-p003-l00-w00-p000");
    expect(byId("baseline").runs[5].run_id).toBe("baseline-rest-p005-l00-w00-p000");
  });

  it("only the baseline runs without a LoRA, and each checkpoint's LoRA is its own step", () => {
    expect(baseline().lora).toBeNull();
    for (const v of checkpoints()) expect(v.lora?.name).toBe(`proof-synth-v3-step${String(v.step).padStart(8, "0")}.safetensors`);
  });

  it("recomputes each summary from its images", () => {
    for (const v of tape.variants) {
      const images = v.runs.flatMap((r) => r.images);
      const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
      expect(v.summary.means.identity).toBeCloseTo(mean(images.map((im) => im.metrics.identity!)), 3);
      expect(v.summary.pass_rate).toBeCloseTo(mean(v.runs.map((r) => r.pass_rate)), 3);
      expect(v.summary.passed + v.summary.review + v.summary.failed).toBe(24);
      expect(v.summary.passed).toBe(images.filter((im) => verdictKey(im.passed) === "pass").length);
    }
  });

  it("supports the verdict the page states", () => {
    const [s320, s640, s1600, base] = ["320", "640", "1600", "baseline"].map(byId);
    expect(RECOMMENDED).toBe("320");
    // 320 and 640 tie on identity and prompt match (to two places)...
    expect(value(s320, "identity").toFixed(2)).toBe(value(s640, "identity").toFixed(2));
    expect(value(s320, "clip_score").toFixed(2)).toBe(value(s640, "clip_score").toFixed(2));
    expect(Math.abs(value(s320, "clip_score") - value(s640, "clip_score"))).toBeLessThan(0.01);
    // ...320 leads on quality and pass rate, and is the best checkpoint on both.
    for (const v of checkpoints().filter((c) => c.id !== "320")) {
      expect(value(s320, "aesthetic")).toBeGreaterThan(value(v, "aesthetic"));
      expect(value(s320, "pass_rate")).toBeGreaterThan(value(v, "pass_rate"));
    }
    // After 640 prompt match falls, and identity ends lower than it started.
    for (const id of ["960", "1280", "1600"]) expect(value(byId(id), "clip_score")).toBeLessThan(value(s640, "clip_score") - 0.05);
    expect(value(s1600, "identity")).toBeLessThan(value(s320, "identity") - 0.04);
    // The LoRA is what carries identity.
    expect(value(base, "identity")).toBeLessThan(0.2);
  });

  it("publishes no path, address or email and no never-run metric", () => {
    const text = JSON.stringify(tape);
    expect(text).not.toMatch(/\/(?:home|Users|root|data|srv|mnt|models)\//);
    expect(text).not.toMatch(/"preference"/);
    expect(text).not.toMatch(/@[\w-]+\.[a-z]{2,}/i);
  });

  it("shortens a prompt by its trigger only", () => {
    expect(shortPrompt("ohwx woman, portrait, white sweater, pose")).toBe("portrait, white sweater, pose");
  });
});
