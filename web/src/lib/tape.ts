// The recorded tape (public/tape/tape.json, written by scripts/demo_tape.py) and the helpers the page uses.
// The page imports it at build time; the browser fetches nothing but the exported page and the WebP images.
import raw from "../../public/tape/tape.json";

export type Metric = "identity" | "clip_score" | "aesthetic" | "safety";
/** proof's per-image verdict: true = passed, null = needs review (near a threshold), false = rejected. */
export type Passed = boolean | null;

export interface TapeImage {
  image_id: string;
  seed: number;
  metrics: Record<Metric, number | null>;
  passed: Passed;
  reject_reasons: string[];
  duplicate_group: number | null;
  tile: string;
  full: string;
}

export interface TapeRun {
  run_id: string;
  prompt: number;
  seeds: number[];
  lora: { name: string; sha256: string; weight: number } | null;
  verdict: { passed: boolean; pending: boolean; reasons: string[] };
  pass_rate: number;
  diversity: number | null;
  n_groups: number;
  n_passed: number;
  n_needs_hitl: number;
  images: TapeImage[];
}

export interface Summary {
  n_images: number;
  means: Record<Metric, number | null>;
  pass_rate: number;
  diversity: number;
  passed: number;
  review: number;
  failed: number;
  runs_passed: number;
}

export interface Variant {
  id: string;
  label: string;
  step: number | null;
  lora: TapeRun["lora"];
  summary: Summary;
  runs: TapeRun[];
}

export interface Picture {
  id: string;
  tile: string;
  w: number;
  h: number;
  full?: string;
  caption?: string;
}

export interface Tape {
  format: "argus-proof-tape";
  version: number;
  prompts: string[];
  variants: Variant[];
  recording: {
    proof_version: string;
    wire_version: string;
    engine: string;
    base_checkpoint: { name: string; sha256: string };
    sampling: { sampler: string; scheduler: string; steps: number; cfg: number; width: number; height: number };
    seeds: number[];
    pass_threshold: number;
    scorers: { metric: string; name: string; model: string }[];
  };
  training: Record<string, string | number>;
  refs: Picture[];
  train: Picture[];
  token_probe: { prompt: number; plain_prompt: string; ohwx: string; plain: string }[];
}

export const tape = raw as unknown as Tape;

/** Where the tape's files are served from (public/tape/ is exported as /tape/). */
export const src = (path: string) => `/tape/${path}`;

/** The checkpoint the page recommends. Kept in one place so the chart, grid and cards agree. */
export const RECOMMENDED = "320";

export const baseline = (t: Tape = tape) => t.variants.find((v) => v.id === "baseline")!;
export const checkpoints = (t: Tape = tape) => t.variants.filter((v) => v.step !== null);

export const METRICS: { key: Metric | "pass_rate"; label: string; blurb: string }[] = [
  { key: "identity", label: "Identity", blurb: "ArcFace similarity to the 8 held-out references" },
  { key: "clip_score", label: "Prompt match", blurb: "CLIPScore: does the image show what the prompt asked for" },
  { key: "aesthetic", label: "Image quality", blurb: "CLIP-IQA, a no-reference quality score" },
  { key: "pass_rate", label: "Pass rate", blurb: "Share of images proof passed without review" },
];

/** A variant's value for a chart metric: an image mean, or the mean of its runs' pass rates. */
export function value(v: Variant, key: Metric | "pass_rate"): number {
  return key === "pass_rate" ? v.summary.pass_rate : (v.summary.means[key] ?? NaN);
}

export const fmt = (x: number | null | undefined, digits = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(digits));

export const VERDICT: Record<"pass" | "review" | "fail", { mark: string; label: string }> = {
  pass: { mark: "✓", label: "Passed" },
  review: { mark: "?", label: "Needs review" },
  fail: { mark: "✕", label: "Rejected" },
};

export const verdictKey = (p: Passed) => (p === true ? "pass" : p === null ? "review" : "fail");

/** How a run's verdict reads: a run that is neither passed nor pending was rejected. */
export function runVerdict(v: Pick<TapeRun["verdict"], "passed" | "pending">): { text: string; tone: string } {
  if (v.passed) return { text: "✓ passed", tone: "text-success" };
  if (v.pending) return { text: "? pending review", tone: "text-warning" };
  return { text: "✕ failed", tone: "text-error" };
}

/** The image for one (variant, prompt, seed) cell of the grid. */
export function cell(v: Variant, prompt: number, seed: number): TapeImage {
  const run = v.runs.find((r) => r.prompt === prompt);
  const image = run?.images.find((im) => im.seed === seed);
  if (!image) throw new Error(`no image for ${v.id} prompt ${prompt} seed ${seed}`);
  return image;
}

/** A prompt without its trigger, for a short column label. */
export const shortPrompt = (text: string) => text.replace(/^ohwx woman,\s*/, "");
