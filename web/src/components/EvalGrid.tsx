"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { fmt, type Metric, type Passed, VERDICT, verdictKey } from "@/lib/tape";

import { VerdictBadge } from "./ui";

/** One scored image, slimmed for the client (the page passes these from the tape at build time). */
export interface GridImage {
  tile: string;
  full: string;
  metrics: Record<Metric, number | null>;
  passed: Passed;
  reasons: string[];
}

export interface GridRow {
  id: string;
  label: string;
  recommended: boolean;
  /** [prompt][seed index] */
  cells: GridImage[][];
  /** Proof's run verdict per prompt: its reasons, first line first. */
  runs: { reasons: string[]; passed: boolean; pending: boolean }[];
}

export interface GridRef {
  id: string;
  tile: string;
  full: string;
  w: number;
  h: number;
}

type Open = { kind: "eval"; row: number; prompt: number } | { kind: "ref"; index: number };

const METRIC_ROWS: { key: Metric; label: string }[] = [
  { key: "identity", label: "Identity (ArcFace)" },
  { key: "clip_score", label: "Prompt match (CLIPScore)" },
  { key: "aesthetic", label: "Image quality (CLIP-IQA)" },
  { key: "safety", label: "Safety (NudeNet)" },
];

// Fixed columns on a phone (the box scrolls); shared-out columns from sm up. Never `1fr` under min-w-max: it would
// resolve to the images' intrinsic 512 px.
const COLS = "grid-cols-[5.25rem_repeat(8,5.25rem)] sm:grid-cols-[6.5rem_repeat(8,minmax(0,1fr))]";
const TILE_BTN =
  "group relative block w-full overflow-hidden rounded-md border border-base-600/60 bg-base-900 focus-visible:outline-offset-1 hover:border-interactive";

/**
 * The evaluation grid: a row per variant (no LoRA, then each checkpoint), a column per prompt, one seed at a time.
 * Each tile opens a lightbox with the 1024 px image, its four scores and proof's verdict. The grid scrolls sideways
 * inside its own box on a phone, so the page never does.
 */
export function EvalGrid({ prompts, rows, refs, seeds }: { prompts: string[]; rows: GridRow[]; refs: GridRef[]; seeds: number[] }) {
  const [seed, setSeed] = useState(0);
  const [open, setOpen] = useState<Open | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = dialog.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const move = useCallback(
    (dPrompt: number, dRow: number) =>
      setOpen((o) => {
        if (!o) return o;
        if (o.kind === "ref") return { kind: "ref", index: (o.index + dPrompt + refs.length) % refs.length };
        return {
          kind: "eval",
          prompt: (o.prompt + dPrompt + prompts.length) % prompts.length,
          row: Math.min(rows.length - 1, Math.max(0, o.row + dRow)),
        };
      }),
    [prompts.length, rows.length, refs.length],
  );

  const onKey = (e: React.KeyboardEvent) => {
    const keys: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const k = keys[e.key];
    if (!k) return;
    e.preventDefault();
    move(k[0], k[1]);
  };

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <div role="group" aria-label="Seed" className="inline-flex items-center gap-1 rounded-md border border-base-600 bg-base-800 p-1">
          <span className="px-2 text-small text-text-secondary">Seed</span>
          {seeds.map((s, i) => (
            <button
              key={s}
              type="button"
              aria-pressed={i === seed}
              onClick={() => setSeed(i)}
              className={`min-h-9 min-w-9 rounded-sm px-2 text-small font-semibold ${i === seed ? "bg-accent-primary text-on-accent" : "text-text-secondary hover:bg-base-700 hover:text-text-primary"}`}
            >
              {s}
            </button>
          ))}
        </div>
        <p className="text-caption text-text-muted">
          Tiles show proof&apos;s verdict (✓ passed, ? needs review) and the identity score. Tap one for the full image and all its scores.
        </p>
      </div>

      <div className="-mx-4 overflow-x-auto px-4 pb-2 sm:mx-0 sm:px-0" tabIndex={0} aria-label="Evaluation grid, scrolls sideways">
        <div className={`grid min-w-max gap-1.5 sm:min-w-0 ${COLS}`}>
          <div className="sticky left-0 z-10 bg-base-900" />
          {prompts.map((p, i) => (
            <div key={i} className="min-w-0 px-0.5 pb-1 text-caption leading-tight text-text-muted" title={p}>
              <span className="font-semibold text-text-secondary">P{i + 1}</span> <span className="line-clamp-2">{p}</span>
            </div>
          ))}

          <RowLabel label="Held-out refs" note="never trained on" />
          {refs.map((r, i) => (
            <button key={r.id} type="button" className={TILE_BTN} onClick={() => setOpen({ kind: "ref", index: i })} aria-label={`Held-out reference ${i + 1}`}>
              <img src={r.tile} alt="" width={r.w} height={r.h} loading="lazy" decoding="async" className="aspect-square h-auto w-full object-cover" />
            </button>
          ))}

          {rows.map((row, ri) => (
            <Row key={row.id} row={row} ri={ri} seed={seed} onOpen={(prompt) => setOpen({ kind: "eval", row: ri, prompt })} />
          ))}
        </div>
      </div>

      <dialog ref={dialog} className="lightbox" onClose={() => setOpen(null)} onKeyDown={onKey} aria-labelledby="lightbox-title">
        {open ? <LightboxBody open={open} seed={seed} seeds={seeds} rows={rows} refs={refs} prompts={prompts} onMove={move} onClose={() => setOpen(null)} /> : null}
      </dialog>
    </div>
  );
}

function RowLabel({ label, note, recommended = false }: { label: string; note?: string; recommended?: boolean }) {
  return (
    <div className="sticky left-0 z-10 flex flex-col justify-center bg-base-900 pr-1.5">
      <span className={`text-small font-semibold ${recommended ? "text-interactive" : "text-text-primary"}`}>{label}</span>
      {note ? <span className={`text-caption leading-tight ${recommended ? "font-semibold text-interactive" : "text-text-muted"}`}>{note}</span> : null}
    </div>
  );
}

function Row({ row, ri, seed, onOpen }: { row: GridRow; ri: number; seed: number; onOpen: (prompt: number) => void }) {
  const note = row.recommended ? "ship this" : ri === 0 ? "base SDXL" : undefined;
  return (
    <>
      <RowLabel label={row.label} note={note} recommended={row.recommended} />
      {row.cells.map((bySeed, pi) => {
        const im = bySeed[seed];
        const v = VERDICT[verdictKey(im.passed)];
        return (
          <button
            key={pi}
            type="button"
            className={`${TILE_BTN} ${row.recommended ? "ring-2 ring-accent-primary ring-offset-1 ring-offset-base-900" : ""}`}
            onClick={() => onOpen(pi)}
            aria-label={`${row.label}, prompt ${pi + 1}: ${v.label}, identity ${fmt(im.metrics.identity)}`}
          >
            <img src={im.tile} alt="" width={512} height={512} loading="lazy" decoding="async" className="aspect-square h-auto w-full object-cover" />
            <span className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1 bg-base-900/85 px-1 py-0.5 text-caption leading-tight text-text-primary">
              <span aria-hidden="true" className={im.passed === true ? "text-success" : im.passed === null ? "text-warning" : "text-error"}>
                {v.mark}
              </span>
              <span className="font-mono tabular-nums">{fmt(im.metrics.identity)}</span>
            </span>
          </button>
        );
      })}
    </>
  );
}

function LightboxBody({
  open,
  seed,
  seeds,
  rows,
  refs,
  prompts,
  onMove,
  onClose,
}: {
  open: Open;
  seed: number;
  seeds: number[];
  rows: GridRow[];
  refs: GridRef[];
  prompts: string[];
  onMove: (dPrompt: number, dRow: number) => void;
  onClose: () => void;
}) {
  const navBtn = "min-h-9 rounded-md border border-base-600 px-3 text-small text-text-secondary hover:bg-base-700 hover:text-text-primary disabled:opacity-40";
  if (open.kind === "ref") {
    const r = refs[open.index];
    return (
      <div className="flex max-h-[calc(100dvh-18px)] flex-col">
        <Bar title={`Held-out reference ${open.index + 1} of ${refs.length}`} onClose={onClose} />
        <div className="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto md:grid-cols-[minmax(0,1fr)_20rem]">
          <img src={r.full} alt={`Held-out reference ${open.index + 1}`} width={r.w * 2} height={r.h * 2} className="h-auto max-h-[78dvh] w-full bg-base-900 object-contain" />
          <div className="space-y-3 p-4 text-small text-text-secondary">
            <p>
              One of the eight held-out references. None was trained on: they share no prompt with the training set, and the closest train/ref pair is 16 of 64
              bits apart by perceptual hash.
            </p>
            <p>Every generated image&apos;s identity score is its ArcFace similarity to these eight.</p>
            <div className="flex gap-2">
              <button type="button" className={navBtn} onClick={() => onMove(-1, 0)}>
                ← Prev
              </button>
              <button type="button" className={navBtn} onClick={() => onMove(1, 0)}>
                Next →
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }
  const row = rows[open.row];
  const im = row.cells[open.prompt][seed];
  const run = row.runs[open.prompt];
  return (
    <div className="flex max-h-[calc(100dvh-18px)] flex-col">
      <Bar title={`${row.label} · prompt ${open.prompt + 1} · seed ${seeds[seed]}`} onClose={onClose} />
      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto md:grid-cols-[minmax(0,1fr)_21rem]">
        <img
          src={im.full}
          alt={`${row.label}, seed ${seeds[seed]}: ${prompts[open.prompt]}`}
          width={1024}
          height={1024}
          className="aspect-square h-auto max-h-[78dvh] w-full bg-base-900 object-contain"
        />
        <div className="min-w-0 space-y-3 p-4">
          <div>
            <p className="text-caption uppercase tracking-wide text-text-muted">Prompt</p>
            <p className="font-mono text-small text-text-primary [overflow-wrap:anywhere]">
              ohwx woman, {prompts[open.prompt]}
              {row.id === "baseline" ? <span className="block font-sans text-caption text-text-muted">Same prompt, base SDXL, no LoRA.</span> : null}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <VerdictBadge passed={im.passed} />
            {row.recommended ? <span className="text-caption font-semibold text-interactive">recommended checkpoint</span> : null}
          </div>
          <dl className="divide-y divide-base-600/30 text-small">
            {METRIC_ROWS.map((m) => (
              <div key={m.key} className="flex justify-between gap-3 py-1.5">
                <dt className="text-text-secondary">{m.label}</dt>
                <dd className="font-mono tabular-nums text-text-primary">{fmt(im.metrics[m.key], 3)}</dd>
              </div>
            ))}
          </dl>
          {im.reasons.length ? (
            <ul className="list-disc pl-5 text-small text-error">
              {im.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          ) : im.passed === null ? (
            <p className="text-caption text-text-muted">Needs review: a score sits near its threshold, so proof leaves the call to a person. It is not a failure.</p>
          ) : null}
          <div className="rounded-md bg-base-700 p-2 text-caption text-text-secondary">
            <p className="font-semibold text-text-primary">This prompt&apos;s run ({seeds.length} seeds)</p>
            {run.reasons.map((r) => (
              <p key={r} className="[overflow-wrap:anywhere]">
                {r}
              </p>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className={navBtn} onClick={() => onMove(-1, 0)} aria-label="Previous prompt">
              ← Prompt
            </button>
            <button type="button" className={navBtn} onClick={() => onMove(1, 0)} aria-label="Next prompt">
              Prompt →
            </button>
            <button type="button" className={navBtn} onClick={() => onMove(0, -1)} disabled={open.row === 0} aria-label="Earlier checkpoint">
              ↑ Earlier
            </button>
            <button type="button" className={navBtn} onClick={() => onMove(0, 1)} disabled={open.row === rows.length - 1} aria-label="Later checkpoint">
              ↓ Later
            </button>
          </div>
          <p className="text-caption text-text-muted">Arrow keys move too; Esc closes.</p>
        </div>
      </div>
    </div>
  );
}

function Bar({ title, onClose }: { title: string; onClose: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-base-600/60 px-4 py-2">
      <h3 id="lightbox-title" className="min-w-0 truncate text-small font-semibold text-text-primary">
        {title}
      </h3>
      <button
        type="button"
        onClick={onClose}
        className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-text-secondary hover:bg-base-700 hover:text-text-primary"
        aria-label="Close"
      >
        <span aria-hidden="true">✕</span>
      </button>
    </div>
  );
}
