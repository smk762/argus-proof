import type { ReactNode } from "react";

import { type Passed, VERDICT, verdictKey } from "@/lib/tape";

const TONE = {
  pass: "border-success/50 bg-success/10 text-success",
  review: "border-warning/60 bg-warning/10 text-warning",
  fail: "border-error/50 bg-error/10 text-error",
} as const;

/** proof's per-image verdict as a word with a mark; the colour is never the only signal. */
export function VerdictBadge({ passed, compact = false, className = "" }: { passed: Passed; compact?: boolean; className?: string }) {
  const key = verdictKey(passed);
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-sm border px-1.5 py-0.5 text-caption font-semibold ${TONE[key]} ${className}`}
      title={compact ? VERDICT[key].label : undefined}
    >
      <span aria-hidden="true">{VERDICT[key].mark}</span>
      {compact ? <span className="sr-only">{VERDICT[key].label}</span> : VERDICT[key].label}
    </span>
  );
}

export function Card({ children, className = "", as: Tag = "section" }: { children: ReactNode; className?: string; as?: "section" | "div" | "article" | "aside" }) {
  return <Tag className={`min-w-0 rounded-lg border border-base-600/60 bg-base-800 p-4 ${className}`}>{children}</Tag>;
}

export function SectionTitle({ children, className = "", id }: { children: ReactNode; className?: string; id?: string }) {
  return (
    <h2 id={id} className={`mb-3 font-display text-small font-semibold uppercase tracking-wide text-text-secondary ${className}`}>
      {children}
    </h2>
  );
}

/** A section's lead: a plain-language heading and a sentence or two under it. */
export function Lead({ id, title, children }: { id: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-4 max-w-3xl">
      <h2 id={id} className="font-display text-subheading font-semibold text-text-primary sm:text-heading">
        {title}
      </h2>
      {children ? <div className="mt-1 text-body text-text-secondary">{children}</div> : null}
    </div>
  );
}

/** A label and its value. Long values (hashes) wrap anywhere rather than widening the page. */
export function Kv({ label, children, mono = false }: { label: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-0.5 border-b border-base-600/30 py-2 last:border-b-0 sm:grid-cols-[11rem_minmax(0,1fr)]">
      <dt className="text-small text-text-muted">{label}</dt>
      <dd className={`min-w-0 text-small text-text-primary [overflow-wrap:anywhere] ${mono ? "font-mono text-caption sm:text-small" : ""}`}>{children}</dd>
    </div>
  );
}

export const LINK = "text-interactive underline decoration-interactive/40 underline-offset-2 hover:decoration-interactive";

export function Ext({ href, children, className = "" }: { href: string; children: ReactNode; className?: string }) {
  return (
    <a href={href} className={`${LINK} ${className}`} rel="noopener noreferrer">
      {children}
    </a>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code className="rounded-sm bg-base-700 px-1 py-0.5 font-mono text-[0.9em] [overflow-wrap:anywhere]">{children}</code>;
}
