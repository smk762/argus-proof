import { PANEL, scales, ticks, yRange } from "@/lib/chart";
import { baseline, checkpoints, fmt, METRICS, RECOMMENDED, value } from "@/lib/tape";

/**
 * Four small multiples, one metric each, against training step: one y-scale per panel (never two in one), the
 * no-LoRA baseline as a dashed reference line, and the recommended checkpoint as a shaded band. Values are labelled
 * at the recommended step and the last one; every point has a native tooltip, and the table under the chart has
 * every number.
 */
export function ScoreChart() {
  const points = checkpoints();
  const steps = points.map((v) => v.step!);
  const base = baseline();
  const pick = points.findIndex((v) => v.id === RECOMMENDED);
  return (
    <div className="grid grid-cols-1 gap-3 min-[560px]:grid-cols-2 xl:grid-cols-4">
      {METRICS.map((m) => {
        const ys = points.map((v) => value(v, m.key));
        const b = value(base, m.key);
        const range = yRange([...ys, b]);
        const { x, y } = scales(steps, range);
        const p = PANEL;
        const half = (x(1) - x(0)) / 2;
        const last = ys.length - 1;
        const labelled = new Set([pick, last]);
        // The baseline's label sits at the right unless the last point's value label would land on it.
        const nearLast = Math.abs(y(ys[last]) - y(b)) < 22;
        const path = ys.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
        const summary = `${m.label} by training step: ${points.map((v, i) => `${v.step} ${fmt(ys[i])}`).join(", ")}; no LoRA ${fmt(b)}.`;
        return (
          <figure key={m.key} className="min-w-0 rounded-lg border border-base-600/60 bg-base-800 p-3">
            <figcaption className="mb-1">
              <span className="block text-small font-semibold text-text-primary">{m.label}</span>
              <span className="block text-caption text-text-muted">{m.blurb}</span>
            </figcaption>
            <svg viewBox={`0 0 ${p.width} ${p.height}`} className="h-auto w-full" role="img" aria-label={summary}>
              <rect className="chart-band" x={x(pick) - half} y={p.top - 6} width={half * 2} height={p.height - p.top - p.bottom + 6} rx={4} />
              {ticks(range).map((t) => (
                <g key={t}>
                  <line className="chart-grid" x1={p.left} x2={p.width - p.right} y1={y(t)} y2={y(t)} />
                  <text className="chart-axis" x={p.left - 6} y={y(t) + 4} textAnchor="end">
                    {t.toFixed(2)}
                  </text>
                </g>
              ))}
              {steps.map((s, i) => (
                <text key={s} className="chart-axis" x={x(i)} y={p.height - 8} textAnchor="middle">
                  {s}
                </text>
              ))}
              <line className="chart-base" x1={p.left} x2={p.width - p.right} y1={y(b)} y2={y(b)} />
              <text
                className="chart-note"
                x={nearLast ? p.left + 4 : p.width - p.right}
                y={y(b) + (y(b) > p.height - p.bottom - 14 ? -5 : 13)}
                textAnchor={nearLast ? "start" : "end"}
              >
                no LoRA {fmt(b)}
              </text>
              <path className="chart-line" d={path} />
              {ys.map((v, i) => (
                <g key={steps[i]}>
                  <circle className="chart-hit" cx={x(i)} cy={y(v)} r={14}>
                    <title>{`Step ${steps[i]}: ${m.label} ${fmt(v, 3)}`}</title>
                  </circle>
                  <circle className={i === pick ? "chart-dot-pick" : "chart-dot"} cx={x(i)} cy={y(v)} r={i === pick ? 5 : 4.5} pointerEvents="none" />
                  {labelled.has(i) ? (
                    <text className="chart-label" x={x(i)} y={y(v) - 10} textAnchor="middle" pointerEvents="none">
                      {fmt(v)}
                    </text>
                  ) : null}
                </g>
              ))}
            </svg>
          </figure>
        );
      })}
    </div>
  );
}
