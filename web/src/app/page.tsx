import { EvalGrid, type GridRow } from "@/components/EvalGrid";
import { ScoreChart } from "@/components/ScoreChart";
import { Card, Code, Ext, Kv, Lead, SectionTitle } from "@/components/ui";
import { baseline, fmt, RECOMMENDED, shortPrompt, src, tape, type Variant } from "@/lib/tape";

const { recording: rec, training: tr } = tape;
const DOCS = (pkg: string) => `https://docs.dragonhound.dev/packages/${pkg}/`;
const pick = tape.variants.find((v) => v.id === RECOMMENDED)!;
const base = baseline();
const n = tape.variants.length * tape.prompts.length * rec.seeds.length;

/** How each variant reads, in one tag. Kept next to the numbers it summarises (tape.test.ts checks them). */
function standing(v: Variant): { tag: string; tone: string; text: string } {
  if (v.step === null) return { tag: "Reference", tone: "border-base-600 bg-base-700 text-text-secondary", text: "Base SDXL, same prompts and seeds, no LoRA. Good prompt match, no identity." };
  if (v.id === RECOMMENDED)
    return { tag: "Ship this", tone: "border-success/50 bg-success/10 text-success", text: "Identity learned, best image quality and pass rate of any checkpoint." };
  if (v.step === 640) return { tag: "Also fine", tone: "border-success/50 bg-success/10 text-success", text: "Ties step 320 on identity and prompt match; lower quality and pass rate." };
  return { tag: "Over-trained", tone: "border-warning/60 bg-warning/10 text-warning", text: "Prompt match has dropped; the LoRA is overriding the prompt." };
}

const PIPELINE = [
  {
    pkg: "darkroom",
    name: "argus-darkroom",
    text: "Rendered the subject with FLUX.1 schnell, checking every prompt and output against policy: SingGuard, a MiVOLO age gate of 20–32, then a look by eye.",
  },
  {
    pkg: "curator",
    name: "argus-curator",
    text: "Clustered faces by ArcFace identity, kept one consistent person, and split 31 training images from 8 held-out references.",
  },
  { pkg: "lens", name: "argus-lens", text: "Captioned the training set with the ohwx woman trigger first, keeping framing, clothing, setting and light." },
  { pkg: "forge", name: "argus-forge", text: "Wrote the kohya sd-scripts training config from the curated set." },
  { pkg: "proof", name: "argus-proof", text: "Generated a fixed grid from every checkpoint and the base model, scored each image and gave a verdict." },
];

export default function Home() {
  const rows: GridRow[] = tape.variants.map((v) => ({
    id: v.id,
    label: v.label,
    recommended: v.id === RECOMMENDED,
    cells: v.runs.map((r) =>
      r.images.map((im) => ({ tile: src(im.tile), full: src(im.full), metrics: im.metrics, passed: im.passed, reasons: im.reject_reasons })),
    ),
    runs: v.runs.map((r) => ({ reasons: r.verdict.reasons, passed: r.verdict.passed, pending: r.verdict.pending })),
  }));
  const refs = tape.refs.map((r) => ({ id: r.id, tile: src(r.tile), full: src(r.full!), w: r.w, h: r.h }));

  return (
    <div className="space-y-12">
      {/* 1. Hero: the verdict first. */}
      <section className="max-w-3xl" aria-labelledby="hero">
        <p className="mb-2 text-small font-semibold text-interactive">Train a LoRA, and let Proof find where it peaks</p>
        <h1 id="hero" className="mb-3 font-display text-heading font-semibold leading-tight text-text-primary sm:text-display">
          Ship step {RECOMMENDED}. After 640, it only gets worse.
        </h1>
        <p className="text-body text-text-secondary">
          argus-proof scores a trained LoRA&apos;s checkpoints so you don&apos;t have to pick one by eye. One synthetic character was trained for{" "}
          {tr.max_train_steps} steps, with a checkpoint every {tr.save_every_n_steps}. Proof generated the same grid from each checkpoint and from the base model
          with no LoRA ({n} images), then scored every image for identity, prompt match and quality. The likeness is learned by step 320–640; after that, prompt
          match and image quality fall. That is over-training, and the early checkpoint is the one to ship.
        </p>
        <p className="mt-2 text-small text-text-muted">This is a recorded replay: no backend and no live GPU. Nothing is generated while you browse.</p>
      </section>

      <section aria-label="Headline numbers" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label={`Identity at step ${RECOMMENDED}`} value={fmt(pick.summary.means.identity)} note={`vs ${fmt(base.summary.means.identity)} with no LoRA`} />
        <Stat label="Prompt match" value={fmt(pick.summary.means.clip_score)} note={`falls to ${fmt(tape.variants.at(-1)!.summary.means.clip_score)} by step 1600`} />
        <Stat label="Pass rate" value={fmt(pick.summary.pass_rate)} note={`best checkpoint; ${fmt(tape.variants.at(-1)!.summary.pass_rate)} at 1600`} />
        <Stat label="Images scored" value={String(n)} note={`${tape.prompts.length} prompts × ${rec.seeds.length} seeds × ${tape.variants.length} variants`} />
      </section>

      {/* 2. Scores against training step. */}
      <section aria-labelledby="scores" className="scroll-mt-4">
        <Lead id="scores" title="Where it peaks">
          Each panel is one score, averaged over all {tape.prompts.length * rec.seeds.length} images per checkpoint. The dashed line is base SDXL with no LoRA;
          the shaded column is step {RECOMMENDED}.
        </Lead>
        <ScoreChart />
        <ScoreTable />
      </section>

      {/* 3. The grid. */}
      <section aria-labelledby="grid" className="scroll-mt-4">
        <Lead id="grid" title="See it for yourself">
          The same eight prompts and seeds through every checkpoint. The held-out references on top are the real target: watch the no-LoRA row miss her, the
          early steps find her, and the late steps start ignoring the prompt.
        </Lead>
        <EvalGrid prompts={tape.prompts.map(shortPrompt)} rows={rows} refs={refs} seeds={rec.seeds} />
      </section>

      {/* 4. Per-checkpoint cards with proof's own verdict text. */}
      <section aria-labelledby="checkpoints" className="scroll-mt-4">
        <Lead id="checkpoints" title="Proof's verdict, checkpoint by checkpoint">
          Proof gives each prompt&apos;s run a verdict: it passes when at least {Math.round(rec.pass_threshold * 100)}% of its images pass. &ldquo;Needs
          review&rdquo; (?) means a score sits near a threshold, so a person makes the call; no image here was rejected outright.
        </Lead>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {tape.variants.map((v) => (
            <VariantCard key={v.id} v={v} />
          ))}
        </div>
        <Card className="mt-3">
          <SectionTitle>The verdict, carefully</SectionTitle>
          <ul className="list-disc space-y-1 pl-5 text-small text-text-secondary">
            <li>Steps 320 and 640 tie on identity (0.47) and prompt match (0.83).</li>
            <li>
              Step 320 leads on image quality ({fmt(pick.summary.means.aesthetic)} vs {fmt(tape.variants[2].summary.means.aesthetic)}) and pass rate (
              {fmt(pick.summary.pass_rate)} vs {fmt(tape.variants[2].summary.pass_rate)}).
            </li>
            <li>Ship step 320 or 640. From 960 on, prompt match drops to about 0.76 and identity drifts down: over-trained.</li>
          </ul>
        </Card>
      </section>

      {/* 5. The dataset and how it was made. */}
      <section aria-labelledby="dataset" className="scroll-mt-4">
        <Lead id="dataset" title="The data, and how it was made">
          The subject is a synthetic adult woman in her early twenties, made from text alone. A reverse image search found no real-person match: she is not a
          real person.
        </Lead>
        <ol className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {PIPELINE.map((s, i) => (
            <li key={s.pkg} className="min-w-0 rounded-lg border border-base-600/60 bg-base-800 p-3">
              <span className="text-caption font-semibold text-text-muted">{i + 1}</span>{" "}
              <Ext href={DOCS(s.pkg)} className="text-small font-semibold">
                {s.name}
              </Ext>
              <p className="mt-1 text-small text-text-secondary">{s.text}</p>
            </li>
          ))}
        </ol>

        <h3 className="mb-2 text-small font-semibold text-text-primary">Training set ({tape.train.length} images, with their captions)</h3>
        <div className="-mx-4 overflow-x-auto px-4 pb-2 sm:mx-0 sm:px-0" tabIndex={0} aria-label="Training images, scrolls sideways">
          <ul className="flex gap-2">
            {tape.train.map((t) => (
              <li key={t.id} className="w-28 shrink-0 sm:w-32">
                <img src={src(t.tile)} alt={t.caption} title={t.caption} width={t.w} height={t.h} loading="lazy" decoding="async" className="aspect-[4/5] h-auto w-full rounded-md object-cover" />
              </li>
            ))}
          </ul>
        </div>
        <p className="mb-5 mt-1 text-caption text-text-muted">
          Captions name what changes between images (framing, clothing, setting, light) and leave the face to the trigger, e.g.{" "}
          <Code>{tape.train[1].caption}</Code>
        </p>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <Card>
            <SectionTitle>Training</SectionTitle>
            <dl>
              <Kv label="Trainer">{String(tr.trainer)}</Kv>
              <Kv label="Base model">SDXL base 1.0</Kv>
              <Kv label="LoRA">
                dim {tr.network_dim}, alpha {tr.network_alpha}
              </Kv>
              <Kv label="Optimiser">
                {tr.optimizer_type}, lr {tr.learning_rate} ({tr.lr_scheduler}, {tr.lr_warmup_steps} warmup steps)
              </Kv>
              <Kv label="Steps">
                {tr.max_train_steps} at batch {tr.train_batch_size}, a checkpoint every {tr.save_every_n_steps}
              </Kv>
              <Kv label="Time">
                {tr.duration_minutes} minutes on {String(tr.gpu).replace("1x ", "one ")}
              </Kv>
            </dl>
          </Card>
          <Card>
            <SectionTitle>Evaluation</SectionTitle>
            <dl>
              <Kv label="Grid">
                {tape.prompts.length} prompts (the training captions) × {rec.seeds.length} seeds × {tape.variants.length} variants = {n} images
              </Kv>
              <Kv label="Sampling">
                {rec.base_checkpoint.name.replace(".safetensors", "")}, {rec.sampling.steps} steps, {rec.sampling.sampler} {rec.sampling.scheduler}, CFG{" "}
                {rec.sampling.cfg}, {rec.sampling.width}²
              </Kv>
              <Kv label="References">8 held-out images, sharing no prompt with training; closest train/ref pHash pair 16 of 64 bits</Kv>
              <Kv label="Scorers">
                {rec.scorers
                  .filter((s) => ["identity", "clip_score", "aesthetic", "safety"].includes(s.metric) && s.name !== "held-out-references")
                  .map((s) => s.model)
                  .join(", ")}
              </Kv>
              <Kv label="Recorded with">argus-proof {rec.proof_version}</Kv>
            </dl>
          </Card>
        </div>
      </section>

      {/* 6. Sidebar: the trigger token. */}
      <section aria-labelledby="probe" className="scroll-mt-4">
        <Lead id="probe" title="Is ohwx a clean trigger?">
          Base SDXL, no LoRA, same seed: <Code>ohwx woman, …</Code> on top, plain <Code>woman, …</Code> below. They come out near-identical, so the token
          brings no prior of its own. Whatever it means after training, the LoRA taught it.
        </Lead>
        <div className="-mx-4 overflow-x-auto px-4 pb-2 sm:mx-0 sm:px-0" tabIndex={0} aria-label="Trigger-token comparison, scrolls sideways">
          <div className="grid min-w-max grid-cols-[4.5rem_repeat(8,5.5rem)] gap-1.5 sm:min-w-0 sm:grid-cols-[4.5rem_repeat(8,minmax(0,1fr))]">
            <span className="self-center text-caption font-semibold text-text-secondary">ohwx woman</span>
            {tape.token_probe.map((p) => (
              <img key={p.prompt} src={src(p.ohwx)} alt={`ohwx woman, ${shortPrompt(tape.prompts[p.prompt])}`} width={384} height={384} loading="lazy" decoding="async" className="aspect-square h-auto w-full rounded-md" />
            ))}
            <span className="self-center text-caption font-semibold text-text-secondary">woman</span>
            {tape.token_probe.map((p) => (
              <img key={p.prompt} src={src(p.plain)} alt={p.plain_prompt} width={384} height={384} loading="lazy" decoding="async" className="aspect-square h-auto w-full rounded-md" />
            ))}
          </div>
        </div>
      </section>

      {/* Caveats. */}
      <Card as="aside" className="max-w-3xl">
        <SectionTitle>Honest caveats</SectionTitle>
        <ul className="list-disc space-y-1 pl-5 text-small text-text-secondary">
          <li>
            Absolute identity is modest (about 0.47). A face made from a text prompt is loose: most of the training images themselves sit 0.35–0.50 from their own
            identity anchor by ArcFace distance. The comparison between checkpoints is what counts.
          </li>
          <li>&ldquo;Needs review&rdquo; means near a threshold, not failed. Proof&apos;s quality bounds are not yet calibrated, so it asks rather than guesses.</li>
          <li>One subject, one training run, three seeds. It shows the method, not a benchmark.</li>
        </ul>
      </Card>
    </div>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <Card as="div" className="flex flex-col gap-1">
      <span className="text-caption font-semibold uppercase tracking-wide text-text-muted">{label}</span>
      <span className="font-display text-display font-semibold leading-none text-text-primary">{value}</span>
      <span className="text-caption text-text-secondary">{note}</span>
    </Card>
  );
}

/** Every number the chart draws, as a table (the chart's accessible equivalent). */
function ScoreTable() {
  const cols = ["Identity", "Prompt match", "Quality", "Safety", "Pass rate", "Diversity", "✓ / ? / ✕"];
  return (
    <div className="mt-3 overflow-x-auto rounded-lg border border-base-600/60" tabIndex={0} aria-label="Scores table, scrolls sideways">
      <table className="w-full min-w-[44rem] whitespace-nowrap text-small" aria-label="Mean scores per variant">
        <thead className="bg-base-800 text-left text-caption uppercase tracking-wide text-text-muted">
          <tr>
            <th scope="col" className="px-3 py-2 font-semibold">
              Variant
            </th>
            {cols.map((c) => (
              <th key={c} scope="col" className="px-3 py-2 text-right font-semibold">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-base-600/30">
          {tape.variants.map((v) => {
            const s = v.summary;
            return (
              <tr key={v.id} className={v.id === RECOMMENDED ? "bg-accent-primary/10" : ""}>
                <th scope="row" className="px-3 py-1.5 text-left font-semibold text-text-primary">
                  {v.label}
                  {v.id === RECOMMENDED ? <span className="ml-2 text-caption text-interactive">recommended</span> : null}
                </th>
                {[s.means.identity, s.means.clip_score, s.means.aesthetic, s.means.safety, s.pass_rate, s.diversity].map((x, i) => (
                  <td key={i} className="px-3 py-1.5 text-right font-mono tabular-nums">
                    {fmt(x)}
                  </td>
                ))}
                <td className="px-3 py-1.5 text-right font-mono tabular-nums">
                  {s.passed} / {s.review} / {s.failed}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function VariantCard({ v }: { v: Variant }) {
  const st = standing(v);
  const s = v.summary;
  return (
    <Card as="article" className={v.id === RECOMMENDED ? "border-accent-primary" : ""}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="font-display text-subheading font-semibold text-text-primary">{v.label}</h3>
        <span className={`rounded-sm border px-1.5 py-0.5 text-caption font-semibold ${st.tone}`}>{st.tag}</span>
      </div>
      <p className="mb-3 text-small text-text-secondary">{st.text}</p>
      <dl className="grid grid-cols-3 gap-2 text-center">
        <Mini label="Identity" value={fmt(s.means.identity)} />
        <Mini label="Prompt" value={fmt(s.means.clip_score)} />
        <Mini label="Quality" value={fmt(s.means.aesthetic)} />
        <Mini label="Pass rate" value={fmt(s.pass_rate)} />
        <Mini label="Runs passed" value={`${s.runs_passed}/${v.runs.length}`} />
        <Mini label="✓ / ? / ✕" value={`${s.passed}/${s.review}/${s.failed}`} />
      </dl>
      <details className="mt-3 text-small">
        <summary className="cursor-pointer text-interactive">Proof&apos;s reasons, per prompt</summary>
        <ol className="mt-2 space-y-1.5">
          {v.runs.map((r) => (
            <li key={r.run_id} className="text-caption text-text-secondary">
              <span className="font-semibold text-text-primary">P{r.prompt + 1}</span>{" "}
              <span className={r.verdict.passed ? "text-success" : "text-warning"}>{r.verdict.passed ? "✓ passed" : "? pending review"}</span>
              <span className="block [overflow-wrap:anywhere]">{r.verdict.reasons.join("; ")}</span>
            </li>
          ))}
        </ol>
        {v.lora ? (
          <p className="mt-2 font-mono text-caption text-text-muted [overflow-wrap:anywhere]">
            {v.lora.name} · sha256 {v.lora.sha256.slice(0, 16)}…
          </p>
        ) : null}
      </details>
    </Card>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-base-700 px-1 py-1.5">
      <dt className="text-caption text-text-muted">{label}</dt>
      <dd className="font-mono text-small font-semibold tabular-nums text-text-primary">{value}</dd>
    </div>
  );
}
