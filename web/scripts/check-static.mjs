// Post-build check for the hosted demo (`make pages`): the export is pure static files, and it publishes nothing it
// must not.
//
//   - every page carries the hash-based CSP <meta>, and _headers is there for the host
//   - no page loads anything from another origin (no <img>, <script> or <link> to anywhere else)
//   - the tape is in the export, every image it lists is there, and there is no image under tape/ it doesn't list
//   - the tape carries no never-run metric (preference) and no path from the recording host or container
//   - no home or server path, private address or email address in the tape or any page
//   - the whole export stays small (Cloudflare Pages caps a file at 25 MiB; this site aims under 40 MB in all)
//
//   node scripts/check-static.mjs [outDir]   (run by `npm run build`)
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

import { denylistFromEnv, opsecHits } from "./opsec.mjs";

const out = process.argv[2] ?? "out";
const problems = [];
const list = (dir) =>
  existsSync(dir)
    ? readdirSync(dir, { withFileTypes: true, recursive: true })
        .filter((e) => e.isFile())
        .map((e) => join(e.parentPath, e.name))
    : [];
const files = list(out);
const denylist = denylistFromEnv();
const FOREIGN = /<(?:img|script|link|source|iframe)\b[^>]*\b(?:src|href|srcset)="(?:https?:)?\/\/[^"]*"/gi;
const CONTAINER_PATH = /"\/(?:data|models|srv|mnt)\//;

for (const file of files.filter((f) => f.endsWith(".html"))) {
  const html = readFileSync(file, "utf8");
  if (!html.includes('http-equiv="Content-Security-Policy"')) problems.push(`${file} has no CSP <meta>`);
  for (const [tag] of html.matchAll(FOREIGN)) problems.push(`${file} loads from another origin: ${tag.slice(0, 120)}`);
  for (const hit of opsecHits(html, denylist)) problems.push(`${file}:${hit.line}: ${hit.kind} (${hit.match})`);
}
if (!existsSync(join(out, "_headers"))) problems.push("no _headers file");

const root = join(out, "tape");
const tapeJson = join(root, "tape.json");
if (!existsSync(tapeJson)) problems.push(`no tape at ${tapeJson}`);
else {
  const text = readFileSync(tapeJson, "utf8");
  const tape = JSON.parse(text);
  const listed = new Set(["tape.json"]);
  const want = (path) => {
    listed.add(path);
    if (!existsSync(join(root, path))) problems.push(`tape file missing: ${path}`);
  };
  for (const v of tape.variants)
    for (const run of v.runs)
      for (const im of run.images) {
        want(im.tile);
        want(im.full);
        if ("preference" in im.metrics) problems.push(`${im.image_id} carries the never-run preference metric`);
      }
  for (const r of tape.refs) {
    want(r.tile);
    want(r.full);
  }
  for (const t of tape.train) want(t.tile);
  for (const p of tape.token_probe) {
    want(p.ohwx);
    want(p.plain);
  }
  for (const path of list(root)) {
    const rel = relative(root, path);
    if (!listed.has(rel)) problems.push(`tape/${rel} is on disk but the tape doesn't list it`);
    else if (rel !== "tape.json" && !rel.endsWith(".webp")) problems.push(`tape/${rel} is not WebP`);
  }
  if (CONTAINER_PATH.test(text)) problems.push(`${tapeJson} carries a container path`);
  for (const hit of opsecHits(text, denylist)) problems.push(`${tapeJson}:${hit.line}: ${hit.kind} (${hit.match})`);
}

const MiB = 1024 * 1024;
let total = 0;
for (const file of files) {
  const size = statSync(file).size;
  total += size;
  if (size > 25 * MiB) problems.push(`${file} is ${(size / MiB).toFixed(1)} MiB, over Cloudflare Pages' 25 MiB file cap`);
}
if (total > 40 * 1000 * 1000) problems.push(`the export is ${(total / MiB).toFixed(1)} MiB, over the 40 MB budget`);

if (problems.length) {
  console.error(`check-static: ${problems.length} problem(s)\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`check-static: ${files.length} files, ${(total / MiB).toFixed(2)} MiB, pure static, CSP on every page, every tape image listed and present`);
