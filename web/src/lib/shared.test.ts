// The vendored Dragonhound shared layer (vendor/dh/). The files themselves are pinned by
// `node vendor/dh/sync-shared.mjs --check .` in CI; this pins what the app builds from them.
import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DemoStrip, STRIP_SITE } from "@/components/DemoStrip";

import { strip, stripHtml } from "../../vendor/dh/dh-strip.mjs";

const read = (path: string) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");
const squash = (html: string) =>
  html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/>\s+</g, "><")
    .replace(/<(path)([^>]*)\/>/g, "<$1$2></$1>")
    .trim();
// dh-shared 1.2.4+: the static strip carries every label suffix (data-dh-label picks one); the JSX renders the default only.
const defaultSuffixOnly = (html: string) =>
  html
    .replace(/<span class="dh-strip-more" data-dh-label="[a-z]+" hidden>[^<]*<\/span>/g, "")
    .replace(/<span class="dh-strip-more" data-dh-label="[a-z]+">/, '<span class="dh-strip-more">');
// The JSX strip also carries data-dh-site on the <nav> (dh-strip.css marks the current link from it).
const rendered = (current?: string) => renderToStaticMarkup(createElement(DemoStrip, current ? { current } : {})).replace(/<nav ([^>]*) data-dh-site="[a-z]+">/, "<nav $1>");

describe("dh-strip", () => {
  it("renders the shared source's markup", () => {
    expect(rendered()).toBe(squash(stripHtml({ current: STRIP_SITE, label: strip.defaultLabel })));
    expect(rendered("cull")).toBe(squash(stripHtml({ current: "cull", label: strip.defaultLabel })));
  });

  it("matches the vendored static strip", () => {
    // The pasted strip marks no link (dh-strip.css and dh-theme.js do it from data-dh-site); the JSX sets aria-current itself.
    expect(rendered().replace(' aria-current="page"', "")).toBe(defaultSuffixOnly(squash(read("vendor/dh/dh-strip.html"))));
  });

  it("says this is proof on the nav, with the default recorded-output label", () => {
    const html = renderToStaticMarkup(createElement(DemoStrip));
    expect(html).toContain('<nav class="dh-strip" aria-label="Dragonhound Labs demos" data-dh-site="proof">');
    expect(html).toContain(`<span class="dh-strip-more">${strip.labelMore}</span>`);
    expect(strip.defaultLabel).toBe("recorded");
  });

  it("lists Proof and marks its link current, and only it (dh-shared 1.2.10)", () => {
    const html = renderToStaticMarkup(createElement(DemoStrip));
    const current = html.match(/<a [^>]*aria-current="page"[^>]*>/g) ?? [];
    expect(current).toEqual([expect.stringContaining(`data-dh-site="${STRIP_SITE}"`)]);
  });

  it("links the family in the same tab", () => {
    for (const link of [strip.home, ...strip.links]) expect(new URL(link.href).hostname.endsWith("dragonhound.dev")).toBe(true);
    expect(renderToStaticMarkup(createElement(DemoStrip))).not.toContain("target=");
  });
});

describe("dh-theme", () => {
  it("serves the vendored script unchanged", () => {
    expect(read("public/dh-theme.js")).toBe(read("vendor/dh/dh-theme.js"));
  });
});
