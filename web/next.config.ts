import type { NextConfig } from "next";

// The hosted demo (proof.dragonhound.dev) is a pure static export: `make pages` builds out/ from the recorded tape
// in public/tape/, then scripts/csp.mjs gives every page a hash-based CSP and scripts/check-static.mjs checks the
// result. There is no server and nothing calls a model or GPU from the browser.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
