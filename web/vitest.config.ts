import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    // Pure logic runs in node; a component test opts into jsdom with
    // `// @vitest-environment jsdom` at the top of the file.
    environment: "node",
  },
});
