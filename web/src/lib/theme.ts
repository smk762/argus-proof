/**
 * Light/dark is the Dragonhound shared script, public/dh-theme.js (a copy of
 * vendor/dh/dh-theme.js; src/lib/shared.test.ts holds them equal). It runs
 * blocking in <head>, sets `data-theme` on <html> before the first paint from a
 * `dh-theme` cookie (Domain .dragonhound.dev on the live hosts, so the choice
 * follows you between the family's sites), handles clicks on any
 * `[data-dh-theme-toggle]` button and keeps their `aria-pressed` in step.
 *
 * React only reads the effective scheme through the script's API (`dhTheme`),
 * for labels; it never writes the theme or `aria-pressed` itself.
 */

type Scheme = "light" | "dark";

interface DhTheme {
  get(): Scheme | null;
  effective(): Scheme;
  set(theme: Scheme | null): void;
  subscribe(listener: (detail: { theme: Scheme; stored: Scheme | null; source: string }) => void): () => void;
}

declare global {
  interface Window {
    dhTheme?: DhTheme;
  }
}

/** The scheme on screen now. Falls back to the system's if dh-theme.js did not run. */
export function effectiveScheme(): Scheme {
  if (window.dhTheme) return window.dhTheme.effective();
  return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Server snapshot: unknown, rendered as light until hydration reads the real one. */
export const serverScheme = (): Scheme => "light";

/** Calls `listener` when the effective scheme changes (toggle, system, another subdomain). */
export function subscribeScheme(listener: () => void): () => void {
  return window.dhTheme?.subscribe(listener) ?? (() => {});
}
