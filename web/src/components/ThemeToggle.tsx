"use client";

import { useSyncExternalStore } from "react";

import { effectiveScheme, serverScheme, subscribeScheme } from "@/lib/theme";

import { MoonIcon, SunIcon } from "./icons";

/**
 * The family's two-state theme toggle (as in argus-safelight). public/dh-theme.js handles clicks on any
 * `[data-dh-theme-toggle]`, stores the choice in the shared cookie and keeps `aria-pressed` (pressed = dark) in step,
 * so React renders it once and leaves it alone. Both icons are in the markup and globals.css shows the right one.
 */
export function ThemeToggle() {
  const dark = useSyncExternalStore(subscribeScheme, effectiveScheme, serverScheme) === "dark";
  return (
    <button
      type="button"
      data-dh-theme-toggle=""
      aria-label="Dark theme"
      title={dark ? "Dark theme on" : "Dark theme off"}
      // Owned by dh-theme.js, which corrects it before hydration; React never updates it.
      aria-pressed="false"
      suppressHydrationWarning
      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-text-secondary transition-colors hover:bg-base-700 hover:text-text-primary"
    >
      <span className="theme-icon-moon">
        <MoonIcon />
      </span>
      <span className="theme-icon-sun">
        <SunIcon />
      </span>
    </button>
  );
}
