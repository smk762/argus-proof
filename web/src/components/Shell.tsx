import Link from "next/link";
import type { ReactNode } from "react";

import { tape } from "@/lib/tape";

import { ThemeToggle } from "./ThemeToggle";

export const REPO_URL = "https://github.com/smk762/argus-proof";
export const DOCS_URL = "https://docs.dragonhound.dev/packages/proof/";

const NAV = [
  { href: "#scores", label: "Scores" },
  { href: "#grid", label: "Grid" },
  { href: "#dataset", label: "Dataset" },
];

/** Header, main pane and footer. */
export function Shell({ children }: { children: ReactNode }) {
  const { recording } = tape;
  const title = `Recorded from argus-proof ${recording.proof_version}. Nothing runs while you browse.`;
  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-[1280px] flex-col">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-base-600/60 px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-display text-subheading font-semibold text-text-primary">
          {/* The Proof mark (argus-cortex brand/packages/proof/, unchanged); globals.css shows the one for the theme. */}
          <img src="/brand/argus-proof-light.svg" alt="" width={24} height={24} className="mark-light h-6 w-6" />
          <img src="/brand/argus-proof-dark.svg" alt="" width={24} height={24} className="mark-dark h-6 w-6" />
          Argus Proof
        </Link>
        <span className="dh-badge shrink-0" title={title}>
          Recorded
        </span>
        <div className="ml-auto flex items-center gap-1">
          <nav aria-label="Sections" className="hidden items-center gap-1 sm:flex">
            {NAV.map((item) => (
              <a key={item.href} href={item.href} className="rounded-md px-2.5 py-1.5 text-small text-text-secondary hover:bg-base-700 hover:text-text-primary">
                {item.label}
              </a>
            ))}
          </nav>
          <a href={DOCS_URL} className="rounded-md px-2.5 py-1.5 text-small text-text-secondary hover:bg-base-700 hover:text-text-primary">
            Docs
          </a>
          <a href={REPO_URL} className="rounded-md px-2.5 py-1.5 text-small text-text-secondary hover:bg-base-700 hover:text-text-primary">
            GitHub
          </a>
          <ThemeToggle />
        </div>
      </header>
      <main id="main" className="min-w-0 flex-1 px-4 py-5 sm:px-6">
        {children}
      </main>
      <footer className="space-y-1 border-t border-base-600/60 px-4 py-4 text-caption text-text-muted sm:px-6">
        <p>
          A replay of one recorded argus-proof evaluation. Every image and score is from that run, unedited; the images are WebP copies of the 1024 px
          originals. There is no backend: nothing is generated or scored while you browse. The subject is synthetic and is not a real person.
        </p>
        <p>
          Source:{" "}
          <a href={REPO_URL} className="underline decoration-base-600 underline-offset-2 hover:text-text-secondary">
            github.com/smk762/argus-proof
          </a>
          . Docs:{" "}
          <a href={DOCS_URL} className="underline decoration-base-600 underline-offset-2 hover:text-text-secondary">
            docs.dragonhound.dev/packages/proof
          </a>
          .
        </p>
      </footer>
    </div>
  );
}
