// The opsec gate for imported docs. It fails closed: a hit refuses the import,
// and nothing is redacted. Loopback (127.0.0.1, localhost) and the `.env`
// placeholders (<GPU_HOST>, <LAN_HOST>, ...) are fine; anything that names a
// real machine, a home directory or a person is not.

const PATTERNS = [
  ["private IPv4 address", /\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b/g],
  ["link-local IPv4 address", /\b169\.254\.\d{1,3}\.\d{1,3}\b/g],
  ["link-local or unique-local IPv6 address", /\b(?:fe80|f[cd][0-9a-f]{2}):[0-9a-f:]*:[0-9a-f:]+\b/gi],
  ["home directory path", /(?:\/home\/|\/Users\/|\/root\/|~\/)[\w.-]+/g],
  ["server path", /\/(?:srv|mnt|media|var\/www)\/[\w.-]+/g],
  ["email address", /\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b/gi],
];

/** The private denylist, from the environment (a CI secret, never committed): one term per line or comma. */
export function denylistFromEnv(env = process.env) {
  return (env.OPSEC_DENYLIST ?? "")
    .split(/[\n,]/)
    .map((term) => term.trim())
    .filter(Boolean);
}

/**
 * Every hit in `text`, as `{ kind, match, line }`. The denylist is matched
 * case-insensitively and its hits never echo the term, so a CI log cannot
 * leak what the secret holds.
 */
export function opsecHits(text, denylist = []) {
  const hits = [];
  const lineOf = (index) => text.slice(0, index).split("\n").length;
  for (const [kind, pattern] of PATTERNS) {
    for (const found of text.matchAll(pattern)) hits.push({ kind, match: found[0], line: lineOf(found.index) });
  }
  const lower = text.toLowerCase();
  for (const term of denylist) {
    let from = 0;
    for (;;) {
      const index = lower.indexOf(term.toLowerCase(), from);
      if (index < 0) break;
      hits.push({ kind: "denylisted term", match: "(withheld)", line: lineOf(index) });
      from = index + term.length;
    }
  }
  return hits.sort((a, b) => a.line - b.line);
}
