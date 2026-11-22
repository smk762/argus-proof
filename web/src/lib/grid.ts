// The evaluation grid's layout, kept pure so it can be tested: a label column, then one column per prompt.

/** The refs row(s): refs in lines of `cols` (a longer list continues on the next line), each line padded with empty
 * cells to `cols` so the variant rows below start in the label column, whatever the ref and prompt counts. */
export function refLines<T>(refs: T[], cols: number): { items: { ref: T; index: number }[]; pad: number }[] {
  const n = Math.max(1, cols);
  const lines: { items: { ref: T; index: number }[]; pad: number }[] = [];
  for (let i = 0; i < refs.length; i += n) {
    const items = refs.slice(i, i + n).map((ref, j) => ({ ref, index: i + j }));
    lines.push({ items, pad: n - items.length });
  }
  return lines;
}

/** The grid's column count after the label column: the tape's prompt count (at least one). */
export function gridCols(prompts: number): number {
  return Math.max(1, prompts);
}
