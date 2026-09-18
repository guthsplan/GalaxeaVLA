# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Bridge the BEHAVIOR belief-graph pipeline (bgdata) outputs into the field format
consumed by the BeliefGraph* samples builders.

Input  (bgdata PoC, 1 Hz per frame):
    <bg_out>/cot_targets.parquet   columns: task, episode, frame, subtask, belief, delta, effect
                                   (belief/delta/effect carry their 'Belief:/Delta:/Effect:' prefixes)

Output (under --out):
    bg_fields.parquet        (episode, frame, bg_known, bg_belief, bg_delta, bg_effect, bg_observe) — plain strings
    bg_tasks.jsonl           deduplicated strings with task_index ids (tasks-table extension rows)
    bg_frame_indices.parquet (episode, frame, bg_known_index, bg_belief_index, bg_delta_index, bg_effect_index)

bg_known (the conditioning input) is the PREVIOUS snapshot's "Remaining: ... | Known: ..."
(delta+belief of the prior 1 Hz row within the episode), mirroring MemoryCoTBuilder's
prev_memory/memory split: input = state before this chunk, CoT target = state at this chunk.
The first frame of an episode falls back to its own snapshot (BDDL prior state).

Usage:
    python tools/build_bg_fields.py \
        --bg-out /Users/hoyong/local/workspace/out/task045 \
        --out    /Users/hoyong/local/workspace/out/task045/g05_bg_fields \
        --index-offset 100000     # first task_index for the new tasks-table rows
"""
import argparse
import json
import pathlib

import pandas as pd


def strip_prefix(text: str, prefix: str) -> str:
    text = (text or "").strip()
    if text.lower().startswith(prefix.lower() + ":"):
        text = text[len(prefix) + 1 :].strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-out", required=True, help="bgdata task output dir (has cot_targets.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-offset", type=int, default=100000,
                    help="task_index of the first deduplicated bg string (must not collide "
                         "with the existing tasks table)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ct = pd.read_parquet(pathlib.Path(args.bg_out) / "cot_targets.parquet")
    ct = ct.sort_values(["episode", "frame"]).reset_index(drop=True)

    rows = []
    for ep, g in ct.groupby("episode", sort=False):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            prev = g.iloc[i - 1] if i > 0 else r  # first frame: BDDL prior snapshot
            known = f"Remaining: {strip_prefix(prev['delta'], 'Delta')}"
            bel = strip_prefix(prev["belief"], "Belief")
            if bel and bel != "none":
                known += f" | Known: {bel}"
            rows.append(dict(
                episode=int(ep), frame=int(r["frame"]),
                bg_known=known,
                bg_belief=f"Belief: {strip_prefix(r['belief'], 'Belief')}",
                bg_delta=f"Delta: {strip_prefix(r['delta'], 'Delta')}",
                bg_effect=f"Effect: {strip_prefix(r['effect'], 'Effect')}",
                bg_observe=f"Observe: {strip_prefix(r.get('observe', 'none'), 'Observe')}",
                # plain skill text -> joined as atomic_task_index (SubtaskCoT builders reuse
                # the stock atomic_task decode path; the builder re-adds the 'Subtask: ' prefix)
                atomic_task=strip_prefix(r.get('subtask', ''), 'Subtask'),
            ))
    fields = pd.DataFrame(rows)
    fields.to_parquet(out / "bg_fields.parquet", index=False)

    # deduplicate strings -> tasks-table extension + per-frame index columns
    cols = ["bg_known", "bg_belief", "bg_delta", "bg_effect", "bg_observe", "atomic_task"]
    uniq = pd.unique(fields[cols].values.ravel())
    str2idx = {s: args.index_offset + i for i, s in enumerate(uniq)}
    with open(out / "bg_tasks.jsonl", "w") as f:
        for s, idx in str2idx.items():
            f.write(json.dumps({"task_index": idx, "task": s}) + "\n")
    idx_df = fields[["episode", "frame"]].copy()
    for c in cols:
        idx_df[f"{c}_index"] = fields[c].map(str2idx)
    idx_df.to_parquet(out / "bg_frame_indices.parquet", index=False)

    print(f"frames: {len(fields)} · episodes: {fields.episode.nunique()} · "
          f"unique strings: {len(str2idx)} (task_index {args.index_offset}..{args.index_offset + len(str2idx) - 1})")
    print(f"wrote: {out}/bg_fields.parquet, bg_tasks.jsonl, bg_frame_indices.parquet")
    print("\nexample row:")
    ex = fields.iloc[min(240, len(fields) - 1)]
    for c in ["episode", "frame"] + cols:
        print(f"  {c}: {str(ex[c])[:150]}")


if __name__ == "__main__":
    main()
