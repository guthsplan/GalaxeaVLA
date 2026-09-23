"""Cross-task validation report over bgdata outputs.

    python -m bgdata.validate --root <out_root> [--csv report.csv] [--md report.md]

One row per taskNNN directory with the quality signals worth eyeballing before training:
  n_keys                 predicate keys (too many -> the universe is too permissive)
  observe_none_frac      fraction of 1 Hz frames whose Observe: target is 'none'
  subtask_fallback_frac  fraction of frames whose Subtask: fell back to the bare skill text
  unresolved_frac        fraction of annotation segments with an unresolved object token
  accum_mismatch         operator pre/eff accumulation vs labels (per-frame)
  final_progress         mean goal progress at the last frame (should approach 1.0)
  inhand_source          assisted_grasp (tasks 50-99) | geometric (0-49)
"""
import argparse
import json
import pathlib

import pandas as pd


def task_row(d: pathlib.Path) -> dict | None:
    sp = d / "summary.json"
    if not sp.exists():
        return None
    s = json.load(open(sp))
    row = dict(task=int(d.name.replace("task", "")), episodes=s.get("episodes"),
               failed=len(s.get("episodes_failed", [])), n_keys=s.get("n_keys"),
               n_operators=s.get("n_operators"),
               accum_mismatch=s.get("accum_mismatch_overall"),
               boundary_mismatch=s.get("accum_mismatch_boundary"),
               memory_prefix_pass=s.get("memory_prefix_pass_rate"),
               final_progress=s.get("final_progress_mean"),
               frac_complete=s.get("final_progress_frac_complete"),
               unresolved_frac=s.get("segments_unresolved_frac"),
               seconds=(s.get("timing_s") or {}).get("total"))
    cf = d / "config_fitted.json"
    if cf.exists():
        f = json.load(open(cf))["fit_notes"].get("fitted", {})
        row.update(inhand_source=f.get("inhand_source"), reach_dist=f.get("reach_dist"),
                   grasp_dist=f.get("grasp_dist"))
    ts = d / "taskspec.json"
    if ts.exists():
        t = json.load(open(ts))
        row.update(task_name=t.get("task_name"), robot=t.get("robot"),
                   n_relevant=len(t.get("relevant", {})),
                   unresolved_tokens="|".join(t.get("unresolved_tokens", [])),
                   substances="|".join(t.get("substances", [])))
    ct = d / "cot_targets.parquet"
    if ct.exists():
        c = pd.read_parquet(ct)
        if len(c):
            row.update(cot_rows=len(c),
                       observe_none_frac=float((c.observe == "Observe: none").mean()),
                       delta_none_frac=float((c.delta == "Delta: none").mean()),
                       effect_none_frac=float((c.effect == "Effect: none").mean()),
                       subtask_done_frac=float((c.subtask == "Subtask: done").mean()),
                       subtask_fallback_frac=float(c.subtask.str.contains(r"\{\?", regex=True).mean()))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    rows = [r for r in (task_row(d) for d in sorted(pathlib.Path(a.root).glob("task*"))) if r]
    df = pd.DataFrame(rows).sort_values("task")
    if a.csv:
        df.to_csv(a.csv, index=False)
    flags = []
    for _, r in df.iterrows():
        f = []
        if (r.get("n_keys") or 0) > 500: f.append("n_keys>500")
        if (r.get("observe_none_frac") or 0) > 0.5: f.append("observe_none>0.5")
        if (r.get("subtask_fallback_frac") or 0) > 0.2: f.append("subtask_fallback>0.2")
        if (r.get("unresolved_frac") or 0) > 0.2: f.append("unresolved>0.2")
        if (r.get("accum_mismatch") or 0) > 0.15: f.append("accum_mismatch>0.15")
        if (r.get("final_progress") or 0) < 0.5: f.append("final_progress<0.5")
        if r.get("failed"): f.append(f"failed_eps={r['failed']}")
        flags.append(",".join(f))
    df["flags"] = flags
    cols = ["task", "task_name", "episodes", "failed", "n_keys", "n_operators", "accum_mismatch",
            "final_progress", "observe_none_frac", "subtask_fallback_frac", "unresolved_frac",
            "inhand_source", "seconds", "flags"]
    cols = [c for c in cols if c in df.columns]
    text = df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    print(text)
    print(f"\n{len(df)} tasks, {int((df['flags'] == '').sum())} without flags")
    if a.md:
        pathlib.Path(a.md).write_text("```\n" + text + "\n```\n")
    return df


if __name__ == "__main__":
    main()
