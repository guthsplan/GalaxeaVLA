"""End-to-end PoC pipeline for one task.

    python -m bgdata.run --task 45 --out out/task045 [--episodes 5|all] [--fit-episodes 20]

Streaming layout: thresholds are fitted on the first --fit-episodes episodes, then each
episode is decoded, labeled, exported (truth json) and discarded; the belief trace +
sidecar pass reruns over the kept label tables after operator extraction.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from . import belief_trace, figures, goal, inventory, labels, operators, sidecar
from .config import Thresholds


def truth_from_labels(df_ep: pd.DataFrame) -> dict[int, dict]:
    out = {}
    for fr, sub in df_ep.groupby("frame", observed=True):
        out[int(fr)] = {k: (bool(v), bool(vis))
                        for k, v, vis in zip(sub.key, sub.value, sub.visible)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=45)
    ap.add_argument("--out", default="out/task045")
    ap.add_argument("--episodes", default="5")
    ap.add_argument("--fit-episodes", type=int, default=20)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = Thresholds()
    t0 = time.time()
    timing, summary = {}, {}

    # 1. inventory ----------------------------------------------------------
    inv = inventory.resolve_task(args.task)
    eps = [e for e in inv["episodes"] if e["raw_exists"] and e["annotation_exists"]]
    if args.episodes != "all":
        eps = eps[: int(args.episodes)]
    print(f"[inventory] task {args.task} ({inv['task_name']}): {len(eps)} episodes selected")
    (out / "inventory.json").write_text(json.dumps({**inv, "episodes": eps}, indent=2))
    timing["inventory"] = time.time() - t0

    # 2. fit on the first K episodes ---------------------------------------
    t = time.time()
    fit_eds = [labels.EpisodeData(e, thr) for e in eps[: args.fit_episodes]]
    fitted, fit_notes = labels.fit_thresholds(fit_eds, thr)
    thr.save(out / "config_fitted.json", {**fit_notes, "fitted": fitted})
    print(f"[fit] on {len(fit_eds)} episodes: {json.dumps({k: round(v, 4) for k, v in fitted.items() if isinstance(v, float)})}")
    timing["fit"] = time.time() - t

    # 3. labels + truth export (streaming) ----------------------------------
    t = time.time()
    label_dfs, segs_by_ep, n_frames, per_ep_s = {}, {}, {}, []
    for i, e in enumerate(eps):
        te = time.time()
        ed = fit_eds[i] if i < len(fit_eds) else labels.EpisodeData(e, thr)
        ep_id = e["raw_episode_id"]
        df = labels.compute_labels(ed, thr, fitted)
        for c in ("key", "tag", "source"):
            df[c] = df[c].astype("category")
        label_dfs[ep_id] = df
        segs_by_ep[ep_id] = ed.segments
        n_frames[ep_id] = ed.N
        truth = truth_from_labels(df)
        tj = {str(fr): {k: [v, vis] for k, (v, vis) in d.items()} for fr, d in truth.items()}
        (out / f"truth_{ep_id}.json").write_text(json.dumps(tj))
        if i < len(fit_eds):
            fit_eds[i] = None  # free
        del ed
        per_ep_s.append(time.time() - te)
        if (i + 1) % 25 == 0:
            print(f"  labels {i+1}/{len(eps)} ({np.mean(per_ep_s):.2f}s/ep)")
    all_labels = pd.concat(label_dfs.values(), ignore_index=True)
    all_labels.to_parquet(out / "predicates.parquet", index=False)
    timing["labels_truth"] = time.time() - t
    summary["episodes"] = len(eps)
    summary["label_rows"] = len(all_labels)
    summary["labels_s_per_episode"] = round(float(np.mean(per_ep_s)), 2)
    del all_labels

    # 4. operators -----------------------------------------------------------
    t = time.time()
    ops_json, stats = operators.extract_operators(label_dfs, segs_by_ep, n_frames)
    operators.save_operators(ops_json, out / f"operators_task{args.task:03d}.json")
    stats.to_csv(out / "operator_stats.csv", index=False)
    acc = operators.validate_accumulation(ops_json, label_dfs, segs_by_ep)
    acc.to_csv(out / "validation_accumulation.csv", index=False)
    mp = operators.validate_memory_prefix(label_dfs, segs_by_ep)
    mp.to_csv(out / "validation_memory_prefix.csv", index=False)
    summary["accum_mismatch_overall"] = float(acc.mismatch_rate.mean())
    summary["accum_mismatch_boundary"] = float(acc.boundary_mismatch_rate.mean())
    summary["accum_mismatch_worst_keys"] = (
        acc.groupby("key").mismatch_rate.mean().sort_values(ascending=False).head(8).to_dict())
    summary["memory_prefix_pass_rate"] = float(mp.passed.mean()) if len(mp) else None
    summary["memory_prefix_n"] = int(len(mp))
    timing["operators"] = time.time() - t
    print(f"[operators] {len(ops_json)} ops; accum mismatch "
          f"{summary['accum_mismatch_overall']:.3%} (boundary "
          f"{summary['accum_mismatch_boundary']:.3%}); memory_prefix pass "
          f"{summary['memory_prefix_pass_rate']} (n={summary['memory_prefix_n']})")

    # 5. goal -----------------------------------------------------------------
    scene_objs = sorted({o for segs in segs_by_ep.values() for s in segs for o in s["objects"]})
    gout = goal.save_goal(args.task, inv["bddl_path"], scene_objs,
                          [e["raw_episode_id"] for e in eps],
                          out / f"goal_task{args.task:03d}.json")
    print(f"[goal] {gout['goal_lines']}")

    # 6. belief trace + sidecar (streaming) -----------------------------------
    t = time.time()
    fridge = gout["synset_to_scene"].get("electric_refrigerator.n.01_1")
    hotdogs = sorted(v for k, v in gout["synset_to_scene"].items()
                     if k.startswith("hotdog") and v)
    init_lines = []
    for h in hotdogs:
        init_lines.append((f"(inside {h} {fridge})", True))
        init_lines.append((f"(cooked {h})", False))
    goal_specs = [dict(line=l, pred=l.split()[0][1:], instances=hotdogs)
                  for l in gout["goal_lines"]]
    vocab = None
    sidecar_parts = []
    for i, (ep_id, df) in enumerate(label_dfs.items()):
        truth = truth_from_labels(df)
        recs, chlog = belief_trace.run_trace(
            truth, segs_by_ep[ep_id], ops_json, init_lines, goal_specs,
            out / f"belief_trace_{ep_id}.jsonl")
        if vocab is None:
            vocab = sidecar.build_vocab({ep_id: recs}, gout["goal_lines"])
        if i == 0:
            (out / f"confidence_changes_{ep_id}.json").write_text(json.dumps(chlog, indent=1))
            sel = select_snapshot_frames(segs_by_ep[ep_id])
            for tag, fr in sel.items():
                rec = next((r for r in recs if r["step"] >= fr), recs[-1])
                figures.draw_graph(rec, out / f"graph_{tag}_{ep_id}.png", f"[{tag}]")
            h1 = hotdogs[0]
            h2 = hotdogs[1] if len(hotdogs) > 1 else hotdogs[0]
            keys = [f"(inside {h2} {fridge})", f"(open {fridge})", f"(inhand {h1})",
                    "(toggled_on microwave_abzvij_0)", f"(cooked {h1})"]
            figures.draw_timeline(recs, segs_by_ep[ep_id], keys, out / f"timeline_{ep_id}.png")
        sidecar_parts.append(sidecar.build_sidecar({ep_id: recs}, vocab, thr.sidecar_K,
                                                   thr.sidecar_R, args.task))
        if (i + 1) % 25 == 0:
            print(f"  trace {i+1}/{len(label_dfs)}")
    sc = pd.concat(sidecar_parts, ignore_index=True)
    sidecar.save(sc, vocab, out / "sidecar.parquet", out / "predicate_vocab.json")
    timing["trace_sidecar_figures"] = time.time() - t

    timing["total"] = time.time() - t0
    summary["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    summary["files"] = {p.name: p.stat().st_size for p in sorted(out.iterdir())
                        if p.is_file() and not p.name.startswith(("truth_", "belief_trace_"))}
    summary["truth_files"] = sum(1 for p in out.iterdir() if p.name.startswith("truth_"))
    summary["trace_files"] = sum(1 for p in out.iterdir() if p.name.startswith("belief_trace_"))
    summary["out_dir_total_bytes"] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def select_snapshot_frames(segments) -> dict[str, int]:
    """Frames for the 5 story figures."""
    def seg(skill, nth=0):
        hits = [s for s in segments if s["skill"] == skill]
        return hits[nth] if len(hits) > nth else None
    out = {}
    s = seg("open door", 0)
    if s: out["1_before_fridge_open"] = max(0, s["start"] - 30)
    if s: out["2_fridge_open"] = s["end"] + 30
    s = seg("close door", 0)
    if s: out["3_after_leaving_fridge"] = s["end"] + 60
    s = seg("place in", 1) or seg("place in", 0)
    if s: out["4_after_place_in_microwave"] = s["end"] + 30
    s = seg("turn on switch", 0)
    if s: out["5_after_turn_on"] = s["end"] + 30
    return out


if __name__ == "__main__":
    main()
