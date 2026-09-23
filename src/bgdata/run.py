"""End-to-end belief-graph pipeline for one task (any of the 100).

    python -m bgdata.run --task 45 --out out/task045 [--episodes 5|all] [--fit-episodes 20]

Streaming layout: the task spec is read from the first episode's HDF5 (scene_file), thresholds
are fitted on the first --fit-episodes episodes, then each episode is decoded, labeled, exported
(truth json) and discarded; the belief trace + sidecar pass reruns over the kept label tables
after operator extraction.
"""
import argparse
import gzip
import json
import pathlib
import time
import traceback

import numpy as np
import pandas as pd

from . import belief_trace, goal, inventory, labels, operators, sidecar, taskspec
from .config import Thresholds


def truth_from_labels(df_ep: pd.DataFrame) -> dict[int, dict]:
    out = {}
    for fr, sub in df_ep.groupby("frame", observed=True):
        out[int(fr)] = {k: (bool(v), bool(vis))
                        for k, v, vis in zip(sub.key, sub.value, sub.visible)}
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=45)
    ap.add_argument("--out", default="out/task045")
    ap.add_argument("--episodes", default="5")
    ap.add_argument("--fit-episodes", type=int, default=20)
    ap.add_argument("--no-sidecar", action="store_true")
    ap.add_argument("--truth-json", action="store_true", help="also write truth_<ep>.json.gz")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = Thresholds()
    t0 = time.time()
    timing, summary = {}, {}

    # 1. inventory + task spec ----------------------------------------------------------
    inv = inventory.resolve_task(args.task)
    eps = [e for e in inv["episodes"] if e["raw_exists"] and e["annotation_exists"]]
    if args.episodes != "all":
        eps = eps[: int(args.episodes)]
    if not eps:
        raise SystemExit(f"task {args.task}: no episode with raw hdf5 + annotation")
    ann_tokens = set()
    raw_segs = {}
    for e in eps:
        raw_segs[e["raw_episode_id"]] = inventory.segments(inventory.load_annotation(e["annotation_path"]))
        for s in raw_segs[e["raw_episode_id"]]:
            ann_tokens.update(taskspec.flatten_objects(s["objects"]))
    spec = taskspec.load_taskspec(args.task, eps[0]["raw_path"], ann_tokens)
    tg = goal.TaskGoal(spec.bddl_objects, spec.bddl_init, spec.bddl_goal, spec.inst_to_name)
    # one key universe for the whole task, from BDDL + every annotation token's candidates
    universe = labels.key_universe(spec, tg, raw_segs)
    (out / "inventory.json").write_text(json.dumps({**inv, "episodes": eps}, indent=2))
    (out / "taskspec.json").write_text(json.dumps(spec.summary(), indent=2, default=_json_default))
    print(f"[inventory] task {args.task} ({inv['task_name']}): {len(eps)} episodes; "
          f"relevant objects {len(spec.relevant)}; unresolved tokens {spec.unresolved_tokens}")
    timing["inventory"] = time.time() - t0

    # 2. fit on the first K episodes -----------------------------------------------------
    t = time.time()
    fit_eds, failed = [], []
    for e in eps[: args.fit_episodes]:
        try:
            fit_eds.append(labels.EpisodeData(e, spec, thr))
        except Exception as ex:  # a broken episode must not kill the task
            failed.append(dict(episode=e["raw_episode_id"], stage="decode", error=repr(ex)[:300]))
    if not fit_eds:
        raise SystemExit(f"task {args.task}: no episode decoded: {failed}")
    fitted, fit_notes = labels.fit_thresholds(fit_eds, thr)
    thr.save(out / "config_fitted.json", {**fit_notes, "fitted": fitted})
    print(f"[fit] on {len(fit_eds)} episodes: reach_dist={fitted['reach_dist']:.3f} "
          f"grasp_dist={fitted['grasp_dist']:.3f} inhand={fitted['inhand_source']} "
          f"joint_ranges={len(fitted['joint_ranges'])} objs")
    timing["fit"] = time.time() - t

    # 3. labels + truth export (streaming) -------------------------------------------------
    t = time.time()
    # the key universe needs every episode's resolved segments -> resolve first (cheap)
    label_dfs, segs_by_ep, n_frames, per_ep_s = {}, {}, {}, []
    eds_iter = {e["raw_episode_id"]: ed for e, ed in zip(eps[: args.fit_episodes], fit_eds + [None] * 99)}
    for i, e in enumerate(eps):
        te = time.time()
        ep_id = e["raw_episode_id"]
        try:
            ed = eds_iter.get(ep_id) or labels.EpisodeData(e, spec, thr)
        except Exception as ex:
            failed.append(dict(episode=ep_id, stage="decode", error=repr(ex)[:300]))
            continue
        try:
            df = labels.compute_labels(ed, thr, fitted, universe, tg)
        except Exception as ex:
            failed.append(dict(episode=ep_id, stage="labels", error=repr(ex)[:300],
                               tb=traceback.format_exc()[-800:]))
            continue
        for c in ("key", "tag", "source"):
            df[c] = df[c].astype("category")
        label_dfs[ep_id] = df
        segs_by_ep[ep_id] = ed.segments
        n_frames[ep_id] = ed.N
        if args.truth_json:  # 10 Hz truth dump is ~100 MB/episode: opt-in (predicates.parquet has it)
            truth = truth_from_labels(df)
            tj = {str(fr): {k: [v, vis] for k, (v, vis) in d.items()} for fr, d in truth.items()}
            with gzip.open(out / f"truth_{ep_id}.json.gz", "wt") as f:
                json.dump(tj, f)
        summary.setdefault("decode_stats", {})[str(ep_id)] = ed.decode_stats
        eds_iter[ep_id] = None
        del ed
        per_ep_s.append(time.time() - te)
        if (i + 1) % 25 == 0:
            print(f"  labels {i+1}/{len(eps)} ({np.mean(per_ep_s):.2f}s/ep)")
    if not label_dfs:
        raise SystemExit(f"task {args.task}: every episode failed: {failed[:3]}")
    all_labels = pd.concat(label_dfs.values(), ignore_index=True)
    all_labels.to_parquet(out / "predicates.parquet", index=False)
    (out / "segments_resolved.json").write_text(json.dumps(
        {str(k): v for k, v in segs_by_ep.items()}, default=_json_default))
    timing["labels_truth"] = time.time() - t
    summary["episodes"] = len(label_dfs)
    summary["episodes_failed"] = failed
    summary["label_rows"] = int(len(all_labels))
    summary["n_keys"] = int(all_labels.key.nunique())
    summary["keys"] = sorted(all_labels.key.unique().tolist())
    summary["labels_s_per_episode"] = round(float(np.mean(per_ep_s)), 2)
    summary["segments_unresolved_frac"] = float(np.mean(
        [not s["resolved_all"] for segs in segs_by_ep.values() for s in segs] or [0.0]))
    del all_labels

    # 4. operators --------------------------------------------------------------------------
    t = time.time()
    ops_json, stats = operators.extract_operators(label_dfs, segs_by_ep, n_frames)
    operators.save_operators(ops_json, out / f"operators_task{args.task:03d}.json")
    stats.to_csv(out / "operator_stats.csv", index=False)
    acc = operators.validate_accumulation(ops_json, label_dfs, segs_by_ep)
    acc.to_csv(out / "validation_accumulation.csv", index=False)
    mp = operators.validate_memory_prefix(label_dfs, segs_by_ep)
    mp.to_csv(out / "validation_memory_prefix.csv", index=False)
    summary["n_operators"] = len(ops_json)
    summary["accum_mismatch_overall"] = float(acc.mismatch_rate.mean()) if len(acc) else None
    summary["accum_mismatch_boundary"] = float(acc.boundary_mismatch_rate.mean()) if len(acc) else None
    summary["accum_mismatch_worst_keys"] = (
        acc.groupby("key").mismatch_rate.mean().sort_values(ascending=False).head(8).to_dict()
        if len(acc) else {})
    summary["memory_prefix_pass_rate"] = float(mp.passed.dropna().mean()) if len(mp) and mp.passed.notna().any() else None
    summary["memory_prefix_n"] = int(len(mp))
    timing["operators"] = time.time() - t
    print(f"[operators] {len(ops_json)} ops; accum mismatch "
          f"{summary['accum_mismatch_overall']:.3%} (boundary "
          f"{summary['accum_mismatch_boundary']:.3%}); memory_prefix pass "
          f"{summary['memory_prefix_pass_rate']} (n={summary['memory_prefix_n']})")

    # 5. goal ---------------------------------------------------------------------------------
    gout = goal.save_goal(args.task, inv["bddl_path"], tg, [e["raw_episode_id"] for e in eps],
                          out / f"goal_task{args.task:03d}.json")
    print(f"[goal] {gout['goal_lines']}")

    # 6. belief trace + sidecar (streaming) ----------------------------------------------------
    t = time.time()
    vocab = None
    sidecar_parts = []
    final_progress = []
    for i, (ep_id, df) in enumerate(label_dfs.items()):
        truth = truth_from_labels(df)
        recs, chlog = belief_trace.run_trace(
            truth, segs_by_ep[ep_id], ops_json, tg.init_lines, tg,
            out / f"belief_trace_{ep_id}.jsonl.gz")
        final_progress.append(recs[-1]["progress"] if recs else 0.0)
        if vocab is None:
            vocab = sidecar.build_vocab({ep_id: recs}, gout["goal_lines"], extra_keys=summary["keys"])
        if i == 0:
            (out / f"confidence_changes_{ep_id}.json").write_text(json.dumps(chlog, indent=1))
        if not args.no_sidecar:
            sidecar_parts.append(sidecar.build_sidecar({ep_id: recs}, vocab, thr.sidecar_K,
                                                       thr.sidecar_R, args.task))
        if (i + 1) % 25 == 0:
            print(f"  trace {i+1}/{len(label_dfs)}")
    if sidecar_parts:
        sc = pd.concat(sidecar_parts, ignore_index=True)
        sidecar.save(sc, vocab, out / "sidecar.parquet", out / "predicate_vocab.json")
    summary["final_progress_mean"] = float(np.mean(final_progress)) if final_progress else None
    summary["final_progress_frac_complete"] = float(np.mean([p >= 0.999 for p in final_progress])) if final_progress else None
    timing["trace_sidecar"] = time.time() - t

    timing["total"] = time.time() - t0
    summary["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    summary["truth_files"] = sum(1 for p in out.iterdir() if p.name.startswith("truth_"))
    summary["trace_files"] = sum(1 for p in out.iterdir() if p.name.startswith("belief_trace_"))
    summary["out_dir_total_bytes"] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("keys", "decode_stats", "accum_mismatch_worst_keys")},
                     indent=2, default=_json_default))


if __name__ == "__main__":
    main()
