# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Offline replay of the FULL glue path on demo episodes (glue P1-P2 validation, CPU-only).

Per episode it drives BeliefGraphMiddleware exactly as a server would — reset ->
before_infer (OracleEstimator observations) -> stub policy CoT -> after_infer — where
the stub "model" returns the GROUND-TRUTH CoT of that frame (bgdata cot_targets:
Subtask + Delta), i.e. it simulates a perfectly trained model. The resulting belief is
compared frame-by-frame against the recorded reference (belief_trace_*.jsonl).

Expected: high value-agreement; residual disagreement comes only from effect timing
(controller completion vs annotation-boundary application) and is reported.

Usage:
    PYTHONPATH=src python -m g05.belief_graph.serving.replay_harness \
        --bg-out /path/to/bgdata/out/task045 [--task 45] [--episodes 5] \
        [--min-agreement 0.95]
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

from ..estimator import OracleEstimator
from .policy_middleware import BeliefGraphMiddleware
from .task_registry import TaskRegistry


def replay_episode(bg_out: pathlib.Path, task: int, episode: int) -> dict:
    import pandas as pd

    registry = TaskRegistry(bg_out)
    mw = BeliefGraphMiddleware(
        registry, estimator=OracleEstimator(bg_out / f"truth_{episode}.json"),
        belief_every=1)
    mw.reset(task, registry.get(task).goal.task)

    ct = pd.read_parquet(bg_out / "cot_targets.parquet")
    ct = ct[ct.episode == episode].sort_values("frame")
    trace = {}
    with open(bg_out / f"belief_trace_{episode}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            trace[r["step"]] = r["predicates"]

    agree = defaultdict(lambda: [0, 0])   # key-name -> [match, total]
    completes = mism = 0
    for _, row in ct.iterrows():
        fr = int(row["frame"])
        mw.before_infer({}, step=fr)
        res = mw.after_infer(f"{row['subtask']} | {row['delta']}")
        completes += int(res["completed"])
        mism += len(res["hooks"]["delta_mismatch"])
        ref = trace.get(fr, {})
        for key, (val, p, obs, src) in ref.items():
            e = mw.runtime.belief.table.get(key)
            name = key[1:].split()[0]
            agree[name][1] += 1
            if e is not None and e.value == bool(val):
                agree[name][0] += 1
    per_key = {k: m / t for k, (m, t) in sorted(agree.items())}
    total = sum(m for m, _ in agree.values()) / max(1, sum(t for _, t in agree.values()))
    return dict(episode=episode, agreement=total, per_key=per_key,
                skill_completes=completes, delta_mismatches=mism,
                controller_events=[e for e in mw.controller.events])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-out", required=True)
    ap.add_argument("--task", type=int, default=45)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--min-agreement", type=float, default=0.95)
    args = ap.parse_args()
    bg_out = pathlib.Path(args.bg_out)

    eps = sorted(int(p.stem.split("_")[1]) for p in bg_out.glob("truth_*.json"))[: args.episodes]
    results = [replay_episode(bg_out, args.task, ep) for ep in eps]
    print(f"{'episode':>9} {'agreement':>10} {'completes':>10} {'delta_mism':>10}")
    for r in results:
        print(f"{r['episode']:>9} {r['agreement']:>10.4f} {r['skill_completes']:>10} "
              f"{r['delta_mismatches']:>10}")
    worst = min(results, key=lambda r: r["agreement"])
    print("\nworst episode per-key agreement:")
    for k, v in worst["per_key"].items():
        print(f"  {k:12s} {v:.4f}")
    mean = sum(r["agreement"] for r in results) / len(results)
    print(f"\nmean value-agreement vs belief_trace: {mean:.4f} "
          f"(threshold {args.min_agreement})")
    if mean < args.min_agreement:
        print("FAIL: agreement below threshold — inspect effect-timing or wiring")
        sys.exit(1)
    print("PASS: glue P1-P2 replay validation")


if __name__ == "__main__":
    main()
