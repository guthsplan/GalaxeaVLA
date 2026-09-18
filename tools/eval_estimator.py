# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Estimator benchmark harness (hybrid P0, CPU-only) — the gate for EVERY estimator.

Compares an estimator's per-frame output against the privileged truth files of the
bgdata pipeline (demo episodes carry both RGB-D+proprio and truth, so any estimator
is quantitatively evaluable WITHOUT a simulator):

    per predicate name: detection precision/recall (did it return the visible keys?)
                        value accuracy (were the returned values right?)

Built-in estimators (both must score 1.0 — they validate the harness and the
Observe round-trip; a learned model plugs in the same way via CoTObserveEstimator):
    oracle      OracleEstimator(truth_*.json)
    observe-gt  CoTObserveEstimator over the GT 'observe' texts of cot_targets.parquet
                (robot-tag predicates are excluded from Observe by design, so the
                 comparison restricts to geom/appear keys)

Usage:
    PYTHONPATH=src python tools/eval_estimator.py --bg-out <bgdata out/task045> \
        --estimator oracle|observe-gt [--episodes 5]
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from g05.belief_graph.estimator import CoTObserveEstimator, OracleEstimator  # noqa: E402

ROBOT_PREDS = ("inhand", "inhand_left", "inhand_right", "reachable", "visited")


def truth_frames(bg_out: pathlib.Path, episode: int) -> dict[int, dict]:
    data = json.loads((bg_out / f"truth_{episode}.json").read_text())
    return {int(fr): {k: (bool(v), bool(vis)) for k, (v, vis) in d.items()}
            for fr, d in data.items()}


def make_estimator(kind: str, bg_out: pathlib.Path, episode: int):
    if kind == "oracle":
        return OracleEstimator(bg_out / f"truth_{episode}.json"), False
    if kind == "observe-gt":
        import pandas as pd

        ct = pd.read_parquet(bg_out / "cot_targets.parquet")
        ct = ct[ct.episode == episode]
        texts = dict(zip(ct.frame.astype(int), ct.observe))
        return CoTObserveEstimator(lambda obs, step: texts.get(step, "Observe: none")), True
    raise ValueError(kind)


def shorten_map(bg_out: pathlib.Path) -> dict[str, str]:
    # cot_targets shortens container names; invert for comparison with truth keys
    return {"fridge": "fridge_dszchb_0", "microwave": "microwave_abzvij_0",
            "counter": "countertop_kelker_0"}


def unshorten(key: str, m: dict[str, str]) -> str:
    name, *args = key[1:-1].split()
    args = [m.get(a, a) for a in args]
    return f"({name}{''.join(' ' + a for a in args)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-out", required=True)
    ap.add_argument("--estimator", choices=["oracle", "observe-gt"], default="oracle")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    bg_out = pathlib.Path(args.bg_out)
    smap = shorten_map(bg_out)

    eps = sorted(int(p.stem.split("_")[1]) for p in bg_out.glob("truth_*.json"))[: args.episodes]
    stats = defaultdict(lambda: dict(tp=0, fn=0, fp=0, val_ok=0, val_n=0))
    frames_per_ep = 0
    for ep in eps:
        est, skip_robot = make_estimator(args.estimator, bg_out, ep)
        truth = truth_frames(bg_out, ep)
        # 1 Hz frames (cot_targets cadence): every 30th 30fps index present in truth
        frames = sorted(fr for fr in truth if fr % 30 == 0)
        frames_per_ep = len(frames)
        for fr in frames:
            got = {unshorten(k, smap): e for k, e in est.estimate({}, fr).items()}
            visible = {k: v for k, (v, vis) in truth[fr].items() if vis
                       and not (skip_robot and k[1:].split()[0] in ROBOT_PREDS)}
            for k, v in visible.items():
                name = k[1:].split()[0]
                if k in got:
                    stats[name]["tp"] += 1
                    stats[name]["val_n"] += 1
                    stats[name]["val_ok"] += int(got[k].value == v)
                else:
                    stats[name]["fn"] += 1
            for k in got:
                if k not in visible and not (skip_robot and k[1:].split()[0] in ROBOT_PREDS):
                    stats[k[1:].split()[0]]["fp"] += 1

    print(f"estimator={args.estimator} · episodes={len(eps)} · ~{frames_per_ep} frames/ep (1 Hz)")
    print(f"{'predicate':14s} {'recall':>8} {'precision':>10} {'val_acc':>8} {'n_vis':>7}")
    fails = 0
    for name, s in sorted(stats.items()):
        n = s["tp"] + s["fn"]
        rec = s["tp"] / n if n else 1.0
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 1.0
        acc = s["val_ok"] / s["val_n"] if s["val_n"] else 1.0
        print(f"{name:14s} {rec:>8.4f} {prec:>10.4f} {acc:>8.4f} {n:>7}")
        fails += int(min(rec, prec, acc) < 1.0)
    if args.estimator in ("oracle", "observe-gt"):
        if fails:
            print(f"FAIL: {fails} predicate(s) below 1.0 — harness or round-trip broken")
            sys.exit(1)
        print("PASS: harness validated (exact agreement with truth)")


if __name__ == "__main__":
    main()
