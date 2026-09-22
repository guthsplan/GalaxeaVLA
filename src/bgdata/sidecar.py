"""LeRobot-compatible numeric sidecar built from the belief-trace records.

bg.pred_ids  int64[K]  indices into predicate_vocab.json (-1 padding)
bg.pred_vals float32[K] P(pred is True) from the reference belief sim (-1 padding)
bg.pred_obs  int64[K]  1 if observed at this frame
bg.rem_ids   int64[R]  vocab ids of remaining (unsatisfied) goal count lines

PoC caveat (stated in README): pred_vals come from GT labels masked by visibility and run
through the belief memory — NOT from a learned estimator.
"""
import json
import re

import numpy as np
import pandas as pd


def strip_count(line: str) -> str:
    """'(cooked ?x) 1/2 [hotdog.n.02]' -> canonical vocab form with 0/n."""
    return re.sub(r" \d+/(\d+) ", r" 0/\1 ", line)


def build_vocab(records_by_ep: dict[int, list[dict]], goal_lines: list[str]) -> list[str]:
    keys = set()
    for recs in records_by_ep.values():
        for r in recs:
            keys |= set(r["predicates"])
    return sorted(keys) + [strip_count(l) for l in goal_lines]


def build_sidecar(records_by_ep: dict[int, list[dict]], vocab: list[str], K: int, R: int,
                  task: int) -> pd.DataFrame:
    vid = {v: i for i, v in enumerate(vocab)}
    rows = []
    for ep, recs in records_by_ep.items():
        for r in recs:
            ids = np.full(K, -1, dtype=np.int64)
            vals = np.full(K, -1.0, dtype=np.float32)
            obs = np.zeros(K, dtype=np.int64)
            for j, (k, (val, p, o, src)) in enumerate(sorted(r["predicates"].items())[:K]):
                ids[j], vals[j], obs[j] = vid[k], p, int(o)
            rem = np.full(R, -1, dtype=np.int64)
            for j, line in enumerate(r["remaining_goal_lines"][:R]):
                rem[j] = vid[strip_count(line)]
            rows.append({"task": task, "episode": ep, "frame": r["step"],
                         "bg.pred_ids": ids, "bg.pred_vals": vals,
                         "bg.pred_obs": obs, "bg.rem_ids": rem})
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, vocab: list[str], parquet_path: str, vocab_path: str):
    df.to_parquet(parquet_path, index=False)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=1)
