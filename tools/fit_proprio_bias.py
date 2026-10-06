"""Fit the serving-time proprio bias correction (PROPRIO_BIAS in scripts/serve_g05_b1k.py).

In the BEHAVIOR demos the commanded joint position leads the measured one by ~0.007 rad even at rest,
while the eval simulator tracks commands exactly. The model learned action[0] ~= state + b, so in eval
every chunk starts ~b away from the current position and an idle arm drifts by ~b per chunk
(~0.12-0.16 rad per 1000 steps measured on popcorn). Serving with (state - b) restores the demo relation.

b is fitted on STILL demo frames (|delta state| < --still per frame), per action part:
    arms (first 7 dims)  posture-dependent b(s) = [s, sin s, cos s, 1] @ W (ridge), with the input clamped
                         to the demo state range (p0.5..p99.5) and b clipped to the still-offset range (p1..p99)
    torso (lower_body[:4]) constant b (a posture fit extrapolated badly: the demo torso barely moves)

    python tools/fit_proprio_bias.py --dataset <lerobot dataset dir> --out proprio_bias.json

Result on the 5-task LoRA run: the drift fell by 40-75 %, popcorn success did not change (3/20 vs the
5/18 pooled baseline), so the option is off unless PROPRIO_BIAS points to this file.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq

PARTS = {"left_arm": 7, "right_arm": 7, "lower_body": 4}


def _feat(x):
    return np.hstack([x, np.sin(x), np.cos(x), np.ones((len(x), 1))])


def _still_frames(files, key, n, still):
    O, S = [], []
    for f in files:
        t = pq.read_table(f, columns=[f"action.{key}", f"observation.state.{key}", "episode_index"]).to_pandas()
        a = np.stack(t[f"action.{key}"].values)[:, :n]
        s = np.stack(t[f"observation.state.{key}"].values)[:, :n]
        ep = t["episode_index"].values
        same = ep[1:] == ep[:-1]
        mask = same & (np.linalg.norm(np.diff(s, axis=0), axis=1) < still)
        O.append((a - s)[:-1][mask])
        S.append(s[:-1][mask])
    return np.concatenate(O), np.concatenate(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot v3 dataset dir (data/**/*.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--still", type=float, default=2e-4, help="max |delta state| per frame for a still frame")
    ap.add_argument("--ridge", type=float, default=10.0, help="ridge weight (scaled by n_samples * 1e-6)")
    ap.add_argument("--every-file", type=int, default=2, help="use every k-th parquet file")
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.dataset}/data/**/*.parquet", recursive=True))[:: args.every_file]
    out = {"model": "posture_clipped", "W": {}, "x_lo": {}, "x_hi": {}, "b_lo": {}, "b_hi": {}, "bias": {}}
    for key, n in PARTS.items():
        O, S = _still_frames(files, key, n, args.still)
        if key == "lower_body":
            out["bias"][key] = [round(float(v), 6) for v in O.mean(0)]
            print(f"{key}: constant b |b|={np.linalg.norm(O.mean(0)):.4f} (n={len(O)})")
            continue
        X = _feat(S)
        W = np.linalg.solve(X.T @ X + args.ridge * len(X) * 1e-6 * np.eye(X.shape[1]), X.T @ O)
        out["W"][key] = W.tolist()
        out["x_lo"][key], out["x_hi"][key] = np.percentile(S, 0.5, 0).tolist(), np.percentile(S, 99.5, 0).tolist()
        out["b_lo"][key], out["b_hi"][key] = np.percentile(O, 1, 0).tolist(), np.percentile(O, 99, 0).tolist()
        resid = np.median(np.linalg.norm(O - X @ W, axis=1))
        print(f"{key}: posture b(s), still offset median {np.median(np.linalg.norm(O, axis=1)):.4f} -> "
              f"residual {resid:.4f} (n={len(O)})")
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
