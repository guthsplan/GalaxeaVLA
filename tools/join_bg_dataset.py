# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Stage-3 join: build a SCRATCH LeRobot-v3 dataset with the bg_* fields merged in.

The source dataset stays untouched (read-only). The scratch copy contains, for ONE task:
  meta/tasks.parquet   source tasks  +  appended bg_tasks.jsonl rows
  meta/info.json       source info   +  five bg_*_index int64 features
  meta/episodes, meta/stats.json     copied as-is
  data/chunk-XXX/*.parquet           source columns + forward-filled bg_*_index columns
                                     (bg rows are 1 Hz; every 30 fps frame gets the most
                                      recent 1 Hz snapshot — E2E_GUIDE Stage 3 policy)
Videos are NOT copied — use load_images=False for the dry run.

Usage:
    python tools/join_bg_dataset.py --demos <lerobot root> --bg-fields <g05_bg_fields dir> \
        --out <scratch root> --task 45
"""
import argparse
import json
import pathlib
import shutil

import pandas as pd

BG_COLS = ["bg_known_index", "bg_belief_index", "bg_delta_index", "bg_effect_index",
           "bg_observe_index", "atomic_task_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--bg-fields", required=True, help="dir with bg_tasks.jsonl + bg_frame_indices.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=45)
    args = ap.parse_args()
    src = pathlib.Path(args.demos)
    bgf = pathlib.Path(args.bg_fields)
    out = pathlib.Path(args.out)
    chunk = f"chunk-{args.task:03d}"

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / chunk).mkdir(parents=True, exist_ok=True)

    # ---- meta/tasks.parquet: append the deduplicated bg strings ----
    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    bg_rows = [json.loads(l) for l in open(bgf / "bg_tasks.jsonl")]
    max_idx = int(tasks["task_index"].max())
    min_bg = min(r["task_index"] for r in bg_rows)
    assert min_bg > max_idx, (
        f"task_index collision: bg starts at {min_bg} but source tasks go up to {max_idx} — "
        f"regenerate bg fields with --index-offset {max_idx + 1} or higher")
    bg_df = pd.DataFrame({"task_index": [r["task_index"] for r in bg_rows]},
                         index=pd.Index([r["task"] for r in bg_rows], name=tasks.index.name))
    pd.concat([tasks, bg_df]).to_parquet(out / "meta/tasks.parquet")

    # ---- meta/info.json: register the new int64 columns as features ----
    info = json.loads((src / "meta/info.json").read_text())
    for c in BG_COLS:
        info["features"][c] = {"dtype": "int64", "shape": [1], "names": None}
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))

    # ---- meta/episodes + stats: copy ----
    shutil.copytree(src / "meta/episodes", out / "meta/episodes", dirs_exist_ok=True)
    shutil.copy(src / "meta/stats.json", out / "meta/stats.json")

    # ---- episode_index <-> raw_episode_id map (bg frames key on raw id) ----
    epmeta = pd.concat(pd.read_parquet(p) for p in sorted((src / "meta/episodes" / chunk).glob("*.parquet")))
    raw2ep = dict(zip(epmeta.raw_episode_id.astype(int), epmeta.episode_index.astype(int)))

    bg = pd.read_parquet(bgf / "bg_frame_indices.parquet")
    bg["episode_index"] = bg["episode"].map(raw2ep)
    bg = bg.dropna(subset=["episode_index"])
    bg["episode_index"] = bg["episode_index"].astype("int64")
    bg = bg.rename(columns={"frame": "frame_index"}).sort_values(["episode_index", "frame_index"])

    # ---- data chunk: merge_asof per episode (forward fill from the last 1 Hz row) ----
    # pandas merge_asof requires the `on` key GLOBALLY sorted even with `by=`, so sort by
    # frame_index for the merge and restore the file's original row order afterwards.
    bg_sorted = bg[["episode_index", "frame_index"] + BG_COLS].sort_values("frame_index")
    n_files = n_rows = 0
    for f in sorted((src / "data" / chunk).glob("file-*.parquet")):
        df = pd.read_parquet(f)
        order = df.index.copy()
        left = df.sort_values("frame_index", kind="stable")
        merged = pd.merge_asof(
            left, bg_sorted, on="frame_index", by="episode_index", direction="backward")
        merged.index = left.index
        merged = merged.loc[order]
        for c in BG_COLS:
            merged[c] = merged[c].astype("int64")  # bg starts at frame 0, so no NaN expected
        merged.to_parquet(out / "data" / chunk / f.name, index=False)
        n_files += 1
        n_rows += len(merged)
    print(f"scratch dataset: {out}")
    print(f"  tasks: {len(tasks)} + {len(bg_df)} bg strings · data: {n_files} files / {n_rows} rows "
          f"· episodes mapped: {bg.episode_index.nunique()}")


if __name__ == "__main__":
    main()
