#!/usr/bin/env python3
"""Build a task-filtered LeRobot v3 subset of the BEHAVIOR 2026 challenge demos for G0.5.

What it does
------------
* keeps only the episodes whose task_index is in --tasks (default: the 5-task PoC set)
* rewrites data parquets with re-indexed `episode_index` / `index` so that row position == `index`
  (required by the G0.5 v3 loader, which indexes the concatenated parquet table by frame index)
* adds Galaxea-style derived columns so the G0.5 shape_meta can address parts by column:
    action.left_arm(7)  action.left_gripper(1)  action.right_arm(7)  action.right_gripper(1)
    action.lower_body(7) = [torso(4), base_vel(3)]
    observation.state.<same keys>
* symlinks only the RGB mp4 files that the selected episodes reference (depth is dropped from
  info.json so the loader never decodes it)
* rewrites meta/tasks.parquet so the `task` text is the natural-language instruction
  (the original index is the snake_case task name)

Index layout of the raw columns (from the openpi `b1k/R1Pro` robot config):
  action(23): base[0:3] torso[3:7] left_arm[7:14] left_gripper[14] right_arm[15:22] right_gripper[22]
  state(61):  base_qvel[0:3] left_arm_qpos[3:10] left_gripper_qpos[24:26] right_arm_qpos[28:35]
              right_gripper_qpos[49:51] trunk_qpos[53:57]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RGB_KEYS = [
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
]

ACTION_SLICES = {
    "action.left_arm": [(7, 14)],
    "action.left_gripper": [(14, 15)],
    "action.right_arm": [(15, 22)],
    "action.right_gripper": [(22, 23)],
    "action.lower_body": [(3, 7), (0, 3)],  # torso(4) then base velocity(3)
}
STATE_SLICES = {
    "observation.state.left_arm": [(3, 10)],
    "observation.state.left_gripper": [(24, 25)],
    "observation.state.right_arm": [(28, 35)],
    "observation.state.right_gripper": [(49, 50)],
    "observation.state.lower_body": [(53, 57), (0, 3)],  # trunk qpos(4) then base qvel(3)
}


def log(msg):
    print(msg, flush=True)


def to_2d(col: pa.ChunkedArray) -> np.ndarray:
    """list<float>/fixed_size_list<float> column -> (N, D) float32 array."""
    arr = col.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        width = arr.type.list_size
        flat = arr.values.to_numpy(zero_copy_only=False)
        return flat.reshape(-1, width).astype(np.float32)
    lists = arr.to_pylist()
    return np.asarray(lists, dtype=np.float32)


def vec_column(mat: np.ndarray, like_type: pa.DataType) -> pa.Array:
    """(N, D) -> arrow list column with the same list flavour as the source `action` column."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    n, d = mat.shape
    flat = pa.array(mat.reshape(-1), type=pa.float32())
    if d == 1:
        # LeRobot stores shape-[1] features as scalar columns (HF Value("float32")), not 1-element lists
        return flat
    if pa.types.is_fixed_size_list(like_type):
        return pa.FixedSizeListArray.from_arrays(flat, d)
    offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat)


def derive(mat: np.ndarray, slices) -> np.ndarray:
    return np.concatenate([mat[:, a:b] for a, b in slices], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 35, 40, 46])
    ap.add_argument("--copy-videos", action="store_true", help="copy mp4s instead of symlinking")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        if not args.overwrite:
            sys.exit(f"{dst} exists; pass --overwrite to rebuild")
        shutil.rmtree(dst)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)

    info = json.load(open(src / "meta" / "info.json"))
    fps = info["fps"]

    # ---------------- episodes metadata ----------------
    ep_files = sorted((src / "meta" / "episodes").glob("*/*.parquet"))
    ep_tables = []
    for f in ep_files:
        t = pq.read_table(f)
        keep = [c for c in t.column_names if not c.startswith("stats/")]
        ep_tables.append(t.select(keep))
    eps = pa.concat_tables(ep_tables).to_pandas().sort_values("episode_index").reset_index(drop=True)
    sel = eps[eps["task_index"].isin(args.tasks)].copy().sort_values("episode_index").reset_index(drop=True)
    log(f"selected {len(sel)} / {len(eps)} episodes for tasks {args.tasks}")
    for t in args.tasks:
        log(f"  task {t}: {(sel.task_index == t).sum()} episodes")

    sel["old_episode_index"] = sel["episode_index"]
    sel["episode_index"] = np.arange(len(sel), dtype=np.int64)
    lengths = sel["length"].to_numpy().astype(np.int64)
    new_from = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    sel["dataset_from_index"] = new_from
    sel["dataset_to_index"] = new_from + lengths
    old2new_ep = dict(zip(sel["old_episode_index"], sel["episode_index"]))
    old2new_from = dict(zip(sel["old_episode_index"], sel["dataset_from_index"]))

    # ---------------- data parquets ----------------
    src_action_type = None
    file_groups = sel.groupby(["data/chunk_index", "data/file_index"], sort=True).groups
    new_file_idx = {}
    total_rows = 0
    for k, (chunk, fidx) in enumerate(sorted(file_groups.keys())):
        fpath = src / info["data_path"].format(chunk_index=chunk, file_index=fidx)
        t = pq.read_table(fpath)
        if src_action_type is None:
            src_action_type = t.schema.field("action").type
        ep_col = t.column("episode_index").to_numpy()
        wanted = np.isin(ep_col, sel.loc[file_groups[(chunk, fidx)], "old_episode_index"].to_numpy())
        t = t.filter(pa.array(wanted))
        # sort by original global index so rows are episode-contiguous and in time order
        order = np.argsort(t.column("index").to_numpy(), kind="stable")
        t = t.take(pa.array(order))

        old_ep = t.column("episode_index").to_numpy()
        frame = t.column("frame_index").to_numpy()
        new_ep = np.vectorize(old2new_ep.get)(old_ep).astype(np.int64)
        new_index = np.vectorize(old2new_from.get)(old_ep).astype(np.int64) + frame
        assert np.all(np.diff(new_index) == 1), f"non-contiguous rows in {fpath}"
        assert new_index[0] == total_rows, f"row/index mismatch at {fpath}: {new_index[0]} vs {total_rows}"

        act = to_2d(t.column("action"))
        st = to_2d(t.column("observation.state"))
        assert act.shape[1] == 23 and st.shape[1] == 61, (act.shape, st.shape)

        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", pa.array(new_ep))
        t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(new_index))
        for name, sl in ACTION_SLICES.items():
            t = t.append_column(name, vec_column(derive(act, sl), src_action_type))
        for name, sl in STATE_SLICES.items():
            t = t.append_column(name, vec_column(derive(st, sl), src_action_type))

        out = dst / "data" / "chunk-000" / f"file-{k:03d}.parquet"
        pq.write_table(t, out, compression="zstd")
        new_file_idx[(chunk, fidx)] = k
        total_rows += t.num_rows
        log(f"[{k+1}/{len(file_groups)}] {fpath.name}: {t.num_rows} rows -> {out.relative_to(dst)} (cum {total_rows})")

    assert total_rows == int(lengths.sum()), (total_rows, lengths.sum())
    sel["data/file_index"] = [
        new_file_idx[(int(c), int(f))]
        for c, f in zip(sel["data/chunk_index"].to_numpy(), sel["data/file_index"].to_numpy())
    ]
    sel["data/chunk_index"] = 0

    # ---------------- videos (RGB only) ----------------
    for key in RGB_KEYS:
        (dst / "videos" / key / "chunk-000").mkdir(parents=True, exist_ok=True)
        ck, fk = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        mapping = {}
        new_files = []
        for c, f in zip(sel[ck].to_numpy(), sel[fk].to_numpy()):
            if (c, f) not in mapping:
                j = len(mapping)
                mapping[(c, f)] = j
                srcv = src / info["video_path"].format(video_key=key, chunk_index=c, file_index=f)
                dstv = dst / "videos" / key / "chunk-000" / f"file-{j:03d}.mp4"
                if not srcv.exists():
                    sys.exit(f"missing video {srcv}")
                if args.copy_videos:
                    shutil.copyfile(srcv, dstv)
                else:
                    os.symlink(srcv.resolve(), dstv)
            new_files.append(mapping[(c, f)])
        sel[ck] = 0
        sel[fk] = new_files
        log(f"videos/{key}: {len(mapping)} mp4 files linked")

    # ---------------- episodes parquet ----------------
    drop_cols = [c for c in sel.columns if ".depth_linear." in c] + ["old_episode_index"]
    sel_out = sel.drop(columns=drop_cols)
    sel_out["meta/episodes/chunk_index"] = 0
    sel_out["meta/episodes/file_index"] = 0
    pq.write_table(pa.Table.from_pandas(sel_out, preserve_index=False),
                   dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # ---------------- tasks ----------------
    tasks = [json.loads(l) for l in open(src / "meta" / "tasks.jsonl")]
    tasks = sorted(tasks, key=lambda x: x["task_index"])
    tdf = pd.DataFrame({"task_index": [t["task_index"] for t in tasks]},
                       index=pd.Index([t["task"] for t in tasks], name="task"))
    tdf.to_parquet(dst / "meta" / "tasks.parquet")
    with open(dst / "meta" / "tasks.jsonl", "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")

    # ---------------- info.json / stats ----------------
    feats = {k: v for k, v in info["features"].items() if ".depth_linear." not in k}
    for name, sl in list(ACTION_SLICES.items()) + list(STATE_SLICES.items()):
        d = sum(b - a for a, b in sl)
        feats[name] = {"dtype": "float32", "shape": [d], "names": [name.split(".")[-1]]}
    info["features"] = feats
    info["total_episodes"] = int(len(sel))
    info["total_frames"] = int(total_rows)
    info["splits"] = {"train": f"0:{len(sel)}"}
    info["total_chunks"] = 1
    info["subset_of"] = str(src)
    info["subset_tasks"] = args.tasks
    json.dump(info, open(dst / "meta" / "info.json", "w"), indent=4)
    if (src / "meta" / "stats.json").exists():
        shutil.copyfile(src / "meta" / "stats.json", dst / "meta" / "stats.json")

    # ---------------- validation ----------------
    log("validating ...")
    ds = pq.ParquetDataset([str(p) for p in sorted((dst / "data").glob("*/*.parquet"))])
    tbl = ds.read(columns=["index", "episode_index", "frame_index", "task_index"])
    idx = tbl.column("index").to_numpy()
    assert np.array_equal(idx, np.arange(len(idx))), "index column is not 0..N-1"
    ep = tbl.column("episode_index").to_numpy()
    for i in range(len(sel)):
        a, b = int(sel_out.dataset_from_index[i]), int(sel_out.dataset_to_index[i])
        assert np.all(ep[a:b] == i), f"episode {i} rows mismatch"
        assert tbl.column("frame_index").to_numpy()[a] == 0
    ti = tbl.column("task_index").to_numpy()
    log(f"OK: {len(sel)} episodes, {len(idx)} frames, task_index counts: "
        f"{dict(zip(*np.unique(ti, return_counts=True)))}")
    # gripper range sanity (tokenizer binarises gripper on sign)
    g = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet", columns=["action.left_gripper"]).column("action.left_gripper").to_numpy()
    log(f"left gripper action range in first file: min={g.min():.3f} max={g.max():.3f} unique~{np.unique(np.round(g, 2))[:8]}")


if __name__ == "__main__":
    main()
