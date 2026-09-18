#!/usr/bin/env python3
"""Carve a few-episode LeRobot v3 fixture out of an existing subset.

The BEHAVIOR subset is ~1000 episodes / 4.4 M frames / 2.2 GB of parquet, which is
far more than the CoT test gates need. This slices the first N episodes into a
standalone dataset — same schema, same video files (symlinked), re-indexed so
`index` is 0..M-1 and each episode's `dataset_from_index` is correct — so the
builder / alignment / dry-run / smoke gates run in seconds and never touch the
production subset.

    python tools/make_cot_fixture.py --src DIR --dst DIR --episodes 8
    python tools/merge_b1k_cot_labels.py --dataset DST --synthetic --episodes 8
"""
from __future__ import annotations

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if dst.exists():
        if not args.overwrite:
            sys.exit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)

    info = json.loads((src / "meta" / "info.json").read_text())

    ep_files = sorted((src / "meta" / "episodes").glob("*/*.parquet"))
    eps = pd.concat([pq.read_table(f).to_pandas() for f in ep_files], ignore_index=True)
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    sel = eps.iloc[: args.episodes].copy().reset_index(drop=True)

    lengths = sel["length"].to_numpy().astype(np.int64)
    new_from = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    keep_lo = int(sel["dataset_from_index"].iloc[0])
    keep_hi = int(sel["dataset_to_index"].iloc[-1])
    sel["dataset_from_index"] = new_from
    sel["dataset_to_index"] = new_from + lengths
    total = int(lengths.sum())
    print(f"episodes {len(sel)}  frames {total}  (source rows [{keep_lo}, {keep_hi}))")

    # ---- data: the contiguous global row range of the selected episodes ----
    tables = []
    taken = 0
    for path in sorted((src / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        index = table.column("index").to_numpy()
        mask = (index >= keep_lo) & (index < keep_hi)
        if not mask.any():
            if taken:
                break
            continue
        table = table.filter(pa.array(mask))
        tables.append(table)
        taken += table.num_rows
        if taken >= total:
            break
    out = pa.concat_tables(tables)
    assert out.num_rows == total, (out.num_rows, total)

    order = np.argsort(out.column("index").to_numpy(), kind="stable")
    out = out.take(pa.array(order))
    out = out.set_column(
        out.schema.get_field_index("index"), "index", pa.array(np.arange(total, dtype=np.int64))
    )
    # Re-index episodes to 0..N-1 so the fixture is self-contained.
    old_ep = out.column("episode_index").to_numpy()
    remap = {int(o): i for i, o in enumerate(sel["episode_index"].to_numpy())}
    out = out.set_column(
        out.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array(np.array([remap[int(e)] for e in old_ep], dtype=np.int64)),
    )
    sel["episode_index"] = np.arange(len(sel), dtype=np.int64)
    pq.write_table(out, dst / "data" / "chunk-000" / "file-000.parquet", compression="zstd")
    sel["data/chunk_index"] = 0
    sel["data/file_index"] = 0

    # ---- videos: symlink whatever the selected episodes reference ----
    for key in [k for k, v in info["features"].items() if v.get("dtype") == "video"]:
        ck, fk = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        if ck not in sel.columns:
            continue
        (dst / "videos" / key / "chunk-000").mkdir(parents=True, exist_ok=True)
        mapping: dict[tuple[int, int], int] = {}
        new_files = []
        for c, f in zip(sel[ck].to_numpy(), sel[fk].to_numpy()):
            if (int(c), int(f)) not in mapping:
                j = len(mapping)
                mapping[(int(c), int(f))] = j
                srcv = src / info["video_path"].format(video_key=key, chunk_index=c, file_index=f)
                if not srcv.exists():
                    sys.exit(f"missing video {srcv}")
                os.symlink(srcv.resolve(), dst / "videos" / key / "chunk-000" / f"file-{j:03d}.mp4")
            new_files.append(mapping[(int(c), int(f))])
        sel[ck] = 0
        sel[fk] = new_files

    sel["meta/episodes/chunk_index"] = 0
    sel["meta/episodes/file_index"] = 0
    pq.write_table(
        pa.Table.from_pandas(sel, preserve_index=False),
        dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    shutil.copyfile(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    for optional in ("tasks.jsonl", "stats.json"):
        if (src / "meta" / optional).exists():
            shutil.copyfile(src / "meta" / optional, dst / "meta" / optional)

    info["total_episodes"] = int(len(sel))
    info["total_frames"] = total
    info["splits"] = {"train": f"0:{len(sel)}"}
    info["total_chunks"] = 1
    info["fixture_of"] = str(src)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    idx = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet", columns=["index"]).column("index").to_numpy()
    assert np.array_equal(idx, np.arange(total)), "index column is not 0..N-1"
    print(f"OK -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
