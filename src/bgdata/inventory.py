"""Resolve episode links: demos (LeRobot v3) <-> raw HDF5 <-> annotation JSON <-> BDDL."""
import glob
import json
import pathlib

import pandas as pd
import pyarrow.parquet as pq

import os

# Data root: the directory holding b1k_demos/, b1k_raw/, BEHAVIOR-1K-main/.
# Now that bgdata lives inside the GalaxeaVLA repo, the root is no longer derived from
# the package location — set BGDATA_ROOT, or run from the data directory (default: CWD).
WS = pathlib.Path(os.environ.get("BGDATA_ROOT", os.getcwd())).resolve()
DEMOS = WS / "b1k_demos"
RAW = WS / "b1k_raw"
BDDL_ROOT = WS / "BEHAVIOR-1K-main/bddl3/bddl/activity_definitions"


def load_episode_meta() -> pd.DataFrame:
    fs = sorted(glob.glob(str(DEMOS / "meta/episodes/**/*.parquet"), recursive=True))
    return pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)


def load_tasks() -> pd.DataFrame:
    rows = [json.loads(l) for l in open(DEMOS / "meta/tasks.jsonl")]
    return pd.DataFrame(rows).set_index("task_index")


def resolve_task(task_index: int) -> dict:
    tasks = load_tasks()
    task_name = tasks.loc[task_index, "task_name"]
    meta = load_episode_meta()
    eps = meta[meta.task_index == task_index].copy()
    recs = []
    for _, r in eps.iterrows():
        raw_id = int(r.raw_episode_id)
        raw_path = RAW / f"task-{task_index:04d}" / f"episode_{raw_id:08d}.hdf5"
        ann_path = DEMOS / r.annotation_path
        recs.append(
            dict(
                episode_index=int(r.episode_index),
                raw_episode_id=raw_id,
                task_instance_id=int(r.task_instance_id),
                length=int(r.length),
                data_chunk=int(r["data/chunk_index"]),
                data_file=int(r["data/file_index"]),
                raw_path=str(raw_path),
                raw_exists=raw_path.exists(),
                annotation_path=str(ann_path),
                annotation_exists=ann_path.exists(),
            )
        )
    bddl = BDDL_ROOT / task_name / "problem0.bddl"
    return dict(
        task_index=task_index,
        task_name=task_name,
        instruction=tasks.loc[task_index, "task"],
        bddl_path=str(bddl),
        bddl_exists=bddl.exists(),
        episodes=recs,
    )


def demo_parquet_path(chunk: int, file: int) -> pathlib.Path:
    return DEMOS / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"


def load_annotation(path: str) -> dict:
    return json.load(open(path))


def segments(ann: dict) -> list[dict]:
    """Flatten skill_annotation into one record per segment."""
    out = []
    for sk in ann["skill_annotation"]:
        out.append(
            dict(
                skill_idx=sk["skill_idx"],
                skill_id=sk["skill_id"][0],
                skill=sk["skill_description"][0],
                objects=list(sk["object_id"][0]) if sk["object_id"] else [],
                manip=(sk["manipulating_object_id"] or [None])[0],
                memory_prefix=(sk["memory_prefix"] or [""])[0],
                start=sk["frame_duration"][0],
                end=sk["frame_duration"][1],
                skill_type=(sk["skill_type"] or [""])[0],
            )
        )
    return out
