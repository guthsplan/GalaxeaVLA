#!/usr/bin/env python3
"""Convert the BEHAVIOR 2026 bbox / EE-pose exports into CoT sidecars for merge_b1k_cot_labels.py.

Inputs (all under the challenge demos root, $B1K_RAW_DEMOS)
------------------------------------------------------------
* ``behavior_bbox_2025_025fps_v2/task-XXXX/episode_<raw_id>/{frames.jsonl,report.json}``
    0.25 Hz object boxes from the head-camera instance segmentation. Each record:
    ``{"frame_index", "bbox": {obj: [x1,y1,x2,y2]}, "raw_bbox", "filtering"}``, coordinates
    normalized to the 720x720 head image. ``bbox`` (component-filtered) is used, ``raw_bbox``
    and ``filtering`` are dropped. Episodes whose report is incomplete, not ``packable`` or
    has ``length_2025 != length_2026`` are skipped (the 2025 masks would be misaligned).
* ``behavior_ee_pose_0_99/task-XXXX/frames/chunk-*/file-*.parquet``
    30 Hz ``observation.ee_pose.{left,right}`` = [x, y, z, qx, qy, qz, qw] in the robot frame,
    keyed by the 2026 ``episode_index``.
* ``data/chunk-*/file-*.parquet`` column ``observation.robot2cam_pose.<camera>``
    30 Hz camera pose in the same robot frame (USD camera convention: -Z forward, +Y up).

The 2D trace is the EE position projected into the head camera of the same frame::

    p_cam = R_cv^T (p_ee - t_cam),  R_cv = R_cam @ diag(1, -1, -1)     (USD -> OpenCV)
    u = (fx * x / z + cx) / W,      fx = W * focal_length / horizontal_aperture

with the evaluator's head camera (720 px, horizontal_aperture 40, OmniGibson's default focal
length 17 -> fx = 306). A hand is ``visb`` when z > --min-depth and (u, v) is inside the
image; occlusion is NOT tested (the depth videos are not decoded here).

Output (per episode, keyed by raw_episode_id like the replay extractor)
-----------------------------------------------------------------------
    <out>/episode_<raw_id>_cot_strings.json   de-duplicated payload strings
    <out>/episode_<raw_id>_cot_index.jsonl    {"frame_index", "bbox_index", "2d_trace_index"}
    <out>/coverage.json                       per-task / per-episode counts and skip reasons

then:
    python tools/merge_b1k_cot_labels.py --dataset <subset> --cot-sidecar-dir <out> \\
        --sidecar-episode-key raw

Usage
-----
    python tools/build_b1k_bbox_trace_sidecars.py --demos-root $B1K_RAW_DEMOS \\
        --tasks 45 --out $B1K_RAW_DEMOS/cot_sidecars_bbox_trace
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

BBOX_DIR = "behavior_bbox_2025_025fps_v2"
EE_DIR = "behavior_ee_pose_0_99"
ARMS = ("left", "right")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_episodes(root: Path, tasks: List[int]) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    cols = ["episode_index", "task_index", "length", "raw_episode_id", "data/chunk_index", "data/file_index"]
    eps = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    return eps[eps["task_index"].isin(tasks)].sort_values("episode_index").reset_index(drop=True)


def read_bbox(bbox_root: Path, task: int, raw_id: int, length: int, key: str) -> Tuple[Dict[int, dict], Optional[str]]:
    """frame -> {obj: [x1,y1,x2,y2]}, or (empty, skip reason)."""
    ep_dir = bbox_root / f"task-{task:04d}" / f"episode_{raw_id:08d}"
    report_path, frames_path = ep_dir / "report.json", ep_dir / "frames.jsonl"
    if not report_path.exists() or not frames_path.exists():
        return {}, "missing"
    report = json.loads(report_path.read_text())
    if not report.get("complete", False):
        return {}, "incomplete"
    if not report.get("packable", True):
        return {}, "not_packable"
    if report.get("length_2025") != report.get("length_2026"):
        return {}, "length_2025_mismatch"
    if report.get("length_2026") not in (None, length):
        return {}, "length_2026_mismatch"

    out: Dict[int, dict] = {}
    for line in frames_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        frame = int(rec["frame_index"])
        boxes = rec.get(key) or {}
        if not 0 <= frame < length or not boxes:
            continue
        clean = {}
        for name, box in boxes.items():
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(np.clip(v, 0.0, 1.0)) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            clean[name] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
        if clean:
            out[frame] = clean
    return out, None


class EEPoseTable:
    """Lazily loaded per-task EE poses: episode_index -> (frame_index, left(N,7), right(N,7))."""

    def __init__(self, ee_root: Path):
        self.ee_root = ee_root
        self.cache: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def available(self, task: int) -> bool:
        return any((self.ee_root / f"task-{task:04d}" / "frames").glob("chunk-*/*.parquet"))

    def get(self, task: int, episode: int):
        if task not in self.cache:
            files = sorted((self.ee_root / f"task-{task:04d}" / "frames").glob("chunk-*/*.parquet"))
            cols = ["episode_index", "frame_index", "observation.ee_pose.left", "observation.ee_pose.right"]
            df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
            per_ep = {}
            for ep, g in df.groupby("episode_index", sort=False):
                g = g.sort_values("frame_index")
                per_ep[int(ep)] = (
                    g["frame_index"].to_numpy(),
                    np.stack(g["observation.ee_pose.left"].to_numpy()).astype(np.float64),
                    np.stack(g["observation.ee_pose.right"].to_numpy()).astype(np.float64),
                )
            self.cache = {task: per_ep}  # one task at a time keeps memory bounded
        return self.cache[task].get(episode)


def read_cam_poses(root: Path, eps: pd.DataFrame, camera: str) -> Dict[int, np.ndarray]:
    """episode_index -> (N, 7) robot->camera pose, frame-ordered."""
    col = f"observation.robot2cam_pose.{camera}"
    out: Dict[int, np.ndarray] = {}
    for (chunk, fidx), group in eps.groupby(["data/chunk_index", "data/file_index"]):
        path = root / "data" / f"chunk-{int(chunk):03d}" / f"file-{int(fidx):03d}.parquet"
        wanted = [int(e) for e in group["episode_index"]]
        t = pq.read_table(path, columns=["episode_index", "frame_index", col], filters=[("episode_index", "in", wanted)])
        df = t.to_pandas()
        for ep, g in df.groupby("episode_index", sort=False):
            g = g.sort_values("frame_index")
            if not np.array_equal(g["frame_index"].to_numpy(), np.arange(len(g))):
                raise ValueError(f"episode {ep}: frame_index in {path.name} is not 0..N-1")
            out[int(ep)] = np.stack(g[col].to_numpy()).astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project(ee: np.ndarray, cam: np.ndarray, fx: float, size: int, min_depth: float):
    """(N,7) EE poses + (N,7) camera poses -> (N,2) normalized uv, (N,) visible."""
    r_cv = Rotation.from_quat(cam[:, 3:7]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    rel = ee[:, :3] - cam[:, :3]
    p_cam = np.einsum("nij,ni->nj", r_cv, rel)  # R_cv^T @ rel
    z = p_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    c = size / 2.0
    u = (fx * p_cam[:, 0] / safe_z + c) / size
    v = (fx * p_cam[:, 1] / safe_z + c) / size
    visible = (z > min_depth) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return np.stack([u, v], axis=1), visible


def trace_payload(uv: Dict[str, np.ndarray], vis: Dict[str, np.ndarray], k: int) -> Optional[dict]:
    if not any(vis[a][k] for a in ARMS):
        return None  # Trace2DCoTBuilder.can_handle rejects frames with no visible hand anyway
    out = {}
    for arm in ARMS:
        ok = bool(vis[arm][k])
        out[f"uv_{arm}"] = [round(float(uv[arm][k, 0]), 4), round(float(uv[arm][k, 1]), 4)] if ok else None
        out[f"visb_{arm}"] = ok
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_episode(out_dir: Path, raw_id: int, records: Dict[int, Dict[str, dict]]) -> None:
    strings: List[str] = []
    index: Dict[str, int] = {}

    def intern(payload: dict) -> int:
        text = json.dumps(payload, separators=(",", ":"))
        if text not in index:
            index[text] = len(strings)
            strings.append(text)
        return index[text]

    lines = []
    for frame in sorted(records):
        rec = records[frame]
        lines.append(json.dumps({
            "frame_index": frame,
            "bbox_index": intern(rec["bbox"]) if "bbox" in rec else -1,
            "2d_trace_index": intern(rec["trace_2d"]) if "trace_2d" in rec else -1,
        }))
    (out_dir / f"episode_{raw_id}_cot_strings.json").write_text(json.dumps(strings))
    (out_dir / f"episode_{raw_id}_cot_index.jsonl").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos-root", type=Path, required=True, help="$B1K_RAW_DEMOS (LeRobot v3 challenge demos)")
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bbox-dir", type=Path, help=f"default <demos-root>/{BBOX_DIR}")
    ap.add_argument("--ee-pose-dir", type=Path, help=f"default <demos-root>/{EE_DIR}")
    ap.add_argument("--no-bbox", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--bbox-key", default="bbox", choices=["bbox", "raw_bbox"])
    ap.add_argument("--camera", default="zed_link_camera_0")
    ap.add_argument("--image-size", type=int, default=720)
    ap.add_argument("--focal-length", type=float, default=17.0)
    ap.add_argument("--horizontal-aperture", type=float, default=40.0)
    ap.add_argument("--min-depth", type=float, default=0.05, help="metres in front of the camera")
    ap.add_argument("--max-episodes", type=int, default=None, help="per task, for smoke tests")
    args = ap.parse_args()

    root = args.demos_root
    bbox_root = args.bbox_dir or root / BBOX_DIR
    ee = EEPoseTable(args.ee_pose_dir or root / EE_DIR)
    fx = args.image_size * args.focal_length / args.horizontal_aperture
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"head camera {args.camera}: {args.image_size}px, fx = {fx:.2f}")

    eps = read_episodes(root, args.tasks)
    coverage: Dict[str, dict] = {}
    for task, task_eps in eps.groupby("task_index"):
        task = int(task)
        if args.max_episodes:
            task_eps = task_eps.head(args.max_episodes)
        do_trace = not args.no_trace and ee.available(task)
        if not args.no_trace and not do_trace:
            log(f"task {task}: no EE poses under {ee.ee_root}/task-{task:04d} -> bbox only")
        cams = read_cam_poses(root, task_eps, args.camera) if do_trace else {}

        stats = defaultdict(int)
        skips: Dict[str, List[int]] = defaultdict(list)
        for row in task_eps.itertuples():
            ep, raw_id, length = int(row.episode_index), int(row.raw_episode_id), int(row.length)
            records: Dict[int, Dict[str, dict]] = defaultdict(dict)

            if not args.no_bbox:
                boxes, reason = read_bbox(bbox_root, task, raw_id, length, args.bbox_key)
                if reason:
                    skips[f"bbox:{reason}"].append(raw_id)
                for frame, payload in boxes.items():
                    records[frame]["bbox"] = payload
                stats["bbox_frames"] += len(boxes)

            if do_trace:
                pose = ee.get(task, ep)
                cam = cams.get(ep)
                if pose is None or cam is None:
                    skips["trace:missing"].append(raw_id)
                elif len(pose[0]) != length or len(cam) != length or not np.array_equal(pose[0], np.arange(length)):
                    skips["trace:length_mismatch"].append(raw_id)
                else:
                    uv, vis = {}, {}
                    for arm, arr in zip(ARMS, pose[1:]):
                        uv[arm], vis[arm] = project(arr, cam, fx, args.image_size, args.min_depth)
                    for k in range(length):
                        payload = trace_payload(uv, vis, k)
                        if payload is not None:
                            records[k]["trace_2d"] = payload
                            stats["trace_frames"] += 1
                    stats["visible_left"] += int(vis["left"].sum())
                    stats["visible_right"] += int(vis["right"].sum())

            stats["frames"] += length
            if records:
                write_episode(args.out, raw_id, records)
                stats["episodes_written"] += 1

        stats["episodes"] = len(task_eps)
        coverage[str(task)] = {**stats, "skipped": {k: v for k, v in skips.items()}}
        f = max(stats["frames"], 1)
        log(
            f"task {task}: {stats['episodes_written']}/{len(task_eps)} episodes written; "
            f"bbox {stats['bbox_frames']} frames ({100 * stats['bbox_frames'] / f:.2f}%), "
            f"trace {stats['trace_frames']} ({100 * stats['trace_frames'] / f:.1f}%); "
            f"skipped {', '.join(f'{k}={len(v)}' for k, v in skips.items()) or 'none'}"
        )

    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if not any(c.get("episodes_written") for c in coverage.values()):
        log("no episode produced any label")
        return 1
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
