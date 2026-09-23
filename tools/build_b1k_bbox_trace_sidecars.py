#!/usr/bin/env python3
"""Build the non-belief-graph CoT labels (Subtask, BBox, 2D trace) for merge_b1k_cot_labels.py.

Inputs (all under the challenge demos root, $B1K_RAW_DEMOS)
------------------------------------------------------------
* ``behavior_bbox_2025_025fps_v2/task-XXXX/episode_<raw_id>/{frames.jsonl,report.json}``
    0.25 Hz object boxes from the head-camera instance segmentation. Each record:
    ``{"frame_index", "bbox": {obj: [x1,y1,x2,y2]}, "raw_bbox", "filtering"}``, coordinates
    normalized to the 720x720 head image. ``bbox`` (component-filtered) is used, ``raw_bbox``
    and ``filtering`` are dropped. Episodes whose report is incomplete, not ``packable`` or
    has ``length_2025 != length_2026`` are skipped (the 2025 masks would be misaligned).
* ``behavior_2d_trace_00_99/task-XXXX/frames/chunk-*/file-*.parquet``
    30 Hz end-effector position already projected into the head camera
    (``observation.rgb.zed_link_camera_0``, 720x720, fx = fy = 306, cx = cy = 360):
    ``observation.ee_pose_2d.{uv,visb}_{left,right}``, uv normalized with origin top-left and
    quantized to 1/1024 (the <loc> grid Trace2DCoTBuilder emits), ``uv`` null when not
    ``visb``. Visibility is in-frustum only, no occlusion test. Its ``episode_index`` is the
    per-task ``demo_index_within_task`` (0..199), not the dataset's global episode_index (they
    coincide only for task 0); the namespace is detected per task. The export's own
    ``2d_trace_index`` points into a string table that is not shipped and is ignored.
* ``annotations/task-XXXX/episode_<raw_id>.json`` ``skill_annotation``
    Subtask text per skill segment, rendered with the same templates the belief-graph
    pipeline uses (bgdata.operators.TEXT via bgdata.cot_targets.subtask_text), so the text is
    identical with or without belief-graph labels. Frames after the last segment: "done".

Frames where neither hand is visible get no trace label (Trace2DCoTBuilder.can_handle would
reject them anyway).

Output (per episode, keyed by raw_episode_id like the replay extractor)
-----------------------------------------------------------------------
    <out>/episode_<raw_id>_cot_strings.json   de-duplicated payload strings
    <out>/episode_<raw_id>_cot_index.jsonl    {"frame_index", "bbox_index", "2d_trace_index"}
    <out>/subtask_targets.parquet             (episode=raw_id, frame, subtask) at segment starts
    <out>/coverage.json                       per-task / per-episode counts and skip reasons

then:
    python tools/merge_b1k_cot_labels.py --dataset <subset> --cot-sidecar-dir <out> \\
        --subtask-targets <out>/subtask_targets.parquet --sidecar-episode-key raw

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

BBOX_DIR = "behavior_bbox_2025_025fps_v2"
TRACE_DIR = "behavior_2d_trace_00_99"
ARMS = ("left", "right")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_episodes(root: Path, tasks: List[int]) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    cols = ["episode_index", "task_index", "length", "raw_episode_id", "demo_index_within_task",
            "data/chunk_index", "data/file_index"]
    eps = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    return eps[eps["task_index"].isin(tasks)].sort_values("episode_index").reset_index(drop=True)


#: Subtask templates for skills bgdata.operators has no operator for (bgdata would emit the
#: bare skill name). {0}, {1}, ... are the annotation's object_id slots, in order.
EXTRA_SUBTASK_TEXT = {
    "open lid": "open lid {0}",
    "close lid": "close lid {0}",
    "hand over": "hand over {0} from {1} to {2} hand",
    "turn to": "turn {0} to {1}",
    "push to": "push {0} to {1}",
    "place on next to": "place {0} on {1} next to {2}",
    "chop": "chop {1} with {0}",
    "sweep off": "sweep {0} off {1}",
}


def _skill_segments(ann: dict) -> List[dict]:
    """bgdata.inventory.segments, but a skill annotated over several frame intervals
    (``frame_duration: [[s0, e0], [s1, e1]]``) becomes one segment per interval."""
    out = []
    for sk in ann["skill_annotation"]:
        durations = sk["frame_duration"]
        if durations and isinstance(durations[0], list):
            intervals = [tuple(d) for d in durations]
        else:
            intervals = [tuple(durations)]
        objects = list(sk["object_id"][0]) if sk["object_id"] else []
        # an object slot can hold a list (several objects handled together)
        objects = [" and ".join(o) if isinstance(o, list) else o for o in objects]
        for start, end in intervals:
            out.append(dict(
                skill=sk["skill_description"][0],
                objects=objects,
                memory_prefix=(sk["memory_prefix"] or [""])[0],
                start=int(start),
                end=int(end),
            ))
    return sorted(out, key=lambda seg: seg["start"])


def _subtask_text(seg: dict, ops: dict) -> str:
    from bgdata.cot_targets import subtask_text
    from bgdata.operators import SKILL_TO_OP

    if seg["skill"] in SKILL_TO_OP:  # the belief-graph pipeline's own text, verbatim
        return subtask_text(seg, ops)
    template = EXTRA_SUBTASK_TEXT.get(seg["skill"])
    objs = seg["objects"]
    if template is not None:
        try:
            return template.format(*objs)
        except IndexError:
            pass
    return " ".join([seg["skill"], *objs])


def subtask_rows(root: Path, task: int, raw_id: int, length: int) -> Tuple[List[dict], Optional[str]]:
    """Segment-start rows (episode, frame, subtask) from the skill annotation."""
    from bgdata.operators import OP_ARGS, TEXT

    path = root / "annotations" / f"task-{task:04d}" / f"episode_{raw_id:08d}.json"
    if not path.exists():
        return [], "missing"
    ops = {op: {"args": args, "text": TEXT[op]} for op, args in OP_ARGS.items()}
    rows: List[dict] = []
    last_end = 0
    for seg in _skill_segments(json.loads(path.read_text())):
        if not 0 <= seg["start"] < length:
            continue
        text = _subtask_text(seg, ops)
        if rows and rows[-1]["frame"] == seg["start"]:
            rows[-1]["subtask"] = text  # two segments starting on one frame: the later wins
        else:
            rows.append({"episode": raw_id, "frame": seg["start"], "subtask": text})
        last_end = max(last_end, seg["end"])
    if rows and last_end < length:
        rows.append({"episode": raw_id, "frame": last_end, "subtask": "done"})
    return rows, None if rows else "empty"


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


class TraceTable:
    """Lazily loaded per-task 2D traces: key -> (frame_index, {arm: uv (N,2)}, {arm: visb (N,)})."""

    def __init__(self, trace_root: Path):
        self.root = trace_root
        self.cache: Dict[int, dict] = {}

    def _files(self, task: int) -> List[Path]:
        return sorted((self.root / f"task-{task:04d}" / "frames").glob("chunk-*/*.parquet"))

    def available(self, task: int) -> bool:
        return bool(self._files(task))

    def _load(self, task: int) -> dict:
        if task not in self.cache:
            cols = ["episode_index", "frame_index"] + [
                f"observation.ee_pose_2d.{k}_{arm}" for arm in ARMS for k in ("uv", "visb")
            ]
            df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in self._files(task)],
                           ignore_index=True)
            per_ep = {}
            for ep, g in df.groupby("episode_index", sort=False):
                g = g.sort_values("frame_index")
                uv, vis = {}, {}
                for arm in ARMS:
                    vis[arm] = g[f"observation.ee_pose_2d.visb_{arm}"].to_numpy(dtype=bool)
                    uv[arm] = np.array(
                        [x if v else (np.nan, np.nan)
                         for x, v in zip(g[f"observation.ee_pose_2d.uv_{arm}"], vis[arm])],
                        dtype=np.float64,
                    ).reshape(-1, 2)
                per_ep[int(ep)] = (g["frame_index"].to_numpy(), uv, vis)
            self.cache = {task: per_ep}  # one task at a time keeps memory bounded
        return self.cache[task]

    def resolve_keys(self, task: int, task_eps: pd.DataFrame) -> Dict[int, int]:
        """global episode_index -> the key this task's trace files use for it."""
        have = set(self._load(task))
        global_ids = [int(e) for e in task_eps["episode_index"]]
        local_ids = [int(e) for e in task_eps["demo_index_within_task"]]
        if have <= set(global_ids) and have & set(global_ids):
            return dict(zip(global_ids, global_ids))
        if have <= set(local_ids):
            return dict(zip(global_ids, local_ids))
        raise ValueError(
            f"task {task}: 2D-trace episode_index values {sorted(have)[:5]}... match neither the "
            f"global episode_index nor demo_index_within_task of this task"
        )

    def get(self, task: int, key: int):
        return self._load(task).get(key)


def trace_payload(uv: Dict[str, np.ndarray], vis: Dict[str, np.ndarray], k: int) -> Optional[dict]:
    if not any(vis[a][k] for a in ARMS):
        return None  # Trace2DCoTBuilder.can_handle rejects frames with no visible hand anyway
    out = {}
    for arm in ARMS:
        ok = bool(vis[arm][k]) and not np.isnan(uv[arm][k]).any()
        # values are k/1024 already; 6 decimals keeps them exact on the <loc> grid
        out[f"uv_{arm}"] = [round(float(uv[arm][k, 0]), 6), round(float(uv[arm][k, 1]), 6)] if ok else None
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
    ap.add_argument("--trace-dir", type=Path, help=f"default <demos-root>/{TRACE_DIR}")
    ap.add_argument("--no-subtask", action="store_true")
    ap.add_argument("--no-bbox", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--bbox-key", default="bbox", choices=["bbox", "raw_bbox"])
    ap.add_argument("--max-episodes", type=int, default=None, help="per task, for smoke tests")
    args = ap.parse_args()

    root = args.demos_root
    bbox_root = args.bbox_dir or root / BBOX_DIR
    traces = TraceTable(args.trace_dir or root / TRACE_DIR)
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # bgdata
    eps = read_episodes(root, args.tasks)
    coverage: Dict[str, dict] = {}
    subtasks: List[dict] = []
    for task, task_eps in eps.groupby("task_index"):
        task = int(task)
        if args.max_episodes:
            task_eps = task_eps.head(args.max_episodes)
        do_trace = not args.no_trace and traces.available(task)
        if not args.no_trace and not do_trace:
            log(f"task {task}: no 2D traces under {traces.root}/task-{task:04d} -> no trace labels")
        trace_key = traces.resolve_keys(task, task_eps) if do_trace else {}

        stats = defaultdict(int)
        skips: Dict[str, List[int]] = defaultdict(list)
        for row in task_eps.itertuples():
            ep, raw_id, length = int(row.episode_index), int(row.raw_episode_id), int(row.length)
            records: Dict[int, Dict[str, dict]] = defaultdict(dict)

            if not args.no_subtask:
                rows, reason = subtask_rows(root, task, raw_id, length)
                if reason:
                    skips[f"subtask:{reason}"].append(raw_id)
                else:
                    subtasks.extend(rows)
                    stats["subtask_episodes"] += 1

            if not args.no_bbox:
                boxes, reason = read_bbox(bbox_root, task, raw_id, length, args.bbox_key)
                if reason:
                    skips[f"bbox:{reason}"].append(raw_id)
                for frame, payload in boxes.items():
                    records[frame]["bbox"] = payload
                stats["bbox_frames"] += len(boxes)

            if do_trace:
                trace = traces.get(task, trace_key[ep])
                if trace is None:
                    skips["trace:missing"].append(raw_id)
                elif not np.array_equal(trace[0], np.arange(length)):
                    skips["trace:length_mismatch"].append(raw_id)
                else:
                    _, uv, vis = trace
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
            f"task {task}: subtask {stats['subtask_episodes']}/{len(task_eps)} episodes; "
            f"bbox/trace {stats['episodes_written']}/{len(task_eps)} episodes; "
            f"bbox {stats['bbox_frames']} frames ({100 * stats['bbox_frames'] / f:.2f}%), "
            f"trace {stats['trace_frames']} ({100 * stats['trace_frames'] / f:.1f}%); "
            f"skipped {', '.join(f'{k}={len(v)}' for k, v in skips.items()) or 'none'}"
        )

    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if subtasks:
        pd.DataFrame(subtasks).to_parquet(args.out / "subtask_targets.parquet", index=False)
        log(f"subtask_targets.parquet: {len(subtasks)} segment rows")
    if not subtasks and not any(c.get("episodes_written") for c in coverage.values()):
        log("no episode produced any label")
        return 1
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
