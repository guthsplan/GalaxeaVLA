#!/usr/bin/env python3
"""Materialize BEHAVIOR CoT labels into a LeRobot v3 subset, offline.

    raw/subset LeRobot data
        + bbox / trace sidecars   (behavior1k-2026-solution, `cot` branch)
        + belief-graph targets    (behavior1k-2026-solution, `bg` branch)
                |
                v
    materialized BEHAVIOR CoT LeRobot dataset  ->  normal GalaxeaVLA loader

Why offline
-----------
The alternative — scanning a parquet or JSON sidecar inside `__getitem__` — puts
random filesystem IO and duplicated parsing into every DataLoader worker, on
every sample. This writes the labels once into the dataset itself, following the
existing G0.5 indexed-string convention (`<field>_index` int column in the data
parquet -> row of `meta/tasks.parquet` holding the string), so the loader resolves
each one with a single dict hit.

Pipeline
--------
    read  -> align (causal) -> dedupe into a string table -> write -> validate

Only the **read** stage depends on the exact sidecar file formats. It is isolated
in `read_bg_targets` / `read_cot_sidecars` below so the rest of the tool is
schema-independent. `--synthetic` substitutes a generator for that stage, which
exercises alignment, writing and validation without the sidecars present.

Alignment
---------
Every label is attached to observation frame `t`, which is paired with the action
chunk beginning at `t`. Sparse sources are resolved **causally**::

    source_frame = max{ s : s <= t, s in the same episode }

`source_frame > t` is never selected — a belief state from the future would leak
the answer. Episode boundaries are never crossed. Two modes:

* ``snapshot`` (bg_known, bg_belief, bg_observe) — the row is an observation at
  instant `s`, so
  reusing it at `t > s` is genuinely stale. `--max-staleness-frames` bounds it;
  frames past the bound get no label rather than a wrong one.
* ``segment`` (subtask, bg_delta, bg_effect) — the row *defines* the segment it opens,
  which runs to the next annotated frame (or the episode end). Filling that
  segment is exact, not stale, so the staleness bound does not apply.

`meta/cot_alignment.jsonl` records `frame`, `bg_source_frame` and
`staleness_frames` per labelled frame for auditing.

Usage
-----
    python tools/merge_b1k_cot_labels.py --dataset DIR \\
        --bg-targets /path/to/cot_targets.parquet \\
        --cot-sidecar-dir /path/to/cot_out

    # no sidecars needed — generates fixture labels for the test gates
    python tools/merge_b1k_cot_labels.py --dataset DIR --synthetic --episodes 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Field table
# ---------------------------------------------------------------------------
# (label name, parquet column, alignment mode). Must stay in step with
# `_INDEXED_ANNOTATION_FIELDS` in src/g05/data/lerobot/lerobot_dataset_v3.py.
#
# `atomic_task_index` is reused for Subtask rather than adding a parallel
# `subtask_index`: it is the field every upstream subtask builder already reads.
SNAPSHOT = "snapshot"
SEGMENT = "segment"

LABEL_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("subtask", "atomic_task_index", SEGMENT),
    ("bg_known", "bg_known_index", SNAPSHOT),
    ("bg_belief", "bg_belief_index", SNAPSHOT),
    ("bg_delta", "bg_delta_index", SEGMENT),
    ("bg_effect", "bg_effect_index", SEGMENT),
    ("bg_observe", "bg_observe_index", SNAPSHOT),
    ("bbox", "bbox_index", SEGMENT),
    ("trace_2d", "2d_trace_index", SEGMENT),
)
#: Labels that are dense/frame-level and must not be zero-order-held at all.
FRAME_EXACT = {"bbox", "trace_2d"}

#: Prefixes the offline extractors may already have written. The builder owns the
#: prefix (see behavior_cot_builders.with_prefix), so strip it on the way in and
#: store the bare payload. Normalizing on both sides keeps the invariant even if
#: only one of them ran.
# Storage convention, matching tools/build_bg_fields.py exactly:
#   atomic_task  -> stored BARE; SubtaskCoTBuilder / BeliefGraphSubtaskCoTBuilder
#                   unconditionally prepend "Subtask: ".
#   bg_belief / bg_delta / bg_effect / bg_observe
#                -> stored WITH their prefix. BeliefGraph*CoTBuilder calls
#                   _strip_prefix then re-adds, so the prefix still appears exactly
#                   once either way — but keeping it matters for a second reason:
#                   BaseSamplesBuilder._INVALID_STRINGS rejects the bare string
#                   "none", so a stored bare "none" would make the candidate
#                   inapplicable and silently drop every satisfied-goal /
#                   no-known-operator frame from supervision. "Delta: none" passes.
#   bg_known     -> stored as "Remaining: ... | Known: ...", re-serialized by
#                   BeliefGraphBuilder._format_bg_known; nothing is stripped.
KNOWN_PREFIXES = {
    "subtask": "Subtask",
}


def strip_prefix(payload: str, prefix: Optional[str]) -> str:
    """Remove any number of leading ``"<prefix>:"`` occurrences."""
    text = "" if payload is None else str(payload)
    if not prefix:
        return text.strip()
    needle = prefix.lower() + ":"
    while True:
        stripped = text.lstrip()
        if stripped.lower().startswith(needle):
            text = stripped[len(needle) :]
            continue
        return stripped


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Dataset geometry
# ---------------------------------------------------------------------------


@dataclass
class DatasetGeometry:
    """Per-episode frame ranges of the target LeRobot dataset."""

    root: Path
    fps: float
    total_frames: int
    #: episode_index -> (dataset_from_index, dataset_to_index) — a half-open
    #: global row range. Row `dataset_from_index + k` is frame_index `k`.
    ranges: Dict[int, Tuple[int, int]]
    #: raw_episode_id (the demo_id the sidecars and cot_targets key on, e.g. 450010)
    #: -> episode_index. Empty when the episodes table has no raw_episode_id column.
    raw_to_episode: Dict[int, int] = field(default_factory=dict)

    def length(self, episode: int) -> int:
        lo, hi = self.ranges[episode]
        return hi - lo

    def global_row(self, episode: int, frame: int) -> int:
        lo, hi = self.ranges[episode]
        if not 0 <= frame < hi - lo:
            raise IndexError(
                f"frame {frame} out of range for episode {episode} "
                f"(length {hi - lo}); labels must not address frames beyond the episode"
            )
        return lo + frame


def read_geometry(root: Path) -> DatasetGeometry:
    info = json.loads((root / "meta" / "info.json").read_text())
    ep_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not ep_files:
        sys.exit(f"no episode metadata under {root/'meta'/'episodes'}")
    eps = pd.concat([pq.read_table(f).to_pandas() for f in ep_files], ignore_index=True)
    ranges = {
        int(r.episode_index): (int(r.dataset_from_index), int(r.dataset_to_index))
        for r in eps.itertuples()
    }
    raw_to_episode = (
        {int(r.raw_episode_id): int(r.episode_index) for r in eps.itertuples()}
        if "raw_episode_id" in eps.columns else {}
    )
    return DatasetGeometry(
        root=root,
        fps=float(info["fps"]),
        total_frames=int(info["total_frames"]),
        ranges=ranges,
        raw_to_episode=raw_to_episode,
    )


# ---------------------------------------------------------------------------
# Stage 1: read (the only schema-dependent stage)
# ---------------------------------------------------------------------------
# A source row is (episode, frame, {label: payload}). Everything downstream works
# on that shape alone.

SourceRows = Dict[int, Dict[int, Dict[str, str]]]  # episode -> frame -> label -> payload


def _insert(rows: SourceRows, episode: int, frame: int, label: str, payload: str) -> None:
    """Insert one payload, rejecting a duplicate (episode, frame, label) key."""
    bucket = rows.setdefault(int(episode), {}).setdefault(int(frame), {})
    if label in bucket:
        raise ValueError(
            f"duplicate label key (episode={episode}, frame={frame}, label={label}): "
            f"{bucket[label]!r} vs {payload!r}. Each (episode, frame) may carry at most "
            f"one payload per label; deduplicate the source before merging."
        )
    bucket[label] = payload


def read_bg_targets(path: Path, rows: SourceRows, skip: Tuple[str, ...] = ()) -> None:
    """Read `cot_targets.parquet` from the solution repo's `bg` branch.

    Expected columns: ``episode``, ``frame``, and any of ``subtask`` / ``belief``
    / ``delta`` / ``effect`` / ``observe``. ``task`` is carried for provenance and
    ignored here. Column names are matched case-insensitively and a few obvious
    aliases (``episode_index``, ``frame_index``) are accepted, because the exact
    spelling is owned by the other repository. Anything else fails loudly rather
    than being guessed at.
    """
    df = pd.read_parquet(path)
    lower = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in lower:
                return lower[n]
        return None

    ep_col = pick("episode", "episode_index", "ep")
    fr_col = pick("frame", "frame_index", "step")
    if ep_col is None or fr_col is None:
        sys.exit(
            f"{path}: cannot find episode/frame columns. Got {list(df.columns)}; "
            f"expected 'episode' and 'frame' (or *_index)."
        )

    # cot_targets.parquet (bgdata) spells the columns without the bg_ prefix; the
    # canonical loader fields carry it. tools/build_bg_fields.py is the preferred
    # path (it also derives bg_known and bg_observe); this mapping keeps the raw
    # cot_targets.parquet readable directly.
    _SOURCE_TO_LABEL = {
        "subtask": "subtask",
        "belief": "bg_belief",
        "delta": "bg_delta",
        "effect": "bg_effect",
        "observe": "bg_observe",
        "bg_known": "bg_known",
    }
    label_cols = {
        _SOURCE_TO_LABEL[src]: lower[src]
        for src in _SOURCE_TO_LABEL
        if src in lower and _SOURCE_TO_LABEL[src] not in skip
    }
    if skip:
        log(f"  {path.name}: ignoring {sorted(skip)} (supplied by another source)")
    if not label_cols:
        sys.exit(
            f"{path}: none of subtask/belief/delta/effect/observe present. "
            f"Got {list(df.columns)}."
        )
    log(f"  bg targets: {len(df)} rows, labels={sorted(label_cols)}")

    for rec in df.itertuples(index=False):
        d = rec._asdict()
        episode, frame = int(d[ep_col]), int(d[fr_col])
        for label, col in label_cols.items():
            payload = d[col]
            if payload is None or (isinstance(payload, float) and np.isnan(payload)):
                continue
            payload = strip_prefix(str(payload), KNOWN_PREFIXES.get(label))
            if not payload:
                continue
            _insert(rows, episode, frame, label, payload)


def read_cot_sidecars(directory: Path, rows: SourceRows) -> None:
    """Read the `cot` branch's per-episode bbox / trace sidecars.

    Expected per episode::

        episode_<N>_cot_strings.json   string table
        episode_<N>_cot_index.jsonl    one record per frame, with `bbox_index`
                                       and/or `2d_trace_index` into that table

    Each index record is documented to correspond to LeRobot ``frame_index k``
    paired with action ``k``; the record's own frame field is used when present
    and the line position is the fallback.
    """
    index_files = sorted(directory.glob("episode_*_cot_index.jsonl"))
    if not index_files:
        sys.exit(f"no episode_*_cot_index.jsonl under {directory}")

    n_records = 0
    for index_path in index_files:
        stem = index_path.name[len("episode_") : -len("_cot_index.jsonl")]
        try:
            episode = int(stem)
        except ValueError:
            sys.exit(f"cannot parse an episode index out of {index_path.name}")

        strings_path = directory / f"episode_{stem}_cot_strings.json"
        if not strings_path.exists():
            sys.exit(f"{index_path.name} has no matching {strings_path.name}")
        raw_strings = json.loads(strings_path.read_text())
        if isinstance(raw_strings, dict):
            strings = {int(k): v for k, v in raw_strings.items()}
        elif isinstance(raw_strings, list):
            strings = dict(enumerate(raw_strings))
        else:
            sys.exit(f"{strings_path}: expected a list or an object, got {type(raw_strings).__name__}")

        for position, line in enumerate(index_path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame = int(rec.get("frame", rec.get("frame_index", position)))
            for key, label in (("bbox_index", "bbox"), ("2d_trace_index", "trace_2d")):
                if key not in rec or rec[key] is None:
                    continue
                sidx = int(rec[key])
                if sidx < 0:
                    continue
                if sidx not in strings:
                    raise KeyError(
                        f"{index_path.name} frame {frame}: {key}={sidx} is not in "
                        f"{strings_path.name} ({len(strings)} entries)"
                    )
                payload = strings[sidx]
                if not isinstance(payload, str):
                    payload = json.dumps(payload, separators=(",", ":"))
                _validate_payload(label, payload, episode, frame)
                _insert(rows, episode, frame, label, payload)
                n_records += 1
    log(f"  cot sidecars: {len(index_files)} episodes, {n_records} payloads")


def _validate_payload(label: str, payload: str, episode: int, frame: int) -> None:
    """Reject malformed bbox / trace payloads at ingest, not at training time."""
    if label not in FRAME_EXACT:
        return
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"malformed {label} JSON at episode {episode} frame {frame}: {err}"
        ) from err
    if label == "bbox":
        if not isinstance(obj, dict):
            raise ValueError(f"bbox at ep {episode} frame {frame} must be an object, got {type(obj).__name__}")
        for name, box in obj.items():
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                raise ValueError(f"bbox {name!r} at ep {episode} frame {frame} must have 4 coords, got {box!r}")
            if not all(isinstance(v, (int, float)) and -0.01 <= v <= 1.01 for v in box):
                raise ValueError(
                    f"bbox {name!r} at ep {episode} frame {frame} must be normalized to [0,1], got {box!r}"
                )
    elif label == "trace_2d":
        if not isinstance(obj, dict):
            raise ValueError(f"trace_2d at ep {episode} frame {frame} must be an object, got {type(obj).__name__}")
        for side in ("left", "right"):
            uv = obj.get(f"uv_{side}")
            if uv is None:
                continue
            if not (isinstance(uv, (list, tuple)) and len(uv) == 2):
                raise ValueError(f"trace_2d uv_{side} at ep {episode} frame {frame} must be [u,v], got {uv!r}")


def synthesize(geom: DatasetGeometry, episodes: int, target_hz: float, seed: int) -> SourceRows:
    """Generate fixture labels so the test gates can run without the sidecars.

    Deliberately uneven: `delta`/`effect` are sometimes "none" (a valid target),
    bbox/trace are denser than the belief-graph fields, and some frames carry no
    label at all — so the coverage and selection-proportion gates see a realistic
    conditional mix rather than uniform full coverage.
    """
    rng = np.random.default_rng(seed)
    stride = stride_frames(geom.fps, target_hz)
    rows: SourceRows = {}
    objects = ["hotdog_207", "fridge_1", "countertop_3", "plate_88"]
    verbs = ["reach for", "grasp", "lift", "carry", "place"]

    for episode in sorted(geom.ranges)[:episodes]:
        length = geom.length(episode)
        for frame in range(0, length, stride):
            obj = objects[int(rng.integers(len(objects)))]
            verb = verbs[int(rng.integers(len(verbs)))]
            _insert(rows, episode, frame, "subtask", f"{verb} the {obj}")
            _insert(
                rows, episode, frame, "bg_belief",
                f"Belief: (inside {obj} fridge_1) {rng.random():.2f} obs | "
                f"(open fridge_1) {rng.random():.2f} prior",
            )
            _insert(
                rows, episode, frame, "bg_observe",
                f"Observe: (ontop {obj} countertop_3) 1 | (open fridge_1) {int(rng.integers(0, 2))}",
            )
            _insert(
                rows, episode, frame, "bg_delta",
                "Delta: none" if rng.random() < 0.35
                else f"Delta: (cooked ?x) {int(rng.integers(0,3))}/3 [{obj.split('_')[0]}.n.02]",
            )
            _insert(
                rows, episode, frame, "bg_effect",
                "Effect: none" if rng.random() < 0.5
                else f"Effect: (inhand {obj}) 0>1 | (inside {obj} fridge_1) 1>0",
            )
            # bg_known is the PREVIOUS snapshot's "Remaining: ... | Known: ..."
            # (build_bg_fields.py derives it from the prior 1 Hz row).
            _insert(
                rows, episode, frame, "bg_known",
                f"Remaining: (cooked ?x) 0/2 [hotdog.n.02] | "
                f"Known: (ontop {obj} countertop_3) {rng.random():.2f} obs",
            )
        # bbox / trace are frame-exact and denser (every 5th frame here).
        for frame in range(0, length, 5):
            if rng.random() < 0.15:
                continue  # some frames genuinely have no detection
            x1, y1 = rng.random() * 0.5, rng.random() * 0.5
            box = [round(x1, 3), round(y1, 3), round(x1 + 0.2, 3), round(y1 + 0.2, 3)]
            _insert(rows, episode, frame, "bbox", json.dumps({objects[0]: box}, separators=(",", ":")))
            visible = bool(rng.random() < 0.8)
            _insert(
                rows, episode, frame, "trace_2d",
                json.dumps(
                    {
                        "uv_left": [round(float(rng.random()), 3), round(float(rng.random()), 3)] if visible else None,
                        "visb_left": visible,
                        "uv_right": None,
                        "visb_right": False,
                    },
                    separators=(",", ":"),
                ),
            )
    return rows


def stride_frames(fps: float, target_hz: float) -> int:
    """Frames between sparse annotations for a requested rate.

    The BG extractor historically hardcoded ``every=10`` and called it "1 Hz".
    That only holds at 10 fps; the BEHAVIOR demos run at **30 fps**, where
    ``every=10`` is 3 Hz. Deriving the stride from the dataset's own fps removes
    the assumption.
    """
    stride = int(round(fps / target_hz))
    if stride < 1:
        raise ValueError(f"target_hz={target_hz} exceeds the data rate fps={fps}")
    return stride


# ---------------------------------------------------------------------------
# Stage 2: causal alignment
# ---------------------------------------------------------------------------


@dataclass
class AlignmentStats:
    labelled_frames: int = 0
    per_label: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    staleness: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    dropped_stale: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dropped_trailing: int = 0


def align_episode(
    length: int,
    frame_payloads: Dict[int, Dict[str, str]],
    max_staleness: Optional[int],
    stats: AlignmentStats,
    audit: List[dict],
    episode: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Causally expand one episode's sparse rows to every frame.

    Returns ``(payload_index_by_label, source_frame_by_label)``, both arrays of
    length `length`, using -1 for "no label at this frame". The payload index is
    an index into `ordered` (the per-label list of distinct payload strings in
    source order), resolved to a global string-table index by the caller.
    """
    # bbox_replay emits N+1 records for N actions: "the final observation (state N)
    # has no matching action and is emitted as one extra record". Drop exactly that
    # record; anything further out is still a hard error.
    if length in frame_payloads:
        frame_payloads = {f: v for f, v in frame_payloads.items() if f != length}
        stats.dropped_trailing += 1
    annotated = sorted(frame_payloads)
    for frame in annotated:
        if not 0 <= frame < length:
            raise IndexError(
                f"episode {episode}: label at frame {frame} but the episode has {length} frames. "
                f"Labels must address an observation frame that is paired with an action; "
                f"a trailing observation without a matching action is not a training sample."
            )

    payload_of: Dict[str, List[str]] = defaultdict(list)
    out_payload: Dict[str, np.ndarray] = {}
    out_source: Dict[str, np.ndarray] = {}

    for label, _column, mode in LABEL_FIELDS:
        frames_with = [f for f in annotated if label in frame_payloads[f]]
        idx = np.full(length, -1, dtype=np.int64)
        src = np.full(length, -1, dtype=np.int64)
        if not frames_with:
            out_payload[label], out_source[label] = idx, src
            continue

        seen: Dict[str, int] = {}
        for f in frames_with:
            text = frame_payloads[f][label]
            if text not in seen:
                seen[text] = len(payload_of[label])
                payload_of[label].append(text)

        exact_only = label in FRAME_EXACT
        cursor = -1  # index into frames_with of the most recent source <= t
        for t in range(length):
            while cursor + 1 < len(frames_with) and frames_with[cursor + 1] <= t:
                cursor += 1
            if cursor < 0:
                continue  # no source at or before t -> leave unlabelled (never look ahead)
            s = frames_with[cursor]
            assert s <= t, "causal alignment must never select a future source frame"
            if exact_only and s != t:
                continue  # frame-exact labels are not held forward
            stale = t - s
            if mode == SNAPSHOT and max_staleness is not None and stale > max_staleness:
                stats.dropped_stale[label] += 1
                continue
            idx[t] = seen[frame_payloads[s][label]]
            src[t] = s
            stats.per_label[label] += 1
            if stale:
                stats.staleness[label].append(stale)

        out_payload[label], out_source[label] = idx, src

    # Audit rows for the belief-graph snapshot fields, which are the ones whose
    # staleness can actually mislead.
    for t in range(length):
        entry = {"episode": episode, "frame": t}
        interesting = False
        for label in ("bg_belief", "bg_known"):
            s = int(out_source[label][t])
            if s >= 0:
                entry["bg_source_frame"] = s
                entry["staleness_frames"] = t - s
                interesting = True
        if interesting:
            audit.append(entry)
            stats.labelled_frames += 1

    return out_payload, out_source, payload_of


# ---------------------------------------------------------------------------
# Stage 3-5: string table, write, validate
# ---------------------------------------------------------------------------


def merge(
    dataset: Path,
    rows: SourceRows,
    max_staleness: Optional[int],
    dry_run: bool,
    episode_key: str = "auto",
) -> None:
    geom = read_geometry(dataset)
    log(f"dataset : {dataset}")
    log(f"          fps={geom.fps} frames={geom.total_frames} episodes={len(geom.ranges)}")

    # Both label sources key episodes by raw_episode_id (the demo_id, e.g. 450010),
    # a different namespace from LeRobot's episode_index. Remap through
    # meta/episodes when the keys are not already episode indices.
    # `auto` remaps only when some key is not an episode index, which is ambiguous when
    # raw ids happen to be small (task 0: 10, 20, ...); `raw` / `index` say it explicitly.
    if episode_key == "raw" and not geom.raw_to_episode:
        sys.exit("--sidecar-episode-key raw: the dataset's meta/episodes has no raw_episode_id column")
    remap = episode_key == "raw" or (
        episode_key == "auto" and not set(rows) <= set(geom.ranges) and geom.raw_to_episode
    )
    if rows and remap:
        remapped: SourceRows = {}
        unmapped = []
        for key, frames in rows.items():
            ep = geom.raw_to_episode.get(int(key))
            if ep is None:
                unmapped.append(int(key)); continue
            if ep in remapped:
                raise ValueError(f"two source episodes map to episode_index {ep}")
            remapped[ep] = frames
        log(f"remapped {len(remapped)} source episode(s) raw_episode_id -> episode_index"
            + (f"; {len(unmapped)} not in this dataset (skipped): {unmapped[:5]}" if unmapped else ""))
        rows = remapped
    unknown = sorted(set(rows) - set(geom.ranges))
    if unknown:
        raise KeyError(
            f"labels reference episodes absent from the dataset: {unknown[:10]}"
            f"{' ...' if len(unknown) > 10 else ''}. Labels from a different episode "
            f"indexing (e.g. the pre-subset numbering) must be remapped first."
        )

    # ---- existing string table ----
    tasks_path = dataset / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    string_to_index: Dict[str, int] = {
        str(t): int(i) for t, i in zip(tasks.index, tasks["task_index"].to_numpy())
    }
    next_index = (max(string_to_index.values()) + 1) if string_to_index else 0

    # ---- per-episode alignment ----
    stats = AlignmentStats()
    audit: List[dict] = []
    columns: Dict[str, np.ndarray] = {
        column: np.full(geom.total_frames, -1, dtype=np.int64) for _, column, _ in LABEL_FIELDS
    }

    for episode in sorted(rows):
        lo, _hi = geom.ranges[episode]
        length = geom.length(episode)
        payload_idx, _source, payload_of = align_episode(
            length, rows[episode], max_staleness, stats, audit, episode
        )
        for label, column, _mode in LABEL_FIELDS:
            local = payload_idx[label]
            hit = local >= 0
            if not hit.any():
                continue
            # local payload ordinal -> global string-table index, deduped globally
            table_index = np.empty(len(payload_of[label]), dtype=np.int64)
            for ordinal, text in enumerate(payload_of[label]):
                existing = string_to_index.get(text)
                if existing is None:
                    existing = next_index
                    string_to_index[text] = existing
                    next_index += 1
                table_index[ordinal] = existing
            columns[column][lo : lo + length][hit] = table_index[local[hit]]

    log("")
    if stats.dropped_trailing:
        log(f"dropped {stats.dropped_trailing} trailing action-less sidecar record(s) (frame == episode length)")
    log("alignment")
    for label, column, mode in LABEL_FIELDS:
        n = stats.per_label[label]
        if not n:
            log(f"  {label:<10} {column:<20} {'(no source rows)':>28}")
            continue
        stale = stats.staleness[label]
        pct = 100.0 * n / geom.total_frames
        detail = f"mean stale {np.mean(stale):.1f} max {max(stale)}" if stale else "frame-exact"
        dropped = stats.dropped_stale[label]
        drop_note = f", dropped {dropped} over --max-staleness-frames" if dropped else ""
        log(f"  {label:<10} {column:<20} {n:>9} frames ({pct:5.2f}%)  [{mode}] {detail}{drop_note}")

    if dry_run:
        log("\n--dry-run: nothing written")
        return

    # ---- write ----
    active = {c for c in columns if (columns[c] >= 0).any()}
    if not active:
        sys.exit("no labels aligned to any frame; refusing to write empty annotation columns")

    files = sorted((dataset / "data").glob("*/*.parquet"))
    written = 0
    for path in files:
        table = pq.read_table(path)
        index = table.column("index").to_numpy()
        changed = False
        for column in sorted(active):
            values = columns[column][index]
            if column in table.column_names:
                table = table.set_column(
                    table.schema.get_field_index(column), column, pa.array(values, type=pa.int64())
                )
            else:
                table = table.append_column(column, pa.array(values, type=pa.int64()))
            changed = True
        if changed:
            pq.write_table(table, path, compression="zstd")
            written += table.num_rows
    log(f"\nwrote {sorted(active)} into {len(files)} parquet files ({written} rows)")

    # ---- register the columns in info.json ----
    # Mandatory, not cosmetic: LeRobotDatasetV3.load_hf_dataset builds its column
    # list from `info["features"]`, so a column present in the parquet but absent
    # here is silently never loaded, and every CoT candidate quietly becomes
    # inapplicable. That failure mode looks exactly like "the model ignores CoT".
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for column in sorted(active):
        info["features"][column] = {"dtype": "int64", "shape": [1], "names": None}
    info["cot_annotation_columns"] = sorted(active)
    info_path.write_text(json.dumps(info, indent=4))
    log(f"registered {len(active)} annotation columns in meta/info.json features")

    # ---- string table ----
    ordered = sorted(string_to_index.items(), key=lambda kv: kv[1])
    new_tasks = pd.DataFrame(
        {"task_index": [i for _t, i in ordered]},
        index=pd.Index([t for t, _i in ordered], name=tasks.index.name or "task"),
    )
    shutil.copyfile(tasks_path, tasks_path.with_suffix(".parquet.bak"))
    new_tasks.to_parquet(tasks_path)
    log(f"string table: {len(tasks)} -> {len(new_tasks)} entries (backup at {tasks_path.name}.bak)")

    audit_path = dataset / "meta" / "cot_alignment.jsonl"
    with open(audit_path, "w") as fh:
        for entry in audit:
            fh.write(json.dumps(entry) + "\n")
    log(f"audit: {len(audit)} rows -> {audit_path.relative_to(dataset)}")

    validate(dataset, geom, active, string_to_index)


def validate(
    dataset: Path,
    geom: DatasetGeometry,
    active: Iterable[str],
    string_to_index: Dict[str, int],
) -> None:
    log("\nvalidating ...")
    active = sorted(active)
    table = pq.ParquetDataset(
        [str(p) for p in sorted((dataset / "data").glob("*/*.parquet"))]
    ).read(columns=["index", "episode_index", "frame_index"] + active)
    index = table.column("index").to_numpy()
    assert np.array_equal(index, np.arange(len(index))), "index column is not 0..N-1"

    known = set(string_to_index.values())
    reread = pd.read_parquet(dataset / "meta" / "tasks.parquet")
    on_disk = {int(i) for i in reread["task_index"].to_numpy()}
    assert len(on_disk) == len(reread), "task_index is not unique in meta/tasks.parquet"
    assert known <= on_disk, "string table on disk is missing indices referenced by the data"

    for column in active:
        values = table.column(column).to_numpy()
        used = set(np.unique(values[values >= 0]).tolist())
        missing = used - on_disk
        assert not missing, f"{column} references unknown string-table indices {sorted(missing)[:5]}"

    # Episode containment: every labelled row must sit inside its own episode range.
    ep = table.column("episode_index").to_numpy()
    fr = table.column("frame_index").to_numpy()
    for episode, (lo, hi) in geom.ranges.items():
        if lo >= len(ep):
            continue
        assert np.all(ep[lo:hi] == episode), f"episode {episode} rows are not contiguous"
        assert fr[lo] == 0, f"episode {episode} does not start at frame_index 0"
    # The loader only sees columns declared in info.json features.
    info = json.loads((dataset / "meta" / "info.json").read_text())
    unregistered = [c for c in active if c not in info.get("features", {})]
    assert not unregistered, (
        f"annotation columns {unregistered} are in the parquet but not in "
        f"meta/info.json features; LeRobotDatasetV3 would never load them"
    )
    log(f"OK: {len(active)} annotation columns, {len(on_disk)} string-table entries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="LeRobot v3 subset to annotate in place")
    ap.add_argument("--bg-targets", type=Path, help="cot_targets.parquet from the bg branch")
    ap.add_argument("--cot-sidecar-dir", type=Path, help="directory of episode_*_cot_*.json{,l}")
    ap.add_argument(
        "--subtask-targets",
        type=Path,
        help="parquet (episode, frame, subtask) from build_b1k_bbox_trace_sidecars.py; when given, "
        "the subtask column of --bg-targets is ignored",
    )
    ap.add_argument("--synthetic", action="store_true", help="generate fixture labels instead of reading sidecars")
    ap.add_argument("--episodes", type=int, default=8, help="--synthetic: episodes to label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--target-hz",
        type=float,
        default=1.0,
        help="sparse-annotation rate; the frame stride is round(fps / target_hz). "
        "At the BEHAVIOR demos' 30 fps, 1.0 Hz is a stride of 30 frames — the "
        "historical hardcoded every=10 was 3 Hz, not the 1 Hz it was documented as.",
    )
    ap.add_argument(
        "--max-staleness-frames",
        type=int,
        default=None,
        help="drop snapshot labels (bg_known/bg_belief/bg_observe) held forward longer than this",
    )
    ap.add_argument(
        "--sidecar-episode-key",
        choices=["auto", "raw", "index"],
        default="auto",
        help="namespace of the label sources' episode keys: raw_episode_id (e.g. 450010), "
        "the dataset's episode_index, or auto-detect",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.dataset / "meta" / "info.json").exists():
        sys.exit(f"{args.dataset} is not a LeRobot dataset (no meta/info.json)")

    rows: SourceRows = {}
    if args.synthetic:
        geom = read_geometry(args.dataset)
        stride = stride_frames(geom.fps, args.target_hz)
        log(f"synthetic labels: {args.episodes} episodes, target_hz={args.target_hz} -> stride {stride} frames @ {geom.fps} fps")
        rows = synthesize(geom, args.episodes, args.target_hz, args.seed)
    else:
        if not (args.bg_targets or args.cot_sidecar_dir or args.subtask_targets):
            sys.exit("pass --bg-targets, --subtask-targets and/or --cot-sidecar-dir (or --synthetic)")
        log("reading sources")
        if args.subtask_targets:
            read_bg_targets(args.subtask_targets, rows)
        if args.bg_targets:
            read_bg_targets(args.bg_targets, rows, skip=("subtask",) if args.subtask_targets else ())
        if args.cot_sidecar_dir:
            read_cot_sidecars(args.cot_sidecar_dir, rows)

    merge(args.dataset, rows, args.max_staleness_frames, args.dry_run, args.sidecar_episode_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
