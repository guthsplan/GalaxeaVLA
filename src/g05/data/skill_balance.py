"""Skill-balanced frame sampling for the training split (BEHAVIOR skill annotations).

Each training frame gets the skill of the annotated segment it falls in (episode meta
`annotation_path` -> annotations/task-XXXX/episode_YYYYYYYY.json, `skill_annotation`). The
sampling mass of skill k is

    p_k  ∝  F_k ** alpha * weight_k          (F_k: training frames labelled k)

spread uniformly over that skill's frames. alpha = 1 with no weights is the natural
frame distribution; alpha = 0 gives every skill the same mass; `weight_k` multiplies on top
(e.g. "move to=0.3"). `keep_skills` zeroes every other skill (skill-expert runs, see
g05.data.skill_groups). Frames outside every segment (leading frames before the first
segment, gaps, a tail past the last one) take the nearest segment's skill.

Sample `idx` of `n` is drawn by stratified inverse-CDF over the cumulative mass: mass
(idx + U) / n * total. With uniform weights this is exactly `train_frame_stride` jitter
(one random frame per window), so an epoch has the same length either way.

The mass is stored per run (one contiguous segment), not per frame, so the full
100-task set (~10^8 frames) costs a few MB.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence, Set, Union

import numpy as np

from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)

UNANNOTATED = "<unannotated>"


def parse_skill_weights(spec: Union[None, str, Dict[str, float]]) -> Dict[str, float]:
    """`"move to=0.3,pick up from=1.5"` or a mapping -> {skill: weight}."""
    if spec is None or spec == "":
        return {}
    if isinstance(spec, str):
        out = {}
        for item in spec.split(","):
            if not item.strip():
                continue
            name, sep, value = item.rpartition("=")
            if not sep or not name.strip():
                raise ValueError(f"skill_weights entry must be '<skill>=<weight>', got {item!r}")
            out[name.strip()] = float(value)
    else:
        out = {str(k).strip(): float(v) for k, v in dict(spec).items()}
    bad = {k: v for k, v in out.items() if v < 0}
    if bad:
        raise ValueError(f"skill_weights must be >= 0, got {bad}")
    return out


def _episode_segments(ann: dict, length: int) -> List[tuple]:
    """(start, end, skill) runs covering [0, length) in order, from one annotation file."""
    raw = []
    for sk in ann.get("skill_annotation", []):
        skill = (sk.get("skill_description") or [UNANNOTATED])[0]
        fd = sk["frame_duration"]
        # a segment interrupted by another one is stored as a list of intervals
        for b, e in fd if isinstance(fd[0], (list, tuple)) else [fd]:
            b, e = max(0, int(b)), min(length, int(e))
            if e > b:
                raw.append((b, e, skill))
    if not raw:
        return [(0, length, UNANNOTATED)]
    raw.sort()
    # Clip overlaps, then hand each uncovered stretch to its neighbour (leading frames to
    # the first segment, everything else to the segment before it).
    runs = []
    cursor = 0
    for b, e, skill in raw:
        b = max(b, cursor)
        if e <= b:
            continue
        if not runs:
            b = 0
        elif b > cursor:
            pb, _, ps = runs[-1]
            runs[-1] = (pb, b, ps)
        runs.append((b, e, skill))
        cursor = e
    pb, _, ps = runs[-1]
    runs[-1] = (pb, length, ps)
    return runs


def _annotation_file(ds, rel, annotation_root: Optional[str]) -> Optional[Path]:
    roots = [Path(ds.root)] + ([Path(annotation_root)] if annotation_root else [])
    return next((r / rel for r in roots if rel and (r / rel).is_file()), None)


def skill_val_episodes(
    datasets: Sequence,
    episode_from: Sequence[int],
    episode_to: Sequence[int],
    keep_skills: Collection[str],
    per_task: int,
    annotation_root: Optional[str] = None,
) -> Set[int]:
    """Validation episodes of a skill-expert run: per task, the last `per_task` episodes (in
    episode order) that contain a kept skill. Global episode indices over `datasets`.

    The episode-tail split (`val_set_proportion`) takes the last episodes of the whole set,
    i.e. of one task when the episodes are grouped by task, which may hold none of a group's
    skills (b1k_5task: the tail is cook_hot_dogs, the tool skills are in chopping_wood).
    """
    keep = set(keep_skills)
    by_task: Dict[tuple, List[int]] = {}
    ep = 0
    for di, ds in enumerate(datasets):
        meta_eps = ds.meta.episodes
        tasks = meta_eps["task_index"] if "task_index" in meta_eps.column_names else [None] * len(meta_eps)
        for rel, task in zip(meta_eps["annotation_path"], tasks):
            path = _annotation_file(ds, rel, annotation_root)
            if path is not None:
                with open(path) as f:
                    segs = _episode_segments(json.load(f), int(episode_to[ep]) - int(episode_from[ep]))
                if any(skill in keep for _, _, skill in segs):
                    by_task.setdefault((di, task), []).append(ep)
            ep += 1
    if not by_task:
        raise ValueError(f"none of the skills {sorted(keep)} occur in the dataset")
    val = set()
    for eps in by_task.values():
        k = min(per_task, len(eps) - 1)  # keep at least one episode of each task for training
        if k > 0:
            val.update(eps[-k:])
    if not val:
        raise ValueError(f"skill_val_episodes_per_task: every task has a single episode with {sorted(keep)}")
    logger.info(
        f"[skill-balance] per-task validation split: {len(val)} episodes over {len(by_task)} tasks "
        f"with {sorted(keep)} (<= {per_task} per task)"
    )
    return val


class SkillBalancedIndex:
    """Maps sample index -> global frame index with per-skill sampling mass."""

    def __init__(
        self,
        datasets: Sequence,
        episode_from: Sequence[int],
        episode_to: Sequence[int],
        start_idx: int,
        end_idx: int,
        num_samples: int,
        alpha: float = 1.0,
        skill_weights: Optional[Dict[str, float]] = None,
        annotation_root: Optional[str] = None,
        keep_skills: Optional[Sequence[str]] = None,
        episodes: Optional[Collection[int]] = None,
    ):
        """`episodes`: global episode indices to draw from (default: every episode in range)."""
        if alpha < 0:
            raise ValueError(f"skill_balance_alpha must be >= 0, got {alpha}")
        skill_weights = dict(skill_weights or {})
        self.num_samples = int(num_samples)

        run_lo, run_hi, run_skill = [], [], []
        missing = 0
        ep = 0
        for ds in datasets:
            meta_eps = ds.meta.episodes
            if "annotation_path" not in meta_eps.column_names:
                raise KeyError(
                    f"{ds.root}: episode meta has no 'annotation_path' column; skill-balanced "
                    "sampling needs the BEHAVIOR challenge episode meta."
                )
            for rel in meta_eps["annotation_path"]:
                g0, g1 = int(episode_from[ep]), int(episode_to[ep])
                ep += 1
                if g1 <= start_idx or g0 >= end_idx or (episodes is not None and ep - 1 not in episodes):
                    continue
                path = _annotation_file(ds, rel, annotation_root)
                if path is None:
                    missing += 1
                    segs = [(0, g1 - g0, UNANNOTATED)]
                else:
                    with open(path) as f:
                        segs = _episode_segments(json.load(f), g1 - g0)
                for b, e, skill in segs:
                    lo, hi = max(g0 + b, start_idx), min(g0 + e, end_idx)
                    if hi > lo:
                        run_lo.append(lo)
                        run_hi.append(hi)
                        run_skill.append(skill)
        if not run_lo:
            raise RuntimeError("skill-balanced sampling found no frames in the training range")
        if missing:
            logger.warning(
                f"[skill-balance] {missing} episodes have no annotation file (looked in the dataset "
                f"dir and skill_annotation_root={annotation_root}); they count as '{UNANNOTATED}'."
            )

        lo = np.asarray(run_lo, dtype=np.int64)
        hi = np.asarray(run_hi, dtype=np.int64)
        n_frames = hi - lo
        skills = sorted(set(run_skill))
        sid = np.asarray([skills.index(s) for s in run_skill], dtype=np.int64)
        frames_per_skill = np.bincount(sid, weights=n_frames, minlength=len(skills))

        unknown = sorted(set(skill_weights) - set(skills))
        if unknown:
            logger.warning(f"[skill-balance] skill_weights names absent from the training split: {unknown}")
        mult = np.asarray([skill_weights.get(s, 1.0) for s in skills], dtype=np.float64)
        if keep_skills is not None:
            # skill-expert runs (g05.data.skill_groups): every other skill gets no mass
            keep = set(keep_skills)
            if not keep & set(skills):
                raise ValueError(
                    f"none of the skills {sorted(keep)} occur in this split (the validation split is "
                    "the last episodes of the set; skill_val_episodes_per_task=N takes N per task)"
                )
            mult *= np.asarray([s in keep for s in skills], dtype=np.float64)
        mass = frames_per_skill ** alpha * mult
        if mass.sum() <= 0:
            raise ValueError("skill_weights zero out every skill")
        mass /= mass.sum()
        per_frame = mass / frames_per_skill  # sampling density of one frame of each skill

        keep = per_frame[sid] > 0
        self._lo = lo[keep]
        self._hi = hi[keep]
        self._sid = sid[keep]
        self.skills = skills
        self._density = per_frame[sid][keep]
        self._cum = np.cumsum(n_frames[keep] * self._density)
        self._total = float(self._cum[-1])

        share = frames_per_skill / frames_per_skill.sum()
        lines = [f"{'skill':<22}{'frames':>10}{'data %':>9}{'sample %':>10}{'x':>7}"]
        for k in np.argsort(-mass):
            lines.append(
                f"{skills[k]:<22}{int(frames_per_skill[k]):>10}{share[k] * 100:>9.2f}"
                f"{mass[k] * 100:>10.2f}{mass[k] / share[k]:>7.2f}"
            )
        logger.info(
            f"[skill-balance] alpha={alpha}, weights={skill_weights or {}}, "
            f"keep={sorted(keep_skills) if keep_skills is not None else 'all'}, {len(lo)} runs, "
            f"{self.num_samples} samples/epoch\n" + "\n".join(lines)
        )
        self.summary = {s: (int(f), float(m)) for s, f, m in zip(skills, frames_per_skill, mass)}

    def __call__(self, idx: int) -> int:
        m = (idx + np.random.random()) / self.num_samples * self._total
        r = min(int(np.searchsorted(self._cum, m, side="right")), len(self._cum) - 1)
        prev = self._cum[r - 1] if r > 0 else 0.0
        return min(int(self._lo[r] + (m - prev) / self._density[r]), int(self._hi[r]) - 1)

    def skill_of(self, frame: int) -> str:
        """Skill label of a global frame index in the training range."""
        r = int(np.searchsorted(self._lo, frame, side="right")) - 1
        return self.skills[self._sid[r]]
