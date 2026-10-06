"""Skill-balanced start-frame sampler for skill-annotated BEHAVIOR-1K LeRobot v3 datasets (G0.5 training).

Port of openpi/scripts/skill_balance/skill_sampler.py (same budget levels and stride/phase indices, same seeds)
with a rotating segment-level remainder (below), plus the G0.5 glue: segment table from the training dataset's
own meta/episodes + annotations, mapping of segment frames to train-dataset positions (val episodes excluded),
and a DDP sampler with the ResumableDistributedSampler API.

The sampler only decides WHICH START FRAMES become training samples. A sample still reads `action_horizon`
consecutive 30 Hz actions from its start frame, so action chunks are never subsampled. Normalization statistics
are untouched. Prompts/CoT come from per-frame labels, so overlapping annotations do not change a sample.

Segments
--------
A segment is one annotated skill interval of one demo: task, skill, length L, a stable id
"<raw_episode_id>:<skill_idx>:<part>" (nested multi-interval entries: one segment per interval), and
`global_start` = index of its first frame in the index space the sampler emits (LeRobot frame index offline,
train-dataset position after map_segments_to_train_positions). Offset k maps to `global_start + k`.
By default (gap_to_next) unannotated frames between two skills are prepended to the next skill's segment;
unannotated episode heads and tails stay outside every segment and are never sampled.

Budget (stage 1, once)          n_s   = frames of skill s;  w_s = n_s^a / sum n^a
                                B_s   = B w_s;  B_st = B_s / |T_s|
  the skill and task levels are integerised with largest-remainder rounding (children sum exactly to the
  parent; ties broken by a seeded permutation so no task id is systematically favoured).
  Segment level, rotating: a cell (s, t) with budget b over C = |C(s,t)| segments gives every segment the base
  n_i = b // C (the stride/phase stream below) plus, in local epoch e, one extra sample to r = b mod C segments:
  the cell's segments in a seeded order, positions (e r + j) mod C, j < r. Over ceil(C / r) local epochs every
  segment gets its extra, so cells with more segments than budget (n_i = 0) are still all sampled -- in turn,
  at one uniformly random frame each -- instead of leaving some segments at zero forever. Every local epoch
  still has exactly B samples and each cell exactly b. Extra offsets are i.i.d. uniform, RNG(seed, e).
Stride (stage 2, once, base n)  s_nat = max(1, ceil(L/n));  s = min(s_nat, s_max);  capped = s_nat > s_max
                                W = n s;  G = min(max s_nat, s_max)   (global epoch = G local epochs)
Indices (stage 3, local epoch e) g = e // G, p = e % G, per segment rng = RNG(seed, g, segment_id):
  n >= L (s = 1):  r ~ U[0, L);  idx = (r + k) mod L, k < n          (cycling; frames repeat, by design)
  capped:          w ~ U[0, L-W]; perm = rng.permutation(s); idx = w + perm[p % s] + k s   (inside [w, w+W))
  otherwise:       perm = rng.permutation(s); idx = (perm[p % s] + k s) mod L
  the base indices of all segments plus the rotating extras of local epoch e are shuffled with RNG(seed, e).
Base stream: non-capped segments are fully covered once per global epoch; capped ones a fraction W/L per global
epoch (the window moves randomly between global epochs). Segments with base n = 0 get only the rotating
extras (one random frame every ceil(C / r) local epochs).

Stream, DDP and resume
----------------------
The local epochs concatenate into one infinite stream that is a pure function of (segments, B, s_max, alpha,
seed). Optimizer step t with global batch GB = batch_size * world_size consumes stream[t GB, (t+1) GB); rank r
takes the contiguous slice [t GB + r bs, t GB + (r+1) bs). So the global batch sequence does not depend on how
it is split across ranks, and rank streams are disjoint. A DataLoader "epoch" is `steps_per_epoch` per-rank
batches (default ceil(B / GB), about one local epoch); the stream position of (epoch, batch_idx) is
(epoch * steps_per_epoch + batch_idx) * GB, so finetune.py's (epoch, action_batch_idx) checkpoint state resumes
the exact stream. Changing B, s_max, alpha, seed, batch_size, world size or steps_per_epoch breaks that.
"""
from __future__ import annotations

import glob
import json
import math
import os
import zlib
from collections import Counter
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)

_TAG_BUDGET, _TAG_SEGMENT, _TAG_SHUFFLE, _TAG_EXTRA = 0xB0D6E7, 0x5E6, 0x5F1E, 0xE7A


def largest_remainder(weights, total: int, rng: np.random.Generator) -> np.ndarray:
    """Integers proportional to `weights` that sum exactly to `total` (Hamilton / largest remainder)."""
    w = np.asarray(weights, dtype=np.float64)
    if total <= 0 or len(w) == 0:
        return np.zeros(len(w), dtype=np.int64)
    q = w / w.sum() * total
    base = np.floor(q).astype(np.int64)
    rem = int(total - base.sum())
    if rem:
        frac = q - base
        tie = rng.permutation(len(w))                      # seeded tie-break
        order = np.lexsort((tie, -frac))                    # largest fraction first
        base[order[:rem]] += 1
    return base


class SkillBalancedSampler:
    """Single-process infinite stream of start indices (see module docstring)."""

    def __init__(self, segments: pd.DataFrame, B: int, s_max: int, alpha: float = 0.5, seed: int = 0,
                 start_epoch: int = 0, start_pos: int = 0):
        need = {"task", "skill", "segment_id", "length", "global_start"}
        missing = need - set(segments.columns)
        if missing:
            raise ValueError(f"segment table lacks {sorted(missing)}")
        self.seg = segments.sort_values("segment_id", kind="stable").reset_index(drop=True)
        self.B, self.s_max, self.alpha, self.seed = int(B), int(s_max), float(alpha), int(seed)
        self.start_epoch, self.start_pos = int(start_epoch), int(start_pos)
        self._allocate()
        self._memo_g, self._memo = None, None
        self._epoch_cache_e, self._epoch_cache = None, None

    # ---------------- stage 1 + 2 ----------------
    def _allocate(self):
        seg = self.seg
        n_s = seg.groupby("skill").length.sum().sort_index()
        self.skills = list(n_s.index)
        w = n_s.to_numpy(np.float64) ** self.alpha
        self.w_s = pd.Series(w / w.sum(), index=n_s.index)
        rng = lambda *k: np.random.default_rng([self.seed, _TAG_BUDGET, *k])
        B_s = pd.Series(largest_remainder(self.w_s.to_numpy(), self.B, rng(0)), index=n_s.index)
        n = np.zeros(len(seg), dtype=np.int64)
        cell_budget, rot_rows, rot_r = {}, [], []
        for si, s in enumerate(self.skills):
            tasks = sorted(seg.loc[seg.skill == s, "task"].unique())
            B_st = largest_remainder(np.ones(len(tasks)), int(B_s[s]), rng(1, si))
            for t, b in zip(tasks, B_st):
                cell_budget[(s, t)] = int(b)
                rows = np.flatnonzero((seg.skill == s).to_numpy() & (seg.task == t).to_numpy())
                q, r = divmod(int(b), len(rows))
                n[rows] = q
                if r:                                           # rotating remainder, seeded segment order
                    rot_rows.append(rows[rng(2, si, int(t)).permutation(len(rows))])
                    rot_r.append(r)
        self.B_s, self.cell_budget = B_s, cell_budget
        self._rot_rows, self._rot_r = rot_rows, np.array(rot_r, dtype=np.int64)
        self.rot_period = max((math.ceil(len(c) / r) for c, r in zip(rot_rows, rot_r)), default=1)
        L = seg.length.to_numpy(np.int64)
        s_nat = np.maximum(1, np.ceil(L / np.maximum(n, 1))).astype(np.int64)
        s_nat[n == 0] = 1
        self.n, self.L, self.s_nat = n, L, s_nat
        self.s = np.minimum(s_nat, self.s_max)
        self.capped = s_nat > self.s_max
        self.W = self.n * self.s
        active = n > 0
        self.G = int(min(s_nat[active].max(), self.s_max)) if active.any() else 1
        self.gstart = seg.global_start.to_numpy(np.int64)
        self._crc = np.array([zlib.crc32(str(x).encode()) for x in seg.segment_id], dtype=np.int64)
        self._active = np.flatnonzero(active)
        assert int(n.sum() + self._rot_r.sum()) == self.B, (int(n.sum() + self._rot_r.sum()), self.B)

    # ---------------- stage 3 ----------------
    def _draws(self, g: int):
        """Per-segment random draws for global epoch g (window start / cycle start, phase permutation)."""
        if self._memo_g != g:
            memo = {}
            for i in self._active:
                r = np.random.default_rng([self.seed, _TAG_SEGMENT, g, int(self._crc[i])])
                L, W, s = int(self.L[i]), int(self.W[i]), int(self.s[i])
                if self.n[i] >= L:                            # s == 1, cycle from a random start
                    memo[i] = (int(r.integers(0, L)), None)
                elif self.capped[i]:
                    w = int(r.integers(0, L - W + 1))
                    memo[i] = (w, r.permutation(s))
                else:
                    memo[i] = (0, r.permutation(s))
            self._memo_g, self._memo = g, memo
        return self._memo

    def segment_local_indices(self, e: int, i: int) -> np.ndarray:
        g, p = divmod(e, self.G)
        a, perm = self._draws(g)[i]
        n, L, s = int(self.n[i]), int(self.L[i]), int(self.s[i])
        k = np.arange(n, dtype=np.int64)
        if perm is None:
            return (a + k) % L
        off = a + int(perm[p % s])
        if self.capped[i]:
            return off + k * s
        return (off + k * s) % L

    def extra_rows(self, e: int) -> np.ndarray:
        """Segment rows that get the rotating extra sample in local epoch e (one entry per extra)."""
        out = [c[(e * r + np.arange(r)) % len(c)] for c, r in zip(self._rot_rows, self._rot_r)]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def _build_local_epoch_with_rows(self, e: int):
        """(start indices, segment rows) of local epoch e, both length B. Pure function of (seed, e)."""
        frames = np.empty(self.B, dtype=np.int64)
        rows = np.empty(self.B, dtype=np.int64)
        pos = 0
        for i in self._active:
            idx = self.segment_local_indices(e, i)
            frames[pos:pos + len(idx)] = self.gstart[i] + idx
            rows[pos:pos + len(idx)] = i
            pos += len(idx)
        xr = self.extra_rows(e)
        u = np.random.default_rng([self.seed, _TAG_EXTRA, e]).random(len(xr))
        frames[pos:] = self.gstart[xr] + np.minimum((u * self.L[xr]).astype(np.int64), self.L[xr] - 1)
        rows[pos:] = xr
        perm = np.random.default_rng([self.seed, _TAG_SHUFFLE, e]).permutation(self.B)
        return frames[perm], rows[perm]

    def _build_local_epoch(self, e: int) -> np.ndarray:
        if self._epoch_cache_e != e:
            self._epoch_cache_e, self._epoch_cache = e, self._build_local_epoch_with_rows(e)[0]
        return self._epoch_cache

    def stream_slice(self, a: int, b: int) -> np.ndarray:
        """stream[a:b] of the infinite concatenation of local epochs."""
        out, pos = [], int(a)
        while pos < b:
            e, off = divmod(pos, self.B)
            take = min(b - pos, self.B - off)
            out.append(self._build_local_epoch(e)[off:off + take])
            pos += take
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    # ---------------- stage 4 ----------------
    def __iter__(self):
        e, pos = self.start_epoch, self.start_pos
        while True:
            yield from self._build_local_epoch(e)[pos:].tolist()
            pos = 0
            e += 1

    @classmethod
    def from_step(cls, segments, B, s_max, step: int, batch_size: int, **kw):
        consumed = int(step) * int(batch_size)
        return cls(segments, B, s_max, start_epoch=consumed // int(B), start_pos=consumed % int(B), **kw)

    def summary(self) -> str:
        act = self.n > 0
        share = self.B_s / self.B
        nat = self.seg.groupby("skill").length.sum() / self.L.sum()
        amp = share / nat
        return (f"segments {len(self.seg)} ({int(act.sum())} with base budget, {int((~act).sum())} rotating-only; "
                f"{int(self._rot_r.sum()):,} rotating extras / local epoch, period {self.rot_period} local epochs), "
                f"skills {len(self.skills)}, tasks {self.seg.task.nunique()}, frames {int(self.L.sum()):,}; "
                f"B {self.B:,} s_max {self.s_max} alpha {self.alpha} -> G {self.G}, capped segments "
                f"{int(self.capped.sum())}, amplification {amp.min():.2f}x ({amp.idxmin()}) .. "
                f"{amp.max():.2f}x ({amp.idxmax()})")


# ---------------- G0.5 glue ----------------
def _absorb_gaps(ep_rows: list, length: int) -> int:
    """Extend the next skill's segment back over each interior unannotated gap of one episode, in place.

    A gap is a run of frames covered by no segment that has annotated frames on both sides (episode head and
    tail runs are left out). The segment starting where the gap ends (first by segment id if several do)
    absorbs it. Returns the number of frames absorbed."""
    cov = np.zeros(length, dtype=bool)
    for r in ep_rows:
        cov[r["start"]:r["end"]] = True
    edge = np.diff(np.r_[0, (~cov).astype(np.int8), 0])
    absorbed = 0
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if a == 0 or b == length:
            continue
        nxt = min((r for r in ep_rows if r["start"] == b), key=lambda r: r["segment_id"])
        nxt["start"] = int(a)
        absorbed += int(b - a)
    return absorbed


def load_lerobot_segments(dataset_dir: str, annotations_root: str, start_margin: int = 0,
                          min_skill_frames: int = 0, gap_to_next: bool = True) -> pd.DataFrame:
    """Segment table of a LeRobot v3 B1K dataset (full challenge demos or a subset built from them).

    Needs meta/episodes columns episode_index, task_index, raw_episode_id, length, dataset_from_index and the
    annotation JSONs at <annotations_root>/annotations/task-XXXX/episode_<raw_id:08d>.json (e.g. the challenge
    demos root). annotations_root must not be dataset_dir: G0.5's LeRobot loader reads <dataset_dir>/annotations
    as subtask_annotations.jsonl and fails on the skill JSONs. global_start is the LeRobot frame index
    (dataset_from_index + start).
    start_margin: forbid starts in the last `start_margin` frames of a segment (segments shorter keep their first).
    min_skill_frames: drop skills with fewer total frames.
    gap_to_next: unannotated frames between two skills (mostly the approach into the next skill, e.g. reaching
        for the microwave button between "close door" and "turn on switch") become the start of the next skill's
        segment; `ann_start` keeps the annotated start. Only the sampler's segments change: G0.5's per-frame
        subtask label of these frames is still the previous skill (forward-filled).
    """
    if not annotations_root:
        raise ValueError("skill sampler: set skill_sampler.annotations_root (dir containing annotations/task-XXXX)")
    root = annotations_root
    files = sorted(glob.glob(f"{dataset_dir}/meta/episodes/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no meta/episodes parquet under {dataset_dir}")
    cols = ["episode_index", "task_index", "raw_episode_id", "length", "dataset_from_index"]
    meta = pd.concat([pd.read_parquet(f) for f in files])
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise ValueError(f"{dataset_dir}/meta/episodes lacks {missing}; the skill sampler needs them")
    meta = meta.sort_values("episode_index")
    rows, issues = [], Counter()
    for ep in meta.itertuples():
        f = f"{root}/annotations/task-{int(ep.task_index):04d}/episode_{int(ep.raw_episode_id):08d}.json"
        if not os.path.exists(f):
            issues["missing_annotation"] += 1
            continue
        a = json.load(open(f))
        ep_rows = []
        for s in a["skill_annotation"]:
            fd = s["frame_duration"]
            intervals = fd if isinstance(fd[0], list) else [fd]
            for part, (st, en) in enumerate(intervals):
                if en > ep.length:
                    issues["segment_past_episode_end"] += 1
                    en = int(ep.length)
                if en <= st:
                    issues["empty_segment"] += 1
                    continue
                ep_rows.append(dict(task=int(ep.task_index), skill=s["skill_description"][0],
                                    segment_id=f"{int(ep.raw_episode_id)}:{s['skill_idx']}:{part}",
                                    raw_episode_id=int(ep.raw_episode_id), episode_index=int(ep.episode_index),
                                    start=int(st), ann_start=int(st), end=int(en)))
        if gap_to_next and ep_rows:
            issues["gap_frames_to_next_skill"] += _absorb_gaps(ep_rows, int(ep.length))
        for r in ep_rows:
            r["length"] = r["end"] - r["start"]
            r["global_start"] = int(ep.dataset_from_index) + r["start"]
        rows.extend(ep_rows)
    if issues["missing_annotation"] == len(meta):
        raise FileNotFoundError(f"no annotation JSON found under {root}/annotations for any of {len(meta)} episodes")
    if issues:
        logger.info(f"skill sampler segment table: {dict(issues)}")
    seg = pd.DataFrame(rows)
    if start_margin > 0:
        seg = seg.assign(length=np.maximum(1, seg.length - start_margin))
    if min_skill_frames > 0:
        seg = seg[seg.groupby("skill").length.transform("sum") >= min_skill_frames]
    return seg.reset_index(drop=True)


def train_frame_indices(train_dataset):
    """(frames, dataset_dir): LeRobot frame index of every train-dataset position (position i -> frame) and the
    LeRobot dir they index, for the plain G0.5 setup.

    Supported: BaseLerobotDataset over one LeRobot dir, alone or as the only group of an unweighted
    MixtureLerobotDataset. Anything else (several groups, weighted undersampling, overfit mode, rank-sharded
    datasets) maps positions differently and raises instead of guessing.
    """
    from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset

    ds = train_dataset
    if isinstance(ds, MixtureLerobotDataset):
        if len(ds.datasets) != 1:
            raise ValueError(f"skill sampler: mixture with {len(ds.datasets)} dataset groups is not supported")
        if hasattr(ds, "_overfit_len") or int(ds.effective_lengths[0]) != int(ds.actual_lengths[0]):
            raise ValueError("skill sampler: weighted/overfit mixture sampling changes the index map")
        ds = ds.datasets[0]
    if getattr(ds, "is_rank_sharded", False) or hasattr(ds, "_overfit_indices"):
        raise ValueError("skill sampler: rank-sharded or overfit datasets are not supported")
    inner = getattr(getattr(ds, "multi_dataset", None), "_datasets", None)
    if inner is not None and len(inner) != 1:
        raise ValueError(f"skill sampler: dataset group with {len(inner)} LeRobot dirs is not supported")
    if getattr(ds, "_sample_indices", None) is not None:
        frames = np.asarray(ds._sample_indices, dtype=np.int64)
    else:
        frames = np.arange(ds._start_idx, ds._end_idx, dtype=np.int64)
    if len(frames) != len(train_dataset):
        raise ValueError(f"skill sampler: index map has {len(frames)} entries, dataset {len(train_dataset)}")
    return frames, str(ds.dataset_dirs[0])


def map_segments_to_train_positions(seg: pd.DataFrame, frames: np.ndarray) -> pd.DataFrame:
    """Keep segments lying wholly in the train split; global_start becomes the train-dataset position."""
    if len(frames) and not (np.diff(frames) > 0).all():
        raise ValueError("skill sampler: train frame indices are not strictly increasing")
    gs = seg.global_start.to_numpy(np.int64)
    last = gs + seg.length.to_numpy(np.int64) - 1
    p0 = np.clip(np.searchsorted(frames, gs), 0, max(len(frames) - 1, 0))
    p1 = p0 + (last - gs)
    ok = (frames[p0] == gs) & (p1 < len(frames))
    ok[ok] &= frames[p1[ok]] == last[ok]                   # contiguous inside the split
    out = seg[ok].assign(frame_start=gs[ok], global_start=p0[ok])
    return out.reset_index(drop=True)


class DistributedSkillBalancedSampler(Sampler[int]):
    """Per-rank view of the SkillBalancedSampler stream with the ResumableDistributedSampler API
    (set_epoch / set_start_batch / __len__); see "Stream, DDP and resume" in the module docstring."""

    def __init__(self, segments: pd.DataFrame, B: int, s_max: int, batch_size: int, num_replicas: int, rank: int,
                 alpha: float = 0.5, seed: int = 0, steps_per_epoch: Optional[int] = None):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} not in [0, {num_replicas})")
        self.core = SkillBalancedSampler(segments, B, s_max, alpha=alpha, seed=seed)
        self.batch_size, self.num_replicas, self.rank = int(batch_size), int(num_replicas), int(rank)
        self.global_batch = self.batch_size * self.num_replicas
        self.steps_per_epoch = int(steps_per_epoch or math.ceil(self.core.B / self.global_batch))
        self.epoch, self.start_batch_idx = 0, 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = int(start_batch_idx)

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def stream_position(self, epoch: int, batch_idx: int) -> int:
        return (int(epoch) * self.steps_per_epoch + int(batch_idx)) * self.global_batch

    def __iter__(self) -> Iterator[int]:
        bs, r = self.batch_size, self.rank
        for t in range(self.start_batch_idx, self.steps_per_epoch):
            a = self.stream_position(self.epoch, t) + r * bs
            yield from self.core.stream_slice(a, a + bs).tolist()


def build_skill_sampler(cfg_sampler, train_dataset, batch_size: int, num_replicas: int, rank: int,
                        seed: int) -> DistributedSkillBalancedSampler:
    """Skill sampler for finetune.py from the `skill_sampler` config block."""
    frames, dataset_dir = train_frame_indices(train_dataset)
    seg = load_lerobot_segments(dataset_dir, cfg_sampler.get("annotations_root"),
                                start_margin=int(cfg_sampler.get("start_margin", 0) or 0),
                                min_skill_frames=int(cfg_sampler.get("min_skill_frames", 0) or 0),
                                gap_to_next=bool(cfg_sampler.get("gap_to_next", True)))
    n_all = len(seg)
    seg = map_segments_to_train_positions(seg, frames)
    sampler = DistributedSkillBalancedSampler(
        seg, int(cfg_sampler.B), int(cfg_sampler.s_max), batch_size=batch_size, num_replicas=num_replicas,
        rank=rank, alpha=float(cfg_sampler.get("alpha", 0.5)), seed=seed,
        steps_per_epoch=cfg_sampler.get("steps_per_epoch"))
    logger.info(f"[skill_sampler] {dataset_dir}: {len(seg)} / {n_all} segments in the train split; "
                f"{sampler.core.summary()}; global batch {sampler.global_batch} -> "
                f"{sampler.steps_per_epoch} steps per dataloader epoch, "
                f"{sampler.core.B * sampler.core.G / sampler.global_batch:,.0f} steps per global epoch")
    if (sampler.core.n == 0).any():
        logger.info(f"[skill_sampler] {int((sampler.core.n == 0).sum())} segments have no base budget and are "
                    f"sampled only by the rotating extras (each once per {sampler.core.rot_period} local epochs)")
    return sampler
