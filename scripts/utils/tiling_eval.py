"""Fixed, full-coverage validation for checkpoint selection ("tiling" eval).

The periodic eval (train_eval.PeriodicEvaluator) reads the next few hundred consecutive validation frames each
time, so two evals never see the same data and a "best" step mostly reflects which slice it drew. This eval
uses the same chunks every time: in every validation episode, the action chunks starting at frame_index
0, stride, 2 stride, ... With stride = action horizon (32) they tile each episode exactly once.

Each chunk is attributed to the skill segment containing its start frame (skill annotations, unannotated gaps
between skills joined to the next skill as in the skill sampler; episode head/tail chunks get skill "none").
Per chunk and action part (parts_meta) two scores are computed in normalized space, so parts with different
units are comparable: L1 = mean |pred - gt| over the part's valid dims x timesteps, and accuracy = the fraction
of them within 1/256 (the definition of rollout/fm_action_acc). Three predictions are scored:
  fm      the FM head
  ar      the AR decode (dims the decode did not emit are excluded)
  hybrid  what hybrid serving executes: parts in `hybrid_ar_parts` from the AR decode, the rest from FM, and
          FM wherever the AR decode did not emit a dim (inferencer._postprocess_single's fallback)
A chunk's score is the mean over its parts with valid dims (each part counts equally, so a 1-dim gripper weighs
as much as a 7-dim arm). skill_balanced_<pred>_<l1|acc> = mean over skills of the mean over that skill's tasks
of the mean over the (skill, task) cell's chunks; cells with fewer than `min_cell_chunks` chunks and skill
"none" are left out of it but kept in the plain mean <pred>_<l1|acc>. worst_skill_* is the worst skill (max L1,
min accuracy). The 1/256 accuracy is near 0.1 for arms of every head, so L1 is the selection metric.

The model runs in eval mode with a fixed RNG (FM noise) per rank, so an unchanged model gives the same score.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from utils.metric import rollout_and_calculate_metrics

logger = logging.getLogger(__name__)

THRESHOLD = 1 / 256
PREDS = ("hybrid", "fm", "ar")


def tiling_positions(eval_dataset, stride: int, annotations_root: str, gap_to_next: bool = True) -> pd.DataFrame:
    """Eval-dataset positions of the tiling chunks with their episode, frame_index, task and skill."""
    from g05.data.skill_sampler import load_lerobot_segments, train_frame_indices

    frames, ddir = train_frame_indices(eval_dataset)
    base = eval_dataset.datasets[0] if hasattr(eval_dataset, "datasets") else eval_dataset
    ep_from = base.episode_data_index["from"].numpy()
    ep = np.searchsorted(ep_from, frames, side="right") - 1
    fi = frames - ep_from[ep]
    keep = np.flatnonzero(fi % int(stride) == 0)
    df = pd.DataFrame({"position": keep, "frame": frames[keep], "episode_index": ep[keep], "frame_index": fi[keep]})
    seg = load_lerobot_segments(ddir, annotations_root, gap_to_next=gap_to_next)
    seg = seg[seg.episode_index.isin(set(df.episode_index))]
    task_of_ep = seg.groupby("episode_index").task.first()
    skill = np.full(len(df), "none", dtype=object)
    # overlapping annotations: the first segment by id wins, so assign in reverse id order
    for r in seg.sort_values("segment_id", ascending=False).itertuples():
        m = (df.frame.to_numpy() >= r.global_start) & (df.frame.to_numpy() < r.global_start + r.length)
        skill[m] = r.skill
    df["skill"] = skill
    df["task"] = df.episode_index.map(task_of_ep).fillna(-1).astype(int)
    return df


def _part_slices(parts_meta):
    out, off = [], 0
    for part, dim in parts_meta.items():
        out.append((part, slice(off, off + int(dim))))
        off += int(dim)
    return out


def chunk_records(preds: dict, parts_meta, hybrid_ar_parts) -> list:
    """Per-chunk L1 and accuracy (see module docstring) from rollout_and_calculate_metrics(return_preds=True)."""
    gt, fm, valid = preds["action_gt"].float(), preds["action"].float(), preds["valid_dim_mask"].bool()
    ar = preds.get("ar_action_norm")
    arv = preds.get("ar_valid_dim_mask")
    has_ar = isinstance(ar, torch.Tensor) and ar.shape == gt.shape
    if has_ar:
        ar, arv = ar.float(), (arv.bool() & valid)
    hyb = fm.clone()
    parts = _part_slices(parts_meta)
    if has_ar:
        for part, sl in parts:
            if part in hybrid_ar_parts:
                hyb[..., sl] = torch.where(arv[..., sl], ar[..., sl], fm[..., sl])
    err = {"fm": (fm - gt).abs(), "hybrid": (hyb - gt).abs()}
    masks = {"fm": valid, "hybrid": valid}
    if has_ar:
        err["ar"], masks["ar"] = (ar - gt).abs(), arv
    recs = []
    for b in range(gt.shape[0]):
        rec = {}
        for name in err:
            l1s, accs = [], []
            for part, sl in parts:
                m = masks[name][b, :, sl]
                n = int(m.sum())
                if n == 0:
                    continue
                e = err[name][b, :, sl]
                l1s.append(float(e[m].sum()) / n)
                accs.append(float(((e < THRESHOLD) & m).sum()) / n)
                if name == "hybrid":
                    rec[f"part/{part}/l1"], rec[f"part/{part}/acc"] = l1s[-1], accs[-1]
            rec[f"{name}_l1"] = float(np.mean(l1s)) if l1s else np.nan
            rec[f"{name}_acc"] = float(np.mean(accs)) if accs else np.nan
        # per-part FM and AR-with-FM-fallback L1, so any other hybrid composition can be scored from the CSV
        for part, sl in parts:
            m = valid[b, :, sl]
            n = int(m.sum())
            if n == 0:
                continue
            rec[f"part/{part}/fm_l1"] = float(err["fm"][b, :, sl][m].sum()) / n
            if has_ar:
                fb = torch.where(arv[b, :, sl], err["ar"][b, :, sl], err["fm"][b, :, sl])
                rec[f"part/{part}/arfb_l1"] = float(fb[m].sum()) / n
        recs.append(rec)
    return recs


def summarize(df: pd.DataFrame, min_cell_chunks: int) -> dict:
    out = {"eval/tiling/num_chunks": float(len(df))}
    named = df[df.skill != "none"]
    cell_n = named.groupby(["skill", "task"]).size()
    good = cell_n[cell_n >= min_cell_chunks].index
    named = named.set_index(["skill", "task"]).loc[good].reset_index() if len(good) else named.iloc[:0]
    out["eval/tiling/num_cells"] = float(len(good))
    for p in PREDS:
        for kind in ("l1", "acc"):
            col = f"{p}_{kind}"
            if col not in df or df[col].isna().all():
                continue
            out[f"eval/tiling/{col}"] = float(df[col].mean())
            if len(named):
                per_skill = named.groupby(["skill", "task"])[col].mean().groupby(level=0).mean()
                out[f"eval/tiling/skill_balanced_{col}"] = float(per_skill.mean())
                out[f"eval/tiling/worst_skill_{col}"] = float(per_skill.max() if kind == "l1" else per_skill.min())
                if p == "hybrid":
                    for s, v in per_skill.items():
                        out[f"eval/tiling/skill/{s.replace(' ', '_')}/{col}"] = float(v)
    for c in [c for c in df.columns if c.startswith("part/")]:
        part, kind = c.split("/")[1], c.split("/")[2]
        name = f"hybrid_{kind}" if kind in ("l1", "acc") else kind   # fm_l1 / arfb_l1 keep their own name
        out[f"eval/tiling/part/{part}/{name}"] = float(df[c].mean())
    return out


class TilingEvaluator:
    def __init__(self, eval_dataset, cfg_tiling, *, annotations_root, batch_size, num_workers, collate_fn,
                 worker_init_fn, processor, parts_meta, output_dir, rank, world_size, seed):
        self.cfg = cfg_tiling
        self.processor, self.parts_meta = processor, parts_meta
        self.hybrid_ar_parts = set(cfg_tiling.get("hybrid_ar_parts") or [])
        self.min_cell_chunks = int(cfg_tiling.get("min_cell_chunks", 3))
        self.rank, self.world, self.seed = int(rank), int(world_size), int(seed)
        self.out_dir = Path(output_dir) / "eval_tiling"
        self.table = tiling_positions(eval_dataset, int(cfg_tiling.get("stride", 32)), annotations_root,
                                      bool(cfg_tiling.get("gap_to_next", True)))
        self.mine = self.table.iloc[self.rank::self.world].reset_index(drop=True)
        self.loader = DataLoader(Subset(eval_dataset, self.mine.position.tolist()), batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
                                 worker_init_fn=worker_init_fn, persistent_workers=False)
        missing = self.hybrid_ar_parts - set(parts_meta or {})
        if missing:
            raise ValueError(f"eval_tiling.hybrid_ar_parts {sorted(missing)} not in action parts {list(parts_meta)}")
        t = self.table
        logger.info(f"[eval_tiling] {len(t)} chunks (stride {self.cfg.get('stride', 32)}) over "
                    f"{t.episode_index.nunique()} val episodes; skill 'none' {int((t.skill == 'none').sum())}; "
                    f"cells {t[t.skill != 'none'].groupby(['skill', 'task']).size().to_dict()}")

    def evaluate(self, model, accelerator, step: int) -> dict:
        start = time.time()
        was_training = model.training
        model.eval()
        recs = []
        with torch.random.fork_rng(devices=[accelerator.device] if accelerator.device.type == "cuda" else []):
            torch.manual_seed(self.seed + 1000 * self.rank)
            for batch in self.loader:
                _, preds = rollout_and_calculate_metrics(batch, model, accelerator, processor=self.processor,
                                                         return_preds=True, parts_meta=self.parts_meta)
                recs.extend(chunk_records(preds, self.parts_meta, self.hybrid_ar_parts))
        model.train(was_training)
        mine = pd.concat([self.mine, pd.DataFrame(recs)], axis=1)
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, mine)
            df = pd.concat(gathered, ignore_index=True)
        else:
            df = mine
        df = df.sort_values("position").reset_index(drop=True)
        out = summarize(df, self.min_cell_chunks)
        if self.rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.out_dir / f"step_{step:07d}.csv", index=False)
        out["_eval_time_sec"] = time.time() - start
        logger.info(f"[eval_tiling] step {step}: " + ", ".join(
            f"{k.split('/')[-1]}={v:.4f}" for k, v in out.items()
            if k.startswith("eval/tiling/") and "/skill/" not in k and "/part/" not in k)
            + f" ({out['_eval_time_sec']:.0f}s)")
        return out
