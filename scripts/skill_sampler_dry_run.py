"""Dry run of the G0.5 skill sampler on the real train dataset (no model, no training).

Builds the train dataset from the same hydra config as finetune.py, then the DistributedSkillBalancedSampler of
every rank exactly as finetune.py does, and checks:
  [1] index map: emitted positions -> dataset frame -> episode/task of the segment; val episodes excluded
  [2] one local epoch per skill: realized vs allocated, amplification (sample share / frame share)
  [3] G, capped / rotating-only segments (all sampled within one rotation period), coverage of capped segments, max exposure of n > L segments
  [4] DDP: per step, rank r's batch is slot [r bs, (r+1) bs) of the single-process global batch
  [5] resume: set_epoch(e) + set_start_batch(k) reproduces the uninterrupted per-rank stream
  [6] steps per local / global epoch and global epochs in --train-steps

  cd GalaxeaVLA_lora && B1K_SUBSET_DIR=<dataset> PYTHONPATH=src python scripts/skill_sampler_dry_run.py \
      --annotations-root <demos> --B 100000 --s-max 8 --batch-size 2 --world-size 4 --train-steps 50000 \
      task=behavior_cot_lora_bg data.val_split_by_task=true data.val_set_proportion=0.0249
Exits non-zero if a check fails. --out-segments writes the train-split segment table (for budget sweeps).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from g05.data.skill_sampler import (DistributedSkillBalancedSampler, load_lerobot_segments,
                                    map_segments_to_train_positions, train_frame_indices)
from g05.utils.data.processor_utils import instantiate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--s-max", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None, help="default: cfg.seed")
    ap.add_argument("--batch-size", type=int, default=2, help="per GPU")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=50000)
    ap.add_argument("--resume", type=int, nargs=2, default=[1, 1234], metavar=("EPOCH", "BATCH_IDX"))
    ap.add_argument("--start-margin", type=int, default=0)
    ap.add_argument("--gap-to-next", type=int, default=1, help="1: interior gaps join the next skill")
    ap.add_argument("--annotations-root", required=True, help="dir containing annotations/task-XXXX")
    ap.add_argument("--out-segments", default=None)
    ap.add_argument("overrides", nargs="*", help="hydra overrides as for finetune.py")
    args = ap.parse_args()
    ok = True

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent.parent / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=args.overrides)
    seed = cfg.seed if args.seed is None else args.seed
    t0 = time.time()
    train_ds = instantiate_dataset(cfg, is_training_set=True)
    frames, ddir = train_frame_indices(train_ds)
    print(f"train dataset {type(train_ds).__name__}: {len(train_ds):,} positions from {ddir} ({time.time() - t0:.0f}s)")

    seg_all = load_lerobot_segments(ddir, args.annotations_root, start_margin=args.start_margin,
                                    gap_to_next=bool(args.gap_to_next))
    seg = map_segments_to_train_positions(seg_all, frames)
    eps_all, eps_tr = seg_all.episode_index.nunique(), seg.episode_index.nunique()
    print(f"segments {len(seg)} / {len(seg_all)} in the train split; episodes {eps_tr} / {eps_all}; "
          f"per task {seg.groupby('task').episode_index.nunique().to_dict()}")
    if args.out_segments:
        seg.to_parquet(args.out_segments, index=False)
        print(f"wrote {args.out_segments}")

    R, bs = args.world_size, args.batch_size
    samplers = [DistributedSkillBalancedSampler(seg, args.B, args.s_max, bs, R, r, alpha=args.alpha, seed=seed)
                for r in range(R)]
    S = samplers[0].core
    print(S.summary())

    # [1] index map against the dataset's own position -> frame map and episode boundaries
    base = train_ds.datasets[0] if hasattr(train_ds, "datasets") else train_ds
    edi_from = base.episode_data_index["from"].numpy()
    e0 = S._build_local_epoch_with_rows(0)
    pos, row = e0
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pos), size=min(20000, len(pos)), replace=False)
    fr = np.array([base._to_sample_idx(int(p)) for p in pos[pick]])
    want = S.seg.frame_start.to_numpy()[row[pick]] + (pos[pick] - S.gstart[row[pick]])
    ep_of = np.searchsorted(edi_from, fr, side="right") - 1
    ep_ok = ep_of == S.seg.episode_index.to_numpy()[row[pick]]
    inside = (pos >= S.gstart[row]) & (pos < S.gstart[row] + S.L[row])
    print(f"\n[1] {len(pick):,} sampled positions: dataset frame == segment frame {np.mean(fr == want) * 100:.2f} %, "
          f"episode match {ep_ok.mean() * 100:.2f} %; all positions inside their segment: {inside.all()}")
    val_eps = set(seg_all.episode_index) - set(seg.episode_index)
    ok &= bool((fr == want).all() and ep_ok.all() and inside.all())
    print(f"    val episodes excluded: {len(val_eps)} (none sampled: {not set(S.seg.episode_index.to_numpy()[row]) & val_eps})")

    # [2]
    got = pd.Series(S.seg.skill.to_numpy()[row]).value_counts().reindex(S.B_s.index).fillna(0).astype(int)
    frame_share = S.seg.groupby("skill").length.sum() / S.L.sum()
    rep = pd.DataFrame({"frames_%": 100 * frame_share, "tasks": S.seg.groupby("skill").task.nunique(),
                        "target": S.B * S.w_s, "allocated": S.B_s, "realized": got})
    rep["sample_%"] = 100 * rep.realized / S.B
    rep["amp"] = rep["sample_%"] / rep["frames_%"]
    print("\n[2] one local epoch, per skill")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(rep.sort_values("frames_%", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
    if (rep.realized != rep.allocated).any():
        print("    FAIL: realized != allocated"); ok = False

    # [3]
    act = S.n > 0
    short = np.flatnonzero(S.n > S.L)
    expo = max((int(np.bincount(S.segment_local_indices(0, i)).max()) for i in short), default=0)
    print(f"\n[3] G {S.G}; capped {int(S.capped.sum())} / {int(act.sum())} active; base-budget 0 (rotating-only) {int((~act).sum())}; "
          f"n > L segments {len(short)} (max exposure / local epoch {expo})")
    if S.capped.any():
        c = (S.W / S.L)[S.capped]
        print(f"    capped coverage per global epoch W/L: min {c.min():.3f} p50 {np.median(c):.3f}; "
              f"frames covered per global epoch {100 * np.minimum(S.W, S.L)[act].sum() / S.L.sum():.1f} %")
    # rotating remainder: every segment is sampled at least once within one rotation period
    seen = np.zeros(len(S.seg), dtype=bool)
    seen[S._active] = True
    for e in range(S.rot_period):
        seen[S.extra_rows(e)] = True
    print(f"    rotating extras {int(S._rot_r.sum()):,} / local epoch over {len(S._rot_rows)} cells, period "
          f"{S.rot_period} local epochs; segments sampled within one period: {seen.mean() * 100:.2f} %")
    if not seen.all():
        print("    FAIL: a segment is never sampled"); ok = False

    # [4] DDP split over two dataloader epochs (incl. the local-epoch boundaries inside them)
    spe, GB = samplers[0].steps_per_epoch, samplers[0].global_batch
    n_ep = 2
    per_rank = []
    for s in samplers:
        out = []
        for e in range(n_ep):
            s.set_epoch(e); s.set_start_batch(0)
            out.extend(iter(s))
        per_rank.append(np.array(out).reshape(-1, bs))
    stacked = np.stack(per_rank, axis=1).reshape(-1)                 # step-major, rank-minor
    ref = S.stream_slice(0, n_ep * spe * GB)
    same = np.array_equal(stacked, ref)
    print(f"\n[4] world {R} x batch {bs}: {n_ep} dataloader epochs x {spe:,} steps; ranks concatenate to the "
          f"single-process stream: {same}; len(sampler) {len(samplers[0]):,} per rank")
    ok &= same

    # [5] resume
    re, rk = args.resume
    res_ok = True
    for r, s in enumerate(samplers):
        s.set_epoch(re); s.set_start_batch(rk)
        resumed = np.array(list(iter(s)))
        full = per_rank[r].reshape(-1)[(re * spe + rk) * bs:(re + 1) * spe * bs] if re < n_ep else None
        if full is not None:
            res_ok &= np.array_equal(resumed, full)
    print(f"[5] resume at (epoch {re}, batch_idx {rk}) on every rank == uninterrupted stream: {res_ok}")
    ok &= res_ok

    # [6]
    spl, spg = S.B / GB, S.B * S.G / GB
    print(f"[6] global batch {GB}: {spl:,.1f} steps / local epoch, {spg:,.1f} steps / global epoch; "
          f"{args.train_steps:,} steps = {args.train_steps * GB:,} samples = {args.train_steps / spl:.2f} local epochs "
          f"= {args.train_steps / spg:.3f} global epochs (train frames {len(train_ds):,}: "
          f"{args.train_steps * GB / len(train_ds):.2f} uniform epochs)")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
