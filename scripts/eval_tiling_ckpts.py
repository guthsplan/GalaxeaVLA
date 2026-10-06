"""Offline tiling eval (scripts/utils/tiling_eval.py) of saved checkpoints, one GPU per process.

Scores each checkpoint on the same fixed val chunks as the in-training eval_tiling and writes
<out>/<name>/eval_tiling/step_0000000.csv (per chunk, incl. per-part fm_l1 / arfb_l1) and <out>/<name>/summary.json.
The model is built from the task yaml (as serve_g05_b1k.py does) and the val split from the overrides, so
checkpoints from different runs on the same dataset are scored on identical chunks.

  B1K_SUBSET_DIR=<dataset> PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/eval_tiling_ckpts.py \
      --ckpt a.pt b.pt --name v2_s10k v2_s20k --out <dir> --stats <dataset_stats.json> \
      --annotations-root <dir with annotations/> [--override data.val_split_by_task=true ...]
"""
import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accelerate import Accelerator  # noqa: E402

from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint  # noqa: E402
from g05.utils.checkpoint.ckpt_utils import load_config_from_task_yaml  # noqa: E402
from g05.utils.data.data_utils import collate_fn_pad_sequences  # noqa: E402
from g05.utils.data.normalizer import load_dataset_stats_from_json  # noqa: E402
from g05.utils.data.processor_utils import build_processors, instantiate_dataset  # noqa: E402
from utils.metric import resolve_parts_meta  # noqa: E402
from utils.tiling_eval import TilingEvaluator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--name", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True, help="dataset_stats.json the runs were trained with")
    ap.add_argument("--annotations-root", required=True)
    ap.add_argument("--task-yaml", default=str(Path(__file__).resolve().parent.parent / "configs/task/behavior_cot_lora_bg.yaml"))
    ap.add_argument("--hybrid-parts", nargs="*", default=["left_control", "right_control"])
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--override", nargs="*", default=["data.val_split_by_task=true", "data.val_set_proportion=0.0249"])
    args = ap.parse_args()
    assert len(args.ckpt) == len(args.name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    accelerator = Accelerator(mixed_precision="bf16")
    stats = load_dataset_stats_from_json(args.stats)
    tiling_cfg = {"stride": args.stride, "hybrid_ar_parts": list(args.hybrid_parts), "gap_to_next": True,
                  "min_cell_chunks": 3}
    eval_ds = processor = None
    for ckpt, name in zip(args.ckpt, args.name):
        cfg = load_config_from_task_yaml(args.task_yaml, ckpt, list(args.override))
        if eval_ds is None:
            eval_ds = instantiate_dataset(cfg, is_training_set=False)
            processor = build_processors(cfg)
            processor.set_normalizer_from_stats(stats)
            processor.eval()
            eval_ds.set_processor(processor)
            collate = (partial(collate_fn_pad_sequences, padding_input_id=processor.pad_token_id)
                       if cfg.model.get("collate_fn") else None)
        model = load_model_from_checkpoint(cfg.model.model_arch, ckpt, device=str(accelerator.device),
                                           extra_prefixes=["normalizer."], eval_mode=True)
        if hasattr(model, "action_tokenizer"):
            model.action_tokenizer.to(accelerator.device)
        out_dir = Path(args.out) / name
        ev = TilingEvaluator(eval_ds, tiling_cfg, annotations_root=args.annotations_root, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=collate, worker_init_fn=None,
                             processor=processor, parts_meta=resolve_parts_meta(processor=processor),
                             output_dir=out_dir, rank=0, world_size=1, seed=args.seed)
        res = ev.evaluate(model, accelerator, step=0)
        res["ckpt"] = str(ckpt)
        (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
