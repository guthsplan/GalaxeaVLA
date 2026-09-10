#!/usr/bin/env python3
"""Extract the model weights from a G0.5 training checkpoint (model + optimizer, ~34 GB) into a
model-only file (~11 GB) so inference can load it without pulling the optimizer state into RAM.
Uses mmap so RSS stays ~= model size.

  python tools/extract_model_ckpt.py <run>/checkpoints/step_30000.pt   -> <run>/checkpoints/step_30000_model.pt
"""
import sys
import time
from pathlib import Path

import torch

src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_model.pt")
t0 = time.time()
ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
print("keys:", list(ckpt.keys()))
out = {k: v for k, v in ckpt.items() if k not in ("optimizer_state_dict", "scheduler_state_dict", "ema_state_dict")}
sd = out["model_state_dict"]
n = sum(t.numel() for t in sd.values())
print(f"model tensors: {len(sd)}  params: {n/1e9:.3f} B  step: {out.get('step')}")
out["model_state_dict"] = {k: v.clone() for k, v in sd.items()}  # materialise off the mmap
torch.save(out, dst)
print(f"wrote {dst} ({dst.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.0f}s")
