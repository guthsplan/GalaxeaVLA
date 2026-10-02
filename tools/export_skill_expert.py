#!/usr/bin/env python
"""Export a skill-expert training checkpoint as a small expert file for SkillExpertRouter.

A skill-expert run (configs/task/behavior_skill_expert.yaml) trains action-expert LoRA adapters
(+ a few small action-expert modules) on top of a frozen CoT checkpoint. Its checkpoints are
saved merged (W + delta, plain layout) with the raw adapters next to them (lora_state_dict).
This tool keeps only what differs from the base checkpoint:

    lora   the adapters, B pre-multiplied by alpha / r      (checked against the merged weights)
    full   action-expert params trained in full              (I/O projections, time MLP, norm)

and refuses checkpoints whose VLM / vision tower / proprio embedder differ from the base: the
router serves every expert on the base checkpoint's VLM, so anything trained there would be lost.

    python tools/export_skill_expert.py <run>/checkpoints/step_N.pt -o experts/grasp.pt
        [--base <cot ckpt>] [--groups grasp]

--base / --groups default to the run's .hydra/config.yaml (model.pretrained_ckpt and
data.embodiment_datasets.*.skill_groups).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g05.data.skill_groups import ALL, parse_groups  # noqa: E402
from g05.models.g05.skill_router import FORMAT  # noqa: E402

PREFIX = "model."  # policy-level keys -> G05Model-level names
AE = "action_expert."


def _run_config(ckpt: Path) -> Optional[dict]:
    cfg_path = ckpt.resolve().parents[1] / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return None
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=False)


def _config_groups(cfg: dict) -> Optional[str]:
    found = {
        str(v.get("skill_groups"))
        for v in (cfg.get("data", {}).get("embodiment_datasets", {}) or {}).values()
        if isinstance(v, dict) and v.get("skill_groups")
    }
    if len(found) > 1:
        raise SystemExit(f"run config has several skill_groups: {sorted(found)}; pass --groups")
    return found.pop() if found else None


def _strip(key: str) -> str:
    return key[len(PREFIX):] if key.startswith(PREFIX) else key


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    # Frozen Linear weights are stored in bf16 during LoRA training (lora.frozen_dtype).
    if a.shape != b.shape:
        return False
    if a.is_floating_point():
        return torch.equal(a.to(torch.bfloat16), b.to(torch.bfloat16))
    return torch.equal(a, b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="skill-expert training checkpoint (step_N.pt, with lora_state_dict)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, help="checkpoint the run started from (default: run config)")
    ap.add_argument("--groups", help="skill groups the expert covers (default: run config)")
    ap.add_argument("--max-rel-err", type=float, default=1e-3,
                    help="tolerance of the adapter check against the merged weights")
    args = ap.parse_args()

    cfg = _run_config(args.ckpt) or {}
    lora_cfg = cfg.get("model", {}).get("lora", {}) or {}
    base_path = args.base or (Path(cfg["model"]["pretrained_ckpt"]) if cfg.get("model", {}).get("pretrained_ckpt") else None)
    if base_path is None:
        raise SystemExit("no base checkpoint: pass --base (the run config has no model.pretrained_ckpt)")
    groups_spec = args.groups or _config_groups(cfg)
    if not groups_spec:
        raise SystemExit("no skill groups: pass --groups (the run config sets no skill_groups)")
    groups = parse_groups(groups_spec)
    if args.groups and cfg and _config_groups(cfg) and parse_groups(_config_groups(cfg)) != groups:
        raise SystemExit(f"--groups {groups} differs from the run config's {_config_groups(cfg)}")

    print(f"expert ckpt : {args.ckpt}\nbase ckpt   : {base_path}\ngroups      : {groups}")
    ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    adapters = ck.get("lora_state_dict")
    if not adapters:
        raise SystemExit(f"{args.ckpt} has no lora_state_dict (not a LoRA training checkpoint; "
                         "pass the step_N.pt the run wrote, not an extracted *_model.pt)")
    sd = ck["model_state_dict"]
    base_sd = torch.load(base_path, map_location="cpu", mmap=True, weights_only=False)["model_state_dict"]

    # ---- adapters -----------------------------------------------------------------
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, t in adapters.items():
        mod, _, rest = key.partition(".lora_")
        which = rest.split(".")[0]  # A / B
        pairs.setdefault(_strip(mod), {})[which] = t.float()
    outside = [m for m in pairs if not m.startswith(AE)]
    if outside:
        raise SystemExit(f"adapters outside the action expert (e.g. {outside[:3]}): train with "
                         "lora.targets=[action_expert] (configs/task/behavior_skill_expert.yaml)")
    r = int(lora_cfg.get("r", next(iter(pairs.values()))["A"].shape[0]))
    scale = float(lora_cfg["alpha"]) / r if "alpha" in lora_cfg else None
    if scale is None:
        # No run config: recover alpha / r from one merged weight (least squares).
        m = next(iter(pairs))
        ba = pairs[m]["B"] @ pairs[m]["A"]
        delta = sd[PREFIX + m + ".weight"].float() - base_sd[PREFIX + m + ".weight"].to(torch.bfloat16).float()
        scale = float((delta * ba).sum() / (ba * ba).sum().clamp_min(1e-12))
        print(f"no run config: fitted alpha/r = {scale:.4f}")
    worst = 0.0
    lora_out = {}
    for m, ab in sorted(pairs.items()):
        a, b = ab["A"], ab["B"] * scale
        lora_out[m] = {"A": a.contiguous(), "B": b.contiguous()}
        delta = sd[PREFIX + m + ".weight"].float() - base_sd[PREFIX + m + ".weight"].to(torch.bfloat16).float()
        err = float((delta - b @ a).norm() / delta.norm().clamp_min(1e-12)) if delta.norm() > 0 else 0.0
        worst = max(worst, err)
    if worst > args.max_rel_err:
        raise SystemExit(f"adapters do not reproduce the merged weights (worst rel err {worst:.2e}): "
                         "wrong --base checkpoint, or lora.r/alpha differ from the run config")
    print(f"adapters    : {len(lora_out)} layers, r={r}, alpha/r={scale:.4f}, worst rel err {worst:.1e}")

    # ---- everything else ---------------------------------------------------------------
    adapted = {PREFIX + m + ".weight" for m in pairs}
    full, changed_outside = {}, []
    missing = sorted(set(base_sd) - set(sd))
    if missing:
        raise SystemExit(f"keys missing from the expert checkpoint, e.g. {missing[:3]}")
    for key, t in sd.items():
        if key in adapted:
            continue
        if key not in base_sd:
            raise SystemExit(f"{key} is not in the base checkpoint")
        if _same(t, base_sd[key]):
            continue
        if _strip(key).startswith(AE):
            full[_strip(key)] = t.float().clone() if t.is_floating_point() else t.clone()
        else:
            changed_outside.append(key)
    if changed_outside:
        raise SystemExit(
            f"{len(changed_outside)} params outside the action expert differ from the base, e.g. "
            f"{changed_outside[:3]}. The router serves every expert on the base VLM; train with "
            "only action-expert modules trainable."
        )
    n_full = sum(t.numel() for t in full.values())
    print(f"full params : {len(full)} tensors, {n_full / 1e6:.2f}M ({sorted({k.split('.')[1] for k in full})})")

    # ---- fingerprint of the base the expert belongs to -----------------------------------
    vlm_key = next(k for k in base_sd if k.startswith(PREFIX + "vlm.") and base_sd[k].ndim == 2)
    ae_key = sorted(adapted)[0]
    fingerprint = {_strip(k): float(base_sd[k].double().sum()) for k in (vlm_key, ae_key)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "groups": groups,
            "lora": lora_out,
            "full": full,
            "base_fingerprint": fingerprint,
            "meta": {
                "source_ckpt": str(args.ckpt.resolve()),
                "base_ckpt": str(Path(base_path).resolve()),
                "step": ck.get("step"),
                "r": r,
                "alpha_over_r": scale,
            },
        },
        args.out,
    )
    size = args.out.stat().st_size / 2**20
    print(f"wrote {args.out} ({size:.0f} MB){'  [fallback expert]' if groups == [ALL] else ''}")


if __name__ == "__main__":
    main()
