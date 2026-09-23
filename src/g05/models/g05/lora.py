"""LoRA fine-tuning for G0.5.

peft adapters are injected in place into the linear projections selected by ``targets``:

    vlm            ``model.vlm.layers``: full attention, gated-deltanet linear attention, MLP
    action_expert  ``model.action_expert.layers``: attention and MLP

Everything else is frozen except the modules named in ``trainable_modules``, which are
trained in full (e.g. the action expert's 27-D I/O projections and time MLP, whose shapes
are tied to this embodiment's action space and which are too small for a low-rank update).
Entries are prefixes, or regexes full-matched against the module name.

A module can only be adapted if its forward goes through ``nn.Linear.__call__``. The adaLN
``AdaptiveRMSNorm.dense`` layers do not (they call ``F.linear(cond, self.dense.weight)`` to
force fp32), so an adapter there would silently get no gradient; train them in full instead.

Checkpoints stay in the plain, un-adapted layout: ``export_state_dict`` folds every
adapter into its base weight (W + scale * B @ A) and restores the original key names,
so ``load_model_from_checkpoint`` / the policy server load a LoRA run exactly like a full
fine-tune. The raw adapter tensors are saved next to it (``lora_state_dict``) so that
``restore_for_resume`` can un-merge them and continue with the saved optimizer state.

Config (``model.lora``)::

    enabled: true
    r: 32
    alpha: 64
    dropout: 0.05
    targets: [vlm]         # presets above; target_regex (full-match over G05Model names) overrides
    target_regex: null
    trainable_modules: [action_expert, proprio_embedder]   # prefixes / regexes under G05Model
    frozen_dtype: bf16     # storage dtype of frozen weight matrices (fp32 to keep them as loaded)

Memory: frozen weights get no gradient and autocast already runs them in bf16, so storing
them in bf16 halves their footprint (3.4B params: 13.5 GB fp32 -> ~9 GB), which is what makes
this fit a 24 GB card. Trainable params and ``fp32_param_patterns`` (norms, action-expert I/O)
stay fp32. Merged weights are exported in fp32 so a small adapter delta is not rounded away.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: Target presets, full-match regexes relative to G05Model (policy.model). Both towers have
#: identically named q/k/v/o/gate/up/down projections, hence the anchor on ``<tower>.layers``.
#: VLM: in_proj_a / in_proj_b (16-wide gate projections) are left out.
TARGET_PRESETS = {
    "vlm": (
        r"vlm\.layers\.\d+\.("
        r"self_attn\.(q|k|v|o)_proj"
        r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
        r"|mlp\.(gate|up|down)_proj"
        r")"
    ),
    "action_expert": (
        r"action_expert\.layers\.\d+\.("
        r"self_attn\.(q|k|v|o)_proj"
        r"|mlp\.(gate|up|down)_proj"
        r")"
    ),
}
DEFAULT_TARGET_REGEX = TARGET_PRESETS["vlm"]
DEFAULT_TRAINABLE_MODULES = ("action_expert", "proprio_embedder")
ADAPTER = "default"


def _lora_layers(module: nn.Module) -> Dict[str, nn.Module]:
    from peft.tuners.lora import LoraLayer

    return {name: m for name, m in module.named_modules() if isinstance(m, LoraLayer)}


def has_lora(module: nn.Module) -> bool:
    return bool(_lora_layers(module))


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    """``name`` is, or is inside, a module matching one of ``prefixes`` (plain or regex)."""
    return any(re.fullmatch(rf"(?:{p})(\..*)?", name) for p in prefixes)


def apply_lora(policy: nn.Module, cfg, resume_checkpoint: Optional[dict] = None) -> Dict[str, int]:
    """Inject LoRA adapters into ``policy.model`` and set ``requires_grad`` accordingly.

    Must run after the checkpoint is loaded (adapters start as an exact no-op, B = 0) and
    before DDP wrapping / optimizer creation, which only see ``requires_grad`` params.
    With ``resume_checkpoint`` (a checkpoint written by a LoRA run) the saved adapters are
    restored and un-merged, before any weight is down-cast.
    """
    from peft import LoraConfig, inject_adapter_in_model

    core = policy.model
    target_regex = cfg.get("target_regex")
    if not target_regex:
        presets = list(cfg.get("targets", ["vlm"]) or [])
        unknown_presets = [t for t in presets if t not in TARGET_PRESETS]
        if not presets or unknown_presets:
            raise ValueError(f"LoRA targets must be a non-empty subset of {sorted(TARGET_PRESETS)}, got {presets}")
        target_regex = "|".join(f"(?:{TARGET_PRESETS[t]})" for t in presets)
    trainable = list(cfg.get("trainable_modules", DEFAULT_TRAINABLE_MODULES) or [])

    targets = [n for n, m in core.named_modules() if isinstance(m, nn.Linear) and re.fullmatch(target_regex, n)]
    if not targets:
        raise ValueError(f"LoRA target_regex {target_regex!r} matches no nn.Linear under policy.model")

    lora_config = LoraConfig(
        r=int(cfg.get("r", 32)),
        lora_alpha=int(cfg.get("alpha", 64)),
        lora_dropout=float(cfg.get("dropout", 0.0)),
        target_modules=target_regex,  # a str is full-matched against module names by peft
        bias="none",
    )
    inject_adapter_in_model(lora_config, core, adapter_name=ADAPTER)
    if resume_checkpoint is not None:
        restore_for_resume(policy, resume_checkpoint)

    unknown = [p for p in trainable if not any(_matches_prefix(n, [p]) for n, _ in core.named_modules())]
    if unknown:
        raise ValueError(f"LoRA trainable_modules not found under policy.model: {unknown}")

    # peft already freezes non-adapter params of `core`; set every flag explicitly anyway so
    # the result does not depend on peft internals, and so params outside `core` are frozen.
    for p in policy.parameters():
        p.requires_grad_(False)
    for name, p in core.named_parameters():
        if ".lora_" in name or _matches_prefix(name, trainable):
            p.requires_grad_(True)

    frozen_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg.get("frozen_dtype", "bf16")]
    if frozen_dtype is not None:
        # Only Linear / Embedding weights: gated-deltanet A_log / dt_bias / conv1d and the
        # fp32_param_patterns (norms, action-expert I/O) keep their precision.
        # vlm.output_proj (tied to input_proj) is read raw by the fused CE kernel against fp32
        # hidden states, outside autocast, so it keeps the dtype it was loaded in.
        keep_fp32 = list(getattr(policy, "fp32_param_patterns", []) or []) + ["vlm.input_proj", "vlm.output_proj"]
        for mod_name, mod in policy.named_modules():
            if not isinstance(mod, (nn.Linear, nn.Embedding)):
                continue
            for leaf, p in mod.named_parameters(recurse=False):
                name = f"{mod_name}.{leaf}"
                if not p.requires_grad and p.is_floating_point() and not any(k in name for k in keep_fp32):
                    p.data = p.data.to(frozen_dtype)

    n_lora = sum(p.numel() for n, p in core.named_parameters() if ".lora_" in n)
    n_full = sum(p.numel() for n, p in core.named_parameters() if p.requires_grad and ".lora_" not in n)
    n_total = sum(p.numel() for n, p in core.named_parameters() if ".lora_" not in n)
    stats = {
        "lora_layers": len(targets),
        "lora_params": n_lora,
        "full_trainable_params": n_full,
        "frozen_params": n_total - n_full,
    }
    stats["lora_layers_by_tower"] = {
        tower: sum(n.startswith(tower + ".") for n in targets) for tower in ("vlm", "action_expert")
    }
    logger.info(f"LoRA layers by tower: {stats['lora_layers_by_tower']}")
    logger.info(
        f"LoRA r={lora_config.r} alpha={lora_config.lora_alpha} dropout={lora_config.lora_dropout}: "
        f"{len(targets)} layers, {n_lora / 1e6:.1f}M adapter params; fully trainable "
        f"{trainable} = {n_full / 1e6:.1f}M; frozen {(n_total - n_full) / 1e6:.1f}M"
    )
    return stats


@torch.no_grad()
def export_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """``module.state_dict()`` with every LoRA adapter merged into its base weight.

    Keys come back in the un-adapted layout (``...q_proj.weight``), so the result loads
    into a model that never had adapters. Without adapters this is plain ``state_dict()``.
    """
    layers = _lora_layers(module)
    state = module.state_dict()
    if not layers:
        return state

    merged: Dict[str, torch.Tensor] = {}
    for name, layer in layers.items():
        base = layer.get_base_layer()
        delta = layer.get_delta_weight(ADAPTER)
        # fp32 even when the frozen base is stored in bf16 (loaders cast to the model dtype).
        # Moved to CPU one layer at a time: the merged copies of all ~1.4B adapted weights
        # would otherwise need ~5.6 GB of extra GPU memory at every checkpoint.
        merged[f"{name}.weight"] = (base.weight.float() + delta.float()).cpu()
        del delta

    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if ".lora_" in key:
            continue
        key = key.replace(".base_layer.", ".")
        out[key] = merged.get(key, value)
    return out


def adapter_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Only the adapter tensors (``lora_A`` / ``lora_B``), for resuming."""
    return {k: v for k, v in module.state_dict().items() if ".lora_" in k}


@torch.no_grad()
def restore_for_resume(policy: nn.Module, checkpoint: dict) -> None:
    """Undo ``export_state_dict`` on a freshly adapted model loaded from a LoRA checkpoint.

    The loaded base weights are W + delta; load the saved adapters and subtract their delta,
    leaving base = W and adapters = the saved A/B, so the saved optimizer state matches.
    """
    adapters = checkpoint.get("lora_state_dict")
    if not adapters:
        raise ValueError(
            "resume_ckpt has no lora_state_dict: it was not written by a LoRA run. "
            "Warm-start from it with model.pretrained_ckpt instead."
        )
    _, unexpected = policy.load_state_dict(adapters, strict=False)
    not_loaded = [k for k in adapter_state_dict(policy) if k not in adapters]
    if unexpected or not_loaded:
        raise ValueError(
            f"LoRA adapters in resume_ckpt do not match this config: "
            f"unexpected={unexpected[:5]} missing={not_loaded[:5]}"
        )
    for layer in _lora_layers(policy).values():
        base = layer.get_base_layer()
        base.weight.copy_((base.weight.float() - layer.get_delta_weight(ADAPTER).float()).to(base.weight.dtype))
    logger.info(f"LoRA: restored {len(adapters)} adapter tensors and un-merged them from the base weights")


def check_adapter_grads(policy: nn.Module) -> None:
    """Raise if a LoRA parameter got no gradient in the first backward pass.

    ``grad is None`` (not merely zero: lora_A's grad is zero while lora_B is still 0) means
    the adapted layer's forward never ran through the LoRA wrapper.
    """
    dead = sorted({
        name.rsplit(".lora_", 1)[0]
        for name, p in policy.named_parameters()
        if ".lora_" in name and p.requires_grad and p.grad is None
    })
    if dead:
        raise RuntimeError(
            f"{len(dead)} LoRA-adapted layers received no gradient on the first step, e.g. "
            f"{dead[:3]}. Their forward bypasses nn.Linear.__call__; drop them from the LoRA "
            f"targets (and list them in lora.trainable_modules to train them in full)."
        )
    logger.info("LoRA: every adapter received a gradient on the first step")


def summarize_trainable(policy: nn.Module) -> List[str]:
    """Top-level groups of trainable params, for the run log."""
    groups: Dict[str, int] = {}
    for name, p in policy.named_parameters():
        if p.requires_grad:
            key = ".".join(name.split(".")[:2]) + (" (lora)" if ".lora_" in name else "")
            groups[key] = groups.get(key, 0) + p.numel()
    return [f"{k}: {v / 1e6:.2f}M" for k, v in sorted(groups.items())]
