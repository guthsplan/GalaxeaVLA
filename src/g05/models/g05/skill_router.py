"""Skill-routed action experts: one action-expert LoRA per skill group, picked by the CoT subtask.

A CoT policy (predict_cot, continuous_action) generates "Subtask: ..." before acting. The
router maps that text to a skill group (g05.data.skill_groups), switches the action expert to
the group's expert, and the policy runs flow matching on the KV cache / recurrent states of the
prompt context only, never the generated CoT (G05Policy.forward_inference). The VLM is shared
and unchanged: every expert was trained from the same checkpoint the server loads, with the VLM
frozen and only action-expert adapters (+ small action-expert modules) trained
(configs/task/behavior_skill_expert.yaml).

Expert file (tools/export_skill_expert.py), ``torch.save`` of::

    format      "g05_skill_expert_v1"
    groups      ["grasp"] | ["revolute", "linear"] | ["all"]   (all = fallback for every group)
    lora        {"<module>": {"A": [r, in], "B": [out, r] (already x alpha / r)}}
    full        {"<param>": tensor}   action-expert params trained in full (I/O projections, ...)
    base_fingerprint {"<param>": float}   sums of base weights, to catch a wrong base checkpoint
    meta        provenance (source / base checkpoints, step, r, alpha)

Module and param names are relative to G05Model (``action_expert.layers.0.self_attn.q_proj``).

Switching never touches the adapted base weights: each adapted nn.Linear gets a forward hook
that adds ``(x @ A.T) @ B.T`` for the active expert, so a switch is a dict lookup and repeated
switching cannot drift. The few fully trained params are copied in (base values kept aside).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from g05.data.skill_groups import ALL, SKILL_GROUPS, SKILL_TO_GROUP, skill_from_subtask

logger = logging.getLogger(__name__)

FORMAT = "g05_skill_expert_v1"


class _Expert:
    def __init__(self, path: str, payload: dict, device, dtype):
        self.path = path
        self.groups: List[str] = list(payload["groups"])
        self.lora = {
            name: (ab["A"].to(device=device, dtype=dtype), ab["B"].to(device=device, dtype=dtype))
            for name, ab in payload["lora"].items()
        }
        self.full = {name: t for name, t in payload.get("full", {}).items()}
        self.meta = dict(payload.get("meta", {}))

    @property
    def name(self) -> str:
        return "+".join(self.groups)


class SkillExpertRouter:
    """Holds the experts of one policy and switches its action expert between them.

    Args:
        model: G05Model (``policy.model``); its action expert is patched in place.
        expert_paths: expert files. At most one expert may cover a group; an ``all`` expert is
            the fallback for groups without their own expert.
        min_consecutive: switch to a new group only after it was predicted this many calls in
            a row (1 = follow every prediction). Unparsable subtasks never switch.
    """

    def __init__(self, model: nn.Module, expert_paths: Sequence[str], min_consecutive: int = 1):
        if min_consecutive < 1:
            raise ValueError(f"min_consecutive must be >= 1, got {min_consecutive}")
        self.model = model
        self.min_consecutive = int(min_consecutive)
        ae = model.action_expert
        param = next(ae.parameters())
        self.device, self.dtype = param.device, param.dtype

        self.experts: List[_Expert] = []
        self.by_group: Dict[str, _Expert] = {}
        self.fallback: Optional[_Expert] = None
        for path in expert_paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format") != FORMAT:
                raise ValueError(f"{path}: not a skill expert file (format={payload.get('format')!r})")
            expert = _Expert(path, payload, self.device, self.dtype)
            self._check_base(expert, payload.get("base_fingerprint", {}))
            self.experts.append(expert)
            if expert.groups == [ALL]:
                if self.fallback is not None:
                    raise ValueError(f"two '{ALL}' experts: {self.fallback.path} and {path}")
                self.fallback = expert
                continue
            for g in expert.groups:
                if g not in SKILL_GROUPS:
                    raise ValueError(f"{path}: unknown skill group {g!r}")
                if g in self.by_group:
                    raise ValueError(f"skill group {g!r} has two experts: {self.by_group[g].path} and {path}")
                self.by_group[g] = expert
        if not self.experts:
            raise ValueError("SkillExpertRouter needs at least one expert file")

        # Forward hooks on every adapted Linear, and the base values of fully trained params.
        modules = dict(model.named_modules())
        params = dict(model.named_parameters())
        self._hooks = []
        adapted = sorted({n for e in self.experts for n in e.lora})
        for name in adapted:
            mod = modules.get(name)
            if not isinstance(mod, nn.Linear):
                raise ValueError(f"expert adapter target {name!r} is not an nn.Linear of this model")
            self._hooks.append(mod.register_forward_hook(self._make_hook(name)))
        self._base_full = {}
        for name in sorted({n for e in self.experts for n in e.full}):
            if name not in params:
                raise ValueError(f"expert param {name!r} does not exist in this model")
            if not name.startswith("action_expert."):
                raise ValueError(f"expert param {name!r} is outside the action expert")
            self._base_full[name] = params[name].detach().to("cpu", copy=True)
        self._params = params

        self.active: Optional[_Expert] = None
        self.current_group: Optional[str] = None
        self._candidate: Optional[str] = None
        self._streak = 0
        logger.info(
            "[skill-router] experts: %s; fallback: %s; groups without an expert run the base action "
            "expert on the full CoT KV; min_consecutive=%d",
            {g: e.path for g, e in self.by_group.items()},
            self.fallback.path if self.fallback else None,
            self.min_consecutive,
        )

    # ------------------------------------------------------------------
    def _check_base(self, expert: _Expert, fingerprint: Dict[str, float]) -> None:
        params = dict(self.model.named_parameters())
        for name, ref in fingerprint.items():
            if name not in params:
                raise ValueError(f"{expert.path}: fingerprint param {name!r} missing from the model")
            got = float(params[name].detach().double().sum())
            if abs(got - ref) > 1e-2 * max(1.0, abs(ref)):
                raise ValueError(
                    f"{expert.path} was exported against a different base checkpoint "
                    f"({name}: sum {got:.4f} here, {ref:.4f} at export; base was "
                    f"{expert.meta.get('base_ckpt')}). Serve the checkpoint the expert was trained from."
                )

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            expert = self.active
            if expert is None or name not in expert.lora:
                return output
            a, b = expert.lora[name]
            x = inputs[0]
            return output + (x.to(a.dtype) @ a.t() @ b.t()).to(output.dtype)

        return hook

    @torch.no_grad()
    def _activate(self, expert: Optional[_Expert]) -> None:
        if expert is self.active:
            return
        for name, base in self._base_full.items():
            src = expert.full.get(name, base) if expert is not None else base
            self._params[name].data.copy_(src.to(self._params[name].dtype))
        self.active = expert

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """New episode: forget the current group (the next prediction switches immediately)."""
        self.current_group = None
        self._candidate = None
        self._streak = 0

    def route(self, cot_text: Optional[str]) -> dict:
        """Pick the group for one model call and activate its expert.

        Returns ``{"skill", "group" (the group in effect), "expert" (name or None),
        "context_only"}``. ``context_only`` is False when no expert covers the group: the base
        action expert was trained on the full CoT KV and is run that way.
        """
        skill = skill_from_subtask(cot_text)
        predicted = SKILL_TO_GROUP.get(skill) if skill else None
        if predicted is not None:
            if self.current_group is None:
                self.current_group = predicted
                self._candidate, self._streak = None, 0
            elif predicted == self.current_group:
                self._candidate, self._streak = None, 0
            else:
                self._streak = self._streak + 1 if predicted == self._candidate else 1
                self._candidate = predicted
                if self._streak >= self.min_consecutive:
                    self.current_group = predicted
                    self._candidate, self._streak = None, 0
        expert = self.by_group.get(self.current_group) if self.current_group else None
        expert = expert or self.fallback
        self._activate(expert)
        return {
            "skill": skill,
            "group": self.current_group,
            "expert": expert.name if expert is not None else None,
            "context_only": expert is not None,
        }

    def remove(self) -> None:
        """Detach the hooks and restore the base action expert."""
        self._activate(None)
        for h in self._hooks:
            h.remove()
        self._hooks = []
