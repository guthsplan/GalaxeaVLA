#!/usr/bin/env python3
"""Assert the AR-only contract: continuous_action=False must not touch the flow head.

Two independent checks, neither of which needs a GPU or a dataset.

1. **No FM call.** `G05Model.forward` / `G05ModelQwen35.forward` used to run
   `fm_helper.train_step(...)` unconditionally and then zero the result
   (`fm_loss = fm_loss * 0`). That still paid for the prefix-KV rebuild, the
   flow-time/noise sampling and the action-expert forward. Both variants now
   guard the call, so this monkeypatches `train_step` with a tripwire that raises
   and drives `forward` far enough to prove it is never reached — and, with
   `continuous_action=True`, that it still *is*.

2. **AR-only config.** `behavior_cot` resolves with discrete_action=True,
   continuous_action=False, return_continuous_action=False, predict_cot=True,
   pred_eov=True; and `behavior` keeps its original settings.

Usage:
    python tests/test_ar_only_no_fm.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from g05.utils.config.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label + ((": " + detail) if detail else ""))


class _FMTripwire:
    """Stands in for FMHelper; raises if the FM path is entered."""

    def __init__(self):
        self.calls = 0

    def train_step(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("fm_helper.train_step was called")


class _StubARHelper:
    _last_ce_cache = {"action_accuracy": 0.5, "cot_accuracy": 0.25}

    def train_step(self, model, vlm_hidden, labels):
        return torch.zeros((), requires_grad=False), 0.75


def _run_forward(model_cls, continuous_action: bool):
    """Drive <cls>.forward's loss-assembly section with stubs, no real weights.

    We bypass __init__ (object.__new__) and stub only what the loss-assembly path
    touches: vlm_prefill, ar_helper, fm_helper. That exercises the real branch
    logic in the real source file rather than a reimplementation of it.
    """
    m = object.__new__(model_cls)
    tripwire = _FMTripwire()
    m.fm_helper = tripwire
    m.ar_helper = _StubARHelper()

    B, S, H = 1, 8, 4
    split_index = 5
    hidden = torch.zeros(B, S, H)
    kv = [(torch.zeros(B, 1, S, H), torch.zeros(B, 1, S, H))]
    pos = torch.zeros(B, S, dtype=torch.long)

    m.vlm_prefill = lambda *a, **k: (hidden, kv, pos)
    # qwen35 variant also rebuilds a prefix action cache before the FM call
    m._build_prefix_action_kv = lambda vlm_kv, si: vlm_kv

    out = model_cls.forward(
        m,
        input_ids=torch.zeros(B, S, dtype=torch.long),
        attention_mask=torch.ones(B, S, dtype=torch.long),
        pixel_values={"head_rgb": torch.zeros(B, 3, 8, 8)},
        actions=torch.zeros(B, 2, 4),
        action_pad_masks=torch.zeros(B, 2, dtype=torch.bool),
        action_dim_is_pad=torch.zeros(B, 4, dtype=torch.bool),
        split_index=split_index,
        labels=torch.zeros(B, S, dtype=torch.long),
        continuous_action=continuous_action,
        skip_ce_loss=False,
        proprio=None,
    )
    return out, tripwire


def main() -> int:
    from g05.models.g05.g05_model import G05Model
    from g05.models.g05.g05_model_qwen35 import G05ModelQwen35

    print("=" * 72)
    print("1. fm_helper.train_step is NOT called when continuous_action=False")
    print("=" * 72)
    # G05ModelQwen35 is the variant BEHAVIOR uses (configs/model/g05.yaml:32
    # -> g05_policy_qwen35.G05PolicyQwen35). Both are checked: the fix is generic.
    for cls in (G05ModelQwen35, G05Model):
        try:
            out, tripwire = _run_forward(cls, continuous_action=False)
            check(
                f"{cls.__name__}: no FM call, calls={tripwire.calls}",
                tripwire.calls == 0,
            )
            check(
                f"{cls.__name__}: 'fm_loss' absent from loss_dict",
                "fm_loss" not in out,
                f"keys={sorted(out)}",
            )
            check(
                f"{cls.__name__}: 'ce_loss' present (CE is the only objective)",
                "ce_loss" in out,
                f"keys={sorted(out)}",
            )
        except AssertionError as err:
            check(f"{cls.__name__}: no FM call", False, str(err))
        except Exception as err:  # noqa: BLE001
            check(f"{cls.__name__}: forward reached loss assembly", False, f"{type(err).__name__}: {err}")

    print()
    print("=" * 72)
    print("2. continuous_action=True still invokes FM (no regression)")
    print("=" * 72)
    for cls in (G05ModelQwen35, G05Model):
        _, tripwire = None, None
        try:
            _run_forward(cls, continuous_action=True)
            check(f"{cls.__name__}: FM still invoked", False, "tripwire did not fire")
        except AssertionError as err:
            check(
                f"{cls.__name__}: FM still invoked",
                "fm_helper.train_step was called" in str(err),
                str(err),
            )
        except Exception as err:  # noqa: BLE001
            check(f"{cls.__name__}: FM still invoked", False, f"{type(err).__name__}: {err}")

    print()
    print("=" * 72)
    print("3. behavior_cot resolves AR-only; behavior baseline unchanged")
    print("=" * 72)
    os.environ.setdefault("B1K_SUBSET_DIR", "data/b1k_5task_lerobot")
    from resolve_config import resolve_task_config

    cot = resolve_task_config("behavior_cot").model.model_arch
    for key, want in (
        ("discrete_action", True),
        ("continuous_action", False),
        ("return_continuous_action", False),
        ("predict_cot", True),
    ):
        check(f"behavior_cot.{key} == {want}", cot.get(key) == want, repr(cot.get(key)))
    check(
        "behavior_cot.input_preprocessor.pred_eov == True",
        cot.input_preprocessor.pred_eov is True,
        repr(cot.input_preprocessor.pred_eov),
    )

    base = resolve_task_config("behavior").model.model_arch
    for key, want in (
        ("discrete_action", True),
        ("continuous_action", True),
        ("return_continuous_action", True),
        ("predict_cot", False),
    ):
        check(f"behavior.{key} == {want} (baseline)", base.get(key) == want, repr(base.get(key)))
    check(
        "behavior.input_preprocessor.pred_eov == False (baseline)",
        base.input_preprocessor.pred_eov is False,
        repr(base.input_preprocessor.pred_eov),
    )

    print()
    if failures:
        print(f"FAILED — {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All AR-only / FM-skip checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
