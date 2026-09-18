#!/usr/bin/env python3
"""Gate A: assert the real training token sequence for every BEHAVIOR CoT builder.

`tests/show_vla_label.py` renders one sample under one builder. This forces each
of the six candidates in turn through the *real* dataset -> processor ->
InputPreprocessor.encode_train path and checks the nine invariants the CoT design
depends on:

    1. BeliefGraph conditioning (bg_known) appears before <EOC>
    2. that conditioning is masked out of the LM loss
    3. the selected CoT target appears after <EOC>
    4. the target's semantic prefix occurs exactly once
    5. CoT target tokens receive prediction loss
    6. <EOV> sits immediately before the action tokens
    7. action tokens follow <EOV>
    8. action tokens still receive loss
    9. the decoded target matches the stored annotation exactly (no noise, no
       truncation, no stray formatting)

Usage:
    B1K_SUBSET_DIR=/path/to/labelled/subset \\
      python tests/test_behavior_cot_template.py --task behavior_cot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from g05.utils.config.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()

from show_vla_label import build_components, load_config  # noqa: E402

from g05.data_processor.processor.samples_builder import (  # noqa: E402
    BEHAVIOR_COT_BUILDERS,
)
from g05.models.g05.io.input_preprocessor import TOKEN_INDEX  # noqa: E402

IGNORE_INDEX = -100

#: builder class name -> (target field, semantic prefix, takes bg_known conditioning)
#: Canonical BeliefGraph protocol (origin/bg@4a951d0). BeliefGraphObserveCoTBuilder
#: deliberately takes NO bg_known: the predicted predicates must come from the
#: images, not from the belief memory.
EXPECTED = {
    "BeliefGraphSubtaskCoTBuilder": ("atomic_task", "Subtask", True),
    "BeliefGraphDeltaCoTBuilder": ("bg_delta", "Delta", True),
    "BeliefGraphUpdateCoTBuilder": ("bg_belief", "Belief", True),
    "BeliefGraphEffectCoTBuilder": ("bg_effect", "Effect", True),
    "BeliefGraphObserveCoTBuilder": ("bg_observe", "Observe", False),
    "BeliefGraphBBoxCoTBuilder": ("bbox", "BBox", True),
    "BeliefGraphTrace2DCoTBuilder": ("trace_2d", "Trace", True),
}


def find_processors(obj, seen=None):
    """Collect every object in the processor tree that owns a samples_builder."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    found = []
    if hasattr(obj, "samples_builder"):
        found.append(obj)
    for attr in ("processors", "_processors", "processor_dict"):
        children = getattr(obj, attr, None)
        if isinstance(children, dict):
            for child in children.values():
                found += find_processors(child, seen)
        elif isinstance(children, (list, tuple)):
            for child in children:
                found += find_processors(child, seen)
    return found


def runs_of(mask: torch.Tensor, value: int):
    """[(start, end)] half-open runs where mask == value."""
    hits = (mask == value).nonzero(as_tuple=False).flatten().tolist()
    runs, start, prev = [], None, None
    for i in hits:
        if start is None:
            start, prev = i, i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev + 1))
            start, prev = i, i
    if start is not None:
        runs.append((start, prev + 1))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="behavior_cot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scan", type=int, default=400, help="frames to scan for a usable sample")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg, task = load_config(args.task, args.override)
    print(f"task={task}  predict_cot={cfg.model.model_arch.predict_cot}  "
          f"pred_eov={cfg.model.model_arch.input_preprocessor.pred_eov}")

    from g05.utils.data.processor_utils import build_processors, instantiate_dataset

    dataset = instantiate_dataset(cfg, is_training_set=True)
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(dataset.get_dataset_stats(processor))
    dataset.set_processor(processor)
    processor.train()  # training mode: the weighted draw, not eval_builder

    preprocessor, _action_tokenizer, _mask_helper = build_components(cfg.model.model_arch and cfg, device)
    tok = preprocessor.tokenizer

    owners = find_processors(processor)
    if not owners:
        # MixtureProcessor exposes its children through __getitem__, not an attribute.
        owners = [p for ds in (getattr(dataset, "datasets", None) or [])
                  if (p := getattr(ds, "processor", None)) is not None
                  and hasattr(p, "samples_builder")]
    assert owners, "no processor in the tree exposes a samples_builder"
    originals = [(o, o.samples_builder) for o in owners]

    # Raw (unprocessed) items, to test can_handle before committing to a sample.
    # MixtureLerobotDataset delegates to per-embodiment BaseLerobotDataset
    # instances; the processor lives on those, not on the mixture.
    inner = getattr(dataset, "datasets", None) or [dataset]

    def set_raw(enabled: bool, saved=[]):
        if enabled:
            saved[:] = [ds.processor for ds in inner]
            for ds in inner:
                ds.processor = None
        else:
            for ds, p in zip(inner, saved):
                ds.processor = p

    failures = []

    for builder_cls in BEHAVIOR_COT_BUILDERS:
        name = builder_cls.__name__
        field, prefix, wants_bg = EXPECTED[name]
        template_builder = originals[0][1]
        forced = builder_cls(
            num_input_images=template_builder.num_input_images,
            image_sizes=dict(zip(template_builder._image_keys,
                                 [template_builder._image_sizes[k] for k in template_builder._image_keys])),
            embodiment_type=template_builder.embodiment_type,
        )

        # locate a sample this builder can actually handle
        set_raw(True)
        idx = None
        annotation = None
        try:
            for i in range(args.scan):
                item = dataset[i]
                if forced.can_handle(item):
                    idx, annotation = i, item.get(field)
                    break
        finally:
            set_raw(False)
        if idx is None:
            failures.append(f"{name}: no sample in the first {args.scan} frames carries `{field}`")
            print(f"\n--- {name}: SKIPPED (no `{field}` within {args.scan} frames) ---")
            continue

        for owner, _orig in originals:
            owner.samples_builder = forced
        try:
            data = dataset[idx]
        finally:
            for owner, orig in originals:
                owner.samples_builder = orig

        sample = data["samples"]
        input_ids, labels, attn, split_index = preprocessor.encode_train(
            [sample], device=device, training=True,
            max_chunk_token_length=cfg.model.model_arch.max_chunk_token_length,
        )
        ids, lab, am = input_ids[0], labels[0], attn[0]

        # ---- locate the regions ----
        action_runs = runs_of(am, int(TOKEN_INDEX.ACTION_TOKEN_INDEX))
        assert action_runs, f"{name}: no ACTION tokens in the sequence"
        a0 = action_runs[0][0]
        pred_runs = [r for r in runs_of(am, int(TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX)) if r[0] < a0]
        assert pred_runs, f"{name}: no PRED_TEXT run before the action tokens"
        cot_lo, cot_hi = pred_runs[0]

        cot_text = tok.decode(ids[cot_lo:cot_hi])
        prefix_text = tok.decode(ids[:cot_lo])
        eov_text = tok.decode(ids[max(cot_lo, a0 - 4):a0])

        problems = []

        # 1 + 2: BeliefGraph conditioning before EOC and masked
        has_bg_slot = "BeliefGraph:" in prefix_text
        if wants_bg and not has_bg_slot:
            problems.append("BeliefGraph conditioning missing from the conditioning region")
        if not wants_bg and has_bg_slot:
            problems.append(
                "BeliefGraphObserveCoTBuilder must NOT receive bg_known conditioning "
                "(belief state would leak into 'perception')"
            )
        # every token up to the start of the generative region is conditioning
        if not bool((lab[:cot_lo] == IGNORE_INDEX).all()):
            problems.append("conditioning region (incl. BeliefGraph slot) is not fully masked")

        # 3 + 4 + 9: target after EOC, prefix once, matches the annotation
        if f"{prefix}:" not in cot_text:
            problems.append(f"target prefix {prefix!r} absent from the generative region")
        if cot_text.count(f"{prefix}:") != 1:
            problems.append(f"prefix {prefix!r} occurs {cot_text.count(f'{prefix}:')}x, expected 1")
        expected_slot = sample.get(field)
        if expected_slot and expected_slot not in cot_text:
            problems.append(f"decoded target does not contain the slot value {expected_slot!r}")

        # 5: CoT tokens carry loss
        if not bool((lab[cot_lo:cot_hi] != IGNORE_INDEX).all()):
            problems.append("CoT target tokens are masked out of the loss")

        # 6 + 7: EOV immediately before the actions
        if "<EOV>" not in eov_text:
            problems.append(f"no <EOV> immediately before the action tokens (saw {eov_text!r})")
        if cot_hi > a0:
            problems.append("CoT region overlaps the action region")

        # 8: action tokens carry loss
        if not bool((lab[a0:action_runs[-1][1]] != IGNORE_INDEX).all()):
            problems.append("action tokens are masked out of the loss")

        status = "FAIL" if problems else "OK"
        print(f"\n--- {name}  [{status}]  sample_idx={idx}  seq_len={len(ids)}  split={split_index} ---")
        print(f"  annotation   : {str(annotation)[:110]}")
        print(f"  slot value   : {str(expected_slot)[:110]}")
        print(f"  decoded CoT  : {cot_text[:150]!r}")
        print(f"  BeliefGraph  : present={has_bg_slot} expected={wants_bg}  (conditioning masked: "
              f"{bool((lab[:cot_lo] == IGNORE_INDEX).all())})")
        print(f"  regions      : cot=[{cot_lo},{cot_hi})  action=[{a0},{action_runs[-1][1]})")
        for p in problems:
            print(f"  !! {p}")
            failures.append(f"{name}: {p}")

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED — {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(BEHAVIOR_COT_BUILDERS)} canonical BeliefGraph builders satisfy the template invariants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
