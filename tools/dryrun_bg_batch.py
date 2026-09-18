# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""1-batch CPU dry-run BEFORE GPU fine-tuning (test_dataloader_batch.py pattern, reduced
to the belief-graph path): joined scratch dataset -> real LeRobotDataset __getitem__
(bg_*_index decode) -> real MixedSamplesBuilder (5 BG builders) -> resolved final prompt
string with loss-mask spans and lengths.

Tokenization uses the repo tokenizer checkpoint (configs/model/g05.yaml
hf_processor_path) when present; otherwise reports character lengths and marks true
token counts as BLOCKED — obtaining that checkpoint is a training prerequisite anyway.

Usage:
    PYTHONPATH=src python tools/dryrun_bg_batch.py --root <scratch dataset> \
        [--episode 9000] [--scan 300] [--tokenizer checkpoints/qwen3_5_2b_base_processor]
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from g05.data.lerobot.lerobot_dataset_v3 import LeRobotDataset  # noqa: E402
from g05.data_processor.processor.samples_builder import (  # noqa: E402
    BeliefGraphDeltaCoTBuilder, BeliefGraphEffectCoTBuilder, BeliefGraphObserveCoTBuilder,
    BeliefGraphSubtaskCoTBuilder, BeliefGraphUpdateCoTBuilder, MixedSamplesBuilder)
from g05.utils.common.special_tokens import SpecialTokenManager  # noqa: E402

BUILDERS = [BeliefGraphSubtaskCoTBuilder, BeliefGraphDeltaCoTBuilder,
            BeliefGraphUpdateCoTBuilder, BeliefGraphEffectCoTBuilder,
            BeliefGraphObserveCoTBuilder]
SLOT = re.compile(r"<(\w+?)_(text|image|proprio|action)((?:_[!0-9]+)*)>")


def make_sample(item):
    proprio = item["observation.state"].float()
    return {"_instructions": item["task"], "_vlm_action": None,
            "proprio": proprio, "proprio_dim_is_pad": torch.zeros_like(proprio, dtype=torch.bool)}


def render(template: str, samples: dict):
    """Substitute slots; return (final_string, [(span_text, masked)]) — masked = no loss."""
    parts, pos = [], 0
    for m in SLOT.finditer(template):
        if m.start() > pos:
            parts.append((template[pos:m.start()], True))       # literal scaffold: masked
        name, kind, flags = m.group(1), m.group(2), m.group(3)
        masked = "_!" in flags
        limit = re.search(r"_(\d+)", flags)
        if kind == "text":
            v = str(samples.get(name, samples.get("command", "") if name == "command" else ""))
            if name == "command":
                v = str(samples.get("command", ""))
            if limit:
                v = v[: int(limit.group(1))]
            parts.append((v, masked))
        else:
            parts.append((f"[{name.upper()}:{kind}]", masked))  # symbolic non-text slot
        pos = m.end()
    parts.append((template[pos:], True))
    return "".join(p for p, _ in parts), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episode", type=int, default=9000)
    ap.add_argument("--index", type=int, default=1290, help="frame within the episode")
    ap.add_argument("--scan", type=int, default=300, help="frames for the length scan")
    ap.add_argument("--tokenizer", default="checkpoints/qwen3_5_2b_base_processor")
    args = ap.parse_args()

    ds = LeRobotDataset("local/b1k-bg-dryrun", root=args.root, episodes=[args.episode],
                        load_images=False, download_videos=False)
    print(f"dataset: {len(ds)} frames (episode {args.episode})")
    item = ds[min(args.index, len(ds) - 1)]

    print("\n== decoded bg fields at this frame ==")
    for k in ("task", "bg_known", "bg_belief", "bg_delta", "bg_effect", "bg_observe"):
        v = str(item.get(k))
        print(f"  {k:11s}: {v[:110]}{'…' if len(v) > 110 else ''}")
    missing = [k for k in ("bg_known", "bg_belief", "bg_delta", "bg_effect", "bg_observe")
               if item.get(k) in (None, "")]
    assert not missing, f"bg fields missing after join/decode: {missing}"

    defaults = dict(num_input_images=2, image_sizes={"image": (256, 256), "wrist": (256, 256)},
                    embodiment_type="r1pro")
    mixed = MixedSamplesBuilder(
        **defaults,
        candidates=[{"_target_": f"g05.data_processor.processor.samples_builder.{c.__name__}",
                     "weight": 1.0} for c in BUILDERS],
        eval_builder={"_target_":
                      "g05.data_processor.processor.samples_builder.BeliefGraphDeltaCoTBuilder"})
    applicable = [type(b).__name__ for b, _ in mixed._candidates if b.can_handle(item)]
    print(f"\n== can_handle: {len(applicable)}/5 -> {applicable}")
    assert len(applicable) == 5, "all five BG builders must match the joined data"

    # tokenizer (optional)
    tok = None
    tok_path = pathlib.Path(args.tokenizer)
    if tok_path.exists():
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(str(tok_path)).tokenizer
    n_tokens = (lambda s: len(tok.encode(s))) if tok else None

    token_map = SpecialTokenManager.for_model("qwen35")
    print("\n== per-builder final strings (frame", int(item["frame_index"]), ") ==")
    for cls in BUILDERS:
        b = cls(**defaults)
        samples = b.build(dict(item), make_sample(item))
        resolved = token_map.resolve_template(samples["template"])
        final, parts = render(resolved, samples)
        sup = "".join(p for p, masked in parts if not masked)
        tk = f" · {n_tokens(final)} tok(full)/{n_tokens(sup)} tok(target)" if n_tokens else ""
        print(f"\n[{cls.__name__}] {len(final)} chars, target {len(sup)} chars{tk}")
        print("  FULL   | " + final.replace("\n", "⏎")[:220])
        print("  TARGET | " + sup.replace("\n", "⏎")[:180])

    # length scan (token-budget proxy over many frames)
    print(f"\n== length scan over {args.scan} frames ==")
    import numpy as np
    stats = {c.__name__: [] for c in BUILDERS}
    idxs = np.linspace(0, len(ds) - 1, num=min(args.scan, len(ds)), dtype=int)
    for i in idxs:
        it = ds[int(i)]
        for cls in BUILDERS:
            b = cls(**defaults)
            samples = b.build(dict(it), make_sample(it))
            final, parts = render(token_map.resolve_template(samples["template"]), samples)
            sup = "".join(p for p, masked in parts if not masked)
            stats[cls.__name__].append((len(final), len(sup),
                                        n_tokens(final) if n_tokens else -1))
    print(f"{'builder':32s} {'chars(max)':>10} {'target(max)':>11} {'tok(max)':>9}")
    for name, rows in stats.items():
        a = np.array(rows)
        print(f"{name:32s} {a[:,0].max():>10} {a[:,1].max():>11} "
              f"{a[:,2].max() if tok else 'BLOCKED':>9}")
    if not tok:
        print(f"\nNOTE: true token counts BLOCKED — tokenizer checkpoint not found at "
              f"'{args.tokenizer}' (required for training anyway; char counts above are the proxy)")
    print("\nPASS: dry-run — dataset decode, builder gating, template resolution all OK")


if __name__ == "__main__":
    main()
