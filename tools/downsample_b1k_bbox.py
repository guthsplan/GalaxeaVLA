"""Thin a bbox export to one box frame every N frames (default 120 = 0.25 Hz at 30 fps).

behavior_bbox_2025_025fps_v2 (tasks 0-49) has boxes on frames 0, 120, 240, ... only (~0.8 % of frames).
tools/replay_b1k_bbox.py writes every --stride frames (~25 % at stride 4), so mixing the two would make
the BBox CoT builders draw almost all of their samples from the replayed tasks. This keeps the frames
with frame_index % N == 0 so every task has the same density.

    python tools/downsample_b1k_bbox.py --src <replay_out> --dst <replay_out>_025hz [--every 120]

Layout and record format are unchanged (task-XXXX/episode_<raw_id>/{frames.jsonl,report.json}); the
report gains output_fps / downsampled_from / downsample_rule.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--every", type=int, default=120, help="keep frame_index %% every == 0 (120 = 0.25 Hz at 30 fps)")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    if args.dst.exists():
        shutil.rmtree(args.dst)
    n_ep = n_in = n_out = 0
    for rep in sorted(args.src.glob("task-*/episode_*/report.json")):
        ep = rep.parent
        out = args.dst / ep.relative_to(args.src)
        out.mkdir(parents=True)
        lines = [l for l in (ep / "frames.jsonl").read_text().splitlines() if l.strip()]
        keep = [l for l in lines if json.loads(l)["frame_index"] % args.every == 0]
        (out / "frames.jsonl").write_text("\n".join(keep) + ("\n" if keep else ""))
        r = json.loads(rep.read_text())
        r.update(output_fps=args.fps / args.every, sampled_frames=len(keep), downsampled_from=str(ep),
                 downsample_rule=f"frame_index % {args.every} == 0",
                 boxes=sum(len(json.loads(l)["bbox"]) for l in keep))
        (out / "report.json").write_text(json.dumps(r, indent=1))
        n_ep += 1
        n_in += len(lines)
        n_out += len(keep)
    print(f"episodes {n_ep}, box frames {n_in} -> {n_out} -> {args.dst}")


if __name__ == "__main__":
    main()
