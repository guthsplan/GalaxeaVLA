"""G0.5 CoT prediction targets from the pipeline outputs (any task).

Emits per-frame (1 Hz) text targets for the BG CoT formats plus Subtask:
  Belief:  goal-first summary of the belief state (<=8 entries, `key p obs|mem`)
  Delta:   remaining goal count lines ('none' when satisfied)
  Effect:  grounded predicate changes of the segment's operator ('none' outside known ops)
  Observe: predicates VISIBLE at this frame with their values (`key 0|1`) — the
           model-as-estimator target: perception only, no memory (robot-tag predicates
           excluded — those come from proprioception, not vision)
  Subtask: operator text template with memory prefix filled in

All values come from belief_trace_*.jsonl (GT-masked visibility — PoC caveat applies),
operators_taskNNN.json and the (instance-resolved) skill annotations; nothing is invented.

Object naming in every predicate target is the scene INSTANCE id exactly as the skill
annotations and the Subtask: target spell it (fridge_dszchb_0, hotdog_207, ...). SHORT is kept
only so `shorten()` stays importable; it must stay empty.
"""
import gzip
import json
import pathlib

import pandas as pd

from .operators import concretize_effects, op_for, parse_key, seg_op

SHORT: dict[str, str] = {}
CAT_ORDER = {"cooked": 0, "frozen": 0, "on_fire": 0, "toggled_on": 4, "inhand": 1,
             "inside": 2, "ontop": 3, "onfloor": 3, "nextto": 3, "under": 3, "attached": 3,
             "touching": 3, "open": 4, "covered": 5, "filled": 5, "contains": 5, "real": 5}
GOAL_CLASS = {"cooked", "frozen", "on_fire", "toggled_on", "covered", "filled", "contains", "real"}
MAX_ENTRIES = 8


def shorten(key: str) -> str:
    """Identity since the instance-id unification; see SHORT."""
    for k, v in SHORT.items():
        key = key.replace(k, v)
    return key


def belief_text(preds: dict) -> str:
    rows = []
    for key, (val, p, obs, src) in preds.items():
        name, _ = parse_key(key)
        if name in ("inhand_left", "inhand_right", "reachable", "visited"):
            continue  # robot-frame detail; conditioning carries it
        cat = CAT_ORDER.get(name)
        if cat is None:
            continue
        # location/door predicates: only state what holds; goal-class predicates always
        if name not in GOAL_CLASS and not val:
            continue
        rows.append((cat, key, p, obs))
    rows.sort(key=lambda r: (r[0], r[1]))
    ent = [f"{shorten(k)} {p:.2f} {'obs' if o else 'mem'}" for _, k, p, o in rows[:MAX_ENTRIES]]
    return " | ".join(ent) if ent else "none"


def delta_text(remaining: list[str]) -> str:
    return " | ".join(remaining) if remaining else "none"


MAX_OBSERVE = 24  # must cover the full non-robot key set (estimator output is complete)
ROBOT_PREDS = ("inhand", "inhand_left", "inhand_right", "reachable", "visited")


def observe_text(preds: dict) -> str:
    """Visible-this-frame predicates with values — the pure-perception target."""
    rows = []
    for key, (val, p, obs, src) in preds.items():
        if not obs:
            continue
        if parse_key(key)[0] in ROBOT_PREDS:
            continue
        rows.append((key, val))
    rows.sort()
    ent = [f"{shorten(k)} {1 if v else 0}" for k, v in rows[:MAX_OBSERVE]]
    return " | ".join(ent) if ent else "none"


def effect_text(seg: dict | None, ops_json: dict) -> str:
    if seg is None:
        return "none"
    so = seg_op(seg)
    if so is None:
        return "none"
    op_name, _, objs = so
    op = ops_json.get(op_name)
    if not op:
        return "none"
    effs = concretize_effects(op, objs)
    return " | ".join(f"{shorten(k)} {'0>1' if v else '1>0'}" for k, v in effs) or "none"


def subtask_text(seg: dict | None, ops_json: dict) -> str:
    if seg is None:
        return "done"
    objs = [o for o in seg.get("objects", []) if o]
    op_name, args, text = op_for(seg["skill"], len(objs))
    op = ops_json.get(op_name)
    if op and op.get("text"):
        text = op["text"]
    mapping = dict(zip(args, objs))
    mp = seg.get("memory_prefix") or ""
    text = text.replace("{mp}", (mp + " ") if mp else "")
    for a in args:
        text = text.replace("{" + a + "}", mapping.get(a, a))
    return text.strip()


def main(out_dir: str = "out/task045", task: int = 45):
    out = pathlib.Path(out_dir)
    ops_json = json.load(open(out / f"operators_task{task:03d}.json"))
    inv = json.load(open(out / "inventory.json"))
    segs_by_ep = json.load(open(out / "segments_resolved.json"))
    parts = []
    for ep in inv["episodes"]:
        ep_id = str(ep["raw_episode_id"])
        tp = out / f"belief_trace_{ep_id}.jsonl.gz"
        if not tp.exists():
            tp = out / f"belief_trace_{ep_id}.jsonl"
        if not tp.exists():
            continue
        segs = segs_by_ep.get(ep_id, [])
        opener = gzip.open if tp.suffix == ".gz" else open
        with opener(tp, "rt") as fh:
            recs = [json.loads(l) for l in fh]
        for r in recs:
            if r["step"] % 30 != 0:  # 1 Hz targets from the 10 Hz trace
                continue
            fr = r["step"]
            seg = next((s for s in segs if s["start"] <= fr < s["end"]), None)
            parts.append(dict(
                task=task, episode=int(ep_id), frame=fr,
                subtask=f"Subtask: {subtask_text(seg, ops_json)}",
                belief=f"Belief: {belief_text(r['predicates'])}",
                delta=f"Delta: {delta_text(r['remaining_goal_lines'])}",
                effect=f"Effect: {effect_text(seg, ops_json)}",
                observe=f"Observe: {observe_text(r['predicates'])}",
            ))
    df = pd.DataFrame(parts)
    df.to_parquet(out / "cot_targets.parquet", index=False)
    print(f"cot_targets.parquet: {len(df)} rows, {df.episode.nunique() if len(df) else 0} episodes")
    if len(df):
        print(df.iloc[len(df) // 2].to_dict())
    return df


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "out/task045", int(sys.argv[2]) if len(sys.argv) > 2 else 45)
