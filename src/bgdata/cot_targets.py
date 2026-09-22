"""G0.5 CoT prediction targets from existing pipeline outputs.

Emits per-frame (1 Hz) text targets for the BG CoT formats plus Subtask:
  Belief:  goal-first summary of the belief state (<=8 entries, `key p obs|mem`)
  Delta:   remaining goal count lines ('none' when satisfied)
  Effect:  grounded predicate changes of the segment's operator ('none' outside known ops)
  Observe: predicates VISIBLE at this frame with their values (`key 0|1`) — the
           model-as-estimator target: perception only, no memory (robot-tag predicates
           excluded — those come from proprioception, not vision)
  Subtask: operator text template with memory_prefix filled in

All values come from belief_trace_*.jsonl (GT-masked visibility — PoC caveat applies),
operators_task045.json and the skill annotations; nothing is invented.
"""
import json
import pathlib
import re

import pandas as pd

from .operators import SKILL_TO_OP, concretize_effects, parse_key

# Object naming in every predicate target (Belief:/Observe:/Effect:) is the scene INSTANCE id
# exactly as the skill annotations and the Subtask: target spell it (fridge_dszchb_0,
# hotdog_207, countertop_kelker_0, ...). Earlier label sets shortened containers to
# fridge/microwave/counter, which left the model with three vocabularies for one object
# (instance id in Subtask, alias in predicates, synset in Delta goal lines) and forced an
# alias table at serving time. SHORT is kept only so `shorten()` stays importable; it must
# stay empty.
SHORT: dict[str, str] = {}
CAT_ORDER = {"cooked": 0, "inhand": 1, "inside": 2, "ontop": 3, "onfloor": 3,
             "open": 4, "toggled_on": 4}
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
        # location/door predicates: only state what holds; goal predicates always
        if name != "cooked" and not val:
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
    """Visible-this-frame predicates with values — the pure-perception target.
    Robot predicates are excluded (proprio supplies them at eval, not vision)."""
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
    if seg is None or seg["skill"] not in SKILL_TO_OP:
        return "none"
    op = ops_json.get(SKILL_TO_OP[seg["skill"]][0])
    if not op:
        return "none"
    objs = seg["objects"][: len(op["args"])]
    effs = concretize_effects(op, objs)
    return " | ".join(f"{shorten(k)} {'0>1' if v else '1>0'}" for k, v in effs) or "none"


def subtask_text(seg: dict | None, ops_json: dict) -> str:
    if seg is None:
        return "done"
    op = ops_json.get(SKILL_TO_OP.get(seg["skill"], (None,))[0])
    if not op:
        return seg["skill"]
    mapping = dict(zip(op["args"], seg["objects"]))
    text = op["text"].replace("{mp}", (seg["memory_prefix"] + " ") if seg["memory_prefix"] else "")
    for a, o in mapping.items():
        text = text.replace("{" + a + "}", o)
    return text


def build_for_episode(trace_path: str, segments: list[dict], ops_json: dict,
                      task: int, episode: int, every: int = 10) -> pd.DataFrame:
    rows = []
    with open(trace_path) as f:
        for n, line in enumerate(f):
            if n % every:
                continue
            r = json.loads(line)
            fr = r["step"]
            seg = next((s for s in segments if s["start"] <= fr < s["end"]), None)
            rows.append(dict(
                task=task, episode=episode, frame=fr,
                subtask=subtask_text(seg, ops_json),
                belief=f"Belief: {belief_text(r['predicates'])}",
                delta=f"Delta: {delta_text(r['remaining_goal_lines'])}",
                effect=f"Effect: {effect_text(seg, ops_json)}",
                observe=f"Observe: {observe_text(r['predicates'])}",
            ))
    return pd.DataFrame(rows)


def main(out_dir: str = "out/task045", task: int = 45):
    from . import inventory
    out = pathlib.Path(out_dir)
    ops_json = json.load(open(out / f"operators_task{task:03d}.json"))
    inv = inventory.resolve_task(task)
    parts = []
    for e in inv["episodes"]:
        ep = e["raw_episode_id"]
        tp = out / f"belief_trace_{ep}.jsonl"
        if not tp.exists() or not e["annotation_exists"]:
            continue
        segs = inventory.segments(inventory.load_annotation(e["annotation_path"]))
        parts.append(build_for_episode(str(tp), segs, ops_json, task, ep))
    df = pd.concat(parts, ignore_index=True)
    df["subtask"] = "Subtask: " + df["subtask"]
    df.to_parquet(out / "cot_targets.parquet", index=False)
    print(f"cot_targets.parquet: {len(df)} rows, {len(parts)} episodes")
    print(df[df.frame == 2400].iloc[0].to_dict() if (df.frame == 2400).any() else df.iloc[50].to_dict())


if __name__ == "__main__":
    main()
