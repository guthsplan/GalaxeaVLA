"""Operator library extraction from skill-annotation segments + predicate labels (all skills).

pre : predicate (or its negation) true at segment start in >= PRE_SUPPORT of segments
eff : predicate value change start->end consistent in >= EFF_SUPPORT of segments
Arguments are abstracted by their position in object_id; inside/ontop are merged into
the abstract relation in_or_on, and the concrete relation goes to the per-operator `rel`.

The skill vocabulary of the 100-task annotations (skill_summary.csv) is covered by SKILL_TABLE;
any skill not listed falls back to a generic operator named after the skill with positional
arguments ?o ?r ?t, so no task can silently drop its segments.
"""
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

PRE_SUPPORT = 0.95
EFF_SUPPORT = 0.85
MIN_APPLICABLE = 5        # segments where the effect could flip, needed before it is trusted
MIN_APPLICABLE_FRAC = 0.1  # ... and at least this fraction of the op's segments
SETTLE = 6  # frames (30 fps) after segment end before sampling the post state

# skill text -> (operator name, argument names, Subtask text template)
# {mp} = memory prefix ("the other ", "back "), filled by cot_targets.subtask_text
SKILL_TABLE = {
    "move to":            ("move_to",            ["?o"],             "move {mp}to {?o}"),
    "pick up from":       ("pick_up_from",       ["?o", "?r"],       "pick up {mp}{?o} from {?r}"),
    "place on":           ("place_on",           ["?o", "?r"],       "place {?o} on {?r}"),
    "place in":           ("place_in",           ["?o", "?r"],       "place {?o} in {?r}"),
    "place on next to":   ("place_on_next_to",   ["?o", "?r", "?t"], "place {?o} on {?r} next to {?t}"),
    "place in next to":   ("place_in_next_to",   ["?o", "?r", "?t"], "place {?o} in {?r} next to {?t}"),
    "place under":        ("place_under",        ["?o", "?r"],       "place {?o} under {?r}"),
    "push to":            ("push_to",            ["?o", "?r"],       "push {?o} to {?r}"),
    "open door":          ("open_door",          ["?d"],             "open door {?d}"),
    "close door":         ("close_door",         ["?d"],             "close door {?d}"),
    "open drawer":        ("open_drawer",        ["?d"],             "open drawer {?d}"),
    "close drawer":       ("close_drawer",       ["?d"],             "close drawer {?d}"),
    "open lid":           ("open_lid",           ["?d"],             "open lid {?d}"),
    "close lid":          ("close_lid",          ["?d"],             "close lid {?d}"),
    "turn on switch":     ("turn_on_switch",     ["?d"],             "turn on switch {?d}"),
    "turn off switch":    ("turn_off_switch",    ["?d"],             "turn off switch {?d}"),
    "press":              ("press",              ["?d"],             "press {?d}"),
    "chop":               ("chop",               ["?o", "?r"],       "chop {?r} with {?o}"),
    "pour":               ("pour",               ["?o", "?s", "?r"], "pour {?o} from {?s} into {?r}"),
    "sweep surface":      ("sweep_surface",      ["?o", "?r"],       "sweep {?r} with {?o}"),
    "sweep off":          ("sweep_off",          ["?o", "?r"],       "sweep {?o} off {?r}"),
    "wipe hard":          ("wipe_hard",          ["?o", "?r"],       "wipe {?r} with {?o}"),
    "spray":              ("spray",              ["?o", "?r"],       "spray {?r} with {?o}"),
    "hand over":          ("hand_over",          ["?o"],             "hand over {?o}"),
    "turn to":            ("turn_to",            ["?o", "?r"],       "turn {?o} to {?r}"),
    "insert":             ("insert",             ["?o", "?r"],       "insert {?o} into {?r}"),
    "attach":             ("attach",             ["?o", "?r"],       "attach {?o} to {?r}"),
    "hang":               ("hang",               ["?o", "?r"],       "hang {?o} on {?r}"),
    "ignite":             ("ignite",             ["?o", "?r"],       "ignite {?r} with {?o}"),
    "tip over":           ("tip_over",           ["?o"],             "tip over {?o}"),
    "hold":               ("hold",               ["?o"],             "hold {?o}"),
    "release":            ("release",            ["?o"],             "release {?o}"),
    "lift":               ("lift",               ["?o"],             "lift {?o}"),
    "pull tray":          ("pull_tray",          ["?d"],             "pull tray {?d}"),
    "push tray":          ("push_tray",          ["?d"],             "push tray {?d}"),
}
_GENERIC_ARGS = ["?o", "?r", "?t", "?u"]


def op_for(skill: str, n_objects: int | None = None) -> tuple[str, list[str], str]:
    """(op_name, arg_names, text_template) for a skill; generic fallback for unknown skills."""
    if skill in SKILL_TABLE:
        return SKILL_TABLE[skill]
    n = max(1, min(n_objects or 1, len(_GENERIC_ARGS)))
    args = _GENERIC_ARGS[:n]
    text = skill + "".join(" {" + a + "}" for a in args)
    return skill.replace(" ", "_"), args, text


# backwards-compatible views used by cot_targets / belief_trace / validation
SKILL_TO_OP = {k: (v[0], v[1]) for k, v in SKILL_TABLE.items()}
OP_ARGS = {v[0]: v[1] for v in SKILL_TABLE.values()}
TEXT = {v[0]: v[2] for v in SKILL_TABLE.values()}

PRED_RE = re.compile(r"^\((\S+)((?:\s+\S+)*)\)$")


def parse_key(key: str):
    m = PRED_RE.match(key)
    name = m.group(1)
    args = m.group(2).split()
    return name, args


def seg_op(seg: dict):
    """(op_name, arg_names, objs) for a resolved segment, or None when no object is known."""
    objs = [o for o in seg.get("objects", []) if o]
    op, args, _ = op_for(seg["skill"], len(objs))
    objs = objs[: len(args)]
    if len(objs) < len(args):
        return None
    return op, args, objs


def snapshot(df_ep: pd.DataFrame, frame: int) -> dict[str, bool]:
    frames = df_ep["frame"].values
    tgt = frames[np.argmin(np.abs(frames - frame))]
    sub = df_ep[df_ep.frame == tgt]
    snap = dict(zip(sub.key, sub.value))
    snap["(handempty)"] = not any(v for k, v in snap.items() if k.startswith("(inhand "))
    return snap


def abstract_snapshot(snap: dict[str, bool], mapping: dict[str, str]):
    """Keep only predicates whose args are all in `mapping`; merge inside/ontop -> in_or_on."""
    out = {}
    inon = defaultdict(bool)
    inon_concrete = {}
    for key, val in snap.items():
        name, args = parse_key(key)
        # inhand_left/right are folded into (inhand ?o); visited is DERIVED from reachable
        # (sticky), so it is neither a precondition nor an effect candidate
        if name in ("inhand_left", "inhand_right", "visited"):
            continue
        if not all(a in mapping for a in args):
            continue
        aargs = [mapping[a] for a in args]
        akey = f"({name}{''.join(' ' + a for a in aargs)})" if aargs else f"({name})"
        if name in ("inside", "ontop"):
            t = tuple(aargs)
            if val:
                inon_concrete[t] = name
            inon[t] |= val
        else:
            out[akey] = val
    for t, val in inon.items():
        out[f"(in_or_on{''.join(' ' + a for a in t)})"] = val
    return out, inon_concrete


def extract_operators(label_dfs: dict[int, pd.DataFrame], segs_by_ep: dict[int, list[dict]],
                      n_frames_by_ep: dict[int, int]):
    """Returns (operators dict in scaffold schema, stats DataFrame)."""
    per_op = defaultdict(list)
    for ep, segs in segs_by_ep.items():
        df_ep = label_dfs[ep]
        N = n_frames_by_ep[ep]
        for s in segs:
            so = seg_op(s)
            if so is None:
                continue
            op, argnames, objs = so
            mapping = dict(zip(objs, argnames))
            pre_raw = snapshot(df_ep, s["start"])
            post_raw = snapshot(df_ep, min(s["end"] + SETTLE, N - 1))
            pre, _ = abstract_snapshot(pre_raw, mapping)
            post, post_rel = abstract_snapshot(post_raw, mapping)
            per_op[op].append(dict(pre=pre, post=post, post_rel=post_rel, args=argnames,
                                   skill_id=s["skill_id"], skill_type=s["skill_type"],
                                   text=op_for(s["skill"], len(objs))[2]))

    ops_json, stats_rows = {}, []
    for op, recs in per_op.items():
        n = len(recs)
        keys = set()
        for r in recs:
            keys |= set(r["pre"]) | set(r["post"])
        pre_list, eff_list = [], []
        support = {}
        for k in sorted(keys):
            true_at_start = np.mean([r["pre"].get(k, False) for r in recs])
            defined = [r for r in recs if k in r["pre"] and k in r["post"]]
            # an effect is judged where it is APPLICABLE: a door already open when the second
            # "open door" segment starts cannot flip again (the unconditional flip rate then
            # saturates at 50 % and the effect would be lost)
            app_pos = [r for r in defined if not r["pre"][k]]
            app_neg = [r for r in defined if r["pre"][k]]
            true_at_end = np.mean([r["post"][k] for r in defined]) if defined else 0
            min_app = max(MIN_APPLICABLE, MIN_APPLICABLE_FRAC * len(defined))
            pos_change = np.mean([r["post"][k] for r in app_pos]) if len(app_pos) >= min_app else 0
            neg_change = np.mean([not r["post"][k] for r in app_neg]) if len(app_neg) >= min_app else 0
            # ... and must hold at the end of (nearly) every segment, else it is not this op's effect
            if true_at_end < EFF_SUPPORT:
                pos_change = 0
            if true_at_end > 1 - EFF_SUPPORT:
                neg_change = 0
            support[k] = (true_at_start, pos_change, neg_change)
            stats_rows.append(dict(op=op, predicate=k, n_segments=n,
                                   support_pre_true=round(float(true_at_start), 3),
                                   support_post_true=round(float(true_at_end), 3),
                                   n_applicable_pos=len(app_pos), n_applicable_neg=len(app_neg),
                                   support_eff_pos=round(float(pos_change), 3),
                                   support_eff_neg=round(float(neg_change), 3)))
        for k, (t0, pos, neg) in support.items():
            if k == "(handempty)":
                continue
            if pos >= EFF_SUPPORT:
                eff_list.append(k)
            elif neg >= EFF_SUPPORT:
                eff_list.append(f"(not {k})")
        for k, (t0, pos, neg) in support.items():
            if t0 >= PRE_SUPPORT:
                pre_list.append(k)
            elif t0 <= 1 - PRE_SUPPORT and k in eff_list:
                pre_list.append(f"(not {k})")
        rels = [nm for r in recs for nm in r["post_rel"].values()]
        rel = ""
        if any(e.startswith("(in_or_on") for e in eff_list) and rels:
            rel = max(set(rels), key=rels.count)
        disturbs = []
        if any("in_or_on ?o ?r" in e for e in eff_list):
            disturbs = ["(in_or_on ?y ?r)"]
        sid = recs[0]["skill_id"]
        stype = recs[0]["skill_type"] or "uncoordinated"
        ops_json[op] = dict(skill_id=int(sid), args=recs[0]["args"], pre=pre_list, eff=eff_list,
                            text=recs[0]["text"], skill_type=stype, disturbs=disturbs, rel=rel,
                            n_segments=n)
    return ops_json, pd.DataFrame(stats_rows)


# ---------------- validation ----------------
def concretize_effects(op: dict, objs: list[str]) -> list[tuple[str, bool]]:
    """Ground an operator's effects; in_or_on uses op['rel'] for positive, both for negative."""
    mapping = dict(zip(op["args"], objs))
    out = []
    for e in op["eff"]:
        neg = e.startswith("(not ")
        inner = e[5:-1] if neg else e
        name, args = parse_key(inner)
        gargs = [mapping.get(a, a) for a in args]
        if name == "in_or_on":
            names = [op["rel"] or "inside"] if not neg else ["inside", "ontop"]
            for nm in names:
                out.append((f"({nm}{''.join(' ' + a for a in gargs)})", not neg))
        else:
            out.append((f"({name}{''.join(' ' + a for a in gargs)})", not neg))
    return out


def concretize_pre(op: dict, objs: list[str]) -> list[tuple[str, bool]]:
    """Ground positive/negative preconditions (STRIPS execution semantics: pre held while acting)."""
    mapping = dict(zip(op["args"], objs))
    out = []
    for e in op["pre"]:
        neg = e.startswith("(not ")
        inner = e[5:-1] if neg else e
        name, args = parse_key(inner)
        if name in ("handempty", "in_or_on"):
            continue
        gargs = [mapping.get(a, a) for a in args]
        out.append((f"({name}{''.join(' ' + a for a in gargs)})", not neg))
    return out


def _derive_visited(pred: dict):
    for k in list(pred):
        if k.startswith("(reachable ") and pred[k]:
            vk = k.replace("(reachable ", "(visited ")
            if vk in pred:
                pred[vk] = True


def validate_accumulation(ops_json: dict, label_dfs: dict[int, pd.DataFrame],
                          segs_by_ep: dict[int, list[dict]]) -> pd.DataFrame:
    """Accumulate operator pre+eff in annotation order; mismatch vs labels (per-frame + boundary)."""
    rows = []
    for ep, segs in segs_by_ep.items():
        df_ep = label_dfs[ep]
        frames = np.sort(df_ep.frame.unique())
        keys = sorted(k for k in df_ep.key.unique()
                      if parse_key(k)[0] not in ("inhand_left", "inhand_right"))
        gt = df_ep.pivot_table(index="frame", columns="key", values="value", aggfunc="first")
        pred = dict(gt.loc[frames[0]][keys])
        seg_iter = sorted([s for s in segs if seg_op(s) is not None], key=lambda s: s["end"])
        boundary_frames = {frames[np.argmin(np.abs(frames - s["start"]))] for s in seg_iter}
        si = 0
        mism = {k: 0 for k in keys}
        bmism = {k: 0 for k in keys}
        for fr in frames:
            while si < len(seg_iter) and seg_iter[si]["end"] <= fr:
                s = seg_iter[si]
                op_name, _, objs = seg_op(s)
                op = ops_json.get(op_name)
                if op:
                    for gk, gv in concretize_pre(op, objs):
                        if gk in pred:
                            pred[gk] = gv
                    effs = concretize_effects(op, objs)
                    if any(gk.startswith("(reachable ") and gv for gk, gv in effs):
                        for k in pred:
                            if k.startswith("(reachable "):
                                pred[k] = False
                    for gk, gv in effs:
                        if gk in pred:
                            pred[gk] = gv
                    _derive_visited(pred)
                si += 1
            row = gt.loc[fr]
            for k in keys:
                if bool(row[k]) != bool(pred[k]):
                    mism[k] += 1
                    if fr in boundary_frames:
                        bmism[k] += 1
        for k in keys:
            rows.append(dict(episode=ep, key=k, mismatch_rate=mism[k] / len(frames),
                             boundary_mismatch_rate=bmism[k] / max(1, len(boundary_frames))))
    return pd.DataFrame(rows)


def validate_memory_prefix(label_dfs, segs_by_ep) -> pd.DataFrame:
    """'the other': target is in_or_on the source ?r and exactly one other same-category
    instance is NOT in_or_on ?r at segment start. 'back': target visited at start."""
    rows = []
    for ep, segs in segs_by_ep.items():
        df_ep = label_dfs[ep]
        for s in segs:
            mp = s["memory_prefix"]
            if not mp or not s.get("objects"):
                continue
            snap = snapshot(df_ep, s["start"])
            if mp == "the other":
                tgt, src = s["objects"][0], s["objects"][-1]
                cat = tgt.rsplit("_", 1)[0]
                same_cat = sorted({parse_key(k)[1][0] for k in snap
                                   if parse_key(k)[0] in ("inside", "ontop")
                                   and parse_key(k)[1][0].rsplit("_", 1)[0] == cat})

                def at_src(o):
                    return (snap.get(f"(inside {o} {src})", False)
                            or snap.get(f"(ontop {o} {src})", False))
                others = [o for o in same_cat if o != tgt]
                ok = at_src(tgt) and sum(not at_src(o) for o in others) == 1
            elif mp == "back":
                tgt = s["objects"][0]
                ok = snap.get(f"(visited {tgt})", False)
            else:
                ok = None
            rows.append(dict(episode=ep, skill_idx=s["skill_idx"], memory_prefix=mp,
                             skill=s["skill"], target=s["objects"][0], passed=ok))
    return pd.DataFrame(rows)


def save_operators(ops_json: dict, path: str):
    with open(path, "w") as f:
        json.dump(ops_json, f, indent=2)
