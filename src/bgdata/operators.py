"""Operator library extraction from skill-annotation segments + predicate labels.

pre : predicate (or its negation) true at segment start in >= PRE_SUPPORT of segments
eff : predicate value change start->end consistent in >= EFF_SUPPORT of segments
Arguments are abstracted by their position in object_id; inside/ontop are merged into
the abstract relation in_or_on, and the concrete relation goes to the per-operator `rel`.
"""
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

PRE_SUPPORT = 0.95
EFF_SUPPORT = 0.85
SETTLE = 6  # frames (30 fps) after segment end before sampling the post state

SKILL_TO_OP = {
    "move to": ("move_to", ["?o"]),
    "open door": ("open_door", ["?d"]),
    "close door": ("close_door", ["?d"]),
    "pick up from": ("pick_up_from", ["?o", "?r"]),
    "place on": ("place_on", ["?o", "?r"]),
    "place in": ("place_in", ["?o", "?r"]),
    "turn on switch": ("turn_on_switch", ["?d"]),
}
OP_ARGS = {v[0]: v[1] for v in SKILL_TO_OP.values()}
TEXT = {
    "move_to": "move {mp}to {?o}",
    "open_door": "open door {?d}",
    "close_door": "close door {?d}",
    "pick_up_from": "pick up {mp}{?o} from {?r}",
    "place_on": "place {?o} on {?r}",
    "place_in": "place {?o} in {?r}",
    "turn_on_switch": "turn on switch {?d}",
}

PRED_RE = re.compile(r"^\((\S+)((?:\s+\S+)*)\)$")


def parse_key(key: str):
    m = PRED_RE.match(key)
    name = m.group(1)
    args = m.group(2).split()
    return name, args


def snapshot(df_ep: pd.DataFrame, frame: int) -> dict[str, bool]:
    frames = df_ep["frame"].values
    tgt = frames[np.argmin(np.abs(frames - frame))]
    sub = df_ep[df_ep.frame == tgt]
    snap = dict(zip(sub.key, sub.value))
    snap["(handempty)"] = not any(v for k, v in snap.items() if k.startswith("(inhand ") )
    return snap


def abstract_snapshot(snap: dict[str, bool], mapping: dict[str, str]) -> dict[str, bool]:
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
    per_op = defaultdict(list)  # op -> list of (pre_snap, post_snap, concrete_rel, skill_id, skill_type)
    for ep, segs in segs_by_ep.items():
        df_ep = label_dfs[ep]
        N = n_frames_by_ep[ep]
        for s in segs:
            if s["skill"] not in SKILL_TO_OP:
                continue
            op, argnames = SKILL_TO_OP[s["skill"]]
            objs = s["objects"][: len(argnames)]
            if len(objs) < len(argnames):
                continue
            mapping = dict(zip(objs, argnames))
            pre_raw = snapshot(df_ep, s["start"])
            post_raw = snapshot(df_ep, min(s["end"] + SETTLE, N - 1))
            pre, _ = abstract_snapshot(pre_raw, mapping)
            post, post_rel = abstract_snapshot(post_raw, mapping)
            per_op[op].append(dict(pre=pre, post=post, post_rel=post_rel,
                                   skill_id=s["skill_id"], skill_type=s["skill_type"]))

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
            pos_change = np.mean([(not r["pre"][k]) and r["post"][k] for r in defined]) if defined else 0
            neg_change = np.mean([r["pre"][k] and (not r["post"][k]) for r in defined]) if defined else 0
            support[k] = (true_at_start, pos_change, neg_change)
            stats_rows.append(dict(op=op, predicate=k, n_segments=n,
                                   support_pre_true=round(float(true_at_start), 3),
                                   support_eff_pos=round(float(pos_change), 3),
                                   support_eff_neg=round(float(neg_change), 3)))
        # effects: consistent value changes ((handempty) is derived from inhand — pre only)
        for k, (t0, pos, neg) in support.items():
            if k == "(handempty)":
                continue
            if pos >= EFF_SUPPORT:
                eff_list.append(k)
            elif neg >= EFF_SUPPORT:
                eff_list.append(f"(not {k})")
        # preconditions: positive if reliably true at start; negative only for the
        # delete/add-duals of this op's own positive effects (avoids vacuous negatives)
        for k, (t0, pos, neg) in support.items():
            if t0 >= PRE_SUPPORT:
                pre_list.append(k)
            elif t0 <= 1 - PRE_SUPPORT and k in eff_list:
                pre_list.append(f"(not {k})")
        # rel: majority concrete relation among positive in_or_on effects
        rels = [nm for r in recs for nm in r["post_rel"].values()]
        rel = ""
        if any(e.startswith("(in_or_on") for e in eff_list) and rels:
            rel = max(set(rels), key=rels.count)
        disturbs = []
        if any("in_or_on ?o ?r" in e for e in eff_list):
            disturbs = ["(in_or_on ?y ?r)"]
        sid = recs[0]["skill_id"]
        stype = recs[0]["skill_type"] or "uncoordinated"
        args = OP_ARGS[op]
        ops_json[op] = dict(skill_id=int(sid), args=args, pre=pre_list, eff=eff_list,
                            text=TEXT[op], skill_type=stype, disturbs=disturbs, rel=rel)
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
            continue  # ambiguous grounding; skip for assertion
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
    """Accumulate operator pre+eff in annotation order; mismatch vs labels.

    Reports two rates per key: per-frame (all 10 Hz frames) and boundary (only at
    segment-start frames, i.e. steady states between skills).
    """
    rows = []
    for ep, segs in segs_by_ep.items():
        df_ep = label_dfs[ep]
        frames = np.sort(df_ep.frame.unique())
        keys = sorted(k for k in df_ep.key.unique()
                      if parse_key(k)[0] not in ("inhand_left", "inhand_right"))
        gt = df_ep.pivot_table(index="frame", columns="key", values="value", aggfunc="first")
        pred = dict(gt.loc[frames[0]][keys])  # start from initial labels
        seg_iter = sorted([s for s in segs if s["skill"] in SKILL_TO_OP], key=lambda s: s["end"])
        boundary_frames = {frames[np.argmin(np.abs(frames - s["start"]))] for s in seg_iter}
        si = 0
        mism = {k: 0 for k in keys}
        bmism = {k: 0 for k in keys}
        for fr in frames:
            while si < len(seg_iter) and seg_iter[si]["end"] <= fr:
                s = seg_iter[si]
                op = ops_json.get(SKILL_TO_OP[s["skill"]][0])
                if op:
                    objs = s["objects"][: len(op["args"])]
                    # assert preconditions (they held while the skill executed)
                    for gk, gv in concretize_pre(op, objs):
                        if gk in pred:
                            pred[gk] = gv
                    effs = concretize_effects(op, objs)
                    # mobile-base exclusivity: reaching a new target leaves the previous one
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
            if not mp:
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
