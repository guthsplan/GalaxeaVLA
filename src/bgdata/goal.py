"""Grounded goal export: COUNT lines for quantified goals + synset-instance -> scene-name map."""
import csv
import json
import pathlib
import re

import os

# Data root: the directory holding b1k_demos/, b1k_raw/, BEHAVIOR-1K-main/.
# Now that bgdata lives inside the GalaxeaVLA repo, the root is no longer derived from
# the package location — set BGDATA_ROOT, or run from the data directory (default: CWD).
WS = pathlib.Path(os.environ.get("BGDATA_ROOT", os.getcwd())).resolve()
CATEGORY_MAP_CSV = WS / "BEHAVIOR-1K-main/bddl3/bddl/generated_data/category_mapping.csv"


def synset_to_categories() -> dict[str, list[str]]:
    out = {}
    with open(CATEGORY_MAP_CSV) as f:
        for row in csv.DictReader(f):
            syn = row.get("synset") or row.get("Synset")
            cat = row.get("category") or row.get("Category")
            if syn and cat:
                out.setdefault(syn.strip(), []).append(cat.strip())
    return out


def parse_bddl(path: str) -> dict:
    txt = pathlib.Path(path).read_text()
    objects_block = txt.split("(:objects")[1].split(")")[0]
    inst_by_synset = {}
    for line in objects_block.strip().splitlines():
        if "-" not in line:
            continue
        insts, syn = line.rsplit("-", 1)
        inst_by_synset[syn.strip()] = insts.split()
    goal = txt.split("(:goal")[1]
    foralls = re.findall(r"\(forall\s*\(\?\S+\s*-\s*(\S+)\)\s*\((\S+)\s+\?\S+\)\s*\)", goal)
    # :init simple literals: (pred a [b]) and (not (pred a)) — inroom/agent lines skipped later
    init_txt = txt.split("(:init")[1].split("(:goal")[0]
    init_lits = []
    for m in re.finditer(r"(\(not\s*)?\((\w+)((?:\s+[\w.\d_]+)+)\)", init_txt):
        neg, pred, args = bool(m.group(1)), m.group(2), m.group(3).split()
        init_lits.append((pred, args, not neg))
    return dict(instances=inst_by_synset, forall_goals=foralls, init_literals=init_lits)


def ground_goal(bddl_path: str, scene_objects: list[str]) -> dict:
    """COUNT goal lines + synset instance -> scene object mapping (category matching)."""
    parsed = parse_bddl(bddl_path)
    s2c = synset_to_categories()
    by_cat = {}
    for o in scene_objects:
        by_cat.setdefault(o.rsplit("_", 1)[0] if o[-1].isdigit() else o, []).append(o)
        # scene fixture names look like fridge_dszchb_0 -> category = first token(s) before model id
    # robust category extraction: try full prefix matches against known categories
    def scene_category(o):
        parts = o.split("_")
        for k in range(len(parts) - 1, 0, -1):
            cand = "_".join(parts[:k])
            yield cand

    mapping, goal_lines = {}, []
    for syn, insts in parsed["instances"].items():
        cats = s2c.get(syn, [])
        matched = []
        for o in scene_objects:
            for cand in scene_category(o):
                if cand in cats:
                    matched.append(o)
                    break
        for i, inst in enumerate(sorted(insts)):
            mapping[inst] = sorted(matched)[i] if i < len(matched) else None
    for syn, pred in parsed["forall_goals"]:
        n = len(parsed["instances"].get(syn, []))
        goal_lines.append(f"({pred} ?x) 0/{n} [{syn}]")
    # ground :init literals to scene names; drop lines with unmapped instances (inroom, agent, floor)
    init_lines = []
    for pred, args, value in parsed["init_literals"]:
        if pred == "inroom":
            continue
        scene_args = [mapping.get(a) for a in args]
        if any(a is None for a in scene_args):
            continue
        init_lines.append([f"({pred}{''.join(' ' + a for a in scene_args)})", value])
    return dict(goal_lines=goal_lines, mapping=mapping, init_lines=init_lines)


def save_goal(task_index: int, bddl_path: str, scene_objects: list[str], episodes: list[int], path: str):
    g = ground_goal(bddl_path, scene_objects)
    out = dict(task=task_index, bddl=str(bddl_path), goal_lines=g["goal_lines"],
               synset_to_scene=g["mapping"], init_lines=g["init_lines"],
               note="mapping is category-matched; instances under a quantified goal are "
                    "exchangeable and only counted (COUNT lines), never tracked by identity",
               episodes=episodes)
    pathlib.Path(path).write_text(json.dumps(out, indent=2))
    return out
