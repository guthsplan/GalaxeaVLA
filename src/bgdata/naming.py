"""Reconcile the two object-naming systems via synsets.

annotation / scene names        BDDL :objects
  fridge_dszchb_0        <-->     electric_refrigerator.n.01_1
  hotdog_207             <-->     hotdog.n.02_{1,2}

Bridge: scene name --(longest-prefix category parse)--> category
        --(category_mapping.csv)--> synset --(hypernym walk if needed)--> BDDL synset.

`audit_all_tasks()` runs the bridge over every task's annotations vs its BDDL file and
reports, per annotation object: matched (direct / via hierarchy), fixture-not-in-goal,
or unresolved — plus instance-count agreement for quantified synsets.
"""
import csv
import json
import pathlib
import re
from collections import defaultdict

import os

# Data root: the directory holding b1k_demos/, b1k_raw/, BEHAVIOR-1K-main/.
# Now that bgdata lives inside the GalaxeaVLA repo, the root is no longer derived from
# the package location — set BGDATA_ROOT, or run from the data directory (default: CWD).
WS = pathlib.Path(os.environ.get("BGDATA_ROOT", os.getcwd())).resolve()
GEN = WS / "BEHAVIOR-1K-main/bddl3/bddl/generated_data"
BDDL_ROOT = WS / "BEHAVIOR-1K-main/bddl3/bddl/activity_definitions"


def load_cat2syn() -> dict[str, str]:
    out = {}
    with open(GEN / "category_mapping.csv") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "").strip()
            syn = (row.get("synset") or "").strip()
            if cat and syn:
                out[cat] = syn
    return out


def load_parents() -> dict[str, list[str]]:
    """synset -> list of ancestor synsets (from output_hierarchy.json)."""
    tree = json.load(open(GEN / "output_hierarchy.json"))
    parents: dict[str, list[str]] = {}

    def walk(node, chain):
        name = node["name"]
        parents.setdefault(name, list(chain))
        for c in node.get("children", []):
            walk(c, chain + [name])

    walk(tree, [])
    return parents


# annotation freeform names -> canonical category (audited over all 100 tasks; the
# remaining freeform names without any category counterpart stay 'unparsed' on purpose)
ALIASES = {
    "brown_door": "door",
    "kitchen_door": "door",
    "wooden_door": "door",
    "slide_door": "sliding_door",
    "flipup_countertop": "countertop",
    "chair": "straight_chair",
}
# annotation entries that name the robot or its parts, not scene objects
ROBOT_PARTS = {"robot", "left", "right", "robot_r1"}
# custom-synset transition prefixes (see bddl3 synsets.csv is_custom naming convention)
TRANSITION_PREFIXES = ("cooked__", "diced__", "half__", "sliced__", "melted__")


def base_lemma(synset: str) -> str:
    """cooked__diced__chili.n.01 -> chili · beet.n.02 -> beet"""
    lemma = synset.rsplit(".n.", 1)[0]
    changed = True
    while changed:
        changed = False
        for p in TRANSITION_PREFIXES:
            if lemma.startswith(p):
                lemma = lemma[len(p):]
                changed = True
    return lemma


class NameBridge:
    def __init__(self):
        self.cat2syn = load_cat2syn()
        self.parents = load_parents()

    def scene_to_category(self, name: str) -> str | None:
        """Longest-prefix category parse, full name first:
        bar_soap -> bar_soap (not 'bar') · fridge_dszchb_0 -> fridge · hotdog_207 -> hotdog"""
        name = ALIASES.get(name, name)
        parts = name.split("_")
        for k in range(len(parts), 0, -1):
            cand = "_".join(parts[:k])
            if cand in self.cat2syn:
                return cand
        return None

    def scene_to_synset(self, name: str) -> str | None:
        cat = self.scene_to_category(name)
        return self.cat2syn.get(cat) if cat else None

    def match(self, scene_name: str, bddl_synsets: set[str]) -> tuple[str, str | None]:
        """-> (status, matched_bddl_synset).
        status: robot | direct | ancestor | descendant | transition | lemma | fixture | unparsed
        - transition: scene object is a transition product (half_/diced_/cooked_ chain);
          matched to the BDDL synset sharing its base lemma (prefers the base object form)
        - lemma:   annotation shorthand equal to a BDDL synset's lemma (e.g. 'cabinet')
        - fixture: parsed to a synset that is deliberately absent from the goal
        """
        if scene_name in ROBOT_PARTS:
            return "robot", None
        syn = self.scene_to_synset(scene_name)
        if syn is not None:
            if syn in bddl_synsets:
                return "direct", syn
            for anc in self.parents.get(syn, []):        # BDDL uses a more general synset
                if anc in bddl_synsets:
                    return "ancestor", anc
            for b in bddl_synsets:                       # BDDL more specific (rare)
                if syn in self.parents.get(b, []):
                    return "descendant", b
            # transition products: half__X / diced__X / cooked__... hang under a different
            # hypernym (e.g. sandwich.n.01), so the hierarchy walk can't recover them —
            # match by shared base lemma instead
            lem = base_lemma(syn)
            cands = [b for b in bddl_synsets if base_lemma(b) == lem]
            if cands:
                cands.sort(key=lambda b: b.count("__"))  # prefer the un-prefixed base object
                return "transition", cands[0]
        # annotation shorthand: bare synset lemma without instance suffix ('cabinet', 'sink')
        for b in bddl_synsets:
            if b.rsplit(".n.", 1)[0] == scene_name:
                return "lemma", b
        if syn is not None:
            return "fixture", None
        return "unparsed", None


def bddl_objects(task_name: str) -> dict[str, int]:
    """synset -> instance count from the :objects block."""
    txt = (BDDL_ROOT / task_name / "problem0.bddl").read_text()
    block = txt.split("(:objects")[1].split(")")[0]
    out: dict[str, int] = {}
    for line in block.strip().splitlines():
        if "-" not in line:
            continue
        insts, syn = line.rsplit("-", 1)
        out[syn.strip()] = len(insts.split())
    return out


def audit_all_tasks(episodes_per_task: int = 3) -> dict:
    import pandas as pd
    from . import inventory

    bridge = NameBridge()
    tasks = inventory.load_tasks()
    rows, count_rows = [], []
    for tid, trow in tasks.iterrows():
        tname = trow["task_name"]
        bsyn = bddl_objects(tname)
        bset = set(bsyn)
        ann_dir = inventory.DEMOS / "annotations" / f"task-{tid:04d}"
        objs: set[str] = set()
        for p in sorted(ann_dir.glob("*.json"))[:episodes_per_task]:
            ann = json.load(open(p))
            def flatten(x):
                for e in x:
                    if isinstance(e, list):
                        yield from flatten(e)
                    elif isinstance(e, str) and e:
                        yield e
            for sk in ann["skill_annotation"]:
                objs.update(flatten(sk["object_id"]))
        cat_insts = defaultdict(set)
        for o in sorted(objs):
            status, msyn = bridge.match(o, bset)
            rows.append(dict(task=tid, task_name=tname, scene_name=o,
                             category=bridge.scene_to_category(o),
                             synset=bridge.scene_to_synset(o),
                             status=status, bddl_synset=msyn))
            if msyn:
                cat_insts[msyn].add(o)
        for syn, insts in cat_insts.items():
            count_rows.append(dict(task=tid, synset=syn,
                                   n_scene=len(insts), n_bddl=bsyn.get(syn, 0),
                                   count_ok=len(insts) == bsyn.get(syn, 0)))
    return dict(objects=pd.DataFrame(rows), counts=pd.DataFrame(count_rows))


if __name__ == "__main__":
    import pandas as pd
    res = audit_all_tasks()
    df, cnt = res["objects"], res["counts"]
    print("=== per-object match status (all 100 tasks, 3 eps each) ===")
    print(df.status.value_counts().to_string())
    print(f"\nobjects total: {len(df)} · tasks: {df.task.nunique()}")
    print("\n=== unparsed / unmatched examples ===")
    bad = df[df.status.isin(["unparsed", "fixture"])]
    print(bad.groupby(["status", "scene_name"]).size().sort_values(ascending=False).head(20).to_string())
    print("\n=== instance-count agreement (matched synsets) ===")
    print(cnt.count_ok.value_counts().to_string())
    print(cnt[~cnt.count_ok].head(15).to_string())
    out = WS / "out/naming_audit.csv"
    df.to_csv(out, index=False)
    print(f"\nfull table -> {out}")
