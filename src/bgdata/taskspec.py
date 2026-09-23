"""Task specification derived from the data itself, for any of the 100 challenge tasks.

Every raw HDF5 carries a `scene_file` attribute (the full OmniGibson scene JSON it was
recorded from). From it we get, without any per-task hardcoding:
  * metadata.task.inst_to_name       BDDL instance -> scene object name (incl. the robot name)
  * objects_info.init_info[name]      category, model, scale, fixed_base
  * state.registry.object_registry    initial root pose, joint count, non-kinematic state
                                      schema (the exact serialization layout of each block)

bddl3 (`ObjectTaxonomy`, `parse_problem`) supplies synset abilities/params (cook temperature,
openable, fillable, ...) and the goal/init literals; the asset inventory supplies bbox extents;
the optional asset metadata supplies openable joint ids.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from hashlib import md5

import numpy as np

from . import inventory

WS = inventory.WS
INVENTORY_JSON = WS / "BEHAVIOR-1K-main/bddl3/bddl/generated_data/object_inventory.json"
ASSET_META_ROOT = pathlib.Path(os.environ.get("BGDATA_ASSET_META", WS / "behavior-1k-assets/objects"))
# <scene>/json/<scene>_task_<name>_0_0_template.json: full-scene state (poses of fixtures that the
# recorded scene_file leaves out because they never became active)
TASK_INSTANCES_ROOT = pathlib.Path(os.environ.get("BGDATA_TASK_INSTANCES",
                                                  WS / "2026-challenge-task-instances/scenes"))

# serialized float counts of the non-kinematic states (OmniGibson object_states/*.serialize)
NONKIN_SIZE = {
    "Temperature": 1, "MaxTemperature": 1, "ToggledOn": 2, "AttachedTo": 1,
    "SlicerActive": 3, "ParticleRemover": 1, "ParticleApplier": 1,
    # variable: first float = n_systems, then (uuid, value) pairs
    "Saturated": None, "ModifiedParticles": None,
}
ROBOT_PARTS = {"robot", "left", "right", "robot_r1", "agent"}


def get_uuid(name: str) -> int:
    """python_utils.get_uuid: int(float32(md5(name) % 1e8))."""
    return int(np.float32(int(md5(name.encode()).hexdigest(), 16) % 10**8))


def _bddl_lib():
    import bddl  # noqa: F401  (PYTHONPATH must include BEHAVIOR-1K-main/bddl3)
    from bddl.object_taxonomy import ObjectTaxonomy
    from bddl.parsing import parse_problem
    return ObjectTaxonomy(), parse_problem


_TAXONOMY = None


def taxonomy():
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY, _ = _bddl_lib()
    return _TAXONOMY


def parse_bddl(task_name: str):
    """-> (objects: {synset: [inst...]}, init: [literal], goal: [expr]) via bddl3's own parser."""
    _, parse_problem = _bddl_lib()
    _domain, objects, init, goal = parse_problem(task_name, 0, "behavior-1k")
    return objects, init, goal


def synset_of_instance(inst: str) -> str:
    return inst.rsplit("_", 1)[0]


@dataclass
class ObjectInfo:
    name: str
    category: str | None
    model: str | None
    scale: np.ndarray
    extents: np.ndarray | None          # scaled bbox (m); None for objects without a model
    n_joints: int
    non_kin: list[str]                   # serialized non-kinematic states, in order
    fixed_base: bool
    init_pos: np.ndarray
    init_quat: np.ndarray
    init_joint_pos: np.ndarray
    init_non_kin: dict
    bddl_inst: str | None = None         # e.g. hotdog.n.02_1 (None for fixtures / future objects)
    synset: str | None = None
    abilities: dict = field(default_factory=dict)
    openable_joints: list[int] = field(default_factory=list)
    task_relevant: bool = False

    @property
    def uuid(self) -> int:
        return get_uuid(self.name)

    @property
    def movable(self) -> bool:
        return not self.fixed_base

    def has(self, ability: str) -> bool:
        return ability in self.abilities

    def param(self, ability: str, key: str, default=None):
        return (self.abilities.get(ability) or {}).get(key, default)


class TaskSpec:
    def __init__(self, task_index: int, scene_file: dict, config: dict, annotation_objects: set[str]):
        self.task_index = task_index
        self.task_name = config["task"]["activity_name"]
        meta = scene_file["metadata"]["task"]
        self.inst_to_name: dict[str, str] = dict(meta["inst_to_name"])
        self.name_to_inst = {v: k for k, v in self.inst_to_name.items()}
        self.robot = next(v for k, v in self.inst_to_name.items() if k.startswith("agent."))
        self.scene_model = scene_file["init_info"]["args"]["scene_model"]
        reg = dict(scene_file["state"]["registry"]["object_registry"])
        init_info = scene_file["objects_info"]["init_info"]
        # the recorded registry only holds objects that were active at reset; fixtures the task
        # references (countertops, bookcases, floors) come from the scene's task template
        self.template_used = []
        missing = [n for n in init_info if n not in reg]
        if missing:
            tpl = self._task_template(self.scene_model, self.task_name)
            treg = ((tpl or {}).get("state", {}).get("registry", {}).get("object_registry", {}))
            for n in missing:
                if n in treg:
                    reg[n] = treg[n]
                    self.template_used.append(n)
        self.registry_order = list(reg.keys())
        self.substances = sorted(v for k, v in self.inst_to_name.items()
                                 if v not in reg and not k.startswith("agent."))
        bbox = json.load(open(INVENTORY_JSON))["bounding_box_sizes"]
        tax = taxonomy()

        self.objects: dict[str, ObjectInfo] = {}
        for name in self.registry_order:
            st = reg[name]
            args = (init_info.get(name) or {}).get("args", {})
            model = args.get("model")
            scale = np.asarray(args.get("scale", [1.0, 1.0, 1.0]), dtype=float)
            ext = np.asarray(bbox[model], dtype=float) * scale if model in bbox else None
            inst = self.name_to_inst.get(name)
            syn = synset_of_instance(inst) if inst else None
            abilities = {}
            if syn and tax.is_valid_synset(syn):
                abilities = tax.get_abilities(syn)
            info = ObjectInfo(
                name=name, category=args.get("category"), model=model, scale=scale, extents=ext,
                n_joints=len(st.get("joint_pos", [])),
                non_kin=list((st.get("non_kin") or {}).keys()),
                fixed_base=bool(args.get("fixed_base", False)),
                init_pos=np.asarray(st["root_link"]["pos"], float),
                init_quat=np.asarray(st["root_link"]["ori"], float),
                init_joint_pos=np.asarray(st.get("joint_pos", []), float),
                init_non_kin=st.get("non_kin") or {},
                bddl_inst=inst, synset=syn, abilities=abilities,
                openable_joints=self._openable_joints(args.get("category"), model),
                task_relevant=inst is not None and name != self.robot,
            )
            self.objects[name] = info
        self.robot_info = self.objects[self.robot]
        # --- annotation vocabulary -------------------------------------------------------
        # tasks 0-49 name instances (fridge_dszchb_0); tasks 50-99 name categories
        # (electric_refrigerator, tupperware). Both are resolved against the relevant objects,
        # then against scene fixtures; leftovers are substances / transition products.
        self.annotation_objects = sorted(annotation_objects)
        self.token_candidates: dict[str, list[str]] = {}
        self.unresolved_tokens: list[str] = []
        for tok in self.annotation_objects:
            cands = self.resolve_token(tok)
            if cands:
                self.token_candidates[tok] = cands
            elif tok not in ROBOT_PARTS and tok != self.robot:
                self.unresolved_tokens.append(tok)
        # objects that labels are computed for: task-relevant + fixtures the annotator names
        rel = {n for n, o in self.objects.items() if o.task_relevant}
        for tok, cands in self.token_candidates.items():
            rel.update(cands)
        self.relevant = sorted(rel)
        self.bddl_objects, self.bddl_init, self.bddl_goal = parse_bddl(self.task_name)

    # -----------------------------------------------------------------------------------
    @staticmethod
    def _task_template(scene_model: str, task_name: str) -> dict | None:
        for p in (TASK_INSTANCES_ROOT / scene_model / "json" / f"{scene_model}_task_{task_name}_0_0_template.json",
                  TASK_INSTANCES_ROOT / scene_model / "json" / f"{scene_model}_stable.json"):
            if p.exists():
                return json.load(open(p))
        return None

    @staticmethod
    def _openable_joints(category, model) -> list[int]:
        if not category or not model:
            return []
        p = ASSET_META_ROOT / category / model / "misc" / "metadata.json"
        if not p.exists():
            return []
        try:
            ids = json.load(open(p)).get("openable_joint_ids") or []
            # entries are [dof_id, "j_link_N"]; the dof id counts fixed links too, whereas the
            # serialized joint_pos vector is ordered by the movable joints (j_link_N -> N)
            out = []
            for i in ids:
                if isinstance(i, list) and len(i) > 1 and isinstance(i[1], str) and i[1].rsplit("_", 1)[-1].isdigit():
                    out.append(int(i[1].rsplit("_", 1)[-1]))
                else:
                    out.append(int(i[0]) if isinstance(i, list) else int(i))
            return out
        except Exception:
            return []

    def resolve_token(self, tok: str) -> list[str]:
        """Annotation token -> candidate scene names (possibly several for a category)."""
        if tok in self.objects and tok != self.robot:
            return [tok]
        if tok in ROBOT_PARTS or tok == self.robot:
            return []
        # exact category among task-relevant objects, then the category synset lemma
        rel = [n for n, o in self.objects.items() if o.task_relevant]
        by_cat = [n for n in rel if self.objects[n].category == tok]
        if by_cat:
            return by_cat
        tax = taxonomy()
        lemma_hits = []
        for n in rel:
            syn = self.objects[n].synset or ""
            lemma = syn.rsplit(".n.", 1)[0]
            if lemma == tok or lemma.replace("__", "_") == tok:
                lemma_hits.append(n)
            else:
                try:
                    cats = tax.get_categories(syn) if syn else []
                except Exception:
                    cats = []
                if tok in cats:
                    lemma_hits.append(n)
        if lemma_hits:
            return lemma_hits
        # loose prefix on relevant objects (name startswith tok_)
        pref = [n for n in rel if n.startswith(tok + "_")]
        if pref:
            return pref
        # scene fixtures not in the BDDL scope (burner_mjvqii_0, door_bexenl_0 ...)
        fix = [n for n, o in self.objects.items() if o.category == tok or n.startswith(tok + "_")]
        return fix

    # -----------------------------------------------------------------------------------
    def instances_of(self, synset: str) -> list[str]:
        return sorted(n for k, n in self.inst_to_name.items()
                      if synset_of_instance(k) == synset and n != self.robot)

    def scene_name(self, bddl_inst: str) -> str | None:
        return self.inst_to_name.get(bddl_inst)

    def summary(self) -> dict:
        return dict(
            task=self.task_index, task_name=self.task_name, scene_model=self.scene_model,
            robot=self.robot, n_registry=len(self.objects), template_used=self.template_used,
            substances=self.substances,
            relevant={n: dict(category=o.category, synset=o.synset, movable=o.movable,
                              n_joints=o.n_joints, non_kin=o.non_kin,
                              extents=(None if o.extents is None else [round(float(x), 3) for x in o.extents]))
                      for n, o in self.objects.items() if n in self.relevant},
            token_candidates=self.token_candidates, unresolved_tokens=self.unresolved_tokens,
        )


def load_taskspec(task_index: int, raw_path: str, annotation_objects: set[str]) -> TaskSpec:
    import h5py
    with h5py.File(raw_path, "r") as h:
        scene_file = json.loads(h["data"].attrs["scene_file"])
        config = json.loads(h["data"].attrs["config"])
    return TaskSpec(task_index, scene_file, config, annotation_objects)


def flatten_objects(x):
    """object_id entries can nest lists: [['a','b'], 'c'] -> ['a','b','c']."""
    out = []
    for e in x:
        if isinstance(e, list):
            out.extend(flatten_objects(e))
        elif isinstance(e, str) and e:
            out.append(e)
    return out


_TRANS_RE = re.compile(r"^(half|diced|cooked|sliced|melted)_")


def is_transition_product(tok: str) -> bool:
    """half_beet_211_0, diced__chili: created by transition rules during the episode."""
    return bool(_TRANS_RE.match(tok)) or "__" in tok
