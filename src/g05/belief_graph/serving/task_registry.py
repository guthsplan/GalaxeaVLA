# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Per-task belief-graph artifacts from a bgdata output directory.

Expected layout (produced by `python -m bgdata.run --task N --out <dir>`):
    <root>/operators_task{N:03d}.json   operator library
    <root>/goal_task{N:03d}.json        goal COUNT lines + synset_to_scene + init_lines
                                        (init_lines = grounded BDDL :init priors)
Multiple tasks can share one root; artifacts are looked up by task id and cached.
"""
import pathlib
from dataclasses import dataclass

from ..goal import GoalSpec
from ..operators import OperatorLibrary


@dataclass
class TaskArtifacts:
    task_id: int
    lib: OperatorLibrary
    goal: GoalSpec


class TaskRegistry:
    def __init__(self, root: str | pathlib.Path, task_names: dict[int, str] | None = None):
        self.root = pathlib.Path(root)
        self.task_names = task_names or {}
        self._cache: dict[int, TaskArtifacts] = {}

    def get(self, task_id: int, task_text: str = "") -> TaskArtifacts:
        if task_id not in self._cache:
            ops = self.root / f"operators_task{task_id:03d}.json"
            gj = self.root / f"goal_task{task_id:03d}.json"
            for p in (ops, gj):
                if not p.exists():
                    raise FileNotFoundError(
                        f"belief-graph artifact missing: {p} — run the bgdata pipeline "
                        f"for task {task_id} first (python -m bgdata.run --task {task_id})")
            self._cache[task_id] = TaskArtifacts(
                task_id=task_id,
                lib=OperatorLibrary.from_json(ops),
                goal=GoalSpec.from_bgdata_json(
                    gj, task=task_text or self.task_names.get(task_id, f"task {task_id}")),
            )
        return self._cache[task_id]
