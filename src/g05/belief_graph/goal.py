# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Goal spec: COUNT goal lines over exchangeable instances (bgdata goal_task*.json)."""
import json
import pathlib
import re
from dataclasses import dataclass, field


@dataclass
class GoalSpec:
    """count_goals: [{"line": "(cooked ?x) 0/2 [hotdog.n.02]",
                      "pred": "cooked", "instances": ["hotdog_207", "hotdog_208"]}]"""
    task: str
    count_goals: list = field(default_factory=list)
    init_lines: list = field(default_factory=list)   # [(key, value)] BDDL :init priors

    @classmethod
    def from_bgdata_json(cls, path: str | pathlib.Path, task: str = "",
                         init_lines: list | None = None) -> "GoalSpec":
        """Load a bgdata goal_task*.json (goal_lines + synset_to_scene [+ init_lines]).
        Explicit init_lines override the file's grounded BDDL :init priors."""
        g = json.loads(pathlib.Path(path).read_text())
        if init_lines is None and "init_lines" in g:
            init_lines = [(k, bool(v)) for k, v in g["init_lines"]]
        goals = []
        for line in g["goal_lines"]:
            m = re.match(r"\((\S+) \?\S*\) \d+/(\d+) \[(\S+)\]", line)
            pred, _n, synset = m.group(1), m.group(2), m.group(3)
            syn_base = synset.rsplit(".n.", 1)[0]
            insts = sorted(v for k, v in g["synset_to_scene"].items()
                           if k.startswith(syn_base) and v)
            goals.append(dict(line=line, pred=pred, instances=insts))
        return cls(task=task, count_goals=goals, init_lines=list(init_lines or []))

    def compute_delta(self, belief) -> tuple[list[str], int, float]:
        """-> (remaining count lines with live counts, satisfied, progress)."""
        remaining, sat_total, n_total = [], 0, 0
        for g in self.count_goals:
            sat = sum(1 for inst in g["instances"]
                      if belief.holds(f"({g['pred']} {inst})"))
            n = len(g["instances"])
            sat_total += sat
            n_total += n
            if sat < n:
                remaining.append(re.sub(r" \d+/(\d+) ", f" {sat}/\\1 ", g["line"]))
        return remaining, sat_total, (sat_total / n_total if n_total else 1.0)
