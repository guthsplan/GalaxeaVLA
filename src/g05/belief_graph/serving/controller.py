# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Minimal skill controller (glue P2): completion and timeout from the belief.

A tracked skill is COMPLETE when every positive grounded effect holds in the belief
via OBSERVATION (not via its own provisional effect) for `hold_steps` consecutive
ticks — i.e. perception confirmed the postcondition. It FAILS on `timeout_steps`
ticks without completion; the middleware then re-queries the high-level CoT.

This is the reduced form of the bg_pi05 scaffold Controller: no recovery-rate
branching (that needs the perturbation-phase data), just complete/timeout/reject.
"""
from dataclasses import dataclass, field
from typing import Optional

from ..belief import Belief
from ..operators import OperatorLibrary


@dataclass
class SkillController:
    lib: OperatorLibrary
    hold_steps: int = 2          # consecutive belief ticks the effects must hold
    timeout_steps: int = 200     # belief ticks before giving up on a skill
    current: Optional[tuple[str, list[str]]] = None   # (op_name, objs)
    _held: int = 0
    _age: int = 0
    events: list = field(default_factory=list)

    def start(self, op_name: str, objs: list[str], step: int) -> None:
        self.current = (op_name, objs)
        self._held = 0
        self._age = 0
        self.events.append(dict(step=step, event="start", skill=op_name, objs=list(objs)))

    def tick(self, belief: Belief, step: int) -> dict:
        """-> {done: bool, failed: bool}. Call once per belief update."""
        if self.current is None:
            return dict(done=False, failed=False)
        op_name, objs = self.current
        self._age += 1
        effs = self.lib.concretize_effects(self.lib.ops[op_name], objs)
        ok = True
        for key, value in effs:
            e = belief.table.get(key)
            # confirmation must come from observation, not from our own effect write
            if e is None or e.value != value or e.source == "effect":
                ok = False
                break
        self._held = self._held + 1 if ok else 0
        if self._held >= self.hold_steps:
            self.events.append(dict(step=step, event="complete", skill=op_name))
            self.current = None
            return dict(done=True, failed=False, skill=op_name, objs=list(objs))
        if self._age >= self.timeout_steps:
            self.events.append(dict(step=step, event="timeout", skill=op_name))
            self.current = None
            return dict(done=False, failed=True, skill=op_name, objs=list(objs))
        return dict(done=False, failed=False)
