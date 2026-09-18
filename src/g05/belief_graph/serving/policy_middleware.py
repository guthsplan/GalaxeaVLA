# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""BeliefGraphMiddleware (glue P1): the two-line wiring around any policy.

Server integration (scripts/serve_policy*.py):

    mw = BeliefGraphMiddleware(registry=TaskRegistry(bg_artifacts_dir),
                               estimator=None)          # None = model-as-estimator
    ...
    # on client reset / first request of an episode:
    mw.reset(task_id=45, task_text=raw_obs["task"])
    # in the request handler, around the model call:
    data = build_obs_dict(raw_obs, processor)
    data["bg_known"] = mw.before_infer(raw_obs)         # ← inject (the `plan` pattern)
    action = policy(data)                               # AR decode
    cot_text = action.pop("_cot_text", None)            # serve_policy_mem.py pattern
    mw.after_infer(cot_text)                            # ← readback hooks
    resp["bg"] = mw.last_log

after_infer routes the CoT to the runtime: Observe: -> belief observations,
Subtask: -> precondition check (hard violation => rejected=True; the server should
re-query the CoT turn, appending the reason — up to `max_requery`), Delta: ->
cross-check. Skill completion is detected by SkillController on the belief and
applies operator effects.
"""
import json
import pathlib
import time
from typing import Optional

from ..runtime import BeliefGraphRuntime
from .controller import SkillController
from .task_registry import TaskRegistry


class BeliefGraphMiddleware:
    def __init__(self, registry: TaskRegistry, estimator=None, belief_every: int = 4,
                 model_obs_conf: float = 0.9, max_requery: int = 3,
                 log_dir: Optional[str] = None):
        self.registry = registry
        self.estimator = estimator
        self.belief_every = belief_every
        self.model_obs_conf = model_obs_conf
        self.max_requery = max_requery
        self.runtime: Optional[BeliefGraphRuntime] = None
        self.controller: Optional[SkillController] = None
        self.last_log: dict = {}
        self._requery_left = max_requery
        self._log_path = (pathlib.Path(log_dir) / f"bg_{int(time.time())}.jsonl"
                          if log_dir else None)

    # ---- episode lifecycle -------------------------------------------------
    def reset(self, task_id: int, task_text: str = "") -> None:
        art = self.registry.get(task_id, task_text)
        self.runtime = BeliefGraphRuntime(
            art.lib, art.goal, estimator=self.estimator,
            belief_every=self.belief_every, model_obs_conf=self.model_obs_conf)
        self.controller = SkillController(art.lib)
        self._requery_left = self.max_requery
        self.last_log = {}

    # ---- per request ---------------------------------------------------------
    def before_infer(self, raw_obs: dict, step: Optional[int] = None) -> str:
        """Belief update + Δ computation; returns the bg_known string for the data dict."""
        assert self.runtime is not None, "call reset(task_id) before before_infer()"
        out = self.runtime.step(raw_obs, step)
        self.last_log = dict(step=self.runtime.step_count, bg_known=out["bg_known"],
                             remaining=out["remaining"], progress=out["progress"])
        return out["bg_known"]

    def after_infer(self, cot_text: Optional[str]) -> dict:
        """Route the decoded CoT; tick the controller. Returns
        {rejected: bool, requery_prompt: str|None, completed: bool, hooks: dict}."""
        assert self.runtime is not None and self.controller is not None
        result = dict(rejected=False, requery_prompt=None, completed=False, hooks={})
        hooks = self.runtime.on_model_cot(cot_text or "")
        result["hooks"] = hooks
        if hooks["accepted_skill"]:
            self.controller.start(hooks["accepted_skill"],
                                  list(self.runtime.current_skill[1]),
                                  self.runtime.step_count)
            self._requery_left = self.max_requery
        elif hooks["hard_violations"]:
            result["rejected"] = True
            if self._requery_left > 0:
                self._requery_left -= 1
                result["requery_prompt"] = (
                    f"(rejected: violated {' '.join(hooks['hard_violations'])}) "
                    f"{self.last_log.get('bg_known', '')}")
        tick = self.controller.tick(self.runtime.belief, self.runtime.step_count)
        if tick["done"]:
            touched = self.runtime.on_skill_complete(tick["skill"], tick["objs"])
            result["completed"] = True
            self.last_log["effects_applied"] = touched
        self.last_log.update(cot=cot_text, rejected=result["rejected"],
                             completed=result["completed"],
                             delta_mismatch=hooks.get("delta_mismatch", []))
        self._write_log()
        return result

    def _write_log(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(self.last_log) + "\n")
