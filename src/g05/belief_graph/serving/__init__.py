# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Serving glue for the belief-graph runtime (glue P1-P2, CPU-only, framework-free).

Wiring target identified in the serve scripts (glue P0):
  - inject side:  scripts/serve_policy*.py build_obs_dict() copies extra text fields
    from raw_obs into the processor `data` dict (the `plan`/`coarse_task` pattern) —
    `data["bg_known"]` inserted there reaches the BeliefGraph* builders' eval path.
  - readback side: the ChunkedPolicyWrapper pattern of serve_policy_mem.py pops the
    AR-decoded CoT (`action.pop("_cot_text")`) and returns it to the client
    (`resp["cot_text"]`) — that text feeds BeliefGraphMiddleware.after_infer().

Modules:
    task_registry     — per-task artifacts (operators/goal/init) from a bgdata output dir
    controller        — minimal skill completion/timeout from observed effects
    policy_middleware — BeliefGraphMiddleware: before_infer/after_infer around any policy
    replay_harness    — offline validation on demos vs recorded belief traces

Smoke/validation:
    python -m g05.belief_graph.serving.replay_harness --bg-out <bgdata out/task045> [--episodes 5]
"""
from .controller import SkillController
from .policy_middleware import BeliefGraphMiddleware
from .task_registry import TaskRegistry

__all__ = ["BeliefGraphMiddleware", "SkillController", "TaskRegistry"]
