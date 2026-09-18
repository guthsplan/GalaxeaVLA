# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""External belief-graph module for inference (BEHAVIOR bgdata port, torch-free).

Maintains the symbolic scene belief (closed-world memory, NO time decay) online and
serializes it into the `bg_known` field consumed by the BeliefGraph* samples builders.

    runtime  — BeliefGraphRuntime: per-step update loop + bg_known serialization
               + model-CoT (Delta:) cross-check
    belief   — Belief table and the confidence rules (prior/observe/effect/disturb)
    operators— OperatorLibrary (operators_task*.json schema of the bgdata pipeline)
    goal     — GoalSpec (COUNT goal lines) + compute_delta
    estimator— Estimate type, OracleEstimator (truth_*.json, local testing only)

Smoke test: python -m g05.belief_graph.runtime
"""
from .belief import Belief, BeliefEntry, PRIOR_CONF, PROVISIONAL_CONF, DISTURB_FACTOR, DISTURB_FLOOR
from .estimator import Estimate, OracleEstimator
from .goal import GoalSpec
from .operators import OperatorLibrary, parse_key
from .runtime import BeliefGraphRuntime

__all__ = [
    "Belief", "BeliefEntry", "BeliefGraphRuntime", "Estimate", "GoalSpec",
    "OperatorLibrary", "OracleEstimator", "parse_key",
    "PRIOR_CONF", "PROVISIONAL_CONF", "DISTURB_FACTOR", "DISTURB_FLOOR",
]
