# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Belief table + confidence rules (port of bgdata/belief_trace.py BeliefSim).

Closed-world memory: an unobserved predicate keeps its last (value, conf) with NO time
decay. Confidence changes only through:
    BDDL :init prior            -> conf 0.5
    observation                 -> conf = estimator prob (oracle: 0.97/0.03)
    operator effect, unobserved -> conf 0.7 (provisional)
    same-container disturbance  -> conf x0.8, floor 0.6
"""
from dataclasses import dataclass

from .operators import OperatorLibrary, parse_key

PRIOR_CONF = 0.5
PROVISIONAL_CONF = 0.7
DISTURB_FACTOR = 0.8
DISTURB_FLOOR = 0.6

# goal-first ordering for serialization (mirrors bgdata/cot_targets.py CAT_ORDER)
_CAT_ORDER = {"cooked": 0, "toggled_on": 1, "inhand": 2, "inside": 3, "ontop": 3,
              "onfloor": 3, "open": 4, "reachable": 5, "visited": 6}


@dataclass
class BeliefEntry:
    value: bool
    conf: float          # confidence in the stored value (>= 0.5)
    source: str          # prior | geom | appear | robot | effect
    observed: bool       # observed at the current step

    @property
    def p_true(self) -> float:
        return self.conf if self.value else 1.0 - self.conf


class Belief:
    def __init__(self, key_universe: set | None = None):
        """key_universe: known predicate keys; effects grounding outside it are skipped
        (guards against mis-grounded skills — see bgdata ep 450280 annotation error)."""
        self.table: dict[str, BeliefEntry] = {}
        self.universe = key_universe

    # ---- rules ------------------------------------------------------------
    def set_prior(self, key: str, value: bool) -> None:
        self.table[key] = BeliefEntry(value=value, conf=PRIOR_CONF, source="prior", observed=False)

    def observe(self, key: str, p_true: float, source: str = "geom") -> None:
        value = p_true >= 0.5
        self.table[key] = BeliefEntry(value=value, conf=max(p_true, 1.0 - p_true),
                                      source=source, observed=True)

    def begin_step(self) -> None:
        for e in self.table.values():
            e.observed = False

    def apply_operator(self, op: dict, objs: list[str], lib: OperatorLibrary) -> list[str]:
        """Apply a completed operator's effects to predicates NOT observed this step.
        Returns the keys it touched. Includes mobile-base reachable exclusivity and
        same-container disturbance (x0.8, floor 0.6)."""
        effs = lib.concretize_effects(op, objs)
        touched: list[str] = []
        # mobile-base exclusivity: reaching a new target leaves the previous one
        if any(k.startswith("(reachable ") and v for k, v in effs):
            for key, e in self.table.items():
                if key.startswith("(reachable ") and e.value and not e.observed \
                        and not any(key == k for k, _ in effs):
                    self.table[key] = BeliefEntry(False, PROVISIONAL_CONF, "effect", False)
                    touched.append(key)
        for key, value in effs:
            if self.universe is not None and key not in self.universe:
                continue
            e = self.table.get(key)
            if e is not None and e.observed:
                continue  # this step's observation wins
            self.table[key] = BeliefEntry(value, PROVISIONAL_CONF, "effect", False)
            touched.append(key)
        # disturbance: same-container relations (?y ?r) not touched this step
        r = dict(zip(op.get("args", []), objs)).get("?r")
        if r and op.get("disturbs"):
            for key, e in self.table.items():
                name, args = parse_key(key)
                if name in ("inside", "ontop") and len(args) == 2 and args[1] == r \
                        and key not in touched and not e.observed and e.source != "prior":
                    e.conf = max(DISTURB_FLOOR, e.conf * DISTURB_FACTOR)
        return touched

    def disturb(self, key: str) -> None:
        """External confidence demotion (e.g. model-Delta cross-check mismatch)."""
        e = self.table.get(key)
        if e is not None:
            e.conf = max(DISTURB_FLOOR, e.conf * DISTURB_FACTOR)

    # ---- queries ----------------------------------------------------------
    def holds(self, key: str) -> bool:
        e = self.table.get(key)
        return bool(e and e.value)

    def known_items(self, limit: int = 8) -> list[tuple[str, BeliefEntry]]:
        """Goal-first ordering; only positive relations plus goal-class predicates
        (mirrors the training-side serialization in bgdata/cot_targets.py)."""
        rows = []
        for key, e in self.table.items():
            name, _ = parse_key(key)
            cat = _CAT_ORDER.get(name)
            if cat is None or name in ("reachable", "visited"):
                continue
            if name not in ("cooked", "toggled_on") and not e.value:
                continue
            rows.append((cat, key, e))
        rows.sort(key=lambda r: (r[0], r[1]))
        return [(k, e) for _, k, e in rows[:limit]]
