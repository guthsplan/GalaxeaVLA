"""Reference simulation of the belief-memory rules (no learning).

Consumes truth_<episode>.json frame by frame exactly as the test-time estimator would:
an observation enters ONLY when visible is true. Operator effects are applied at
annotation segment ends. NO time decay: confidence changes only via
  prior (0.5) -> observation (0.97) | provisional effect (0.7) | disturbance (x0.8, floor 0.6).

Internal representation: value (bool) + conf (confidence that stored value is correct).
Exported p_true = conf if value else 1 - conf.
"""
import json

from .config import (DISTURB_FACTOR, DISTURB_FLOOR, OBS_CONF_TRUE, PRIOR_CONF,
                     PROVISIONAL_CONF)
from .operators import SKILL_TO_OP, concretize_effects, parse_key


def p_true(entry):
    return entry["conf"] if entry["value"] else 1.0 - entry["conf"]


class BeliefSim:
    def __init__(self, ops_json: dict, init_lines: list[tuple[str, bool]],
                 goal_specs: list[dict], key_universe: set[str] | None = None):
        """goal_specs: [{line, pred, instances}] — count goals over exchangeable instances.
        key_universe: known predicate keys; operator effects grounding outside it are skipped
        (guards against annotation errors such as open door ['hotdog_208'] in ep 450280)."""
        self.ops = ops_json
        self.table = {}
        self.goals = goal_specs
        self.universe = key_universe
        self.change_log = []
        for key, val in init_lines:
            self.table[key] = dict(value=val, conf=PRIOR_CONF, source="prior", observed=False)

    def _log(self, step, key, old_p, rule):
        new_p = p_true(self.table[key])
        if old_p is None or abs(new_p - old_p) > 1e-9:
            self.change_log.append(dict(step=step, key=key, p_before=old_p, p_after=new_p, rule=rule))

    def observe(self, step, truth_frame: dict):
        observed = set()
        for key, (val, vis) in truth_frame.items():
            e = self.table.get(key)
            if not vis:
                if e is not None:
                    e["observed"] = False
                continue
            old = p_true(e) if e else None
            tag = "appear" if parse_key(key)[0] in ("cooked", "toggled_on", "open") else "geom"
            self.table[key] = dict(value=bool(val), conf=OBS_CONF_TRUE, source=tag, observed=True)
            self._log(step, key, old, "prior->observed" if (e and e["source"] == "prior") else "observed")
            observed.add(key)
        return observed

    def apply_operator(self, step, skill: str, objs: list[str], observed: set):
        op = self.ops.get(SKILL_TO_OP.get(skill, (None,))[0])
        if not op:
            return
        touched = set()
        effs = concretize_effects(op, objs)
        # mobile-base exclusivity: a positive (reachable X) effect implies leaving everywhere else
        if any(gk.startswith("(reachable ") and gv for gk, gv in effs):
            for key in list(self.table):
                if key.startswith("(reachable ") and key not in observed \
                        and not any(key == gk for gk, _ in effs):
                    e = self.table[key]
                    if e["value"]:
                        old = p_true(e)
                        self.table[key] = dict(value=False, conf=PROVISIONAL_CONF,
                                               source="effect", observed=False)
                        self._log(step, key, old, "provisional effect 0.7 (exclusive reachable)")
        for gk, gv in effs:
            if self.universe is not None and gk not in self.universe:
                continue  # mis-grounded effect (annotation error); skip
            touched.add(gk)
            if gk in observed:
                continue  # the observation this step wins
            e = self.table.get(gk)
            old = p_true(e) if e else None
            self.table[gk] = dict(value=gv, conf=PROVISIONAL_CONF, source="effect", observed=False)
            self._log(step, gk, old, "provisional effect 0.7")
        # disturbance: same-container relations (?y ?r) not touched this step
        mapping = dict(zip(op["args"], objs))
        r = mapping.get("?r")
        if r and op.get("disturbs"):
            for key, e in self.table.items():
                name, args = parse_key(key)
                if name in ("inside", "ontop") and len(args) == 2 and args[1] == r \
                        and key not in touched and key not in observed and e["source"] != "prior":
                    old = p_true(e)
                    e["conf"] = max(DISTURB_FLOOR, e["conf"] * DISTURB_FACTOR)
                    self._log(step, key, old, "disturbance x0.8")

    def remaining_goals(self):
        rem = []
        total_sat = total_n = 0
        for g in self.goals:
            sat = sum(1 for inst in g["instances"]
                      if self.table.get(f"({g['pred']} {inst})", {}).get("value", False))
            n = len(g["instances"])
            total_sat += sat
            total_n += n
            if sat < n:
                rem.append(g["line"].replace(" 0/", f" {sat}/"))
        return rem, (total_sat / total_n if total_n else 1.0)

    def snapshot(self):
        return {k: [e["value"], round(p_true(e), 4), e["observed"], e["source"]]
                for k, e in self.table.items()}


def run_trace(truth: dict[int, dict], segments: list[dict], ops_json: dict,
              init_lines, goal_specs, out_jsonl: str):
    frames = sorted(truth)
    universe = set(truth[frames[0]])
    sim = BeliefSim(ops_json, init_lines, goal_specs, key_universe=universe)
    seg_ends = sorted([s for s in segments if s["skill"] in SKILL_TO_OP], key=lambda s: s["end"])
    si = 0
    records = []
    with open(out_jsonl, "w") as f:
        for fr in frames:
            observed = sim.observe(fr, truth[fr])
            while si < len(seg_ends) and seg_ends[si]["end"] <= fr:
                s = seg_ends[si]
                op_args = s["objects"][: len(SKILL_TO_OP[s["skill"]][1])]
                sim.apply_operator(fr, s["skill"], op_args, observed)
                si += 1
            cur = next((s for s in segments if s["start"] <= fr < s["end"]), None)
            rem, prog = sim.remaining_goals()
            rec = dict(step=int(fr),
                       skill=(f"{cur['skill']} {cur['objects']}" if cur else "done"),
                       predicates=sim.snapshot(), remaining_goal_lines=rem,
                       progress=round(prog, 4))
            f.write(json.dumps(rec) + "\n")
            records.append(rec)
    return records, sim.change_log
