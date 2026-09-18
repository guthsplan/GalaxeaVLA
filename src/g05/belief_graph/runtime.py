# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""BeliefGraphRuntime — the inference-side loop that fills data["bg_known"].

Serving integration (per control step, BEFORE the processor/builder runs):

    runtime = BeliefGraphRuntime(
        lib=OperatorLibrary.from_json("operators_task045.json"),
        goal=GoalSpec.from_bgdata_json("goal_task045.json", task="cook hot dogs",
                                       init_lines=[("(inside hotdog_207 fridge_dszchb_0)", True), ...]),
        estimator=OracleEstimator("truth_450010.json"),   # eval: hybrid RGB-D estimator
    )
    out = runtime.step(obs, step)                # belief update every `belief_every` calls
    data["bg_known"] = out["bg_known"]           # -> BeliefGraph* builders (eval path)
    ...
    # after AR decoding:
    runtime.on_model_cot(cot_text)               # 'Subtask:' -> skill tracking + precondition
                                                 # check; 'Delta:' -> cross-check vs belief
    runtime.on_skill_complete()                  # controller-detected completion -> effects

The bg_known text matches the TRAINING serialization of tools/build_bg_fields.py
("Remaining: ... | Known: <key> <p:.2f> obs|mem | ...") so train/inference prompts agree.

Smoke test: python -m g05.belief_graph.runtime
"""
import re
from typing import Optional

from .belief import Belief
from .estimator import EstimatorProtocol
from .goal import GoalSpec
from .operators import OperatorLibrary


class BeliefGraphRuntime:
    def __init__(self, lib: OperatorLibrary, goal: GoalSpec,
                 estimator: Optional[EstimatorProtocol] = None, belief_every: int = 4,
                 known_limit: int = 8, key_universe: Optional[set] = None,
                 model_obs_conf: float = 0.9):
        """estimator=None runs in pure model-as-estimator mode: observations arrive
        only through 'Observe:' CoT spans via on_model_cot(). model_obs_conf is the
        calibrated confidence assigned to model observations (< oracle 0.97)."""
        self.lib, self.goal, self.estimator = lib, goal, estimator
        self.belief_every = belief_every
        self.known_limit = known_limit
        self.model_obs_conf = model_obs_conf
        self._universe = key_universe
        self.reset()

    def reset(self) -> None:
        self.belief = Belief(self._universe)
        for key, value in self.goal.init_lines:
            self.belief.set_prior(key, value)
        self.step_count = 0
        self.current_skill: Optional[tuple[str, list[str]]] = None  # (op_name, objs)
        self.events: list[dict] = []

    # ---- per control step ---------------------------------------------------
    def step(self, obs: dict, step: Optional[int] = None) -> dict:
        self.step_count += 1
        step = self.step_count if step is None else step
        if self.belief_every <= 1 or self.step_count % self.belief_every == 1:
            self.belief.begin_step()
            if self.estimator is not None:
                for key, est in self.estimator.estimate(obs, step).items():
                    self.belief.observe(key, est.prob, est.source)
        remaining, satisfied, progress = self.goal.compute_delta(self.belief)
        return dict(bg_known=self.serialize_known(remaining),
                    remaining=remaining, satisfied=satisfied, progress=progress)

    def serialize_known(self, remaining: Optional[list] = None) -> str:
        if remaining is None:
            remaining, _, _ = self.goal.compute_delta(self.belief)
        text = "Remaining: " + (" | ".join(remaining) if remaining else "none")
        ks = [f"{k} {e.p_true:.2f} {'obs' if e.observed else 'mem'}"
              for k, e in self.belief.known_items(self.known_limit)]
        if ks:
            text += " | Known: " + " | ".join(ks)
        return text

    # ---- model-output hooks -------------------------------------------------
    def on_model_cot(self, cot_text: str) -> dict:
        """Consume the AR-decoded CoT span. Handles:
        'Observe:' — model-as-estimator: parsed predicates enter the belief table as
            observations at model_obs_conf (perception internalized in the model,
            memory rules still guaranteed externally);
        'Subtask:' — skill tracking + precondition check;
        'Delta:'   — cross-check against the external belief; a mismatched goal
            predicate gets its confidence demoted x0.8 to prioritise re-observation
            (the model's Delta prediction is never written into the belief)."""
        from .estimator import parse_observe

        out: dict = {"accepted_skill": None, "hard_violations": [], "delta_mismatch": [],
                     "observed": []}
        for key, est in parse_observe(cot_text, self.model_obs_conf).items():
            if self._universe is not None and key not in self._universe:
                continue
            self.belief.observe(key, est.prob, est.source)
            out["observed"].append(key)
        m = re.search(r"Subtask:\s*([^|\n]+)", cot_text)
        if m:
            parsed = self.lib.match_skill_text(m.group(1))
            if parsed:
                op_name, objs = parsed
                accept, hard, _soft = self.lib.check_preconditions(
                    self.lib.ops[op_name], objs, self.belief)
                out["hard_violations"] = hard
                if accept:
                    self.current_skill = (op_name, objs)
                    out["accepted_skill"] = op_name
                else:
                    self.events.append(dict(step=self.step_count, event="reject",
                                            skill=op_name, hard=hard))
        m = re.search(r"Delta:\s*([^\n]+)", cot_text)
        if m:
            model_rem = [] if m.group(1).strip().lower() == "none" \
                else [s.strip() for s in m.group(1).split("|")]
            belief_rem, _, _ = self.goal.compute_delta(self.belief)
            if sorted(model_rem) != sorted(belief_rem):
                for g in self.goal.count_goals:
                    for inst in g["instances"]:
                        key = f"({g['pred']} {inst})"
                        if key in self.belief.table:
                            self.belief.disturb(key)
                            out["delta_mismatch"].append(key)
                self.events.append(dict(step=self.step_count, event="delta_mismatch",
                                        model=model_rem, belief=belief_rem))
        return out

    def on_skill_complete(self, op_name: Optional[str] = None,
                          objs: Optional[list] = None) -> list[str]:
        """Apply the completed skill's operator effects (provisional 0.7 + disturbance).
        Defaults to the currently tracked skill."""
        if op_name is None:
            if self.current_skill is None:
                return []
            op_name, objs = self.current_skill
        touched = self.belief.apply_operator(self.lib.ops[op_name], objs or [], self.lib)
        self.current_skill = None
        return touched


# ====================================================================== #
#  Smoke test: python -m g05.belief_graph.runtime                        #
# ====================================================================== #

if __name__ == "__main__":
    from .estimator import Estimate

    OPS = {
        "move_to": dict(skill_id=1, args=["?o"], pre=[], eff=["(reachable ?o)"],
                        text="move {mp}to {?o}", skill_type="navigation", disturbs=[], rel=""),
        "open_door": dict(skill_id=10, args=["?d"], pre=["(not (open ?d))", "(reachable ?d)"],
                          eff=["(open ?d)"], text="open door {?d}", skill_type="coordinated",
                          disturbs=[], rel=""),
        "pick_up_from": dict(skill_id=2, args=["?o", "?r"],
                             pre=["(in_or_on ?o ?r)", "(not (inhand ?o))", "(reachable ?r)"],
                             eff=["(not (in_or_on ?o ?r))", "(inhand ?o)"],
                             text="pick up {mp}{?o} from {?r}", skill_type="uncoordinated",
                             disturbs=["(in_or_on ?y ?r)"], rel=""),
    }
    lib = OperatorLibrary(OPS)
    goal = GoalSpec(task="cook hot dogs",
                    count_goals=[dict(line="(cooked ?x) 0/2 [hotdog.n.02]", pred="cooked",
                                      instances=["hotdog_207", "hotdog_208"])],
                    init_lines=[("(inside hotdog_207 fridge)", True),
                                ("(inside hotdog_208 fridge)", True),
                                ("(cooked hotdog_207)", False),
                                ("(cooked hotdog_208)", False)])

    # scripted estimator: step1 nothing visible; step5 fridge open + contents; step9 silent
    SCRIPT = {
        1: {},
        5: {"(open fridge)": Estimate(True, 0.97, "appear"),
            "(inside hotdog_207 fridge)": Estimate(True, 0.97, "geom"),
            "(cooked hotdog_207)": Estimate(False, 0.03, "appear")},
        9: {},
    }

    class Scripted:
        def estimate(self, obs, step):
            return SCRIPT.get(step, {})

    rt = BeliefGraphRuntime(lib, goal, Scripted(), belief_every=4)

    # step 1: only priors -> bg_known shows 0.50 prior entries and full Remaining
    out1 = rt.step({}, 1)
    assert "Remaining: (cooked ?x) 0/2 [hotdog.n.02]" in out1["bg_known"], out1
    assert "(cooked hotdog_207) 0.50 mem" in out1["bg_known"], out1
    print("  ✓ step1: BDDL prior 0.5 visible in bg_known before any observation")

    # steps 2-4 do not touch belief (belief_every=4); step 5 observes
    rt.step({}, 2); rt.step({}, 3); rt.step({}, 4)
    out5 = rt.step({}, 5)
    assert "(cooked hotdog_207) 0.03 obs" in out5["bg_known"], out5
    assert "(inside hotdog_207 fridge) 0.97 obs" in out5["bg_known"], out5
    print("  ✓ step5: observations enter at estimator confidence (obs flag)")

    # steps 6-9: nothing visible -> values persist with UNCHANGED conf (no decay)
    rt.step({}, 6); rt.step({}, 7); rt.step({}, 8)
    out9 = rt.step({}, 9)
    assert "(inside hotdog_207 fridge) 0.97 mem" in out9["bg_known"], out9
    print("  ✓ step9: memory keeps conf 0.97 with no time decay (obs -> mem)")

    # model Subtask -> precondition check -> completion -> provisional effect 0.7
    hooks = rt.on_model_cot("Subtask: pick up hotdog_207 from fridge")
    assert hooks["accepted_skill"] == "pick_up_from", hooks
    touched = rt.on_skill_complete()
    e = rt.belief.table["(inhand hotdog_207)"]
    assert e.value and abs(e.conf - 0.7) < 1e-9 and e.source == "effect", e
    assert not rt.belief.table["(inside hotdog_207 fridge)"].value, "pick clears in_or_on"
    # disturbance: the other hotdog's inside-fridge conf dropped x0.8 (0.5 prior is exempt;
    # observe it first to make it eligible)
    rt.belief.observe("(inside hotdog_208 fridge)", 0.97)
    rt.belief.begin_step()
    rt.current_skill = ("pick_up_from", ["hotdog_207", "fridge"])
    rt.on_skill_complete()
    e208 = rt.belief.table["(inside hotdog_208 fridge)"]
    assert abs(e208.conf - 0.97 * 0.8) < 1e-9, e208
    print("  ✓ skill completion: provisional effect 0.7 + same-container disturbance x0.8")

    # Delta cross-check: model claims goal satisfied -> mismatch demotes goal predicates
    hooks = rt.on_model_cot("Delta: none")
    assert hooks["delta_mismatch"], hooks
    print("  ✓ Delta cross-check: mismatch demotes goal-predicate confidence (never writes values)")

    # model-as-estimator: Observe: CoT feeds the EXTERNAL belief as observations
    rt2 = BeliefGraphRuntime(lib, goal, estimator=None, model_obs_conf=0.9)
    out0 = rt2.step({}, 1)
    assert "(cooked hotdog_207) 0.50 mem" in out0["bg_known"], out0  # priors only
    hooks = rt2.on_model_cot(
        "Observe: (open fridge) 1 | (inside hotdog_207 fridge) 1 | (cooked hotdog_207) 0")
    assert set(hooks["observed"]) == {"(open fridge)", "(inside hotdog_207 fridge)",
                                      "(cooked hotdog_207)"}, hooks
    e = rt2.belief.table["(cooked hotdog_207)"]
    assert not e.value and abs(e.conf - 0.9) < 1e-9 and e.observed, e
    assert rt2.belief.table["(open fridge)"].source == "appear"
    assert "(cooked hotdog_207) 0.10 obs" in rt2.serialize_known(), rt2.serialize_known()
    # a combined span still routes each part correctly: Delta mismatch demotes x0.8
    hooks2 = rt2.on_model_cot("Observe: (toggled_on microwave) 1 | Delta: none")
    assert hooks2["observed"] == ["(toggled_on microwave)"] and hooks2["delta_mismatch"], hooks2
    assert abs(rt2.belief.table["(cooked hotdog_207)"].conf - 0.9 * 0.8) < 1e-9
    print("  ✓ Observe CoT (model-as-estimator): calibrated conf + combined-span routing")

    # serializer format is exactly what BeliefGraphBuilder passes through
    txt = rt.serialize_known()
    assert txt.startswith("Remaining: (cooked ?x) 0/2 [hotdog.n.02]"), txt
    assert " | Known: " in txt and re.search(r"\) \d\.\d\d (obs|mem)", txt), txt
    print("  ✓ bg_known serialization matches the training format (build_bg_fields.py)")

    print("\nAll belief_graph runtime cases OK.")
