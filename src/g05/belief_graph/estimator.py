# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Estimator interface + oracle implementation.

At evaluation the estimates must come from RGB-D + proprio only (challenge rule):
geometric predicates from depth rules, appearance predicates (cooked/open/toggled_on)
from a learned crop classifier, robot predicates from proprioception. That hybrid
estimator is future work; OracleEstimator (privileged truth files) is for LOCAL
testing of this runtime only and must never ship in a submission.
"""
import json
import pathlib
from typing import Callable, NamedTuple, Protocol

_APPEAR = ("cooked", "toggled_on", "open")


class Estimate(NamedTuple):
    value: bool
    prob: float          # P(pred is True)
    source: str          # geom | appear | robot


class EstimatorProtocol(Protocol):
    def estimate(self, obs: dict, step: int) -> dict[str, Estimate]:
        """Return estimates ONLY for predicates actually observed this step."""
        ...


_OBS_ENTRY = None  # compiled lazily in parse_observe


def parse_observe(text: str, model_obs_conf: float = 0.9) -> dict[str, Estimate]:
    """Parse a model 'Observe:' CoT span into estimates (model-as-estimator).

    Input:  "Observe: (open fridge) 1 | (inside hotdog_207 fridge) 1 | (cooked hotdog_207) 0"
            (the exact target format of bgdata cot_targets 'observe' / bg_observe)
    Output: {key: Estimate} with prob = model_obs_conf for value 1 and
            1 - model_obs_conf for value 0. model_obs_conf must be CALIBRATED offline
            against truth files (tools such as the estimator benchmark harness) —
            0.9 is a conservative default below the oracle's 0.97.
    'Observe: none' and malformed entries yield {}.
    """
    import re as _re

    global _OBS_ENTRY
    if _OBS_ENTRY is None:
        _OBS_ENTRY = _re.compile(r"(\([^()]+\))\s+([01])\s*$")
    m = _re.search(r"Observe:\s*([^\n]+)", text)
    if not m:
        return {}
    body = m.group(1).strip()
    if body.lower() in ("none", ""):
        return {}
    out: dict[str, Estimate] = {}
    for part in body.split("|"):
        em = _OBS_ENTRY.match(part.strip())
        if not em:
            continue
        key, val = em.group(1), em.group(2) == "1"
        name = key[1:].split()[0]
        src = "appear" if name in _APPEAR else "geom"
        out[key] = Estimate(value=val, prob=model_obs_conf if val else 1.0 - model_obs_conf,
                            source=src)
    return out


class CoTObserveEstimator:
    """EstimatorProtocol adapter for the offline benchmark harness: wraps a callable
    model_fn(obs, step) -> CoT text and parses its 'Observe:' span. In the live serving
    loop the parsing goes through BeliefGraphRuntime.on_model_cot() instead (the model
    is invoked once by the policy server, not by the estimator)."""

    def __init__(self, model_fn, model_obs_conf: float = 0.9):
        self.model_fn = model_fn
        self.model_obs_conf = model_obs_conf

    def estimate(self, obs: dict, step: int) -> dict[str, Estimate]:
        return parse_observe(self.model_fn(obs, step), self.model_obs_conf)


class OracleEstimator:
    """Privileged truth-based estimator (bgdata truth_<episode>.json).

    truth format: {frame(str): {key: [value, visible]}} — exactly the file produced
    by `python -m bgdata.run`. Only visible entries become estimates (0.97/0.03),
    mirroring OracleEstimator in the bg_pi05 scaffold.
    """

    def __init__(self, truth: str | pathlib.Path | Callable[[int], dict]):
        if callable(truth):
            self._truth = truth
        else:
            data = json.loads(pathlib.Path(truth).read_text())
            self._truth = lambda step: {k: (v, vis) for k, (v, vis) in
                                        data.get(str(step), {}).items()}

    def estimate(self, obs: dict, step: int) -> dict[str, Estimate]:
        out = {}
        for key, (value, visible) in self._truth(step).items():
            if not visible:
                continue
            name = key[1:].split()[0]
            src = "appear" if name in _APPEAR else (
                "robot" if name.startswith(("inhand", "reachable", "visited")) else "geom")
            out[key] = Estimate(value=bool(value), prob=0.97 if value else 0.03, source=src)
        return out
