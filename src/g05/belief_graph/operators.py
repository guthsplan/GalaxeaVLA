# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Operator library — the exact JSON schema produced by bgdata/operators.py:

    {op_name: {skill_id, args, pre, eff, text, skill_type, disturbs, rel}}

eff entries use the abstract relation (in_or_on ?o ?r); a positive one is made
concrete via the per-operator `rel` (inside|ontop), a negative one clears both.
"""
import json
import pathlib
import re

_PRED_RE = re.compile(r"^\((\S+)((?:\s+\S+)*)\)$")


def parse_key(key: str) -> tuple[str, list[str]]:
    m = _PRED_RE.match(key)
    if not m:
        return key, []
    return m.group(1), m.group(2).split()


class OperatorLibrary:
    def __init__(self, ops: dict[str, dict]):
        self.ops = ops

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "OperatorLibrary":
        return cls(json.loads(pathlib.Path(path).read_text()))

    def concretize_effects(self, op: dict, objs: list[str]) -> list[tuple[str, bool]]:
        mapping = dict(zip(op["args"], objs))
        out: list[tuple[str, bool]] = []
        for e in op["eff"]:
            neg = e.startswith("(not ")
            inner = e[5:-1] if neg else e
            name, args = parse_key(inner)
            gargs = [mapping.get(a, a) for a in args]
            if name == "in_or_on":
                names = [op.get("rel") or "inside"] if not neg else ["inside", "ontop"]
                for nm in names:
                    out.append((f"({nm}{''.join(' ' + a for a in gargs)})", not neg))
            else:
                out.append((f"({name}{''.join(' ' + a for a in gargs)})", not neg))
        return out

    def check_preconditions(self, op: dict, objs: list[str], belief) -> tuple[bool, list[str], list[str]]:
        """-> (accept, hard_violations, soft_warnings).
        hard: an OBSERVED predicate contradicts the precondition (reject the skill);
        soft: contradiction against an unobserved (memory) predicate (warn only)."""
        mapping = dict(zip(op["args"], objs))
        hard, soft = [], []
        for p in op.get("pre", []):
            neg = p.startswith("(not ")
            inner = p[5:-1] if neg else p
            name, args = parse_key(inner)
            if name in ("handempty", "in_or_on"):
                continue  # aggregate/abstract: skip online grounding
            key = f"({name}{''.join(' ' + mapping.get(a, a) for a in args)})"
            e = belief.table.get(key)
            if e is None:
                continue
            expected = not neg
            if e.value != expected:
                (hard if e.observed else soft).append(p)
        return (len(hard) == 0), hard, soft

    def match_skill_text(self, text: str) -> tuple[str, list[str]] | None:
        """Parse a model Subtask output back to (op_name, objects) via the op text
        templates, e.g. 'pick up the other hotdog_207 from fridge_dszchb_0'."""
        text = text.strip().rstrip(".")
        for name, op in self.ops.items():
            tpl = op.get("text", "")
            pat = re.escape(tpl).replace(r"\{mp\}", r"(?:[\w ]+? )?")
            for arg in op.get("args", []):
                pat = pat.replace(re.escape("{" + arg + "}"), r"(\S+)")
            m = re.fullmatch(pat, text)
            if m:
                return name, list(m.groups())
        return None
