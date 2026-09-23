"""BDDL goal grounding + evaluation for any task (bddl3 parser AST).

Goal AST shapes (bddl3 `parse_problem`):
  ["forall", ["?x", "-", synset], body]      ["exists", ["?x", "-", synset], body]
  ["forn", ["2"], ["?x", "-", synset], body]  ["forpairs", ["?x","-",s1], ["?y","-",s2], body]
  ["and", e...] ["or", e...] ["not", e]       [pred, arg...]   (args: "?x" | "?inst" | "inst")

Every top-level conjunct becomes one COUNT line `"<body> sat/n [synsets]"`, evaluated against a
predicate table {key: bool} whose keys use SCENE object names -- the same names the labels,
operators and Subtask text use. A missing key evaluates to unknown (None) and counts as
unsatisfied, so the belief can only claim what it has evidence for.
"""
from __future__ import annotations

import json
import pathlib
import re

QUANT = {"forall", "exists", "forn", "forpairs", "fornpairs"}
LOGIC = {"and", "or", "not"}
# predicates whose second argument is a substance (particle system), not a rigid object
SUBSTANCE_PREDS = {"covered", "filled", "contains", "saturated", "insource"}
SKIP_INIT = {"inroom", "future", "insource"}


def synset_of(inst: str) -> str:
    return inst.rsplit("_", 1)[0]


def _is_instance(tok: str) -> bool:
    return bool(re.search(r"\.n\.\d+_\d+$", tok))


class GroundedGoal:
    """One goal conjunct with its variable domains resolved to scene names."""

    def __init__(self, expr, line: str, synsets: list[str]):
        self.expr = expr            # AST with variables ("?x") and scene-name constants
        self.line = line            # canonical text, "... 0/n [syn]"
        self.synsets = synsets

    def to_json(self):
        return dict(line=self.line, expr=self.expr, synsets=self.synsets)

    @classmethod
    def from_json(cls, d):
        return cls(d["expr"], d["line"], d["synsets"])


class TaskGoal:
    def __init__(self, bddl_objects: dict, bddl_init: list, bddl_goal: list, inst_to_name: dict):
        self.inst_to_name = dict(inst_to_name)
        self.domains: dict[str, list[str]] = {}          # synset -> scene names (only mapped ones)
        for syn, insts in bddl_objects.items():
            names = []
            for i in insts:
                if i.endswith("_*"):  # wildcard instance list: every mapped instance of the synset
                    pre = i[:-1]
                    names += [inst_to_name[k] for k in sorted(inst_to_name) if k.startswith(pre)]
                else:
                    names.append(inst_to_name.get(i))
            self.domains[syn] = [n for n in dict.fromkeys(names) if n]
        conj = list(bddl_goal)
        if len(conj) == 1 and isinstance(conj[0], list) and conj[0] and conj[0][0] == "and":
            conj = conj[0][1:]
        self.goals: list[GroundedGoal] = [self._ground_conjunct(c) for c in conj]
        self.init_lines: list[tuple[str, bool]] = self._ground_init(bddl_init)

    # ---- grounding --------------------------------------------------------------
    def _name(self, tok: str) -> str:
        """'?electric_refrigerator.n.01_1' / 'hotdog.n.02_1' -> scene name; variables unchanged."""
        t = tok[1:] if tok.startswith("?") else tok
        if _is_instance(t):
            return self.inst_to_name.get(t, t)
        return tok  # a variable such as ?hotdog.n.02

    def _ground(self, e):
        if isinstance(e, str):
            return self._name(e)
        if not e:
            return e
        head = e[0]
        if head in QUANT:
            return [head] + [x if (isinstance(x, list) and len(x) == 3 and x[1] == "-") or
                             (isinstance(x, list) and len(x) == 1 and x[0].isdigit())
                             else self._ground(x) for x in e[1:]]
        if head in LOGIC:
            return [head] + [self._ground(x) for x in e[1:]]
        return [head] + [self._name(a) for a in e[1:]]

    def _ground_conjunct(self, c) -> GroundedGoal:
        g = self._ground(c)
        syns = []
        n = self._count_total(g, {})
        line = f"{render(g)} 0/{n}"
        if syns_in(g):
            line += " [" + ",".join(syns_in(g)) + "]"
        return GroundedGoal(g, line, syns_in(g))

    def _ground_init(self, init):
        out = []
        for lit in init:
            neg = lit[0] == "not"
            atom = lit[1] if neg else lit
            pred = atom[0]
            if pred in SKIP_INIT:
                continue
            args = [self._name(a) for a in atom[1:]]
            if any(a.startswith("agent.") or "agent.n" in a for a in atom[1:]):
                continue
            if any(_is_instance(a) for a in args):  # unmapped instance (substance without object)
                if pred not in SUBSTANCE_PREDS:
                    continue
                args = [a if not _is_instance(a) else synset_of(a).split(".n.")[0] for a in args]
            out.append((key(pred, args), not neg))
        return out

    # ---- evaluation --------------------------------------------------------------
    def _count_total(self, g, env) -> int:
        head = g[0] if isinstance(g, list) else None
        if head == "forall":
            return len(self.domains.get(g[1][2], []))
        if head == "forn":
            return int(g[1][0])
        if head == "forpairs":
            return min(len(self.domains.get(g[1][2], [])), len(self.domains.get(g[2][2], []))) or 1
        return 1

    def count(self, g: GroundedGoal, lookup) -> tuple[int, int]:
        """(satisfied, total) for a conjunct. lookup(key) -> bool | None."""
        e = g.expr
        head = e[0]
        if head == "forall":
            var, syn = e[1][0], e[1][2]
            insts = self.domains.get(syn, [])
            sat = sum(1 for o in insts if evaluate(e[2], {var: o}, lookup, self.domains) is True)
            return sat, len(insts)
        if head == "forn":
            k = int(e[1][0])
            var, syn = e[2][0], e[2][2]
            insts = self.domains.get(syn, [])
            sat = sum(1 for o in insts if evaluate(e[3], {var: o}, lookup, self.domains) is True)
            return min(sat, k), k
        if head == "forpairs":
            v1, s1 = e[1][0], e[1][2]
            v2, s2 = e[2][0], e[2][2]
            a, b = self.domains.get(s1, []), self.domains.get(s2, [])
            used = set()
            sat = 0
            for x in a:
                for y in b:
                    if y in used:
                        continue
                    if evaluate(e[3], {v1: x, v2: y}, lookup, self.domains) is True:
                        used.add(y)
                        sat += 1
                        break
            return sat, min(len(a), len(b)) or 1
        if head == "exists":
            var, syn = e[1][0], e[1][2]
            ok = any(evaluate(e[2], {var: o}, lookup, self.domains) is True for o in self.domains.get(syn, []))
            return int(ok), 1
        return int(evaluate(e, {}, lookup, self.domains) is True), 1

    def remaining(self, lookup) -> tuple[list[str], float]:
        rem, s_tot, n_tot = [], 0, 0
        for g in self.goals:
            sat, n = self.count(g, lookup)
            s_tot += sat
            n_tot += n
            if sat < n:
                rem.append(g.line.replace(" 0/", f" {sat}/", 1))
        return rem, (s_tot / n_tot if n_tot else 1.0)

    # ---- universe ---------------------------------------------------------------
    def atoms(self) -> set[tuple[str, tuple[str, ...]]]:
        """Every grounded (pred, args) the goal or init can ask about."""
        out = set()
        for g in self.goals:
            out |= _atoms(g.expr, {}, self.domains)
        for k, _ in self.init_lines:
            name, args = parse_key(k)
            out.add((name, tuple(args)))
        return out

    def forced_literals(self) -> list[tuple[str, bool]]:
        """(key, value) pairs that MUST hold once the task succeeds: literals reached through
        forall / and only (never through exists / or / forn / forpairs, whose witness is unknown)."""
        out = {}

        def walk(e, env):
            if isinstance(e, str) or not e:
                return
            head = e[0]
            if head == "forall":
                binders = [p for p in e[1:] if isinstance(p, list) and len(p) == 3 and p[1] == "-"]
                envs = [dict(env)]
                for var, _, syn in binders:
                    envs = [dict(en, **{var: o}) for en in envs for o in self.domains.get(syn, [])]
                for en in envs:
                    walk(e[-1], en)
            elif head == "and":
                for x in e[1:]:
                    walk(x, env)
            elif head == "not":
                inner = e[1]
                if isinstance(inner, list) and inner and inner[0] not in QUANT and inner[0] not in LOGIC:
                    args = [env.get(a, a) for a in inner[1:]]
                    if not any(a.startswith("?") for a in args):
                        out[key(inner[0], args)] = False
            elif head in QUANT or head in LOGIC:
                return
            else:
                args = [env.get(a, a) for a in e[1:]]
                if not any(a.startswith("?") for a in args):
                    out[key(head, args)] = True
        for g in self.goals:
            walk(g.expr, {})
        return sorted(out.items())

    def to_json(self) -> dict:
        return dict(goal_lines=[g.line for g in self.goals],
                    goals=[g.to_json() for g in self.goals],
                    init_lines=[[k, v] for k, v in self.init_lines],
                    domains=self.domains, synset_to_scene=self.inst_to_name)


# ---- helpers ----------------------------------------------------------------------
def key(pred: str, args) -> str:
    return f"({pred}{''.join(' ' + a for a in args)})"


_KEY_RE = re.compile(r"^\((\S+)((?:\s+\S+)*)\)$")


def parse_key(k: str):
    m = _KEY_RE.match(k)
    return m.group(1), m.group(2).split()


def _all3(vals):
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


def _any3(vals):
    if any(v is True for v in vals):
        return True
    return None if any(v is None for v in vals) else False


def evaluate(e, env: dict, lookup, domains: dict | None = None):
    """3-valued: True / False / None(unknown). `domains` grounds quantifiers nested in a body
    (forall x (exists y (inside x y)))."""
    if isinstance(e, str):
        return None
    head = e[0]
    if head == "not":
        v = evaluate(e[1], env, lookup, domains)
        return None if v is None else (not v)
    if head == "and":
        return _all3([evaluate(x, env, lookup, domains) for x in e[1:]])
    if head == "or":
        return _any3([evaluate(x, env, lookup, domains) for x in e[1:]])
    if head in QUANT:
        if domains is None:
            return None
        if head == "forall":
            var, syn = e[1][0], e[1][2]
            return _all3([evaluate(e[2], {**env, var: o}, lookup, domains) for o in domains.get(syn, [])])
        if head == "exists":
            var, syn = e[1][0], e[1][2]
            return _any3([evaluate(e[2], {**env, var: o}, lookup, domains) for o in domains.get(syn, [])])
        if head == "forn":
            k = int(e[1][0])
            var, syn = e[2][0], e[2][2]
            vals = [evaluate(e[3], {**env, var: o}, lookup, domains) for o in domains.get(syn, [])]
            if sum(v is True for v in vals) >= k:
                return True
            return None if sum(v is not False for v in vals) >= k else False
        if head in ("forpairs", "fornpairs"):
            if head == "fornpairs":
                k = int(e[1][0]); b1, b2, body = e[2], e[3], e[4]
            else:
                b1, b2, body = e[1], e[2], e[3]
                k = None
            a, b = domains.get(b1[2], []), domains.get(b2[2], [])
            need = k if k is not None else min(len(a), len(b))
            used, sat, unknown = set(), 0, False
            for x in a:
                for y in b:
                    if y in used:
                        continue
                    v = evaluate(body, {**env, b1[0]: x, b2[0]: y}, lookup, domains)
                    if v is True:
                        used.add(y); sat += 1
                        break
                    if v is None:
                        unknown = True
            return True if sat >= need else (None if unknown else False)
        return None
    args = [env.get(a, a) for a in e[1:]]
    if any(a.startswith("?") for a in args):
        return None
    return lookup(key(head, args))


def render(e) -> str:
    """Canonical text with variables shortened to ?x / ?y (stable across runs)."""
    return _render(e, {})


def _short_vars(e, m):
    if isinstance(e, list) and e and e[0] in QUANT:
        for part in e[1:]:
            if isinstance(part, list) and len(part) == 3 and part[1] == "-":
                if part[0] not in m:
                    m[part[0]] = "?x" if not m else ("?y" if len(m) == 1 else f"?v{len(m)}")
        for part in e[1:]:
            _short_vars(part, m)
    elif isinstance(e, list):
        for part in e[1:]:
            _short_vars(part, m)


def _render(e, m) -> str:
    if not m:
        _short_vars(e, m)
    if isinstance(e, str):
        return m.get(e, e)
    head = e[0]
    if head in QUANT:
        body = e[-1]
        return _render(body, m)
    if head == "not":
        return f"(not {_render(e[1], m)})"
    if head in ("and", "or"):
        return f"({head} {' '.join(_render(x, m) for x in e[1:])})"
    return f"({head}{''.join(' ' + m.get(a, a) for a in e[1:])})"


def syns_in(e) -> list[str]:
    out = []
    if isinstance(e, list) and e and e[0] in QUANT:
        for part in e[1:]:
            if isinstance(part, list) and len(part) == 3 and part[1] == "-":
                out.append(part[2])
            else:
                out += syns_in(part)
    elif isinstance(e, list):
        for part in e[1:]:
            out += syns_in(part)
    return out


def _atoms(e, env, domains) -> set:
    if isinstance(e, str) or not e:
        return set()
    head = e[0]
    if head in QUANT:
        binders = [p for p in e[1:] if isinstance(p, list) and len(p) == 3 and p[1] == "-"]
        body = e[-1]
        out = set()
        envs = [dict(env)]
        for var, _, syn in binders:
            envs = [dict(en, **{var: o}) for en in envs for o in domains.get(syn, [])]
        for en in envs:
            out |= _atoms(body, en, domains)
        return out
    if head in LOGIC:
        out = set()
        for x in e[1:]:
            out |= _atoms(x, env, domains)
        return out
    args = tuple(env.get(a, a) for a in e[1:])
    if any(a.startswith("?") for a in args):
        return set()
    return {(head, args)}


def save_goal(task_index: int, bddl_path: str, tg: TaskGoal, episodes: list[int], path: str) -> dict:
    out = dict(task=task_index, bddl=str(bddl_path), **tg.to_json(),
               note="COUNT lines: quantified instances are exchangeable and only counted; "
                    "keys use scene instance names, identical to Subtask text",
               episodes=episodes)
    pathlib.Path(path).write_text(json.dumps(out, indent=2))
    return out
