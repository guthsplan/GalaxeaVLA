"""BEHAVIOR skill groups for skill-specific action experts.

The 35 skills of the 2026 challenge skill annotations (`skill_description`), grouped by the
motion the action expert has to produce, not by meaning. The main axis is the gripper: whether
the segment closes it, opens it or never switches it (the most abrupt part of an action chunk),
then whether the base or the arms drive the motion. Frame shares are over the 20k challenge
demos; motion figures are medians over 38k segments of 64 tasks (gripper switches per segment
on average).

    navigation    38.4%  base drives (moving 91% of the time), arms still, no gripper switch
    grasp         30.3%  approach and close the gripper on an object or a handle (1.1 switches)
    place         17.0%  carry and open the gripper at a target (1.0 switches)
    push_contact   9.0%  push / close without the gripper (0.0 switches), long (17 s)
    tool           3.1%  held-tool strokes without a gripper switch, short (6 s), one arm
    inplace        2.1%  base fully still (2.5%), several gripper switches (1.7), short (8 s):
                         switches, buttons, hand-overs, insert / attach

Opening and closing the same articulated object land in different groups on purpose: opening
grasps the handle (open door 1.6 switches, open drawer 2.0, open lid 1.0) while closing pushes it
shut (0.0). "turn to" turns an object toward a target ("turn food_processor_90 to robot") with
the arms; it is not base navigation.

Used on both sides of a skill-expert run:
  * training: `skill_groups` on the dataset keeps only the frames whose annotated skill is in
    the expert's groups (see `skill_weights_for_groups`);
  * inference: the CoT subtask the policy generates ("Subtask: pick up hotdog_207 from
    fridge_0") is mapped back to its skill and group (`skill_from_subtask`), which picks the
    expert (g05.models.g05.skill_router).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Union

SKILL_GROUPS: Dict[str, tuple] = {
    "navigation": ("move to",),
    "grasp": ("pick up from", "lift", "open door", "open drawer", "open lid", "pull tray", "tip over"),
    "place": ("place in", "place on", "place on next to", "place in next to", "place under", "hang"),
    "push_contact": ("push to", "push tray", "close door", "close lid", "close drawer", "turn to"),
    "tool": ("chop", "sweep surface", "sweep off", "pour", "wipe hard", "spray"),
    "inplace": (
        "turn on switch", "turn off switch", "press", "ignite",
        "hand over", "insert", "hold", "release", "attach",
    ),
}

#: Pseudo-group of an expert trained on every skill (the fallback expert at inference).
ALL = "all"

SKILL_TO_GROUP: Dict[str, str] = {s: g for g, skills in SKILL_GROUPS.items() for s in skills}
ALL_SKILLS: tuple = tuple(SKILL_TO_GROUP)


def parse_groups(spec: Union[None, str, Iterable[str]]) -> List[str]:
    """`"grasp,place"` / `["grasp", "place"]` / `"all"` -> validated group names, in order."""
    if spec is None or spec == "":
        return []
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    groups = [str(g).strip() for g in items if str(g).strip()]
    if ALL in groups:
        if len(groups) > 1:
            raise ValueError(f"skill group '{ALL}' cannot be combined with other groups: {groups}")
        return [ALL]
    unknown = [g for g in groups if g not in SKILL_GROUPS]
    if unknown:
        raise ValueError(f"unknown skill groups {unknown}; known: {sorted(SKILL_GROUPS)} or '{ALL}'")
    return list(dict.fromkeys(groups))


def skills_of(groups: Sequence[str]) -> List[str]:
    groups = parse_groups(groups)
    if groups == [ALL]:
        return list(ALL_SKILLS)
    return [s for g in groups for s in SKILL_GROUPS[g]]


def skill_weights_for_groups(groups: Sequence[str], skills_present: Iterable[str]) -> Dict[str, float]:
    """Skill-sampling multipliers that keep only `groups`: 1 inside, 0 for everything else.

    `skills_present` are the skill labels found in the data, so that skills missing from
    SKILL_GROUPS (and the unannotated filler label) are dropped as well.
    """
    keep = set(skills_of(groups))
    return {s: (1.0 if s in keep else 0.0) for s in set(skills_present) | keep}


# ---------------------------------------------------------------------------
# Subtask text -> skill
# ---------------------------------------------------------------------------
# The Subtask CoT labels are rendered from the skill annotation by
# tools/build_b1k_bbox_trace_sidecars.py::_subtask_text:
#   bgdata.operators.TEXT     move {mp}to {o} | pick up {mp}{o} from {r} | place {o} on {r}
#                             place {o} in {r} | open door {d} | close door {d} | turn on switch {d}
#   EXTRA_SUBTASK_TEXT        open lid {0} | close lid {0} | hand over {0} from {1} to {2} hand
#                             turn {0} to {1} | push {0} to {1} | place {0} on {1} next to {2}
#                             chop {1} with {0} | sweep {0} off {1}
#   everything else           "<skill> <objects...>"
# Object names are scene names (hotdog_207, fridge_dszchb_0); a slot holding several objects is
# joined with " and ", so a slot can contain spaces.
_PREFIX_SKILLS = sorted(ALL_SKILLS, key=len, reverse=True)
_TEMPLATE_RULES = (
    (re.compile(r"^move\b"), "move to"),
    (re.compile(r"^pick up\b"), "pick up from"),
    (re.compile(r"^place .+ on .+ next to\b"), "place on next to"),
    (re.compile(r"^place .+ in\b"), "place in"),
    (re.compile(r"^place .+ on\b"), "place on"),
    (re.compile(r"^push .+ to\b"), "push to"),
    (re.compile(r"^turn .+ to\b"), "turn to"),
    (re.compile(r"^sweep .+ off\b"), "sweep off"),
)


def _clean_subtask(text: str) -> str:
    t = (text or "").strip()
    # The generated span is "Subtask: <subtask>|Action: " (trimmed at <EOV>); BBoxSubtask CoT
    # puts a "BBox: ... |" span in front.
    m = re.search(r"subtask:\s*", t, flags=re.IGNORECASE)
    if m:
        t = t[m.end():]
    t = re.split(r"\||\baction:", t, maxsplit=1, flags=re.IGNORECASE)[0]
    return " ".join(t.lower().split())


def skill_from_subtask(text: Optional[str]) -> Optional[str]:
    """Skill name of a Subtask CoT text, or None when it matches no known skill."""
    t = _clean_subtask(text)
    if not t:
        return None
    for skill in _PREFIX_SKILLS:
        if t == skill or t.startswith(skill + " "):
            return skill
    for pattern, skill in _TEMPLATE_RULES:
        if pattern.search(t):
            return skill
    return None


def group_from_subtask(text: Optional[str]) -> Optional[str]:
    skill = skill_from_subtask(text)
    return SKILL_TO_GROUP.get(skill) if skill else None
