"""Skill groups and Subtask-text routing (g05.data.skill_groups)."""
import pytest

from g05.data.skill_groups import (
    ALL, ALL_SKILLS, SKILL_GROUPS, SKILL_TO_GROUP, group_from_subtask, parse_groups,
    skill_from_subtask, skill_weights_for_groups, skills_of,
)


def test_groups_partition_the_35_challenge_skills():
    skills = [s for g in SKILL_GROUPS.values() for s in g]
    assert len(skills) == len(set(skills)) == len(ALL_SKILLS) == 35


def test_parse_groups():
    assert parse_groups("grasp, place") == ["grasp", "place"]
    assert parse_groups(["revolute", "revolute"]) == ["revolute"]
    assert parse_groups("all") == [ALL]
    assert parse_groups(None) == []
    with pytest.raises(ValueError):
        parse_groups("grasping")
    with pytest.raises(ValueError):
        parse_groups("all,grasp")
    assert skills_of(["all"]) == list(ALL_SKILLS)
    assert skills_of("revolute") == ["open door", "close door", "open lid", "close lid"]


def test_skill_weights_for_groups_drop_unknown_labels():
    w = skill_weights_for_groups(["press"], ["press", "move to", "<unannotated>"])
    assert w["press"] == 1.0 and w["move to"] == 0.0 and w["<unannotated>"] == 0.0


# Renderings of tools/build_b1k_bbox_trace_sidecars.py::_subtask_text (bgdata TEXT, the
# EXTRA_SUBTASK_TEXT templates, and the "<skill> <objects>" default).
@pytest.mark.parametrize("text,skill", [
    ("move to fridge_dszchb_0", "move to"),
    ("move same to countertop_1", "move to"),
    ("pick up hotdog_207 from fridge_dszchb_0", "pick up from"),
    ("pick up other hotdog_208 from plate_1", "pick up from"),
    ("place hotdog_207 on plate_1", "place on"),
    ("place hotdog_207 in microwave_3", "place in"),
    ("place bowl_1 on table_2 next to cup_3", "place on next to"),
    ("place in next to apple_1 bowl_2 cup_3", "place in next to"),
    ("open door fridge_dszchb_0", "open door"),
    ("close lid pot_1", "close lid"),
    ("open drawer cabinet_2", "open drawer"),
    ("push chair_1 to table_2", "push to"),
    ("push tray oven_1", "push tray"),
    ("turn radio_1 to wall_2", "turn to"),
    ("turn on switch radio_1", "turn on switch"),
    ("turn off switch stove_2", "turn off switch"),
    ("hand over cup_1 from left to right hand", "hand over"),
    ("chop apple_1 with knife_2", "chop"),
    ("sweep half_log_176_0 and half_log_176_1 off driveway_umalys_0", "sweep off"),
    ("sweep surface table_1 brush_2", "sweep surface"),
    ("wipe hard table_1 rag_2", "wipe hard"),
])
def test_skill_from_subtask_renderings(text, skill):
    assert skill_from_subtask(text) == skill
    assert group_from_subtask(text) == SKILL_TO_GROUP[skill]


def test_skill_from_generated_span():
    # what generate_text returns for SubtaskCoTBuilder / BBoxSubtaskCoTBuilder
    assert skill_from_subtask("Subtask: pick up hotdog_207 from fridge_0|Action: ") == "pick up from"
    assert skill_from_subtask("BBox: [1, 2, 3, 4] | Subtask: open door fridge_0 | Action:") == "open door"
    assert skill_from_subtask("  SUBTASK:   Place  cup_1  on  table_2 ") == "place on"
    for junk in (None, "", "done", "Subtask: ", "Subtask: dance with robot_1"):
        assert skill_from_subtask(junk) is None
