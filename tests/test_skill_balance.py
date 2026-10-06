"""Skill-balanced training-frame sampling (g05.data.skill_balance) on synthetic annotations."""
import collections
import json
import types

import numpy as np
import pytest

from g05.data.skill_balance import SkillBalancedIndex, _episode_segments, parse_skill_weights


def _ann(*segs):
    return {"skill_annotation": [{"skill_description": [s], "frame_duration": fd} for s, fd in segs]}


def test_parse_skill_weights():
    assert parse_skill_weights("move to=0.3, pick up from=1.5") == {"move to": 0.3, "pick up from": 1.5}
    assert parse_skill_weights({"move to": 0}) == {"move to": 0.0}
    assert parse_skill_weights(None) == {}
    with pytest.raises(ValueError):
        parse_skill_weights("move to")


def test_segments_fill_gaps_and_split_intervals():
    # leading frames -> first segment, gap -> previous segment, tail -> last segment
    assert _episode_segments(_ann(("a", [5, 10]), ("b", [12, 20])), 25) == [(0, 12, "a"), (12, 25, "b")]
    # an interrupted segment is a list of intervals
    assert _episode_segments(_ann(("m", [[0, 4], [6, 8]]), ("p", [4, 6])), 8) == [
        (0, 4, "m"), (4, 6, "p"), (6, 8, "m")]


class _Cols(dict):
    """Stands in for the HF episode table: column_names + column access."""
    column_names = property(lambda self: list(self))


def _index(tmp_path, alpha, weights, n):
    # two episodes: 900 frames "move to" + 100 "pick"; 500 "move to" + 500 "pick"
    paths = []
    for i, segs in enumerate([[("move to", [0, 900]), ("pick", [900, 1000])],
                              [("move to", [0, 500]), ("pick", [500, 1000])]]):
        rel = f"annotations/ep{i}.json"
        (tmp_path / "annotations").mkdir(exist_ok=True)
        (tmp_path / rel).write_text(json.dumps(_ann(*segs)))
        paths.append(rel)
    meta = types.SimpleNamespace(episodes=_Cols({"annotation_path": paths}))
    ds = types.SimpleNamespace(meta=meta, root=str(tmp_path))
    return SkillBalancedIndex([ds], [0, 1000], [1000, 2000], 0, 2000, n, alpha, weights)


@pytest.mark.parametrize("alpha,weights,expected_move", [(1.0, {}, 0.7), (0.0, {}, 0.5), (1.0, {"move to": 3 / 7}, 0.5)])
def test_sampling_mass(tmp_path, alpha, weights, expected_move):
    idx = _index(tmp_path, alpha, weights, n=200)
    np.random.seed(0)
    frames = [idx(i) for _ in range(50) for i in range(200)]
    assert all(0 <= f < 2000 for f in frames)
    c = collections.Counter(idx.skill_of(f) for f in frames)
    assert c["move to"] / len(frames) == pytest.approx(expected_move, abs=0.02)


def test_uniform_is_stride_jitter(tmp_path):
    # alpha 1, no weights: sample i lands in window [i*10, i*10 + 10), like train_frame_stride=10
    idx = _index(tmp_path, 1.0, {}, n=200)
    for i in range(200):
        assert i * 10 <= idx(i) < i * 10 + 10


def test_keep_skills_filters_every_other_skill(tmp_path):
    # skill-expert runs: only "pick" frames are drawn, uniformly over them
    _index(tmp_path, 1.0, {}, n=200)  # writes the two annotation files
    idx = SkillBalancedIndex(*_index_args(tmp_path), 200, 1.0, {}, keep_skills=["pick"])
    np.random.seed(0)
    frames = [idx(i) for _ in range(20) for i in range(200)]
    assert {idx.skill_of(f) for f in frames} == {"pick"}
    with pytest.raises(ValueError):
        SkillBalancedIndex(*_index_args(tmp_path), 200, 1.0, {}, keep_skills=["pour"])


def _index_args(tmp_path):
    meta = types.SimpleNamespace(episodes=_Cols({"annotation_path": ["annotations/ep0.json", "annotations/ep1.json"]}))
    ds = types.SimpleNamespace(meta=meta, root=str(tmp_path))
    return [ds], [0, 1000], [1000, 2000], 0, 2000


def test_per_task_val_split(tmp_path):
    from g05.data.skill_balance import skill_val_episodes

    # 6 episodes of 100 frames: task 0 = eps 0-2 ("chop" in 0, 1), task 1 = eps 3-5 ("chop" in 3, 4, 5)
    (tmp_path / "annotations").mkdir()
    skills = ["chop", "chop", "move to", "chop", "chop", "chop"]
    paths = []
    for i, sk in enumerate(skills):
        rel = f"annotations/ep{i}.json"
        (tmp_path / rel).write_text(json.dumps(_ann(("move to", [0, 50]), (sk, [50, 100]))))
        paths.append(rel)
    meta = types.SimpleNamespace(episodes=_Cols({"annotation_path": paths, "task_index": [0, 0, 0, 1, 1, 1]}))
    ds = types.SimpleNamespace(meta=meta, root=str(tmp_path))
    ep_from, ep_to = [100 * i for i in range(6)], [100 * (i + 1) for i in range(6)]

    # last chop episode of each task; ep 2 (no chop) is never validation
    val = skill_val_episodes([ds], ep_from, ep_to, ["chop"], 1)
    assert val == {1, 5}
    # one chop episode of each task always stays in training
    assert skill_val_episodes([ds], ep_from, ep_to, ["chop"], 5) == {1, 4, 5}
    with pytest.raises(ValueError):
        skill_val_episodes([ds], ep_from, ep_to, ["pour"], 1)

    train = SkillBalancedIndex([ds], ep_from, ep_to, 0, 600, 500, keep_skills=["chop"],
                               episodes=set(range(6)) - val)
    vidx = SkillBalancedIndex([ds], ep_from, ep_to, 0, 600, 100, keep_skills=["chop"], episodes=val)
    np.random.seed(0)
    tf = {train(i) // 100 for i in range(500)}
    vf = {vidx(i) // 100 for i in range(100)}
    assert tf == {0, 3, 4} and vf == {1, 5}
    assert all(train.skill_of(train(i)) == "chop" for i in range(500))
