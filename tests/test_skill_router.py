"""SkillExpertRouter (g05.models.g05.skill_router) on a toy model with an action-expert layout."""
import pytest
import torch
import torch.nn as nn

from g05.models.g05.skill_router import FORMAT, SkillExpertRouter


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.vlm = nn.Linear(4, 4)
        self.action_expert = nn.Module()
        self.action_expert.layers = nn.ModuleList([_Layer(), _Layer()])
        self.action_expert.output_proj = nn.Linear(8, 3)


Q0 = "action_expert.layers.0.self_attn.q_proj"


def _expert(tmp_path, model, groups, seed, full=True, fingerprint=None):
    g = torch.Generator().manual_seed(seed)
    lora = {Q0: {"A": torch.randn(2, 8, generator=g), "B": torch.randn(8, 2, generator=g)}}
    payload = {
        "format": FORMAT,
        "groups": groups,
        "lora": lora,
        "full": {"action_expert.output_proj.weight": torch.randn(3, 8, generator=g)} if full else {},
        "base_fingerprint": fingerprint
        if fingerprint is not None
        else {"vlm.weight": float(model.vlm.weight.double().sum())},
        "meta": {},
    }
    path = tmp_path / f"{'_'.join(groups)}.pt"
    torch.save(payload, path)
    return path, payload


def _q0_out(model, x):
    return model.action_expert.layers[0].self_attn.q_proj(x)


def test_switching_adds_the_adapter_and_swaps_full_params(tmp_path):
    model = _Toy()
    x = torch.randn(5, 8)
    base_q = _q0_out(model, x).detach().clone()
    base_out = model.action_expert.output_proj.weight.detach().clone()
    p_grasp, e_grasp = _expert(tmp_path, model, ["grasp"], 1)
    p_place, e_place = _expert(tmp_path, model, ["place", "revolute"], 2)
    router = SkillExpertRouter(model, [str(p_grasp), str(p_place)])

    r = router.route("Subtask: pick up cup_1 from table_2|Action: ")
    assert r == {"skill": "pick up from", "group": "grasp", "expert": "grasp", "context_only": True}
    a, b = e_grasp["lora"][Q0]["A"], e_grasp["lora"][Q0]["B"]
    torch.testing.assert_close(_q0_out(model, x), base_q + x @ a.t() @ b.t())
    torch.testing.assert_close(model.action_expert.output_proj.weight, e_grasp["full"]["action_expert.output_proj.weight"])

    r = router.route("Subtask: open door fridge_0")
    assert r["expert"] == "place+revolute" and r["group"] == "revolute"
    a, b = e_place["lora"][Q0]["A"], e_place["lora"][Q0]["B"]
    torch.testing.assert_close(_q0_out(model, x), base_q + x @ a.t() @ b.t())

    # no expert for navigation and no fallback: base action expert, full-KV conditioning
    r = router.route("Subtask: move to table_2")
    assert r["expert"] is None and r["context_only"] is False
    torch.testing.assert_close(_q0_out(model, x), base_q)
    torch.testing.assert_close(model.action_expert.output_proj.weight, base_out)

    # many switches leave the base weights untouched
    for t in ["pick up a from b", "open door c", "move to d"] * 5:
        router.route(t)
    torch.testing.assert_close(_q0_out(model, x), base_q)
    router.remove()
    torch.testing.assert_close(_q0_out(model, x), base_q)


def test_fallback_and_unparsable_subtasks(tmp_path):
    model = _Toy()
    p_all, _ = _expert(tmp_path, model, ["all"], 3, full=False)
    p_grasp, _ = _expert(tmp_path, model, ["grasp"], 1)
    router = SkillExpertRouter(model, [str(p_grasp), str(p_all)])
    assert router.route("Subtask: move to table_2")["expert"] == "all"
    assert router.route("Subtask: pick up a from b")["expert"] == "grasp"
    # garbage keeps the current group
    r = router.route("Subtask: ???")
    assert r["skill"] is None and r["group"] == "grasp" and r["expert"] == "grasp"
    router.reset()
    assert router.route(None) == {"skill": None, "group": None, "expert": "all", "context_only": True}


def test_min_consecutive(tmp_path):
    model = _Toy()
    p_grasp, _ = _expert(tmp_path, model, ["grasp"], 1)
    p_place, _ = _expert(tmp_path, model, ["place"], 2)
    router = SkillExpertRouter(model, [str(p_grasp), str(p_place)], min_consecutive=2)
    assert router.route("pick up a from b")["group"] == "grasp"   # first prediction switches at once
    assert router.route("place a on b")["group"] == "grasp"       # 1 of 2
    assert router.route("pick up a from b")["group"] == "grasp"   # streak broken
    assert router.route("place a on b")["group"] == "grasp"
    assert router.route("place a on b")["group"] == "place"


def test_rejects_wrong_base_and_duplicate_groups(tmp_path):
    model = _Toy()
    p_bad, _ = _expert(tmp_path, model, ["grasp"], 1, fingerprint={"vlm.weight": 1e6})
    with pytest.raises(ValueError, match="different base checkpoint"):
        SkillExpertRouter(model, [str(p_bad)])
    p1, _ = _expert(tmp_path, model, ["grasp"], 1)
    p2, _ = _expert(tmp_path, model, ["grasp", "place"], 2)
    with pytest.raises(ValueError, match="two experts"):
        SkillExpertRouter(model, [str(p1), str(p2)])
