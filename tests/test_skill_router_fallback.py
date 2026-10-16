"""SkillExpertRouter without an `all` expert: groups without an expert and (unrouted="fallback") subtasks that
map to no group run the base action expert, restored exactly."""
import torch
import torch.nn as nn

from g05.models.g05.skill_router import FORMAT, SkillExpertRouter


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.action_expert = nn.Sequential(nn.Linear(4, 4))


def _expert_file(tmp_path, model, group):
    lin = "action_expert.0"
    payload = {
        "format": FORMAT,
        "groups": [group],
        "lora": {lin: {"A": torch.randn(2, 4), "B": torch.randn(4, 2)}},
        "full": {"action_expert.0.bias": torch.randn(4)},
        "meta": {"fm_context": "cot"},
        "base_fingerprint": {n: float(p.detach().double().sum()) for n, p in model.named_parameters()},
    }
    path = tmp_path / f"{group}.pt"
    torch.save(payload, path)
    return str(path)


def _run(router, model, x, subtask):
    route = router.route(subtask)
    return route, model.action_expert(x)


def test_no_all_expert_falls_back_to_base(tmp_path):
    model = _Toy()
    x = torch.randn(3, 4)
    base_out = model.action_expert(x).detach().clone()
    path = _expert_file(tmp_path, model, "grasp")

    for unrouted, done_expert in (("keep", "grasp"), ("fallback", None)):
        router = SkillExpertRouter(model, [path], unrouted=unrouted)
        assert router.fallback is None
        r, out = _run(router, model, x, "Subtask: pick up apple_1 from table_2|Action: ")
        assert r["expert"] == "grasp" and not torch.allclose(out, base_out)
        # unrouted subtask: keep -> the grasp expert stays; fallback -> base for this call
        r, out = _run(router, model, x, "Subtask: done|Action: ")
        assert r["expert"] == done_expert
        assert torch.allclose(out, base_out) == (done_expert is None)
        # a group without an expert (navigation) -> base, bit-exact
        r, out = _run(router, model, x, "Subtask: move to table_2|Action: ")
        assert r["group"] == "navigation" and r["expert"] is None
        assert torch.equal(out, base_out)
        router.remove()
        assert torch.equal(model.action_expert(x), base_out)
