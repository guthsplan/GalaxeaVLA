"""tools/export_skill_expert.py on toy checkpoints, and the exported file in SkillExpertRouter."""
import subprocess
import sys
from pathlib import Path

import torch

TOOL = Path(__file__).resolve().parents[1] / "tools" / "export_skill_expert.py"
Q = "action_expert.layers.0.self_attn.q_proj"


def _ckpts(tmp_path, touch_vlm=False):
    g = torch.Generator().manual_seed(0)
    base = {
        "model.vlm.layers.0.mlp.down_proj.weight": torch.randn(4, 4, generator=g),
        f"model.{Q}.weight": torch.randn(8, 8, generator=g),
        "model.action_expert.output_proj.weight": torch.randn(3, 8, generator=g),
        "model.action_expert.norm.weight": torch.ones(8),
    }
    a, b = torch.randn(2, 8, generator=g), torch.randn(8, 2, generator=g)
    scale = 64 / 2
    expert = dict(base)
    # what export_state_dict writes: frozen weights in bf16, merged = base + scale * B @ A
    expert[f"model.{Q}.weight"] = base[f"model.{Q}.weight"].to(torch.bfloat16).float() + scale * b @ a
    expert["model.vlm.layers.0.mlp.down_proj.weight"] = base["model.vlm.layers.0.mlp.down_proj.weight"].to(torch.bfloat16)
    expert["model.action_expert.output_proj.weight"] = base["model.action_expert.output_proj.weight"] + 1.0
    if touch_vlm:
        expert["model.vlm.layers.0.mlp.down_proj.weight"] = expert["model.vlm.layers.0.mlp.down_proj.weight"] + 1
    torch.save({"model_state_dict": base}, tmp_path / "base.pt")
    torch.save({"model_state_dict": expert, "step": 7,
                "lora_state_dict": {f"model.{Q}.lora_A.default.weight": a, f"model.{Q}.lora_B.default.weight": b}},
               tmp_path / "step_7.pt")
    return a, b, scale


def _run(tmp_path, *extra):
    return subprocess.run([sys.executable, str(TOOL), str(tmp_path / "step_7.pt"), "-o", str(tmp_path / "e.pt"),
                           "--base", str(tmp_path / "base.pt"), *extra], capture_output=True, text=True)


def test_export_and_route(tmp_path):
    a, b, scale = _ckpts(tmp_path)
    r = _run(tmp_path, "--groups", "grasp")
    assert r.returncode == 0, r.stdout + r.stderr
    e = torch.load(tmp_path / "e.pt", weights_only=False)
    assert e["groups"] == ["grasp"] and set(e["lora"]) == {Q}
    torch.testing.assert_close(e["lora"][Q]["B"] @ e["lora"][Q]["A"], scale * b @ a, rtol=1e-4, atol=1e-4)
    assert set(e["full"]) == {"action_expert.output_proj.weight"}

    from test_skill_router import _Toy, _q0_out
    from g05.models.g05.skill_router import SkillExpertRouter

    # the toy model has its own base weights: drop the fingerprint, keep adapters and full params
    model = _Toy()
    e["base_fingerprint"] = {}
    torch.save(e, tmp_path / "e2.pt")
    x = torch.randn(2, 8)
    before = _q0_out(model, x).detach()
    router = SkillExpertRouter(model, [str(tmp_path / "e2.pt")])
    router.route("Subtask: pick up a from b")
    torch.testing.assert_close(_q0_out(model, x), before + x @ (scale * b @ a).t(), rtol=1e-4, atol=1e-4)


def test_export_refuses_a_changed_vlm(tmp_path):
    _ckpts(tmp_path, touch_vlm=True)
    r = _run(tmp_path, "--groups", "grasp")
    assert r.returncode != 0 and "outside the action expert differ" in (r.stdout + r.stderr)


def test_export_refuses_a_wrong_base(tmp_path):
    _ckpts(tmp_path)
    ck = torch.load(tmp_path / "base.pt")
    ck["model_state_dict"][f"model.{Q}.weight"] += 5.0
    torch.save(ck, tmp_path / "base.pt")
    r = _run(tmp_path, "--groups", "grasp")
    assert r.returncode != 0 and "do not reproduce the merged weights" in (r.stdout + r.stderr)


def test_base_from_manifest_and_resume_chain(tmp_path):
    import json

    sys.path.insert(0, str(TOOL.parent))
    import export_skill_expert as ex

    def run(name, cfg, manifest=None):
        d = tmp_path / name
        (d / ".hydra").mkdir(parents=True)
        (d / "checkpoints").mkdir()
        (d / ".hydra" / "config.yaml").write_text(cfg)
        if manifest is not None:
            (d / "run_manifest.json").write_text(json.dumps(manifest))
        return d / "checkpoints" / "step_5.pt"

    first = run("first", "resume_ckpt: null\nmodel:\n  pretrained_ckpt: /cot/best.pt\n")
    second = run("second", f"resume_ckpt: {first}\nmodel:\n  pretrained_ckpt: g05-base.pt\n")
    third = run("third", f"resume_ckpt: {second}\nmodel:\n  pretrained_ckpt: g05-base.pt\n")
    inplace = tmp_path / "inplace" / "checkpoints" / "step_9.pt"
    run("inplace", f"resume_ckpt: {inplace}\nmodel:\n  pretrained_ckpt: g05-base.pt\n")
    for ck in (first, second, third):
        assert ex._config_base(ck, ex._run_config(ck)) == Path("/cot/best.pt")
    # resumed into its own directory: the config no longer knows; the manifest does
    assert ex._config_base(inplace, ex._run_config(inplace)) is None
    assert ex._manifest_base(inplace) is None
    (tmp_path / "inplace" / "run_manifest.json").write_text(json.dumps({"base_checkpoint": "/cot/best.pt"}))
    assert ex._manifest_base(inplace) == Path("/cot/best.pt")


def test_fm_context_from_the_run_config():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "export_skill_expert", Path(__file__).resolve().parents[1] / "tools" / "export_skill_expert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    builder = lambda name: {"model": {"processor": {"samples_builder": {
        "eval_builder": {"_target_": f"g05.data_processor.processor.samples_builder.{name}"}}}}}
    assert mod.fm_context_of(builder("SkillExpertContextBuilder")) == "prompt"
    assert mod.fm_context_of(builder("SubtaskCoTBuilder")) == "cot"
    assert mod.fm_context_of({**builder("SubtaskCoTBuilder"), "skill_expert_fm_context": "prompt"}) == "prompt"
