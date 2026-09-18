# G0.5 on BEHAVIOR-1K 2026 (R1Pro / OmniGibson)

Team-side additions on the `behavior2026` branch of the GalaxeaVLA fork. Everything here is
driven from the competition repo (`behavior1k-2026-solution/scripts/*.sh`); this file documents
what lives inside GalaxeaVLA itself.

> **CoT post-training** (canonical BeliefGraph protocol, AR-only) is documented
> separately in [`COT.md`](COT.md). This file covers the no-CoT action baseline.

## Files added / changed vs. upstream

| Path | Purpose |
|------|---------|
| `configs/data/behavior.yaml` | LeRobot-v3 data config for the BEHAVIOR demos (per-part columns, 3 RGB cameras, 27-D grouped layout) |
| `configs/task/behavior.yaml` | Post-training task config derived from `r1pro_wbc` (single-frame obs, no CoT, no VLM co-training) |
| `src/g05/data_processor/transforms/relative_action_partial.py` | `RelativeJointTransformPartial`: torso targets relative, base velocity absolute |
| `scripts/serve_g05_b1k.py` | Websocket policy server speaking the `omnigibson.eval.eval` protocol |
| `tools/make_b1k_subset.py` | Builds a task-filtered LeRobot-v3 subset with derived `action.*` / `observation.state.*` columns |
| `configs/task/behavior_cot.yaml` | CoT post-training: canonical BeliefGraph candidates, AR-only (no FM). See `COT.md` |
| `src/g05/data_processor/processor/samples_builder.py` | `BeliefGraph*` builder family (ported from the bg delivery snapshot) + BG-conditioned BBox/Trace |
| `src/g05/belief_graph/` | External belief module for inference (belief memory, operators, goal delta, estimator, runtime, serving middleware) |
| `tools/build_bg_fields.py`, `tools/join_bg_dataset.py`, `tools/eval_estimator.py` | bgdata -> `bg_*_index` field conversion, scratch-dataset join, estimator gate |
| `tools/merge_b1k_cot_labels.py`, `tools/make_cot_fixture.py` | Multi-episode CoT materialization with causal alignment; few-episode test fixture |
| `tools/extract_model_ckpt.py` | Strips optimizer state from a training checkpoint (~34 GB -> ~11 GB) for inference |
| `src/g05/models/g05/qwen35/vision.py`, `src/g05/tokenizer/.../modular_actioncodec2v2.py` | `G05_DISABLE_FLASH_ATTN=1` opt-out for GPUs where FA4 kernels cannot be built (e.g. sm_120) |

## Layouts

```
BEHAVIOR action(23): base_vel[0:3] torso[3:7] left_arm[7:14] left_gripper[14] right_arm[15:22] right_gripper[22]
BEHAVIOR state(61):  base_qvel[0:3] left_arm[3:10] left_gripper[24:26] right_arm[28:35] right_gripper[49:51] trunk[53:57]
G0.5 parts:          left_arm(7) left_gripper(1) right_arm(7) right_gripper(1) lower_body(7)=[torso(4), base_vel(3)]
```

Grouped 27-D order (`configs/data/parts_meta/r1pro.yaml`):
`left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7)`.

## Environment variables

| Var | Meaning |
|-----|---------|
| `B1K_SUBSET_DIR` | LeRobot-v3 dataset dir used by `configs/data/behavior.yaml` (default `data/b1k_5task_lerobot`, repo-relative) |
| `B1K_TASKS_JSONL` | `meta/tasks.jsonl` used by the server to map `task_name -> instruction` (default `$B1K_SUBSET_DIR/meta/tasks.jsonl`) |
| `G05_OUTPUT_DIR` | Hydra output root (`<root>/behavior/<EXP_NAME>/`) |
| `G05_DISABLE_FLASH_ATTN` | `1` to force the eager attention path |

## Commands (run from the GalaxeaVLA root, inside its venv)

```bash
# subset from the raw challenge demos
python tools/make_b1k_subset.py --src /path/to/behavior-1k-2026-challenge-demos --dst $B1K_SUBSET_DIR --tasks 0 1 35 40 46

# resolved config / smoke
python tools/resolve_config.py behavior --key data.embodiment_datasets
bash scripts/run/finetune.sh 1 behavior --dry-run
bash scripts/run/finetune.sh 1 behavior --test model.max_steps=40

# training
EXP_NAME=g05_b1k5_run1 bash scripts/run/finetune.sh 1 behavior model.max_steps=30000

# inference server for the official evaluator
python tools/extract_model_ckpt.py $G05_OUTPUT_DIR/behavior/<exp>/checkpoints/step_30000.pt
python scripts/serve_g05_b1k.py --ckpt_path .../step_30000_model.pt --task-name turning_on_radio --port 8000
python scripts/serve_g05_b1k.py --ckpt_path ... --task-name turning_on_radio --selftest   # one synthetic forward pass
```
