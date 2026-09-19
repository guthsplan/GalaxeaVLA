# BEHAVIOR-1K 2026 — G0.5 CoT post-training (canonical BeliefGraph protocol, AR-only)

This is the single current description of BEHAVIOR CoT training. Where an older
document disagrees, this file wins; see **Obsolete** at the bottom for what was
replaced.

Task config: `configs/task/behavior_cot.yaml` (inherits `behavior.yaml`).
Baseline no-CoT training is `configs/task/behavior.yaml` and is unaffected.

## Token sequence

Resolved for the qwen35-base processor (`checkpoints/qwen3_5_2b_base_processor`),
3 cameras, one sampled CoT target:

```
CONDITIONING — masked, label = -100, no LM loss
  <image0><image1><image2>                        3 × 64 image tokens
  Embodiment: behavior_r1pro;
  Task: <instruction>
  BeliefGraph: Remaining: ... | Known: ...        bg_known — PRIOR 1 Hz snapshot
  State: <proprio>;
  <prompt>                                        per-builder instruction text
<EOC>
GENERATIVE — supervised, single AR cross-entropy
  <one sampled CoT target>                        Subtask: / Delta: / Belief: /
                                                  Effect: / Observe: / BBox: / Trace:
  |                                               separator, also supervised
  Action:                                         static text, also supervised
  <EOV>                                           supervised because pred_eov=true
  <left_control_0><action….>…<lower_body_1>…      ActionCodec RVQ tokens
  |<eos>                                          supervised
```

Control-token order is `<EOC>` … `Action: <EOV>` … `<eos>`. There is **no `<EOA>`**.
Everything after `<EOC>` receives loss, including the `|` separators and the
literal `Action: ` — `InputPreprocessor._parse_control_tokens` converts post-EOC
static segments into unmasked dynamic text segments.

Exactly **one** CoT format is emitted per sample. `MixedSamplesBuilder` filters
candidates by `can_handle(data)` and draws one by weight from the applicable
subset; if none apply it falls back to `BaseSamplesBuilder` (no CoT).

## Fields

Conditioning (masked, pre-`<EOC>`, never a target):

| dataset column | decoded field | slot |
|---|---|---|
| `bg_known_index` | `bg_known` | `BeliefGraph: Remaining: … \| Known: …` |

`bg_known` is the **previous** 1 Hz snapshot, mirroring `prev_memory`/`memory`:
input = state before this chunk, target = state at this chunk. Derived by
`tools/build_bg_fields.py`.

Prediction targets (post-`<EOC>`, clean — no robustness noise is ever applied):

| dataset column | decoded field | prefix | builder |
|---|---|---|---|
| `atomic_task_index` | `atomic_task` | `Subtask:` | `BeliefGraphSubtaskCoTBuilder` |
| `bg_delta_index` | `bg_delta` | `Delta:` | `BeliefGraphDeltaCoTBuilder` |
| `bg_belief_index` | `bg_belief` | `Belief:` | `BeliefGraphUpdateCoTBuilder` |
| `bg_effect_index` | `bg_effect` | `Effect:` | `BeliefGraphEffectCoTBuilder` |
| `bg_observe_index` | `bg_observe` | `Observe:` | `BeliefGraphObserveCoTBuilder` |
| `bbox_index` | `bbox` | `BBox:` | `BeliefGraphBBoxCoTBuilder` |
| `2d_trace_index` | `trace_2d` | `Trace:` | `BeliefGraphTrace2DCoTBuilder` |

`BeliefGraphObserveCoTBuilder` deliberately takes **no** `bg_known`: the predicted
predicates must come from the images, not from belief memory. At inference
`g05.belief_graph.BeliefGraphRuntime.on_model_cot` parses that span back into the
external belief table ("model-as-estimator").

All columns are int indices into `meta/tasks.parquet`, resolved by
`LeRobotDatasetMetadata.lookup_task_text` — an O(1) dict built once per metadata
object. An index absent from the table raises rather than decoding to `""`.

Field names must not contain the substrings `observation` or `action`:
`BaseLerobotDataset.__getitem__` filters those out of the processor payload.

## Weights

In `configs/task/behavior_cot.yaml` only — never in the launcher:

```
Subtask 3 · Delta 2 · Belief 2 · Observe 2 · Effect 1 · BBox 1 · Trace 1
```

Relative, renormalised over the candidates applicable to each sample, so the
empirical mix reflects label coverage too. `scripts/cot_coverage.py` in the
solution repo reports both.

## Training objective — AR only

```
discrete_action: true      continuous_action: false      return_continuous_action: false
predict_cot: true          input_preprocessor.pred_eov: true
```

With $\mathcal{S}$ the supervised positions ($y_i \neq -100$):

```
L = w_ce · (1/|S|) · Σ_{i∈S} −log p_θ(y_i | x_<i)
```

One next-token cross-entropy over CoT text, separators, `<EOV>` and the
ActionCodec tokens. **There is no flow-matching term.** `G05Model.forward` /
`G05ModelQwen35.forward` call `fm_helper.train_step` only under
`if continuous_action:`, so with `continuous_action=false` the flow head's
prefix-KV rebuild, time/noise sampling and forward pass never execute and
`fm_loss` is absent from `loss_dict` (this replaced an earlier
`fm_loss = fm_loss * 0`, which paid the full FM cost and discarded it).

Continuous robot actions are obtained by **`ActionCodec.decode(action_tokens)`**,
not by flow matching.

**DDP requirement that follows from skipping FM.** Because the flow head's parameters
never enter the autograd graph when `continuous_action=false`, DDP with the default
`find_unused_parameters=false` raises *"Expected to have finished reduction in the prior
iteration"* on the **second** optimizer step (a one-step smoke test will not catch it).
`behavior_cot.yaml` therefore sets `model.find_unused_parameters: true`, the same setting
`libero` / `libero_ar` / `bridge` / `robotwin` / `so100` use. Upstream's `fm_loss * 0`
avoided this only by paying for the full FM forward. A cleaner alternative — freezing the
FM parameters (`requires_grad=False`) when `continuous_action=false` — is a code change
not yet made.

Metrics: `train/cot_accuracy` and `train/action_token_accuracy` split by token-ID
range in `ar_helper.py`. Note `cot_accuracy` covers *all* non-action supervised
tokens, so it includes the separators, `Action: ` and `<EOV>` and is optimistically
biased relative to semantic CoT quality.

## Inference

```
prefill (conditioning, stops at <EOC>)
  → AR generate CoT text, stop at <EOV>          g05_policy.py:1028, predict_cot
  → AR generate ActionCodec tokens               generate_action, discrete_action
  → ActionCodec.decode → continuous action chunk
```

No FM inference: `G05Policy.generate_action` guards `inference_fm` with
`if self.continuous_action:`, which is false here.

Validation/inference builder selection is deterministic — `eval_builder`
(`BeliefGraphSubtaskCoTBuilder`), never the training-time weighted draw.

## Action path — unchanged

Same ActionCodec, chunk size 32, `parts_meta`
`left_control 9 / left_gripper 1 / right_control 9 / right_gripper 1 / lower_body 7`,
gripper sign binarisation, `RelativeJointTransform` on arms,
`RelativeJointTransformPartial{lower_body: 4}`, and the BEHAVIOR 23-D → 27-D
mapping. `configs/task/behavior.yaml` and `configs/data/behavior.yaml` are
untouched; the resolved diff between `behavior` and `behavior_cot` is confined to
the CoT/AR-only keys and `logger.project`.

## Data flow

```
raw demos ──bbox_replay.py──▶ episode_<demo_id>_cot_{strings.json,index.jsonl}   (bbox, trace_2d)
bgdata ────cot_targets.py───▶ cot_targets.parquet (task, episode, frame, subtask, belief, delta, effect)
                                   │
                        tools/build_bg_fields.py   → bg_fields.parquet, bg_tasks.jsonl,
                                   │                  bg_frame_indices.parquet (derives bg_known, bg_observe)
                        tools/join_bg_dataset.py   → scratch LeRobot dataset, one task, source read-only
                   or   tools/merge_b1k_cot_labels.py → multi-episode materialization with causal
                                   │                    alignment + staleness audit
                                   ▼
                        B1K_COT_SUBSET_DIR ──▶ scripts/train_g05_cot.sh ──▶ scripts/run/finetune.sh
```

`episode` in both label sources is the **`raw_episode_id`** (`demo_id`,
e.g. `450010`), not the LeRobot `episode_index`. `meta/episodes/*.parquet` carries
`raw_episode_id` for the join.

Alignment: `source_frame = max{s : s ≤ t}` within an episode, never crossing an
episode boundary and never selecting a future frame. `bg_known`/`bg_belief`/
`bg_observe` are snapshots (staleness is real, boundable); `subtask`/`bg_delta`/
`bg_effect` are segment-level. `bbox`/`trace_2d` are frame-exact and never held
forward.

## Obsolete

Replaced by the canonical protocol above; do not reintroduce:

- `BGcond:` slot and the `bgcond` field → **`BeliefGraph:`** / **`bg_known`**
- bare `belief` / `delta` / `effect` fields and `belief_index` / `delta_index` /
  `effect_index` columns → **`bg_belief`/`bg_delta`/`bg_effect`** and their
  `bg_*_index` columns
- `BG*CoTBuilder` classes in `behavior_cot_builders.py` (file deleted) →
  `BeliefGraph*CoTBuilder` in `samples_builder.py`
- `<EOA>` as the action terminator → `<eos>`
- any FM loss term for `behavior_cot` → single AR CE only

`G05_REDESIGN.md` §1 in the solution repo still shows `<EOV> action codes <EOA>`
and all four CoT formats on one line. Both are wrong against the code: there is no
`<EOA>`, and exactly one format is emitted per sample.
