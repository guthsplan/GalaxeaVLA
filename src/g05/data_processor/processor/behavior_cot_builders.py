# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""BEHAVIOR-1K 2026 CoT SamplesBuilders (belief-graph conditioned).

These live in their own module so BEHAVIOR-specific semantics stay out of the
generic `samples_builder.py` hierarchy. They subclass the upstream builders
wherever the upstream one already does the work (bbox / trace formatting,
can_handle JSON validation), and add exactly two things:

1. **BG conditioning input** (`BGConditionedMixin`)
   An optional ``BGcond: <bgcond_text_!>`` slot placed *before* ``<EOC>``, in the
   same position `MemorySamplesBuilder` puts its ``Memory:`` slot. The ``_!``
   suffix marks the slot masked, so InputPreprocessor writes IGNORE_INDEX into
   its labels and it receives no language-model loss. It is conditioning that the
   external belief system supplies at inference time, not something the model
   predicts.

   When a sample carries no ``bgcond`` the slot is dropped from the template
   entirely, rather than emitting a dangling ``BGcond: ``. BG conditioning is
   therefore orthogonal to which CoT target is chosen: bbox / trace CoT keeps
   working on samples that have no belief graph at all.

2. **One CoT prediction target** after ``<EOC>``
   Exactly one of Subtask / Belief / Delta / Effect / BBox / Trace, selected by
   `MixedSamplesBuilder` at training time from the candidates that
   `can_handle()` accepts. Targets are kept clean: no robustness noise is ever
   applied to them (see "BG noise policy" below).

Resulting template (qwen35-base, 3 cameras, BG conditioning present)::

    <image0_image_!><image1_image_!><image2_image_!>
    Embodiment: <embodiment_text_!>; Task: <command_text_!_200>
    BGcond: <bgcond_text_!> State: <proprio_proprio_!>;<prompt_text_!>
    <EOC><belief_text>|Action: <EOV><action_action>|<|endoftext|>

    |<-------------- conditioning, masked -------------->|<-- CoT target -->|<-- action -->|

Note the actual G0.5 control-token order, which differs from the schematic in
`G05_REDESIGN.md`: ``<EOC>`` opens the generative region and ``<EOV>`` sits
immediately before the action tokens. There is no ``<EOA>``; the sequence
terminates with ``<eos>`` (``<|endoftext|>`` for the qwen35 processors).

BG noise policy
---------------
Section 7A of the design calls for the "existing conditioning noise policy" on
BGcond. G0.5 has no text-conditioning noise mechanism — `input_action_corruption`
corrupts input *actions* and `BuiltinProprioMLPDropoutProcessor` drops *proprio*;
neither touches text slots. Rather than invent a runtime one, BG robustness noise
belongs at materialization time, applied to ``bgcond_index`` only. The prediction
targets (``belief`` / ``delta`` / ``effect``) are materialized clean, so the
train-time path here cannot contaminate them.

Smoke test: python -m g05.data_processor.processor.behavior_cot_builders
"""

from typing import Any, Dict, Optional
import logging

from .samples_builder import (
    BaseSamplesBuilder,
    BBoxCoTBuilder,
    Trace2DCoTBuilder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefix handling
# ---------------------------------------------------------------------------
# Section 8: every generated target must carry its semantic prefix exactly once.
# Two producers can each believe they own it — the offline extractor writes
# "Subtask: pick up the mug" while SubtaskCoTBuilder writes f"Subtask: {...}" —
# which yields "Subtask: Subtask: pick up the mug".
#
# Canonical responsibility: **the builder owns the prefix, the stored annotation
# is the bare payload.** `with_prefix` additionally strips any prefixes the
# payload already carries, so a sidecar written under the old convention is
# normalized instead of doubled. Normalizing in both places is deliberate: the
# materialization tool strips on write, this strips on read, and the invariant
# holds even if only one of them ran.


def with_prefix(prefix: str, payload: Any) -> str:
    """``"<prefix>: <payload>"`` with the prefix present exactly once.

    Strips any number of leading ``"<prefix>:"`` occurrences (case-insensitive,
    tolerant of surrounding whitespace) before prepending exactly one.

    >>> with_prefix("Subtask", "pick up the mug")
    'Subtask: pick up the mug'
    >>> with_prefix("Subtask", "Subtask: Subtask: pick up the mug")
    'Subtask: pick up the mug'
    """
    text = "" if payload is None else str(payload)
    lowered_prefix = prefix.lower() + ":"
    while True:
        stripped = text.lstrip()
        if stripped.lower().startswith(lowered_prefix):
            text = stripped[len(lowered_prefix) :]
            continue
        text = stripped
        break
    return f"{prefix}: {text}".rstrip() if text else f"{prefix}:"


# ---------------------------------------------------------------------------
# BG conditioning mixin
# ---------------------------------------------------------------------------


class BGConditionedMixin:
    """Adds an optional masked ``BGcond:`` slot before ``<EOC>``.

    Mix in *before* the concrete builder so the template property resolves here::

        class BGSubtaskCoTBuilder(BGConditionedMixin, SubtaskCoTBuilder): ...

    The template varies per sample (present vs. absent BG conditioning), but
    `BaseSamplesBuilder.build` reads ``self.template`` with no access to ``data``.
    `build` therefore records whether the current sample has BG conditioning
    before delegating upward. A builder instance is only ever used sequentially
    within one process — `MixedSamplesBuilder` holds one instance per candidate
    and DataLoader workers each hold their own copy — so this carries no
    cross-sample risk, and it is reset in a `finally` regardless.
    """

    #: Slot text emitted between the Task and State slots when BG cond is present.
    _BGCOND_SLOT = "BGcond: <bgcond_text_!> "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bgcond_active: bool = False

    @staticmethod
    def has_bgcond(data: Dict[str, Any]) -> bool:
        value = data.get("bgcond")
        if value is None:
            return False
        text = str(value).strip()
        return bool(text) and text.lower() not in BaseSamplesBuilder._INVALID_STRINGS

    def _cot_slot(self) -> str:
        """Template fragment for the post-EOC prediction target. Override this."""
        raise NotImplementedError

    def _prompt_text(self) -> str:
        """Value of the masked ``<prompt_text_!>`` instruction slot. Override this."""
        raise NotImplementedError

    @property
    def template(self) -> str:
        bgcond = self._BGCOND_SLOT if self._bgcond_active else ""
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            f"{bgcond}State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n"
            f"<EOC>{self._cot_slot()}|"
            "Action: <EOV><action_action>|<eos>"
        )

    def build(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        self._bgcond_active = self.has_bgcond(data)
        try:
            return super().build(data, sample)
        finally:
            self._bgcond_active = False

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        # Deliberately does NOT call super(): the BG builders below own their slot
        # values outright, so an upstream _populate_extra_samples cannot reintroduce
        # a second prefix. BGcond is conditioning only — never a loss target.
        if self._bgcond_active:
            samples["bgcond"] = str(data["bgcond"]).strip()
        samples["prompt"] = self._prompt_text()


class _BGTextTargetBuilder(BGConditionedMixin, BaseSamplesBuilder):
    """BG-conditioned builder whose CoT target is a single prefixed text field.

    Subclasses set `annotation_key` (the decoded dataset field), `prefix` (the
    semantic prefix the builder owns) and `prompt` (the masked instruction slot).
    """

    #: Dataset field holding the bare payload, decoded from ``<field>_index``.
    annotation_key: str = ""
    #: Semantic prefix, applied exactly once by `with_prefix`.
    prefix: str = ""
    #: Masked instruction text placed in the ``<prompt_text_!>`` slot.
    prompt: str = ""
    #: Payloads that are meaningful supervision despite looking empty.
    #: `BaseSamplesBuilder._INVALID_STRINGS` rejects "none", but "Delta: none"
    #: and "Effect: none" are exactly the targets that teach the model when
    #: *nothing* changed. Without this, every no-op frame would silently drop out
    #: of the Delta/Effect candidate pools and skew the sampling distribution
    #: toward frames where something happened.
    valid_empty_payloads: frozenset = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.annotation_key:
            cls.required_fields = (cls.annotation_key,)
            cls.eval_required_fields = ()

    def can_handle(self, data: Dict[str, Any]) -> bool:
        value = data.get(self.annotation_key)
        if isinstance(value, str) and value.strip().lower() in self.valid_empty_payloads:
            return True
        return super().can_handle(data)

    def _cot_slot(self) -> str:
        return f"<{self.annotation_key}_text>"

    def _prompt_text(self) -> str:
        return self.prompt

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        super()._populate_extra_samples(data, samples)
        samples[self.annotation_key] = with_prefix(self.prefix, data.get(self.annotation_key, ""))


# ---------------------------------------------------------------------------
# Concrete targets
# ---------------------------------------------------------------------------


class BGSubtaskCoTBuilder(_BGTextTargetBuilder):
    """``Subtask: <atomic_task>``.

    Reuses the established ``atomic_task_index -> atomic_task`` field rather than
    introducing a parallel ``subtask`` one, so BEHAVIOR subtask annotations land in
    the same slot every upstream subtask builder already reads.
    """

    annotation_key = "atomic_task"
    prefix = "Subtask"
    prompt = "predict subtask"


class BGBeliefCoTBuilder(_BGTextTargetBuilder):
    """``Belief: (cooked hotdog_207) 0.93 obs | ...`` — current belief-graph state."""

    annotation_key = "belief"
    prefix = "Belief"
    prompt = "predict belief state"


class BGDeltaCoTBuilder(_BGTextTargetBuilder):
    """``Delta: (cooked ?x) 1/2 [hotdog.n.02]`` — progress toward the goal.

    ``Delta: none`` is a valid target.
    """

    annotation_key = "delta"
    prefix = "Delta"
    prompt = "predict belief delta"
    valid_empty_payloads = frozenset({"none"})


class BGEffectCoTBuilder(_BGTextTargetBuilder):
    """``Effect: (inhand hotdog_207) 0>1 | ...`` — predicted effect of the action chunk.

    ``Effect: none`` is a valid target.
    """

    annotation_key = "effect"
    prefix = "Effect"
    prompt = "predict action effect"
    valid_empty_payloads = frozenset({"none"})


class BGBBoxCoTBuilder(BGConditionedMixin, BBoxCoTBuilder):
    """``BBox: <name> <loc><loc><loc><loc>; ...``.

    Inherits `BBoxCoTBuilder._format_bbox_json` (which already emits the ``BBox: ``
    prefix and the paligemma-style ``<locNNNN>`` encoding) plus its non-empty-JSON
    `can_handle` / `can_handle_for_eval` checks, unchanged.
    """

    required_fields = ("bbox",)
    eval_required_fields = ()

    def _cot_slot(self) -> str:
        return "<bbox_text>"

    def _prompt_text(self) -> str:
        return "predict bbox"

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        super()._populate_extra_samples(data, samples)  # BGcond + prompt
        # _format_bbox_json owns the "BBox: " prefix; with_prefix collapses it to
        # one occurrence so the invariant is checked here too, not just assumed.
        formatted = self._format_bbox_json(data.get("bbox", "{}"))
        samples["bbox"] = with_prefix("BBox", formatted[len("BBox: ") :] if formatted else "")


class BGTrace2DCoTBuilder(BGConditionedMixin, Trace2DCoTBuilder):
    """``Trace: Left <loc0543><loc0436>; Right None`` — projected gripper contact points.

    Inherits `Trace2DCoTBuilder._format_trace_2d_json` and its "at least one arm
    visible" `can_handle` check, unchanged.
    """

    required_fields = ("trace_2d",)
    eval_required_fields = ()

    def _cot_slot(self) -> str:
        return "<trace_2d_text>"

    def _prompt_text(self) -> str:
        return "predict 2d trace of gripper"

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        super()._populate_extra_samples(data, samples)  # BGcond + prompt
        formatted = self._format_trace_2d_json(data.get("trace_2d", "{}"))
        samples["trace_2d"] = with_prefix(
            "Trace", formatted[len("Trace: ") :] if formatted else ""
        )


#: Every BEHAVIOR CoT candidate, in the order the design document lists them.
BEHAVIOR_COT_BUILDERS = (
    BGSubtaskCoTBuilder,
    BGDeltaCoTBuilder,
    BGBeliefCoTBuilder,
    BGEffectCoTBuilder,
    BGBBoxCoTBuilder,
    BGTrace2DCoTBuilder,
)


# ====================================================================== #
#  Smoke test: python -m g05.data_processor.processor.behavior_cot_builders
# ====================================================================== #

if __name__ == "__main__":
    import types as _types

    from g05.utils.common.special_tokens import SpecialTokenManager

    DEFAULTS = dict(
        num_input_images=3,
        image_sizes={"head_rgb": (224, 224), "left_wrist_rgb": (224, 224), "right_wrist_rgb": (224, 224)},
        embodiment_type="behavior_r1pro",
    )

    _mock_inst = _types.SimpleNamespace(
        bos_token="<|im_start|>", eos_token="<|endoftext|>", pad_token=None
    )
    TOKEN_MAPS = [
        ("qwen35-base", SpecialTokenManager.for_model("qwen35")),
        ("qwen35-instruct", SpecialTokenManager.for_model("qwen35", tokenizer=_mock_inst)),
    ]

    DATA = {
        "bgcond": "(inside hotdog_207 fridge_1) 0.88 obs | (open fridge_1) 0.41 prior",
        "atomic_task": "pick up the other hotdog from fridge",
        "belief": "(cooked hotdog_207) 0.93 obs | (inhand hotdog_207) 0.02 prior",
        "delta": "(cooked ?x) 1/2 [hotdog.n.02]",
        "effect": "(inhand hotdog_207) 0>1 | (inside hotdog_207 fridge_1) 1>0",
        "bbox": '{"hotdog_207": [0.113, 0.479, 0.28, 0.688], "fridge_1": [0.0, 0.1, 0.6, 0.95]}',
        "trace_2d": '{"uv_left": [0.53, 0.42], "visb_left": true, "uv_right": null, "visb_right": false}',
    }

    print("=" * 78)
    print("BEHAVIOR CoT builders — templates and slot values")
    print("=" * 78)

    for cls in BEHAVIOR_COT_BUILDERS:
        builder = cls(**DEFAULTS)
        assert builder.can_handle(DATA), f"{cls.__name__} should handle the full sample"
        builder._bgcond_active = builder.has_bgcond(DATA)
        slots: Dict[str, Any] = {}
        builder._populate_extra_samples(DATA, slots)
        print(f"\n--- {cls.__name__} ---")
        print(f"  template : {builder.template!r}")
        resolved = TOKEN_MAPS[0][1].resolve_template(builder.template)
        for line in resolved.split("\n"):
            print(f"      {line}")
        for key, value in slots.items():
            print(f"  slot {key:<12} = {value!r}")
        builder._bgcond_active = False

    # ---- prefix appears exactly once ----
    print("\n" + "=" * 78)
    print("Prefix invariants")
    print("=" * 78)

    assert with_prefix("Subtask", "pick up") == "Subtask: pick up"
    assert with_prefix("Subtask", "Subtask: pick up") == "Subtask: pick up"
    assert with_prefix("Subtask", "Subtask: Subtask:  pick up") == "Subtask: pick up"
    assert with_prefix("Subtask", "subtask: pick up") == "Subtask: pick up"
    assert with_prefix("Delta", "none") == "Delta: none"
    print("  ✓ with_prefix collapses repeated prefixes")

    PREFIXES = {
        "BGSubtaskCoTBuilder": ("atomic_task", "Subtask"),
        "BGBeliefCoTBuilder": ("belief", "Belief"),
        "BGDeltaCoTBuilder": ("delta", "Delta"),
        "BGEffectCoTBuilder": ("effect", "Effect"),
        "BGBBoxCoTBuilder": ("bbox", "BBox"),
        "BGTrace2DCoTBuilder": ("trace_2d", "Trace"),
    }
    # Sidecars written under the old convention already carry the prefix; feeding
    # those in must still yield exactly one.
    DIRTY = dict(DATA)
    DIRTY["atomic_task"] = "Subtask: pick up the other hotdog from fridge"
    DIRTY["belief"] = "Belief: (cooked hotdog_207) 0.93 obs"
    DIRTY["delta"] = "Delta: none"
    DIRTY["effect"] = "Effect: none"

    for cls in BEHAVIOR_COT_BUILDERS:
        key, prefix = PREFIXES[cls.__name__]
        for label, payload in (("clean", DATA), ("pre-prefixed", DIRTY)):
            b = cls(**DEFAULTS)
            out: Dict[str, Any] = {}
            b._populate_extra_samples(payload, out)
            target = out[key]
            count = target.count(f"{prefix}: ")
            assert count == 1, f"{cls.__name__} [{label}]: prefix x{count} in {target!r}"
            assert target.startswith(f"{prefix}: "), f"{cls.__name__}: {target!r}"
    print(f"  ✓ all {len(BEHAVIOR_COT_BUILDERS)} builders emit their prefix exactly once")

    # ---- Delta/Effect "none" stays in the candidate pool ----
    for cls, key in ((BGDeltaCoTBuilder, "delta"), (BGEffectCoTBuilder, "effect")):
        b = cls(**DEFAULTS)
        assert b.can_handle({key: "none"}), f"{cls.__name__} must accept 'none'"
        assert not b.can_handle({key: ""}), f"{cls.__name__} must reject empty"
        assert not b.can_handle({key: "null"}), f"{cls.__name__} must reject 'null'"
        assert not b.can_handle({}), f"{cls.__name__} must reject a missing field"
    print("  ✓ Delta/Effect accept 'none' but still reject empty/null/missing")

    # ---- missing annotation -> candidate inapplicable, never a crash ----
    for cls in BEHAVIOR_COT_BUILDERS:
        b = cls(**DEFAULTS)
        assert not b.can_handle({}), f"{cls.__name__} must be inapplicable with no annotations"
    print("  ✓ every builder is inapplicable (not fatal) without its annotation")

    # ---- BGcond presence toggles the slot, and is always masked ----
    b = BGBeliefCoTBuilder(**DEFAULTS)
    b._bgcond_active = True
    with_bg = b.template
    b._bgcond_active = False
    without_bg = b.template
    assert "BGcond: <bgcond_text_!>" in with_bg
    assert "BGcond" not in without_bg, "empty BG cond must drop the slot, not emit 'BGcond: '"
    assert with_bg.index("BGcond") < with_bg.index("<EOC>"), "BGcond must precede EOC"
    assert "<bgcond_text_!>" in with_bg, "BGcond slot must be masked (trailing _!)"
    assert with_bg.index("<EOC>") < with_bg.index("<belief_text>"), "target must follow EOC"
    assert with_bg.index("<belief_text>") < with_bg.index("<EOV>"), "target must precede EOV"
    assert with_bg.index("<EOV>") < with_bg.index("<action_action>"), "actions must follow EOV"
    print("  ✓ BGcond before EOC and masked; target after EOC; actions after EOV")

    print("\n" + "=" * 78)
    print("All cases OK.")
    print("=" * 78)
