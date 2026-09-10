# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Relative-joint transform applied to only the leading dims of a part.

Used for BEHAVIOR's `lower_body` = [torso joint positions (4), base velocity (3)]:
the torso targets are converted to deltas w.r.t. the current torso state, while the
base velocity command is left absolute. Mirrors RelativeJointTransform semantics.
"""
from typing import Dict

import torch

from g05.data_processor.transforms.relative_action import RelativeJointTransform


class RelativeJointTransformPartial(RelativeJointTransform):
    """
    keys_dims: {part_key: n_leading_dims_to_make_relative}
    Forward:  action[key][..., :n] -= state[key][..., -1:, :n]
    Backward: action[key][..., :n] += state[key][..., -1:, :n]
    """

    invertible = True

    def __init__(self, keys_dims: Dict[str, int], fast_forward: bool = True):
        super().__init__(keys=list(keys_dims.keys()), fast_forward=fast_forward)
        self.keys_dims = {k: int(v) for k, v in keys_dims.items()}

    def _apply(self, batch: Dict, sign: float) -> Dict:
        if "action" not in batch or "state" not in batch:
            return batch
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for k, n in self.keys_dims.items():
            if k not in batch["action"] or k not in batch["state"]:
                continue
            action = batch["action"][k]
            state_last = batch["state"][k][..., -1:, :n]
            head = action[..., :n] + sign * state_last
            out_batch["action"][k] = torch.cat([head, action[..., n:]], dim=-1)
        return out_batch

    def forward(self, batch: Dict):
        return self._apply(batch, -1.0)

    def _forward(self, batch: Dict):
        return self._apply(batch, -1.0)

    def _forward_fast(self, batch: Dict):
        return self._apply(batch, -1.0)

    def backward(self, batch: Dict):
        return self._apply(batch, +1.0)
