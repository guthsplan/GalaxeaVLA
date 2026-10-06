"""Action-chunk second moment for correlated flow-matching noise.

Implements the estimation step (Eq. 7) of "Correlated Noise for Flow Matching" from
*Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR
Challenge* (arXiv:2512.06951, Sec. 5):

    Sigma_hat = 1/N sum_n vec(a_n) vec(a_n)^T,    a_n in R^{H x D} a NORMALIZED action chunk

i.e. the second moment over time AND action dimensions jointly (H*D x H*D). The rest of the
method already lives in FMHelper (helpers/fm_helper.py):

    Sigma_reg = beta * Sigma_hat + (1 - beta) * I          (Eq. 8, fm.correlation_beta = 0.5)
    Sigma_reg = L L^T,   eps = L z,  z ~ N(0, I)           (Eq. 9-10)
    x_t = t * eps + (1 - t) * a                            (Eq. 11, training)

and the same correlated eps starts the ODE at inference (FMHelper.infer -> _sample_noise).

``FMHelper._load_cholesky`` reads a factor F with F F^T = Sigma_hat (it forms Sigma_hat = F F^T
itself before shrinking), so ``save_factor`` writes that F. Sigma_hat is usually singular
(padded action dims are constant zero), so F comes from an eigendecomposition rather than a
Cholesky factorization; only F F^T is ever used.

The chunks are taken from the training DataLoader itself (scripts/finetune.py
``estimate_action_cov=<path>``), so normalization, the grouped 27-D action layout and the
horizon are exactly what the FM head is trained on.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ActionSecondMoment:
    """Running E[vec(a) vec(a)^T] over [B, H, D] action chunks (float64)."""

    def __init__(self, horizon: int, action_dim: int):
        self.horizon = horizon
        self.action_dim = action_dim
        n = horizon * action_dim
        self.sum_outer = torch.zeros(n, n, dtype=torch.float64)
        self.count = 0
        self.skipped_padded = 0
        self.skipped_embodiment = 0

    def update(
        self,
        actions: torch.Tensor,
        step_is_pad: Optional[torch.Tensor] = None,
        keep: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch. Chunks with any padded time step (episode end) are skipped."""
        a = actions.detach().to("cpu", torch.float64)
        if a.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"actions {tuple(a.shape)} do not match H={self.horizon}, D={self.action_dim}")
        mask = torch.ones(a.shape[0], dtype=torch.bool)
        if keep is not None:
            self.skipped_embodiment += int((~keep).sum())
            mask &= keep.cpu()
        if step_is_pad is not None:
            padded = step_is_pad.detach().cpu().bool().any(dim=1)
            self.skipped_padded += int((padded & mask).sum())
            mask &= ~padded
        if not mask.any():
            return
        x = a[mask].reshape(int(mask.sum()), -1)
        self.sum_outer += x.T @ x
        self.count += x.shape[0]

    def sigma(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("no action chunks accumulated")
        return self.sum_outer / self.count


def save_factor(sigma: torch.Tensor, path: str | Path, meta: dict) -> dict:
    """Write F (F F^T = sigma) to ``path`` (.npy, float32), sigma alongside, and a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sigma = 0.5 * (sigma + sigma.T)
    evals, evecs = torch.linalg.eigh(sigma)
    factor = evecs * evals.clamp_min(0.0).sqrt().unsqueeze(0)
    np.save(path, factor.to(torch.float32).numpy())
    np.save(path.with_name(path.stem + "_sigma.npy"), sigma.to(torch.float32).numpy())

    diag = torch.diagonal(sigma)
    std = diag.clamp_min(1e-12).sqrt()
    corr = sigma / (std[:, None] * std[None, :])
    off = corr[~torch.eye(corr.shape[0], dtype=torch.bool)]
    summary = {
        **meta,
        "dim": int(sigma.shape[0]),
        "rank": int((evals > 1e-6 * evals.max()).sum()),
        "trace": float(diag.sum()),
        "diag_mean": float(diag.mean()),
        "zero_variance_dims": int((diag < 1e-8).sum()),
        "mean_abs_offdiag_corr": float(off.abs().mean()),
        "top10_eig_fraction": float(evals.flip(0)[:10].sum() / evals.clamp_min(0).sum()),
        "factor_path": str(path),
        "sigma_path": str(path.with_name(path.stem + "_sigma.npy")),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def estimate_from_dataloader(
    dataloader: Iterable,
    horizon: int,
    action_dim: int,
    num_samples: int,
    embodiments: Optional[list] = None,
    log_every: int = 500,
) -> ActionSecondMoment:
    """Accumulate the second moment of ``batch["action"]`` until ``num_samples`` chunks are in."""
    acc = ActionSecondMoment(horizon, action_dim)
    wanted = set(embodiments or [])
    t0 = time.monotonic()
    for i, batch in enumerate(dataloader):
        actions = batch["action"]
        keep = None
        if wanted:
            embs = [s.get("embodiment") for s in batch["samples"]]
            keep = torch.tensor([e in wanted for e in embs], dtype=torch.bool)
        acc.update(actions, batch.get("action_is_pad"), keep)
        if log_every and i % log_every == 0:
            rate = acc.count / max(time.monotonic() - t0, 1e-6)
            logger.info(f"[action cov] {acc.count}/{num_samples} chunks ({rate:.1f}/s)")
        if acc.count >= num_samples:
            break
    return acc
