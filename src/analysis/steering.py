"""Latent steering: edit a latent along a discovered physical direction and decode the result (Step 3).

Builds on :mod:`analysis.intervention` (which discovers a direction and tests decoded-state
controllability) to support the *real-video* steering loop: learn a direction for a physical quantity
on a labelled (synthetic) cache, then add ``alpha * direction`` to the latents of a real Physics-IQ
clip and **decode the steered latent to pixels** with the trained transformer decoder.

Because real videos have no ground-truth state, controllability is verified two ways:

* **independent readout** — a regressor fit on the *labelled* cache predicts the quantity from pooled
  latents; applied to the steered real latents it should move monotonically with ``alpha`` (this is
  also the Step-3 *transfer* test: does a synthetic direction carry to real data?);
* **decoded frames** — eyeballed (and, where the decoder has a state head, read out directly).

The discipline from ``intervention`` carries over: a real direction produces a smooth, monotonic change;
an ineffective one does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..encoders.feature_extractor import LatentDataset
from .intervention import apply_intervention, discover_direction


def clip_features_and_scalar(
    latent_dir: str | Path, layer: int, group: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clip-level pooled features ``(N, D)`` and a scalar magnitude of ``group`` per clip ``(N,)``.

    The scalar is the per-frame L2 magnitude over the state columns whose name contains ``group``
    (masked to valid columns), averaged over frames — a sign-free target suitable for a 1-D direction
    (e.g. speed for ``vel``, gravity for ``gravity``). Only clips with valid state for the group count.
    """
    ds = LatentDataset(latent_dir, layers=[layer])
    feats, scal, ids = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        keys = s["state_keys"]
        mask = s["state_mask"].numpy()
        cols = [j for j, k in enumerate(keys) if group in k and (j >= len(mask) or mask[j] > 0)]
        if not cols:
            continue
        state = s["state"].numpy()[:, cols]               # (T, |cols|)
        mag = float(np.sqrt((state ** 2).sum(1)).mean())  # mean per-frame magnitude
        feats.append(s["layers"][layer].numpy().mean(0))  # (D,)
        scal.append(mag)
        ids.append(s["id"])
    if not feats:
        raise ValueError(f"No clips with valid state for group '{group}' in {latent_dir}.")
    return np.stack(feats, 0), np.asarray(scal, dtype=np.float32), ids


def fit_readout(features: np.ndarray, scalar: np.ndarray, alpha: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a standardised Ridge readout ``features -> scalar``; returns a predict function ``(M,D)->(M,)``.

    Used as an *independent* verification of steering on data without ground-truth labels.
    """
    from sklearn.linear_model import Ridge

    mu, sd = features.mean(0, keepdims=True), features.std(0, keepdims=True) + 1e-8
    model = Ridge(alpha=alpha).fit((features - mu) / sd, scalar)

    def predict(x: np.ndarray) -> np.ndarray:
        return model.predict((x - mu) / sd)

    return predict


@torch.no_grad()
def decode_intervention(
    decoder: Any,
    latents: dict[int, torch.Tensor],
    grid: tuple[int, int, int],
    direction: torch.Tensor,
    layer: int,
    alphas: list[float],
) -> dict[float, Any]:
    """Decode the latents at each intervention strength. Returns ``{alpha: DecoderOutput}``.

    The decoder is run on ``latents + alpha * direction`` (applied to ``layer``); the caller pulls
    ``.frames`` and/or ``.state`` from each output.
    """
    out: dict[float, Any] = {}
    for a in alphas:
        perturbed = apply_intervention(latents, direction, a, layer)
        out[a] = decoder(perturbed, grid)
    return out


@torch.no_grad()
def readout_along_direction(
    latents: dict[int, torch.Tensor],
    direction: torch.Tensor,
    layer: int,
    alphas: list[float],
    readout: Callable[[np.ndarray], np.ndarray],
) -> list[dict[str, float]]:
    """Independent-readout controllability curve: predicted quantity vs ``alpha`` (latent-space only).

    Pools the steered latents and applies the fitted ``readout`` — no decoder needed, so it works on
    real data with no ground-truth state.
    """
    rows = []
    base = latents[layer]  # (B, L, D)
    d = direction.view(1, 1, -1).to(base.device)
    for a in alphas:
        pooled = (base + a * d).mean(dim=(0, 1)).cpu().numpy()[None, :]  # (1, D)
        rows.append({"alpha": float(a), "readout": float(readout(pooled)[0])})
    return rows


def steer_to_target(
    latents: dict[int, torch.Tensor],
    direction: torch.Tensor,
    layer: int,
    target: float,
    readout: Callable[[np.ndarray], np.ndarray],
    search: tuple[float, float] = (-6.0, 6.0),
    steps: int = 49,
) -> tuple[float, dict[int, torch.Tensor]]:
    """Find ``alpha`` along ``direction`` whose independent readout best matches ``target``.

    Grid-searches ``alpha`` over ``search`` and returns ``(best_alpha, steered_latents)``. Concrete,
    decoder-free counterpart to the gradient-based ideal — sufficient for 1-D physical directions.
    """
    alphas = list(np.linspace(search[0], search[1], steps))
    rows = readout_along_direction(latents, direction, layer, alphas, readout)
    best = min(rows, key=lambda r: abs(r["readout"] - target))
    return best["alpha"], apply_intervention(latents, direction, best["alpha"], layer)


def interpolate_trajectories(
    latents_fail: dict[int, torch.Tensor],
    latents_success: dict[int, torch.Tensor],
    steps: int = 8,
) -> list[dict[int, torch.Tensor]]:
    """Linearly interpolate between a failed and a successful latent trajectory (robotics use, Step 4).

    Returns ``steps`` latent dicts from ``latents_fail`` (t=0) to ``latents_success`` (t=1). Decoding
    these shows whether the latent path from failure to success passes through plausible intermediate
    states — the bridge from "detect" to "steer" for the robotics question.
    """
    ts = np.linspace(0.0, 1.0, steps)
    keys = set(latents_fail) & set(latents_success)
    return [{k: (1 - t) * latents_fail[k] + t * latents_success[k] for k in keys} for t in ts]


def discover_quantity_direction(
    latent_dir: str | Path, layer: int, group: str, method: str = "regression"
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Convenience: learn a unit direction *and* an independent readout for ``group`` from a cache.

    Returns ``(direction (D,), readout_fn)`` — the two objects Step 3 steering needs.
    """
    feats, scal, _ = clip_features_and_scalar(latent_dir, layer, group)
    direction = discover_direction(feats, scal, method=method)
    readout = fit_readout(feats, scal)
    return direction, readout
