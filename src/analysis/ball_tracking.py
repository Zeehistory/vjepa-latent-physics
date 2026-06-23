"""Measure the ball's motion directly from rendered/decoded pixels (Step 2 velocity verification).

The clean moving-ball dataset is a dark disk on a white background, so we can recover the ball's
per-frame center by an intensity-weighted centroid of the *dark* pixels — no learned tracker needed.
From the centroid track we read off the empirical **velocity** (mean inter-frame displacement) and
**speed**. This is the objective, pixel-level evidence for velocity steering: after we add ``alpha*d_v``
to the latents and decode, does the *decoded ball actually move faster/slower/in a new direction*?

These functions take ``(T, C, H, W)`` float frames in ``[0, 1]`` (the decoder's output) and return
normalized-coordinate quantities (units of image width per frame), matching the dataset's state units.
"""

from __future__ import annotations

import numpy as np
import torch


def ball_centroids(frames: torch.Tensor, darkness_thresh: float = 0.5) -> np.ndarray:
    """Per-frame ball center ``(T, 2)`` in normalized [0,1] coords (x, y), NaN if the ball is absent.

    Works on a dark-ball / light-background scene: weight each pixel by how *dark* it is relative to the
    background and take the weighted centroid. Frames with negligible dark mass (e.g. the ball fully
    occluded) yield NaN so callers can skip them.
    """
    x = frames.detach().cpu().float()
    if x.dim() != 4:
        raise ValueError(f"expected (T,C,H,W), got {tuple(x.shape)}")
    t, _c, h, w = x.shape
    gray = x.mean(dim=1)  # (T, H, W), ~1 background, ~0 ball
    dark = (1.0 - gray).clamp(min=0.0)
    dark = torch.where(dark > (1.0 - darkness_thresh), dark, torch.zeros_like(dark))
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    mass = dark.sum(dim=(1, 2))  # (T,)
    cx = (dark * xs).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    cy = (dark * ys).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    out = torch.stack([cx, cy], dim=1).numpy()
    out[mass.numpy() < 1e-3] = np.nan
    return out


def measured_velocity(frames: torch.Tensor, darkness_thresh: float = 0.5) -> dict[str, float]:
    """Empirical velocity from the decoded frames.

    Returns ``{vel_x, vel_y, speed, n_valid}`` where velocity is the mean inter-frame displacement of
    the centroid (normalized units per frame) over frames where the ball is visible. ``speed`` is its
    magnitude. ``n_valid`` is how many consecutive-visible frame pairs contributed.
    """
    c = ball_centroids(frames, darkness_thresh)
    disp = np.diff(c, axis=0)  # (T-1, 2)
    valid = ~np.isnan(disp).any(axis=1)
    if valid.sum() == 0:
        return {"vel_x": float("nan"), "vel_y": float("nan"), "speed": float("nan"), "n_valid": 0}
    v = disp[valid].mean(axis=0)
    return {
        "vel_x": float(v[0]), "vel_y": float(v[1]),
        "speed": float(np.linalg.norm(v)), "n_valid": int(valid.sum()),
    }
