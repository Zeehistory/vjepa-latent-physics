"""Synthetic robot reach task with success/failure labels (Step 4 development fallback).

A 2D end-effector (bright dot) starts at a base and moves toward a target (ring). Episodes are either
*successful* (the effector reaches the target) or *failed* (it stalls short or veers off), giving a
clean achievable-vs-non-achievable contrast for developing the robotics detect-and-steer pipeline
without DROID access. The sample contract matches :mod:`data.droid`:

* ``category`` is ``"success"`` / ``"failure"`` (flows through extraction/pooling unchanged),
* ``state`` holds the per-frame end-effector ``[x, y, vx, vy, dist_to_target]`` (valid mask),
* ``meta`` carries ``success`` and the synthetic ``actions`` (per-frame velocity commands).

Deterministic given ``seed`` so caches are reproducible. Roughly half the episodes succeed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_registry import register_dataset
from .synthetic_physics import Body, _render
from .video_transforms import VideoTransform

_STATE_KEYS = ["ee_x", "ee_y", "ee_vx", "ee_vy", "dist_to_target"]


class RobotToyDataset(Dataset):
    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        self.num_frames = cfg.num_frames
        self.image_size = cfg.image_size
        self.num_clips = cfg.num_clips
        self.seed = cfg.seed
        self.transform = VideoTransform(
            image_size=encoder_image_size, num_frames=encoder_frames, do_normalize=True)
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return self.num_clips

    def _generate(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        success = bool(idx % 2 == 0)  # balanced success / failure
        base = np.array([rng.uniform(0.15, 0.3), rng.uniform(0.7, 0.85)])
        target = np.array([rng.uniform(0.6, 0.85), rng.uniform(0.15, 0.35)])
        ee = base.copy()
        # success: head to target; failure: undershoot or veer to a wrong offset.
        if success:
            goal = target
            gain = rng.uniform(0.18, 0.28)
        else:
            mode = rng.integers(0, 2)
            goal = base + (target - base) * rng.uniform(0.3, 0.55) if mode == 0 else \
                target + rng.uniform(-0.4, 0.4, size=2)
            gain = rng.uniform(0.10, 0.20)

        frames, states = [], []
        prev = ee.copy()
        for _t in range(self.num_frames):
            vel = (goal - ee) * gain + rng.normal(0, 0.004, size=2)
            ee = np.clip(ee + vel, 0.03, 0.97)
            actual_vel = ee - prev
            prev = ee.copy()
            dist = float(np.linalg.norm(ee - target))
            target_ring = Body(target, np.zeros(2), 0.06, 1.0, (0.3, 0.9, 0.4))
            effector = Body(ee, actual_vel, 0.035, 1.0, (0.95, 0.85, 0.2))
            base_marker = Body(base, np.zeros(2), 0.025, 1.0, (0.5, 0.5, 0.9))
            frames.append(_render([target_ring, base_marker, effector], self.image_size))
            states.append([ee[0], ee[1], actual_vel[0], actual_vel[1], dist])

        frame_arr = np.stack(frames, 0).transpose(0, 3, 1, 2)  # (T, C, H, W)
        actions = np.diff(np.stack([s[:2] for s in states]), axis=0, prepend=base[None]).tolist()
        return {
            "id": f"{'succ' if success else 'fail'}_{idx:05d}",
            "frames": torch.from_numpy(frame_arr).float(),
            "state": torch.tensor(states, dtype=torch.float32),
            "state_mask": torch.ones(len(_STATE_KEYS)),
            "state_keys": list(_STATE_KEYS),
            "category": "success" if success else "failure",
            "meta": {"success": float(success), "actions": actions, "goal": goal.tolist(),
                     "target": target.tolist()},
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._cache.get(idx)
        if sample is None:
            sample = self._generate(idx)
            self._cache[idx] = sample
        out = dict(sample)
        out["encoder_input"] = self.transform(sample["frames"])
        return out


@register_dataset("robot_toy")
def _build_robot_toy(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return RobotToyDataset(cfg, encoder_image_size, encoder_frames)
