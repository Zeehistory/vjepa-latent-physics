"""DROID robotics dataset loader (Step 4).

DROID is a large-scale robot-manipulation dataset. Here robotics is the final stage: testing whether
the latent difference between *achievable* (successful) and *non-achievable* (failed) actions can be
detected and steered. The loader mirrors the standard sample contract so the rest of the pipeline
(extraction, decoding, probing, steering) targets it unchanged, with two conventions:

* the binary **success** label is exposed through ``category`` (``"success"`` / ``"failure"``) so it
  flows through latent extraction / pooling with zero pipeline changes (the same machinery used for
  Physics-IQ categories);
* per-frame **actions** populate ``state`` (with a valid ``state_mask``), and ``success`` /
  ``language_goal`` / raw actions are kept in ``meta``.

The loader is graceful: it expects a ``manifest.json`` listing episodes and raises a clear, actionable
error if ``cfg.root`` is unset or empty. For development without DROID access, use the registered
``robot_toy`` dataset (:mod:`data.robot_toy`), which produces the same contract synthetically.

manifest.json schema (one object per episode):
    {"path": "videos/ep0001.mp4", "success": 1, "language_goal": "put cup on shelf",
     "actions": [[...], ...], "split": "train"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..utils.video_io import load_video
from .dataset_registry import register_dataset
from .video_transforms import VideoTransform


class DroidDataset(Dataset):
    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        if not cfg.root:
            raise ValueError(
                "DROID requires `data.root` to point at a prepared dataset directory containing "
                "`manifest.json` (see src/data/droid.py for the schema). For development without DROID, "
                "use the `robot_toy` dataset (configs/data/robot_toy.yaml)."
            )
        self.root = Path(cfg.root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"DROID manifest not found: {manifest}")
        self.cfg = cfg
        self.num_frames = cfg.num_frames
        self.image_size = cfg.image_size
        self.transform = VideoTransform(
            image_size=encoder_image_size, num_frames=encoder_frames, do_normalize=True)
        records = json.loads(manifest.read_text())
        self.samples = [r for r in records if r.get("split", "train") == cfg.split]
        if not self.samples:
            raise RuntimeError(f"No DROID episodes for split='{cfg.split}' under {self.root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.samples[idx]
        path = rec["path"]
        if not Path(path).is_absolute():
            path = str(self.root / path)
        frames = load_video(path, num_frames=self.num_frames, image_size=self.image_size)
        success = float(rec.get("success", 0))

        actions = rec.get("actions")
        if actions is not None:
            act = torch.tensor(actions, dtype=torch.float32)
            if act.shape[0] != self.num_frames:  # resample action stream to the sampled frames
                idxs = torch.linspace(0, act.shape[0] - 1, self.num_frames).round().long()
                act = act[idxs]
            state, state_mask = act, torch.ones(act.shape[-1])
        else:
            state, state_mask = torch.zeros(self.num_frames, 1), torch.zeros(1)

        return {
            "id": rec.get("id", Path(path).stem),
            "frames": frames,
            "encoder_input": self.transform(frames),
            "state": state,
            "state_mask": state_mask,
            "state_keys": [f"action_{i}" for i in range(state.shape[-1])],
            "category": "success" if success >= 0.5 else "failure",
            "meta": {"success": success, "language_goal": rec.get("language_goal", ""),
                     "path": path, "split": rec.get("split", "train")},
        }


@register_dataset("droid")
def _build_droid(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return DroidDataset(cfg, encoder_image_size, encoder_frames)
