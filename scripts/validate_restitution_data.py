#!/usr/bin/env python
"""Validate scene_restitution data + ball tracker before cluster jobs (no GPU).

    python scripts/validate_restitution_data.py
    python scripts/validate_restitution_data.py --output_dir outputs/demos/restitution --n_scenes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.analysis.ball_tracking import measured_bounce
from src.data.moving_ball import MovingBall
from src.utils.video_io import save_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/demos/restitution")
    p.add_argument("--n_scenes", type=int, default=1, help="scenes to tracker-check")
    p.add_argument("--write_demos", action="store_true", default=True)
    p.add_argument("--no_demos", action="store_true", help="skip mp4 export")
    args = p.parse_args()
    if args.no_demos:
        args.write_demos = False

    out = Path(args.output_dir)
    if args.write_demos:
        out.mkdir(parents=True, exist_ok=True)

    gen = MovingBall(
        image_size=256, num_frames=16, fps=4, scenario="scene_restitution",
        clips_per_scene=8, speed_range=(0.018, 0.032), restitution_range=(0.35, 0.95),
        radius_range=(0.11, 0.11), seed=0,
    )
    K = gen.clips_per_scene
    ok = True

    # --- scene 0: write rank demos + shared frame-0 check ---
    clips0 = [gen.generate(r) for r in range(K)]
    f0 = clips0[0].frames[0]
    max_dev = max(float((f0 - c.frames[0]).abs().max()) for c in clips0[1:])
    print(f"[validate] scene0 frame0 max_dev across ranks: {max_dev:.2e}")
    ok = ok and max_dev < 1e-6

    for r, c in enumerate(clips0):
        if args.write_demos:
            mp4 = out / f"scene000_rank{r}_e{c.meta['restitution']:.2f}.mp4"
            save_video(c.frames, mp4, fps=gen.fps)
        bf = int(c.meta["bounce_frame"])
        print(f"  rank {r}: e={c.meta['restitution']:.3f}  bounce_f={bf}  "
              f"gt_speed_ratio={c.meta['speed_ratio']:.3f}")
        ok = ok and 3 <= bf <= 12

    if args.write_demos:
        print(f"[validate] demos -> {out.resolve()}")

    # --- tracker oracle on multiple scenes ---
    errs: list[float] = []
    cor_e: list[float] = []
    cor_r: list[float] = []
    for scene in range(args.n_scenes):
        es, rs = [], []
        for r in range(K):
            c = gen.generate(scene * K + r)
            m = measured_bounce(c.frames, incoming_hint={"bounce_frame": c.meta["bounce_frame"]})
            gt, tr = c.meta["speed_ratio"], m["speed_ratio"]
            e = c.meta["restitution"]
            if np.isfinite(tr) and np.isfinite(gt):
                errs.append(abs(tr - gt))
            if np.isfinite(tr):
                es.append(e)
                rs.append(tr)
            print(f"  scene{scene} r{r}: e={e:.2f}  gt_ratio={gt:.3f}  tracked={tr:.3f}")
        if len(es) >= 2:
            cor_e.extend(es)
            cor_r.extend(rs)

    mean_err = float(np.mean(errs)) if errs else float("nan")
    rho = float(np.corrcoef(cor_e, cor_r)[0, 1]) if len(cor_e) >= 2 else float("nan")
    print(f"[validate] tracker mean |tracked-gt_ratio| = {mean_err:.4f}  "
          f"corr(e, tracked_ratio) = {rho:.4f}")
    ok = ok and np.isfinite(mean_err) and mean_err < 0.15
    ok = ok and np.isfinite(rho) and rho > 0.85

    if ok:
        print("[validate] ALL CHECKS PASSED")
    else:
        print("[validate] *** CHECK FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
