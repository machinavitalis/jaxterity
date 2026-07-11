#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Vision cross-check: catch model/calibration mismatch from a camera (#6).

Geometry is the calibration oracle; vision is the *cross-check* — the cheap way
to notice the model and reality have drifted apart before a motor moves (the
idea behind FANUC's iRVision mastering). The mechanism, demonstrated here in
simulation:

1. render the silhouette the model *expects* at a pose (from a fixed camera),
2. compare it to what the camera *actually* sees,
3. if the silhouette IoU drops below a threshold, flag a mismatch.

Here "actual" is the same model with a small injected joint error (a stand-in
for a calibration offset that is wrong, or a model that doesn't match the real
unit). On real hardware, "actual" is a frame from a wrist/overhead camera and
the only extra requirement is a known camera pose (hand-eye). Output: a montage
(expected vs actual vs overlay) and the IoU verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from calib_planner_generic import GenericModel


def silhouette(renderer, gm: GenericModel, q: dict, cam) -> np.ndarray:
    """Boolean mask of the arm's pixels at pose ``q`` (segmentation render)."""
    gm.set_q(q)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(gm.data, camera=cam)
    seg = renderer.render()  # (H, W, 2): [...,0]=object id, [...,1]=object type
    geom_id = seg[..., 0]
    arm = np.zeros(geom_id.shape, dtype=bool)
    for g in gm._arm_geoms:
        arm |= geom_id == g
    return arm


def rgb(renderer, gm: GenericModel, q: dict, cam) -> np.ndarray:
    gm.set_q(q)
    renderer.disable_segmentation_rendering()
    renderer.update_scene(gm.data, camera=cam)
    return renderer.render()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--gripper-joints", default="Jaw")
    p.add_argument("--joint", default=None, help="joint to perturb (default: first)")
    p.add_argument("--error-rad", type=float, default=0.12, help="injected error")
    p.add_argument("--threshold", type=float, default=0.92, help="IoU mismatch floor")
    p.add_argument("--out", default="examples/media/vision_check.png")
    args = p.parse_args()

    grip = tuple(s for s in args.gripper_joints.split(",") if s)
    gm = GenericModel(args.scene, gripper_joints=grip)
    gm.model.vis.global_.offwidth = 1280
    gm.model.vis.global_.offheight = 960
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.1, 0.16]
    cam.distance, cam.azimuth, cam.elevation = 0.62, 150, -16
    renderer = mujoco.Renderer(gm.model, 600, 760)

    # A nominal inspection pose, and an "actual" with one joint mis-calibrated.
    expected = {j: 0.3 for j in gm.arm_joints}
    joint = args.joint or gm.arm_joints[0]
    good = dict(expected)
    bad = {**expected, joint: expected[joint] + args.error_rad}

    sil_exp = silhouette(renderer, gm, expected, cam)
    iou_good = iou(sil_exp, silhouette(renderer, gm, good, cam))
    iou_bad = iou(sil_exp, silhouette(renderer, gm, bad, cam))

    print(
        f"expected vs matching actual : IoU = {iou_good:.3f}  -> "
        f"{'OK' if iou_good >= args.threshold else 'MISMATCH'}"
    )
    print(
        f"expected vs mis-cal actual  : IoU = {iou_bad:.3f}  -> "
        f"{'OK' if iou_bad >= args.threshold else 'MISMATCH'}  "
        f"(injected {args.error_rad:+.2f} rad on {joint})"
    )

    # Montage: expected | actual(mis-cal) | overlay (expected=green, actual=red).
    img_exp = rgb(renderer, gm, expected, cam)
    img_bad = rgb(renderer, gm, bad, cam)
    overlay = np.full_like(img_exp, 235)
    overlay[sil_exp] = [40, 180, 70]
    sil_bad = silhouette(renderer, gm, bad, cam)
    red = np.zeros_like(img_exp)
    red[sil_bad] = [220, 60, 60]
    overlay = np.where(
        sil_bad[..., None], (0.5 * overlay + 0.5 * red).astype(np.uint8), overlay
    )
    montage = np.concatenate([img_exp, img_bad, overlay], axis=1)

    import cv2

    bgr = montage[..., ::-1].copy()
    labels = [
        ("EXPECTED (model)", 0),
        ("ACTUAL (mis-calibrated)", 760),
        (f"OVERLAY  IoU={iou_bad:.2f}", 1520),
    ]
    for text, x in labels:
        cv2.putText(
            bgr,
            text,
            (x + 14, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            text,
            (x + 14, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 230, 90),
            1,
            cv2.LINE_AA,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(out, bgr[..., ::-1])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
