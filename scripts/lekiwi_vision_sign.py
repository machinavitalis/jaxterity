#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Vision feedback: read a joint's motion *direction* from an external camera.

The bus can position a joint but can't tell which way the link physically moves —
that's the one thing offset/scale calibration can't recover, and the reason we
needed a human. This removes the human: an external webcam watches the arm, the
script nudges one joint, and the *image-space* motion of the arm is measured by
signed frame-differencing (the dark arm against the light wall makes the pixels
it moves *into* get darker and the ones it *leaves* get lighter). That observed
direction, matched against the model's predicted motion, gives the joint's sign.

This module is the perception half — capture + per-joint motion vector. The
model-correlation / sign solve consumes these vectors. Runs on the Pi with the
Feetech SDK + OpenCV; nudges are small, gentle, and returned.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from lekiwi_autosetup import (
    ADDR_GOAL_POSITION,
    ADDR_GOAL_SPEED,
    ADDR_TORQUE_ENABLE,
    ADDR_TORQUE_LIMIT,
    DEFAULT_BAUD,
    LAYOUTS,
    Bus,
)

NUDGE_TICKS = 280  # ~25° probe
NUDGE_SPEED = 300
NUDGE_MAX_LOAD = 0.30
DIFF_THRESHOLD = 18  # grey-level change that counts as motion


class Camera:
    """The external webcam at a fixed resolution, with warm-up reads."""

    def __init__(
        self,
        device: str = "/dev/video2",
        width=960,
        height=720,
        gain=170,
        brightness=150,
    ):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_GAIN, gain)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        for _ in range(8):  # let auto-exposure settle
            self.cap.read()

    def gray(self) -> np.ndarray:
        for _ in range(3):  # flush stale buffered frames
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)

    def close(self):
        self.cap.release()


def motion_vector(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Image-space motion (dx, dy, confidence) of the dark arm from a->b.

    ``b`` darker than ``a`` = the arm moved *into* those pixels (occupied);
    lighter = it *left* (vacated). The vector from the vacated centroid to the
    occupied centroid is the motion direction. ``dy>0`` is downward in the image.
    """
    diff = b - a
    occupied = diff < -DIFF_THRESHOLD
    vacated = diff > DIFF_THRESHOLD
    n = int(min(occupied.sum(), vacated.sum()))
    if n < 80:
        return 0.0, 0.0, n
    oy, ox = np.nonzero(occupied)
    vy, vx = np.nonzero(vacated)
    dx = float(ox.mean() - vx.mean())
    dy = float(oy.mean() - vy.mean())
    return dx, dy, n


def nudge(
    bus: Bus,
    sid: int,
    delta: int,
    *,
    speed=NUDGE_SPEED,
    max_load=NUDGE_MAX_LOAD,
    settle=1.4,
) -> None:
    """Gently move servo ``sid`` by ``delta`` ticks (position mode) and wait."""
    start = bus.position(sid)
    bus._write2(sid, ADDR_TORQUE_LIMIT, max(1, min(1000, round(max_load * 1000))))
    bus._write1(sid, ADDR_TORQUE_ENABLE, 1)
    bus._write2(sid, ADDR_GOAL_SPEED, max(1, speed))
    bus._write2(sid, ADDR_GOAL_POSITION, max(0, min(4095, start + delta)))
    time.sleep(settle)


def probe(bus: Bus, cam: Camera, sid: int, delta: int) -> tuple[float, float, int]:
    """Nudge servo ``sid`` by ``delta`` ticks, measure the image motion, return."""
    start = bus.position(sid)
    a = cam.gray()
    nudge(bus, sid, delta)
    b = cam.gray()
    # return to start
    bus._write2(sid, ADDR_GOAL_POSITION, start)
    time.sleep(1.2)
    bus.set_torque(sid, False)
    return motion_vector(a, b)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--device", default="/dev/video2")
    p.add_argument("--robot", choices=tuple(LAYOUTS), default="lekiwi")
    p.add_argument("--joint", default=None, help="probe one joint (default: all arm)")
    p.add_argument("--delta", type=int, default=NUDGE_TICKS)
    args = p.parse_args(argv)

    specs = [s for s in LAYOUTS[args.robot] if s.kind == "arm"]
    if args.joint:
        specs = [s for s in specs if s.name == args.joint]
    bus = Bus(args.port, args.baud)
    cam = Camera(args.device)
    try:
        print("probing joints (nudge +%d ticks, read image motion):" % args.delta)
        for s in specs:
            dx, dy, n = probe(bus, cam, s.servo_id, args.delta)
            mag = (dx * dx + dy * dy) ** 0.5
            direction = ""
            if n >= 80 and mag > 5:
                horiz = "right" if dx > 0 else "left"
                vert = "down" if dy > 0 else "up"
                direction = f"{vert}+{horiz}  (dx={dx:+.0f}, dy={dy:+.0f})"
            else:
                direction = "no clear motion (n=%d, mag=%.0f)" % (n, mag)
            print(f"  {s.name:14s} id={s.servo_id}: {direction}  [px={n}]")
    finally:
        for s in specs:
            try:
                bus.set_torque(s.servo_id, False)
            except Exception:  # noqa: BLE001
                pass
        cam.close()
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
