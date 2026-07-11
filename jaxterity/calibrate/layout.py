# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Hardware layout for the Feetech-servo robots the setup routine supports.

This module owns the *per-model facts* the bring-up needs but the URDF does not
carry: which bus ID drives which joint, whether a joint is a limited arm joint
or a continuous wheel, the encoder direction relative to the URDF axis
(``sign``), and the servo's tick resolution. The zoo URDF supplies the joint
*limits* (the angles the touch-off anchors to); this supplies the *wiring*.

The numbers here are the published Feetech STS3215 / SO-101 / LeKiwi
conventions. The per-joint ``sign`` is the one fact that cannot be measured from
hard stops alone (two stops fix the offset and the scale magnitude but not the
direction — see :func:`jaxterity.calibrate.offset.touch_off`); it is a one-time
per-model constant and is marked for hardware confirmation on first bring-up.
"""

from __future__ import annotations

import dataclasses
import math

#: Feetech STS3215 single-turn encoder resolution (counts per revolution). The
#: 12-bit absolute magnetic encoder reports 0..4095 over 360°.
STS3215_RESOLUTION = 4096

#: Encoder counts per radian for the STS3215 (``RESOLUTION / 2π``).
STS3215_TICKS_PER_RAD = STS3215_RESOLUTION / (2.0 * math.pi)


@dataclasses.dataclass(frozen=True)
class JointSpec:
    """Wiring and convention for one servo-driven joint.

    Args
    ----
    name : str
        URDF joint name for arm joints; a synthetic name for wheels (which are
        not in the arm-only URDF).
    servo_id : int
        Bus ID of the servo driving this joint.
    kind : str
        ``"arm"`` (limited, position-controlled, calibratable by touch-off) or
        ``"wheel"`` (continuous, velocity-controlled, no meaningful offset).
    sign : int
        Encoder direction relative to the URDF axis (+1 or -1). For wheels,
        the positive-rotation convention. **Confirm on first hardware run.**
    ticks_per_rad : float
        Encoder resolution, counts per radian.
    """

    name: str
    servo_id: int
    kind: str
    sign: int = 1
    ticks_per_rad: float = STS3215_TICKS_PER_RAD


@dataclasses.dataclass(frozen=True)
class RobotLayout:
    """The full servo layout for a supported robot.

    Args
    ----
    name : str
        Layout key (e.g. ``"lekiwi"``, ``"so101"``).
    urdf_robot : str
        The zoo robot whose URDF supplies the arm joint limits.
    joints : tuple[JointSpec, ...]
        Every servo on the bus, in bring-up order.
    """

    name: str
    urdf_robot: str
    joints: tuple[JointSpec, ...]

    @property
    def expected_ids(self) -> tuple[int, ...]:
        """Servo IDs this layout expects on the bus (sorted)."""
        return tuple(sorted(j.servo_id for j in self.joints))

    @property
    def arm_joints(self) -> tuple[JointSpec, ...]:
        """The limited arm joints (touch-off calibratable), in order."""
        return tuple(j for j in self.joints if j.kind == "arm")

    @property
    def wheel_joints(self) -> tuple[JointSpec, ...]:
        """The continuous wheel joints, in order."""
        return tuple(j for j in self.joints if j.kind == "wheel")


#: The six SO-101 arm joints, in LeRobot's canonical motor order, on bus IDs
#: 1..6. Names match the vendored zoo URDF (``jaxterity.zoo.so100``) so the
#: touch-off can read each joint's limits straight from the model.
_SO101_ARM = (
    JointSpec("shoulder_pan", 1, "arm"),
    JointSpec("shoulder_lift", 2, "arm"),
    JointSpec("elbow_flex", 3, "arm"),
    JointSpec("wrist_flex", 4, "arm"),
    JointSpec("wrist_roll", 5, "arm"),
    JointSpec("gripper", 6, "arm"),
)

#: Bench SO-101 arm on its own: just the six arm servos.
SO101 = RobotLayout(name="so101", urdf_robot="so100", joints=_SO101_ARM)

#: LeKiwi: the SO-101 arm (IDs 1-6) plus three omni-wheel drive servos
#: (IDs 7=left, 8=back, 9=right) run in continuous-rotation mode.
LEKIWI = RobotLayout(
    name="lekiwi",
    urdf_robot="so100",
    joints=(
        *_SO101_ARM,
        JointSpec("wheel_left", 7, "wheel"),
        JointSpec("wheel_back", 8, "wheel"),
        JointSpec("wheel_right", 9, "wheel"),
    ),
)

#: Every layout the setup routine knows, keyed by ``--robot`` value.
LAYOUTS: dict[str, RobotLayout] = {layout.name: layout for layout in (SO101, LEKIWI)}


def get_layout(name: str) -> RobotLayout:
    """Look up a robot layout by name.

    Args
    ----
    name : str
        Layout key (``"lekiwi"`` or ``"so101"``), case-insensitive.

    Returns
    -------
    RobotLayout
        The matching layout.

    Raises
    ------
    KeyError
        If ``name`` is not a known layout (the message lists the known ones).
    """
    key = name.strip().lower()
    if key not in LAYOUTS:
        known = ", ".join(sorted(LAYOUTS))
        raise KeyError(f"Unknown robot layout {name!r}. Known layouts: {known}.")
    return LAYOUTS[key]


__all__ = [
    "STS3215_RESOLUTION",
    "STS3215_TICKS_PER_RAD",
    "JointSpec",
    "RobotLayout",
    "SO101",
    "LEKIWI",
    "LAYOUTS",
    "get_layout",
]
