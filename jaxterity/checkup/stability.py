# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Tip-over (ZMP) validation of planned checkup motions on a mobile base.

WHY THIS EXISTS. A checkup excitation that is safe for a *bolted-down* arm can
tip an arm on a small mobile base: the T-228 v0 checkup captured around each
joint's band center, which extended a LeKiwi's arm forward and rocked the base
off its rear wheel (hardware finding, 2026-07-28). Joint-limit guards do not
see the base. This module adds the missing gate: given the arm model we
already have, a declared :class:`BaseProfile` (support polygon + base mass),
and a *planned* joint trajectory, it computes the quasi-dynamic
zero-moment-point (ZMP) trajectory and its **signed margin inside the support
polygon**. Planned motions whose worst-case margin is below the profile's
floor must be re-posed, slowed, or amplitude-derated (see
:mod:`jaxterity.checkup.plan`).

The mechanics: sample the planned arm trajectory, run forward kinematics per
sample (per-body CoM positions from the same MuJoCo model the fits use),
transform into the base frame via the declared arm-mount transform, add the
base's own static mass, finite-difference each body's CoM acceleration, and
form the flat-ground ZMP

    x_zmp = Σ m_i [x_i (z̈_i + g) − z_i ẍ_i] / Σ m_i (z̈_i + g)

(and likewise y). The margin is the minimum over the trajectory of the ZMP's
signed distance inside the support polygon. Assumptions, stated honestly:
perfect tracking of the commanded trajectory (commanded accelerations are the
hazard being screened), rigid base and wheels (no suspension compliance,
which on a real LeKiwi adds rocking — hence a conservative margin floor), and
a **representative, operator-overridable** base profile: :data:`LEKIWI_BASE`
ships plausible-but-unmeasured geometry/mass values; measure your unit to
tighten them.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..errors import CalibrationError
from ..robot import Robot

_GRAVITY = 9.81

#: The capture scripts' Schroeder multisine tone set (Hz) — keep in sync with
#: ``scripts/lekiwi_commanded_capture.py``.
MULTISINE_HZ = (0.25, 0.5, 0.9, 1.4, 2.1)


@dataclasses.dataclass(frozen=True)
class BaseProfile:
    """A mobile base, as the stability gate sees it (base frame: x forward).

    ``support_polygon_xy`` are the wheel ground-contact points (m, CCW).
    ``base_mass`` / ``base_com_xyz`` cover everything that is not the arm
    (plate, battery, wheel motors). ``arm_mount_xyz`` / ``arm_mount_yaw``
    place the arm model's base frame on the base. ``margin_min_m`` is the
    smallest acceptable ZMP margin — the pass/fail floor.
    """

    name: str
    support_polygon_xy: tuple[tuple[float, float], ...]
    base_mass: float
    base_com_xyz: tuple[float, float, float]
    arm_mount_xyz: tuple[float, float, float]
    arm_mount_yaw: float = 0.0
    margin_min_m: float = 0.012


#: REPRESENTATIVE LeKiwi profile — plausible geometry/mass, NOT measured from
#: a unit. Two front omni wheels + one rear, arm mounted at the front edge
#: (which is why forward-extended poses tip it). Override with your unit's
#: measurements for a trustworthy margin; the *relative* verdicts (tucked pose
#: safe, extended pose not) are robust to moderate profile error.
LEKIWI_BASE = BaseProfile(
    name="lekiwi",
    support_polygon_xy=((0.0575, 0.0996), (-0.115, 0.0), (0.0575, -0.0996)),
    base_mass=1.8,
    base_com_xyz=(-0.02, 0.0, 0.04),
    arm_mount_xyz=(0.09, 0.0, 0.05),
    arm_mount_yaw=0.0,
    # 20 mm floor, calibrated against the 2026-07-28 hardware finding: the
    # tip-over occurred on capture segments this model scores at <= 8.5 mm,
    # and the post-extension poses it scores ~12 mm still rocked the base —
    # omni-roller compliance and rocking resonance are unmodeled, so the
    # floor carries their budget.
    margin_min_m=0.020,
)


@dataclasses.dataclass(frozen=True)
class StabilityReport:
    """The ZMP verdict for one planned trajectory segment.

    ``min_margin_m`` is the worst signed ZMP margin over the segment
    (positive = inside the support polygon); ``ok`` compares it against the
    profile's floor. ``static_margin_m`` is the margin of the segment's start
    pose at rest — if *that* is already below the floor, no excitation from
    this pose can pass and the pose itself must change.
    """

    name: str
    min_margin_m: float
    static_margin_m: float
    margin_floor_m: float

    @property
    def ok(self) -> bool:
        """Whether the whole segment stays inside the margin floor."""
        return self.min_margin_m >= self.margin_floor_m


def polygon_margin(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Signed distance of points inside a convex CCW polygon (negative = out).

    Args
    ----
    points_xy : np.ndarray
        Query points, shape ``(n, 2)``.
    polygon : np.ndarray
        Convex polygon vertices in CCW order, shape ``(k, 2)``.

    Returns
    -------
    np.ndarray
        Per-point minimum signed distance to the polygon edges, shape ``(n,)``.
    """
    pts = np.atleast_2d(np.asarray(points_xy, float))
    poly = np.asarray(polygon, float)
    margins = np.full(pts.shape[0], np.inf)
    k = poly.shape[0]
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        edge = b - a
        norm = float(np.hypot(*edge))
        # Inward normal of a CCW edge is (-dy, dx) / |edge|.
        n_in = np.array([-edge[1], edge[0]]) / norm
        margins = np.minimum(margins, (pts - a) @ n_in)
    return margins


def schroeder_multisine(t: np.ndarray, period: float) -> np.ndarray:
    """The capture scripts' Schroeder multisine at nominal amplitude 1.

    Mirrors ``scripts/lekiwi_commanded_capture.py`` exactly (same tone set,
    Schroeder phases, ``sqrt(2n)`` normalization) so the validated trajectory
    is the commanded one. NOTE the script's normalization is loose: this
    5-tone set truly peaks at ~1.51, so the commanded swing exceeds the
    nominal amplitude by ~50% — faithfulness here matters more than
    prettiness, and the ZMP margins price the real excursion in.
    """
    ks = [max(1, round(f * period)) for f in MULTISINE_HZ]
    n = len(ks)
    phases = [-math.pi * i * (i + 1) / n for i in range(n)]
    tt = np.asarray(t, float)
    raw = np.zeros_like(tt)
    for k, p in zip(ks, phases):
        raw = raw + np.sin(2 * np.pi * k * tt / period + p)
    return raw / math.sqrt(2.0 * n)


def _arm_com_series(
    robot: Robot, q_t: np.ndarray, dof_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample world CoM of every arm body along a joint trajectory.

    Returns ``(masses, xipos)`` with shapes ``(n_bodies,)`` and
    ``(T, n_bodies, 3)``, in the arm model's own base frame. One
    ``mj_forward`` per sample — kinematics only, cheap at capture lengths.
    """
    import mujoco

    mj = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    data = mujoco.MjData(mj)
    dofadr = {n: int(mj.joint(n).dofadr[0]) for n in dof_names}
    body_ids = [b for b in range(1, mj.nbody) if mj.body_mass[b] > 0.0]
    masses = np.array([mj.body_mass[b] for b in body_ids])
    xipos = np.empty((q_t.shape[0], len(body_ids), 3))
    for k in range(q_t.shape[0]):
        data.qpos[:] = 0.0
        for j, name in enumerate(dof_names):
            data.qpos[dofadr[name]] = q_t[k, j]
        mujoco.mj_forward(mj, data)
        xipos[k] = data.xipos[body_ids]
    return masses, xipos


def zmp_margins(
    robot: Robot,
    q_t: np.ndarray,
    dt: float,
    dof_names: tuple[str, ...],
    base: BaseProfile,
) -> np.ndarray:
    """Per-sample ZMP margin (m, signed, positive inside) of a joint trajectory.

    Args
    ----
    robot : Robot
        The arm model (masses/kinematics source).
    q_t : np.ndarray
        Planned joint positions, shape ``(T, n_dof)``, radians, in
        ``dof_names`` order. ``T >= 3`` (accelerations are finite-differenced).
    dt : float
        Sample period, s.
    dof_names : tuple[str, ...]
        DOF order of ``q_t`` (:func:`jaxterity.calibrate.active.dof_joint_names`).
    base : BaseProfile
        The declared base under the arm.

    Returns
    -------
    np.ndarray
        Signed ZMP margin per sample, shape ``(T,)``.

    Raises
    ------
    CalibrationError
        If the trajectory is too short to differentiate.
    """
    q_t = np.atleast_2d(np.asarray(q_t, float))
    if q_t.shape[0] < 3:
        raise CalibrationError(
            f"zmp_margins: trajectory has {q_t.shape[0]} samples; need >= 3 "
            "to estimate accelerations."
        )
    masses, xipos_arm = _arm_com_series(robot, q_t, dof_names)

    # Arm frame -> base frame (declared mount transform).
    cy, sy = math.cos(base.arm_mount_yaw), math.sin(base.arm_mount_yaw)
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    xipos = xipos_arm @ rot.T + np.asarray(base.arm_mount_xyz)

    acc = np.gradient(np.gradient(xipos, dt, axis=0), dt, axis=0)

    # Base static mass enters with zero acceleration.
    m_all = np.concatenate([masses, [base.base_mass]])
    pos_all = np.concatenate(
        [xipos, np.broadcast_to(base.base_com_xyz, (q_t.shape[0], 1, 3))], axis=1
    )
    acc_all = np.concatenate([acc, np.zeros((q_t.shape[0], 1, 3))], axis=1)

    fz = m_all[None, :] * (acc_all[:, :, 2] + _GRAVITY)  # (T, n)
    denom = fz.sum(axis=1)
    zmp_x = (
        fz * pos_all[:, :, 0] - m_all[None, :] * pos_all[:, :, 2] * acc_all[:, :, 0]
    ).sum(axis=1) / denom
    zmp_y = (
        fz * pos_all[:, :, 1] - m_all[None, :] * pos_all[:, :, 2] * acc_all[:, :, 1]
    ).sum(axis=1) / denom
    return polygon_margin(
        np.stack([zmp_x, zmp_y], axis=1), np.asarray(base.support_polygon_xy)
    )


def multisine_trajectory(
    pose_rad: np.ndarray,
    joint_index: int,
    center_rad: float,
    amp_rad: float,
    *,
    duration_s: float,
    eval_hz: float = 25.0,
) -> tuple[np.ndarray, float]:
    """The commanded single-joint multisine as a full-arm joint trajectory.

    The excited joint follows ``center + amp * schroeder(t)``; every other
    joint holds ``pose_rad``. Returns ``(q_t, dt)`` at the evaluation rate.
    """
    dt = 1.0 / eval_hz
    t = np.arange(0.0, duration_s, dt)
    q_t = np.tile(np.asarray(pose_rad, float), (t.shape[0], 1))
    q_t[:, joint_index] = center_rad + amp_rad * schroeder_multisine(t, duration_s)
    return q_t, dt


def load_trace_trajectory(
    pose_rad: np.ndarray,
    joint_indices: tuple[int, ...],
    amps_rad: tuple[float, ...],
    *,
    duration_s: float,
    eval_hz: float = 25.0,
) -> tuple[np.ndarray, float]:
    """The synchronized (in-phase) multi-joint load-trace trajectory.

    All commanded joints ride the same Schroeder waveform about their pose
    values — the worst-case in-phase rocking the battery trace produces.
    """
    dt = 1.0 / eval_hz
    t = np.arange(0.0, duration_s, dt)
    wave = schroeder_multisine(t, duration_s)
    q_t = np.tile(np.asarray(pose_rad, float), (t.shape[0], 1))
    for j, amp in zip(joint_indices, amps_rad):
        q_t[:, j] = q_t[0, j] + amp * wave
    return q_t, dt


def transit_trajectory(
    pose_from: np.ndarray,
    pose_to: np.ndarray,
    *,
    speed_rad_s: float,
    eval_hz: float = 25.0,
) -> tuple[np.ndarray, float]:
    """A bounded-speed simultaneous ramp between two poses (what a guarded
    transit commands), with a short hold at each end so the finite-difference
    sees the start/stop transients."""
    dt = 1.0 / eval_hz
    delta = np.asarray(pose_to, float) - np.asarray(pose_from, float)
    t_move = float(np.max(np.abs(delta))) / max(speed_rad_s, 1e-6)
    n_move = max(int(t_move * eval_hz), 2)
    ramp = np.linspace(0.0, 1.0, n_move)[:, None]
    hold = np.ones((int(0.5 * eval_hz), 1))
    s = np.concatenate([0.0 * hold, ramp, hold])
    return np.asarray(pose_from, float)[None, :] + s * delta[None, :], dt


__all__ = [
    "MULTISINE_HZ",
    "BaseProfile",
    "LEKIWI_BASE",
    "StabilityReport",
    "polygon_margin",
    "schroeder_multisine",
    "zmp_margins",
    "multisine_trajectory",
    "load_trace_trajectory",
    "transit_trajectory",
]
