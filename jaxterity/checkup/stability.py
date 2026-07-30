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
hazard being screened), rigid base and wheels by default, and a
**representative, operator-overridable** base profile: :data:`LEKIWI_BASE`
ships plausible-but-unmeasured geometry/mass values; measure your unit to
tighten them.

The commanded waveform is **versioned** (:data:`MULTISINE_KIND`): the scripts
normalize their multisine by its measured peak, the earlier ``√(2n)`` estimate
(:data:`MULTISINE_KIND_V1`) overshot by 1.52×, and a margin is only meaningful
against the waveform that was actually commanded — so every entry point that
builds a trajectory takes a ``kind`` and every capture/plan records one. Score a
historical session with its own recorded kind, not the current default.

The rigid-wheel assumption is now liftable: pass a
:class:`jaxterity.physics.compliance.SuspensionProfile` to :func:`zmp_margins`
and the base tilt under load is subtracted from the margin instead of being
carried inside the floor. That is how the floor comes down honestly —
:func:`jaxterity.physics.compliance.margin_floor_credit` says how much of the
20 mm is explained by modeled compliance, and the remainder is what is still
riding on the 2026-07-28 hardware finding. The floor itself stays where it is
until a measured suspension (not the representative one) says otherwise.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

from ..errors import CalibrationError
from ..robot import Robot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..physics.compliance import SuspensionProfile

_GRAVITY = 9.81

#: The single-joint capture's Schroeder multisine tone set (Hz) — keep in sync
#: with ``scripts/lekiwi_commanded_capture.py``.
MULTISINE_HZ = (0.25, 0.5, 0.9, 1.4, 2.1)

#: The battery load trace's tone set (Hz) — deliberately different from
#: :data:`MULTISINE_HZ` so the aggregate current is broadband; keep in sync with
#: ``scripts/lekiwi_load_capture.py``.
LOAD_MULTISINE_HZ = (0.3, 0.6, 1.0, 1.5, 2.2)

#: Waveform-normalization versions, because the normalization is part of the
#: commanded motion and a plan is only valid for the waveform it vetted.
#:
#: ``v1`` divided the raw Schroeder sum by ``√(2n)``, a *crest-factor estimate*
#: that this 5-tone set overshoots by 1.52× — an ``amp=200`` command actually
#: swung ±303/−234 ticks (the ~46° travel seen on hardware). ``v2`` divides by
#: the sum's true peak, so the commanded swing never exceeds the requested
#: amplitude. The sum is asymmetric (+4.795/−3.696 at the 16 s period), so even
#: under ``v2`` the *peak-to-peak* is 1.77× the amplitude, not 2× — use
#: :func:`multisine_peak_to_peak` rather than assuming.
MULTISINE_KIND_V1 = "schroeder_multisine"
MULTISINE_KIND = "schroeder_multisine_v2"
_MULTISINE_KINDS = (MULTISINE_KIND_V1, MULTISINE_KIND)

#: Per-servo phase step the load capture applies so the servos do not all peak
#: together (mirrors ``scripts/lekiwi_load_capture.py``), rad.
_LOAD_PHASE_STEP = 1.7

#: Samples per second used to locate the raw sum's extrema. The sum is
#: band-limited (highest tone ~2.2 Hz), so this is ~450× oversampled — the peak
#: it finds is exact to well under a tick.
_EXTREMA_RATE_HZ = 1000.0


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
class StabilityConstraint:
    """A tip-over floor an excitation *designer* must respect, not just report.

    :class:`StabilityReport` is a verdict on a motion someone already planned.
    This is the same floor turned into a constraint an optimizer carries while
    it searches — see
    :func:`jaxterity.objects.probe.design_probe`, whose Fisher ascent projects
    every iterate back inside it. That is the difference between a gate that
    rejects your probe and a gate that shapes it.

    Args
    ----
    base : BaseProfile
        The base under the arm.
    suspension : SuspensionProfile | None
        Optional support compliance (see :func:`zmp_margins`).
    eval_hz : float
        Rate at which the candidate trajectory is resampled before
        differentiating. Higher is more faithful and more expensive; the
        default matches the capture planner's.
    margin_floor_m : float | None
        Override the floor; defaults to ``base.margin_min_m``.
    """

    base: BaseProfile
    suspension: "SuspensionProfile | None" = None
    eval_hz: float = 25.0
    margin_floor_m: float | None = None

    @property
    def floor(self) -> float:
        """The margin floor actually in force, m."""
        return (
            self.base.margin_min_m
            if self.margin_floor_m is None
            else float(self.margin_floor_m)
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


def _raw_schroeder(
    t: np.ndarray,
    period: float,
    tones: tuple[float, ...],
    phase: float = 0.0,
) -> np.ndarray:
    """The un-normalized Schroeder sum, tones snapped to 1/period harmonics."""
    ks = [max(1, round(f * period)) for f in tones]
    n = len(ks)
    phases = [-math.pi * i * (i + 1) / n + phase for i in range(n)]
    tt = np.asarray(t, float)
    raw = np.zeros_like(tt)
    for k, p in zip(ks, phases):
        raw = raw + np.sin(2 * np.pi * k * tt / period + p)
    return raw


def multisine_extrema(
    period: float,
    tones: tuple[float, ...] = MULTISINE_HZ,
    phase: float = 0.0,
) -> tuple[float, float]:
    """Max and min of the un-normalized Schroeder sum over one period.

    Evaluated on a dense grid (:data:`_EXTREMA_RATE_HZ`) rather than in closed
    form — a sum of harmonics has no useful analytic extremum, and the signal is
    band-limited enough that a heavily oversampled scan is exact to well below
    one encoder tick. Deterministic: no RNG, no solver.

    Args
    ----
    period : float
        Waveform period, s (the capture duration).
    tones : tuple[float, ...]
        Tone set, Hz.
    phase : float
        Extra phase offset applied to every tone, rad.

    Returns
    -------
    tuple[float, float]
        ``(max, min)`` of the raw sum. Asymmetric in general.

    Units
    -----
    s, Hz in; dimensionless out.

    Examples
    --------
    >>> hi, lo = multisine_extrema(16.0)
    >>> round(hi, 3), round(lo, 3)
    (4.795, -3.696)
    """
    t = np.arange(0.0, period, 1.0 / _EXTREMA_RATE_HZ)
    raw = _raw_schroeder(t, period, tones, phase)
    return float(np.max(raw)), float(np.min(raw))


def _normalizer(period: float, kind: str, tones: tuple[float, ...]) -> float:
    """The divisor a given waveform version applies to the raw sum."""
    if kind == MULTISINE_KIND_V1:
        return math.sqrt(2.0 * len(tones))
    if kind == MULTISINE_KIND:
        hi, lo = multisine_extrema(period, tones)
        return max(abs(hi), abs(lo))
    raise CalibrationError(
        f"Unknown multisine kind {kind!r}; expected one of {_MULTISINE_KINDS}. "
        "The kind is recorded in every capture's `excitation.kind` and in the "
        "capture plan — read it from the artifact rather than assuming, because "
        "the two versions command different swings for the same amplitude."
    )


def multisine_peak_to_peak(
    period: float,
    kind: str = MULTISINE_KIND,
    tones: tuple[float, ...] = MULTISINE_HZ,
) -> float:
    """Commanded peak-to-peak swing per unit of requested amplitude.

    The number to divide a *measured* peak-to-peak by when scoring how well a
    joint followed its command. It is **not 2**: the Schroeder sum is
    asymmetric, so even the true-peak-normalized ``v2`` waveform swings 1.77×
    its amplitude, and the legacy ``v1`` normalization swings 2.69×.

    Args
    ----
    period : float
        Waveform period, s.
    kind : str
        Waveform version (:data:`MULTISINE_KIND` or
        :data:`MULTISINE_KIND_V1`).
    tones : tuple[float, ...]
        Tone set, Hz.

    Returns
    -------
    float
        Peak-to-peak per unit amplitude, dimensionless.

    Raises
    ------
    CalibrationError
        On an unrecognized ``kind``.

    Examples
    --------
    >>> round(multisine_peak_to_peak(16.0, MULTISINE_KIND), 3)
    1.771
    >>> round(multisine_peak_to_peak(16.0, MULTISINE_KIND_V1), 3)
    2.685
    """
    hi, lo = multisine_extrema(period, tones)
    return (hi - lo) / _normalizer(period, kind, tones)


def schroeder_multisine(
    t: np.ndarray,
    period: float,
    *,
    kind: str = MULTISINE_KIND,
    tones: tuple[float, ...] = MULTISINE_HZ,
    phase: float = 0.0,
) -> np.ndarray:
    """The capture scripts' Schroeder multisine at nominal amplitude 1.

    Mirrors the on-device scripts exactly — same tone set, same Schroeder
    phases, same normalization — so the trajectory the ZMP gate validates is
    the trajectory the hardware commands. That fidelity is the whole point, and
    it is why the normalization is *versioned* rather than silently improved:
    a capture (or plan) recorded under ``v1`` must still be scored against the
    ``v1`` waveform, or a historical postmortem would exonerate a motion that
    really did tip the robot.

    Args
    ----
    t : numpy.ndarray
        Sample times, s.
    period : float
        Waveform period, s.
    kind : str
        :data:`MULTISINE_KIND` (true-peak normalized, what the scripts command
        today — ``max|w| = 1``) or :data:`MULTISINE_KIND_V1` (the legacy
        ``√(2n)`` estimate, which overshoots to ~1.52).
    tones : tuple[float, ...]
        Tone set, Hz (:data:`MULTISINE_HZ` for single-joint captures,
        :data:`LOAD_MULTISINE_HZ` for the battery load trace).
    phase : float
        Extra phase offset applied to every tone, rad — the load capture gives
        each servo a different one so they do not all peak together.

    Returns
    -------
    numpy.ndarray
        The waveform at nominal amplitude 1, same shape as ``t``.

    Raises
    ------
    CalibrationError
        On an unrecognized ``kind``.

    Units
    -----
    s in; dimensionless out.
    """
    return _raw_schroeder(t, period, tones, phase) / _normalizer(period, kind, tones)


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
    suspension: "SuspensionProfile | None" = None,
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
    suspension : SuspensionProfile | None
        Optional support-spring model
        (:class:`jaxterity.physics.compliance.SuspensionProfile`). When given,
        the rigid-base margin is reduced by the centre-of-mass walk the base
        tilt produces under the same load — the "rigid base and wheels"
        assumption in this module's docstring, lifted. When ``None`` (the
        default) the behaviour is exactly as before, and the unmodeled
        compliance stays inside ``base.margin_min_m``.

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
    zmp = np.stack([zmp_x, zmp_y], axis=1)
    margins = polygon_margin(zmp, np.asarray(base.support_polygon_xy))
    if suspension is not None:
        from ..physics.compliance import compliant_margin

        margins = compliant_margin(
            margins, zmp, suspension, float(m_all.sum()) * _GRAVITY
        )
    return margins


def multisine_trajectory(
    pose_rad: np.ndarray,
    joint_index: int,
    center_rad: float,
    amp_rad: float,
    *,
    duration_s: float,
    eval_hz: float = 25.0,
    kind: str = MULTISINE_KIND,
) -> tuple[np.ndarray, float]:
    """The commanded single-joint multisine as a full-arm joint trajectory.

    The excited joint follows ``center + amp * schroeder(t)``; every other
    joint holds ``pose_rad``. Returns ``(q_t, dt)`` at the evaluation rate.
    ``kind`` selects the waveform version — pass the one recorded in the
    capture or plan being validated (see :func:`schroeder_multisine`).
    """
    dt = 1.0 / eval_hz
    t = np.arange(0.0, duration_s, dt)
    q_t = np.tile(np.asarray(pose_rad, float), (t.shape[0], 1))
    q_t[:, joint_index] = center_rad + amp_rad * schroeder_multisine(
        t, duration_s, kind=kind
    )
    return q_t, dt


def load_trace_trajectory(
    pose_rad: np.ndarray,
    joint_indices: tuple[int, ...],
    amps_rad: tuple[float, ...],
    *,
    duration_s: float,
    eval_hz: float = 25.0,
    kind: str = MULTISINE_KIND,
    in_phase: bool = True,
) -> tuple[np.ndarray, float]:
    """The multi-joint battery load-trace trajectory.

    Uses :data:`LOAD_MULTISINE_HZ` — the load capture runs a *different* tone
    set from the single-joint captures, so validating it against
    :data:`MULTISINE_HZ` would be checking a motion nobody commands.

    ``in_phase=True`` (the default) puts every commanded joint on the same
    waveform. That is **deliberately not** what the script does: it offsets each
    servo's phase so the aggregate current is broader-band and the servos do not
    all peak together. Riding them in phase is the worse rocking case, which is
    the right side to err on for a tip-over gate — so the gate keeps it, and
    says so here rather than claiming to mirror the script exactly. Pass
    ``in_phase=False`` to reproduce the script's staggered phases (the honest
    *expected* motion, useful for postmortems).

    Args
    ----
    pose_rad : numpy.ndarray
        Held arm pose, rad, shape ``(n_dof,)``.
    joint_indices : tuple[int, ...]
        DOF indices of the commanded joints.
    amps_rad : tuple[float, ...]
        Per-joint amplitude, rad.
    duration_s : float
        Trace duration, s.
    eval_hz : float
        Trajectory evaluation rate, Hz.
    kind : str
        Waveform version (see :func:`schroeder_multisine`).
    in_phase : bool
        Worst-case synchronized rocking (default) vs the script's staggered
        phases.

    Returns
    -------
    tuple[numpy.ndarray, float]
        ``(q_t, dt)``.

    Units
    -----
    rad, s, Hz.
    """
    dt = 1.0 / eval_hz
    t = np.arange(0.0, duration_s, dt)
    q_t = np.tile(np.asarray(pose_rad, float), (t.shape[0], 1))
    for k, (j, amp) in enumerate(zip(joint_indices, amps_rad)):
        # The script's per-servo phase step; 0 for the in-phase worst case.
        phase = 0.0 if in_phase else k * _LOAD_PHASE_STEP
        wave = schroeder_multisine(
            t, duration_s, kind=kind, tones=LOAD_MULTISINE_HZ, phase=phase
        )
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
    "LOAD_MULTISINE_HZ",
    "MULTISINE_HZ",
    "MULTISINE_KIND",
    "MULTISINE_KIND_V1",
    "BaseProfile",
    "LEKIWI_BASE",
    "StabilityConstraint",
    "StabilityReport",
    "multisine_extrema",
    "multisine_peak_to_peak",
    "polygon_margin",
    "schroeder_multisine",
    "zmp_margins",
    "multisine_trajectory",
    "load_trace_trajectory",
    "transit_trajectory",
]
