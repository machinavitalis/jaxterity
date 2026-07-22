# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-unit attested digital twin.

A :class:`Twin` composes a calibrated :class:`~jaxterity.robot.Robot` with the
off-body physics a *specific physical unit* exhibits (per-servo heat; a shared
battery for a mobile base) and the unit's identity (serial + provenance). It is
the object the predictive self-check rolls forward, and the unit
identity the attestation chain (ADR-008) binds a deployment to.

:func:`so101_twin` builds a bench SO-101 twin from the calibrated zoo robot with
representative (un-fitted) thermal defaults; :func:`lekiwi_twin` adds the shared
battery (the brownout path); :func:`unitree_g1_twin` builds the 29-servo G1
unit twin with per-joint-class thermal defaults, a nominal per-servo
actuation scale, and a **declared drift-parameter schema**
(:func:`g1_drift_parameters`) — the canonical fit-target set the online
re-fit / residual-monitor / unit-variance work consumes. The off-body sysid
that replaces the representative defaults with per-unit fits lives in
:func:`jaxterity.calibrate.fit_servo_thermal`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .. import zoo
from ..robot import JointType
from .twin import Twin, TwinRollout

#: Representative bench-SO-101 servo-thermal defaults (per DOF), pending the
#: per-unit off-body sysid. Plausible lumped values for a hobby
#: serial-bus servo: ~15 J/K winding+case mass, ~4 K/W to still air, ~6 W per
#: (N·m)² (≈ R_winding / kt² for kt≈1 N·m/A, R≈6 Ω). NOT a fitted calibration:
#: ``scripts/fit_twin_defaults.py`` (2026-07-22) ran ``fit_servo_thermal`` on
#: the real per-servo ``temp_C`` traces in
#: ``test/calibrate/data/lekiwi/commanded_*.json`` and found them
#: non-identifying (1 °C register quantization, 14–16 s excitations ≪ the
#: servo thermal time constant → non-physical fits, r² ≈ 0), so the
#: representative values stand until a dedicated long-duration heating capture.
_SO101_C_TH = 15.0
_SO101_R_TH = 4.0
_SO101_K_PWR = 6.0

#: LeKiwi shared-battery defaults (3S LiPo bus). ``V_oc``/``R_int`` are FITTED
#: from the real aggregate-load capture
#: ``test/calibrate/data/lekiwi/aggregate_load_trace.json`` by
#: ``scripts/fit_twin_defaults.py`` (2026-07-22): linear-LS Thévenin fit
#: (``fit_battery_sag``), r² = 0.66 on a real 0.70 V sag, servos (1, 2, 3),
#: 960 samples. ``R_int`` is in the duty-current frame (I = i_stall·duty,
#: i_stall = 2.7 A — see ``jaxterity/calibrate/battery.py``). The per-servo
#: torque→current gain ``k_i`` is NOT identified by that trace (the fit
#: measures current from PWM duty, not torque) and stays representative:
#: k_i≈1.5 A/(N·m) (≈ 1/kt for kt≈0.67 N·m/A).
_LEKIWI_V_OC = 12.166
_LEKIWI_R_INT = 0.0653
_LEKIWI_K_I = 1.5


def so101_twin(
    serial: str,
    *,
    C_th: float = _SO101_C_TH,
    R_th: float = _SO101_R_TH,
    k_pwr: float = _SO101_K_PWR,
    T_amb: float = 298.15,
) -> Twin:
    """Build a bench SO-101 twin from the calibrated zoo robot.

    Loads ``zoo.calibrated("so100")`` (the SO-100/SO-101 6-DoF arm) and attaches
    a per-DOF :class:`~jaxterity.physics.thermal.ServoThermal` with uniform
    representative defaults. No battery (bench unit). The thermal defaults are
    placeholders, not a per-unit fit.

    Args
    ----
    serial : str
        The unit's serial number.
    C_th, R_th, k_pwr : float
        Per-servo thermal capacitance (J/K), resistance (K/W), and torque²→power
        gain (W/(N·m)²), applied uniformly across the 6 servos.
    T_amb : float
        Ambient temperature, K.

    Returns
    -------
    Twin
        A bench SO-101 twin with 6 thermal nodes and no battery.

    Examples
    --------
    >>> twin = so101_twin("SO101-0001")  # doctest: +SKIP
    >>> twin.n_servos
    6
    """
    robot = zoo.calibrated("so100")
    # Number of actuated DOFs (nv) from the built system (excludes FIXED joints).
    n = robot.build_system(actuation=True)._nv
    servo_names = tuple(f"servo_{i}" for i in range(n))
    return Twin(
        robot,
        serial=serial,
        C_th=jnp.full(n, C_th),
        R_th=jnp.full(n, R_th),
        k_pwr=jnp.full(n, k_pwr),
        T_amb=T_amb,
        servo_names=servo_names,
    )


def lekiwi_twin(
    serial: str,
    *,
    C_th: float = _SO101_C_TH,
    R_th: float = _SO101_R_TH,
    k_pwr: float = _SO101_K_PWR,
    V_oc: float = _LEKIWI_V_OC,
    R_int: float = _LEKIWI_R_INT,
    k_i: float = _LEKIWI_K_I,
    T_amb: float = 298.15,
) -> Twin:
    """Build a LeKiwi-style twin: the SO-100 arm on a shared battery.

    Like :func:`so101_twin` but with a :class:`~jaxterity.physics.battery.BatterySag`
    so the predicted *bus voltage* (and the brownout veto) is exercised — the
    mobile-base electrical path the bench unit lacks.

    Honest scope: this attaches the shared-battery model to the SO-100 arm robot;
    the zoo has no LeKiwi *mobile base* URDF yet, so there is no wheel/base body
    in the dynamics — only the arm + the shared-bus electrical model. The default
    ``V_oc``/``R_int`` are fitted from a real aggregate-load capture of one unit
    (see ``scripts/fit_twin_defaults.py``); ``k_i`` and the thermal defaults are
    representative, not fits.

    Args
    ----
    serial : str
        The unit's serial number.
    C_th, R_th, k_pwr : float
        Per-servo thermal parameters (see :func:`so101_twin`).
    V_oc, R_int : float
        Battery open-circuit voltage (V) and internal resistance (Ω).
    k_i : float
        Per-servo torque→current gain, A per N·m (applied uniformly).
    T_amb : float
        Ambient temperature, K.

    Returns
    -------
    Twin
        A battery-bearing twin (``has_battery`` is True).

    Examples
    --------
    >>> twin = lekiwi_twin("LK-0001")  # doctest: +SKIP
    >>> twin.has_battery
    True
    """
    robot = zoo.calibrated("so100")
    n = robot.build_system(actuation=True)._nv
    servo_names = tuple(f"servo_{i}" for i in range(n))
    return Twin(
        robot,
        serial=serial,
        C_th=jnp.full(n, C_th),
        R_th=jnp.full(n, R_th),
        k_pwr=jnp.full(n, k_pwr),
        V_oc=V_oc,
        R_int=R_int,
        k_i=tuple([k_i] * n),
        T_amb=T_amb,
        servo_names=servo_names,
    )


#: Representative G1 per-servo thermal defaults by actuator class, keyed by
#: the upstream effort rating (N·m, from the model's ``actuatorfrcrange``).
#: Plausible lumped values for the three Unitree joint-motor sizes and the two
#: wrist sizes — bigger motor: more thermal mass, lower resistance to ambient,
#: lower copper loss per torque². NOT a fitted calibration — no G1 hardware
#: captures exist yet; these stay representative until real G1 temperature
#: traces land (the per-unit fit is the off-body sysid path /
#: the runtime-demo online re-fit).
_G1_THERMAL_BY_EFFORT = {
    139: (180.0, 1.2, 0.02),  # hip roll, knee
    88: (150.0, 1.5, 0.04),  # hip pitch/yaw, waist yaw
    50: (90.0, 2.0, 0.10),  # ankles, waist roll/pitch
    25: (40.0, 3.0, 0.35),  # shoulders, elbow, wrist roll
    5: (15.0, 4.5, 2.0),  # wrist pitch/yaw
}

#: Joint-name fragment → effort class (N·m). Checked in order; first match
#: wins (so ``waist_yaw`` matches before the generic ``waist``, ``wrist_roll``
#: before the generic ``wrist``, ``hip_roll`` before the generic ``hip``).
_G1_EFFORT_BY_FRAGMENT = (
    ("hip_roll", 139),
    ("knee", 139),
    ("hip_", 88),
    ("waist_yaw", 88),
    ("ankle", 50),
    ("waist_", 50),
    ("shoulder", 25),
    ("elbow", 25),
    ("wrist_roll", 25),
    ("wrist_", 5),
)


def _g1_effort_class(joint_name: str) -> int:
    """The G1 joint's actuator effort class (N·m) from its name."""
    for fragment, effort in _G1_EFFORT_BY_FRAGMENT:
        if fragment in joint_name:
            return effort
    raise ValueError(
        f"Joint {joint_name!r} matches no G1 actuator class fragment "
        f"({[f for f, _ in _G1_EFFORT_BY_FRAGMENT]}). The G1 twin factory "
        "only understands the 29 upstream joint names."
    )


class DriftParameter(NamedTuple):
    """One declared drift parameter — a canonical fit target.

    The schema the online re-fit (windowed sysid), the residual monitor, and
    the unit-variance sampling all consume: each entry names one leaf of
    :meth:`Twin.identifiable_params`, with physically-motivated fit bounds.

    Fields
    ------
    path : str
        Canonical key into :meth:`Twin.identifiable_params` — e.g.
        ``"robot.left_knee_joint.friction"`` (a robot leaf) or
        ``"offbody.K_t_scale"`` (a per-servo off-body vector leaf).
    lower, upper : float
        Fit bounds (elementwise, for vector leaves).
    units : str
        SI units of the leaf (``"-"`` for dimensionless).
    description : str
        What the parameter is and why it drifts.
    """

    path: str
    lower: float
    upper: float
    units: str
    description: str


def g1_drift_parameters() -> tuple[DriftParameter, ...]:
    """The G1's declared drift-parameter schema.

    The fit-target set for the runtime work: per-leg-joint Coulomb friction
    (gear/bearing wear — the classic slow drift and the fault-precursor
    channel), the per-servo actuation scale (motor torque-constant drift /
    manufacturing variance), and the per-servo thermal resistance (cooling
    degradation). Every path resolves into
    :meth:`Twin.identifiable_params` of :func:`unitree_g1_twin`'s output.

    Returns
    -------
    tuple[DriftParameter, ...]
        Twelve leg-joint friction scalars + two per-servo vector leaves.
    """
    leg_joints = tuple(
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    )
    frictions = tuple(
        DriftParameter(
            path=f"robot.{joint}.friction",
            lower=0.0,
            upper=2.0,
            units="N·m",
            description="Coulomb friction loss — gear/bearing wear drift.",
        )
        for joint in leg_joints
    )
    return frictions + (
        DriftParameter(
            path="offbody.K_t_scale",
            lower=0.5,
            upper=1.5,
            units="-",
            description=(
                "Per-servo actuation (torque-constant) scale — motor-constant "
                "drift and per-unit manufacturing variance."
            ),
        ),
        DriftParameter(
            path="offbody.R_th",
            lower=0.5,
            upper=8.0,
            units="K/W",
            description="Per-servo thermal resistance — cooling degradation.",
        ),
    )


def unitree_g1_twin(
    serial: str,
    *,
    T_amb: float = 298.15,
) -> Twin:
    """Build a Unitree G1 unit twin from the (fixed-base) zoo robot.

    Loads ``zoo.load("unitree_g1")`` — the fixed-base 29-DoF form, whose DOFs
    map 1:1 onto the 29 joint servos — and attaches per-servo thermal
    defaults by actuator class plus a nominal (all-ones) actuation scale, so
    every leaf of the declared drift schema (:func:`g1_drift_parameters`)
    exists and is fittable from day 0. Servo names are the joint names in DOF
    order, so dashboards and recipes address servos by joint.

    Honest scope: the G1 zoo entry is uncalibrated (no ``calibrated.json``
    yet) and the thermal/actuation defaults are representative, not fits —
    this twin is the *container* the per-unit fits land in.

    Args
    ----
    serial : str
        The physical unit's serial number.
    T_amb : float
        Ambient temperature, K.

    Returns
    -------
    Twin
        A bench (battery-less) G1 twin with 29 named servos, thermal
        defaults by joint class, and ``K_t_scale = 1`` everywhere.

    Examples
    --------
    >>> twin = unitree_g1_twin("G1-0047")  # doctest: +SKIP
    >>> twin.n_servos
    29
    """
    robot = zoo.load("unitree_g1")
    joint_names = tuple(
        joint.name
        for joint in robot.joints
        if joint.joint_type not in (JointType.FIXED, JointType.FLOATING)
    )
    thermal = [_G1_THERMAL_BY_EFFORT[_g1_effort_class(name)] for name in joint_names]
    n = len(joint_names)
    return Twin(
        robot,
        serial=serial,
        C_th=jnp.asarray([row[0] for row in thermal]),
        R_th=jnp.asarray([row[1] for row in thermal]),
        k_pwr=jnp.asarray([row[2] for row in thermal]),
        K_t_scale=jnp.ones(n),
        T_amb=T_amb,
        servo_names=joint_names,
    )


# Imported at the bottom: fleet.py consumes the factories defined above, and
# aging.py -> calibrate.thermal -> calibrate.windowed imports DriftParameter
# (defined above), so the aging import must follow that definition to avoid a
# circular import.
from .aging import (  # noqa: E402
    TwinHistory,
    TwinUpdate,
    track_thermal_drift,
    update_twin_thermal,
)
from .fleet import healthy_population, latin_hypercube, sample_g1_units  # noqa: E402

__all__ = [
    "Twin",
    "TwinRollout",
    "DriftParameter",
    "so101_twin",
    "lekiwi_twin",
    "unitree_g1_twin",
    "g1_drift_parameters",
    "latin_hypercube",
    "sample_g1_units",
    "healthy_population",
    "TwinUpdate",
    "TwinHistory",
    "update_twin_thermal",
    "track_thermal_drift",
]
