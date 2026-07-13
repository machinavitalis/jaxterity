# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Hold-and-wiggle calibration excitation.

The "informative wiggles": a stabilized excitation routine that produces one
:class:`~jaxterity.data.JointTelemetry` window suitable for the windowed
drift re-fit (:func:`jaxterity.calibrate.refit_drift_parameters`) — on a new
unit at commissioning time (the per-unit calibration story) or periodically
on a deployed one (the slow recalibration loop).

Design (distilled from the drift-recovery work — each rule was learned from
a concrete failure mode):

- **PD hold + multi-sine overlay.** A fixed-base humanoid cannot be excited
  passively: the knee's gravity equilibrium sits *on* its lower limit and
  the waist is an inverted torso pendulum, so the wiggles ride on a
  stabilizing hold — exactly how a real calibration routine runs. The
  telemetry records the *total* commanded torque, which is all
  equation-error identification needs.
- **Gains and amplitudes from apparent inertia, not actuator class.** Gains
  are ``kp = I·ω²``, ``kd = 2ζωI`` with ``I`` the mass-matrix diagonal at
  the hold pose, giving every joint the same closed-loop dynamics;
  effort-class tables destabilize near-massless distal joints. Amplitudes
  are ``I · min(accel target, displacement-bounded accel)`` so heavy joints
  are strongly excited (identifiable friction / torque constant) while
  light ones never approach their limits.
- **Continuous in time.** Both the hold loop and the multi-sine are
  evaluated continuously inside the integrator: a zero-order-held command
  puts torque discontinuities exactly at the sample instants — where the
  re-fit's central-difference acceleration straddles.
- **Limit-impact-free by construction, verified at run time.** Equation
  error through an active joint-limit impact is non-smooth (ADR-017);
  :func:`simulate_hold_window` raises if any joint comes within the safety
  margin of a limit.

:func:`simulate_hold_window` is the *simulated* collection path (the plant
is a — possibly drifted/sampled — :class:`~jaxterity.robot.Robot`); the
hardware path reuses the same :class:`HoldExcitation` design via the active
links when a real bus is present.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ..data.schema import JointTelemetry
from ..errors import JaxterityError
from ..robot import JointType, Robot

#: Default hold-loop natural frequency (rad/s) and damping ratio.
_OMEGA_DEFAULT = 12.0
_ZETA_DEFAULT = 0.7
#: Default excitation sizing (see the module docstring).
_ACCEL_TARGET_DEFAULT = 60.0
_DISPLACEMENT_FRACTION_DEFAULT = 0.3
_TORQUE_CEILING_DEFAULT = 8.0
#: Run-time limit safety margin, rad.
_LIMIT_MARGIN = 0.02


@dataclasses.dataclass(frozen=True)
class HoldExcitation:
    """A designed hold-and-wiggle routine for one robot (static data).

    Build with :func:`design_hold_excitation`; collect a window with
    :func:`simulate_hold_window` (or replay the same design on hardware).

    Args
    ----
    joint_names : tuple[str, ...]
        Actuated joints in DOF order.
    q_hold : tuple[float, ...]
        The hold pose, rad.
    kp, kd : tuple[float, ...]
        Hold-loop gains per joint (N·m/rad, N·m·s/rad).
    amplitude : tuple[float, ...]
        Multi-sine amplitude per joint, N·m.
    frequency_hz : tuple[float, ...]
        Multi-sine frequency per joint (incommensurate across joints).
    phase : tuple[float, ...]
        Multi-sine phase per joint, rad.
    duration_s : float
        Window length, s.
    sample_rate_hz : float
        Telemetry sample rate, Hz.
    sim_dt : float
        Inner integration step for the simulated path, s.
    """

    joint_names: tuple[str, ...]
    q_hold: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    amplitude: tuple[float, ...]
    frequency_hz: tuple[float, ...]
    phase: tuple[float, ...]
    duration_s: float
    sample_rate_hz: float
    sim_dt: float

    @property
    def n_samples(self) -> int:
        """Telemetry samples in the window."""
        return int(round(self.duration_s * self.sample_rate_hz))


def _actuated_joint_names(robot: Robot) -> tuple[str, ...]:
    return tuple(
        joint.name
        for joint in robot.joints
        if joint.joint_type not in (JointType.FIXED, JointType.FLOATING)
    )


def _mass_diagonal(robot: Robot, q: np.ndarray) -> np.ndarray:
    """Mass-matrix diagonal at a configuration (kg·m² per DOF)."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    return np.diag(full).copy()


def default_hold_pose(robot: Robot) -> np.ndarray:
    """A hold pose with every joint pulled off its limits.

    Zero where zero is comfortably inside the range; otherwise pulled in by
    ``min(0.5 rad, a third of the range)`` from the nearest limit. For the
    G1 this flexes the knees well clear of their lower limit (the straight
    leg *is* that limit, and gravity sag against the hold loop costs a
    couple of tenths of a radian of the margin).

    Returns
    -------
    numpy.ndarray, shape ``(nv,)``
        The pose, rad, in DOF order.
    """
    pose = []
    for joint in robot.joints:
        if joint.joint_type in (JointType.FIXED, JointType.FLOATING):
            continue
        lo, hi = joint.limits.lower, joint.limits.upper
        pull = min(0.5, (hi - lo) / 3.0)
        pose.append(float(np.clip(0.0, lo + pull, hi - pull)))
    return np.asarray(pose)


def design_hold_excitation(
    robot: Robot,
    *,
    duration_s: float = 1.6,
    sample_rate_hz: float = 100.0,
    sim_dt: float = 0.002,
    q_hold=None,
    omega: float = _OMEGA_DEFAULT,
    zeta: float = _ZETA_DEFAULT,
    accel_target: float = _ACCEL_TARGET_DEFAULT,
    displacement_fraction: float = _DISPLACEMENT_FRACTION_DEFAULT,
    torque_ceiling: float = _TORQUE_CEILING_DEFAULT,
) -> HoldExcitation:
    """Design a hold-and-wiggle routine for a (fixed-base) robot.

    See the module docstring for the design rules. The routine is computed
    from the robot's *nominal* model — a drifted unit then runs the same
    routine (that is the point: the telemetry difference is the drift).

    Args
    ----
    robot : Robot
        The robot to excite. Must be a fixed-base form (DOFs ↔ actuated
        joints 1:1).
    duration_s : float
        Window length, s. Default 1.6 (enough for the drift recovery).
    sample_rate_hz : float
        Telemetry rate, Hz. Default 100.
    sim_dt : float
        Inner integration step for the simulated path, s. Default 2 ms.
    q_hold : array-like | None
        Hold pose override; :func:`default_hold_pose` by default.
    omega, zeta : float
        Hold-loop natural frequency (rad/s) and damping ratio.
    accel_target : float
        Per-joint excitation acceleration target, rad/s².
    displacement_fraction : float
        Cap on the quasi-static excitation displacement as a fraction of
        each joint's limit margin.
    torque_ceiling : float
        Excitation amplitude ceiling, N·m.

    Returns
    -------
    HoldExcitation

    Raises
    ------
    JaxterityError
        If the robot has a floating base (the hold design is fixed-base),
        or the sample interval is not a multiple of ``sim_dt``.
    """
    names = _actuated_joint_names(robot)
    if any(j.joint_type is JointType.FLOATING for j in robot.joints):
        raise JaxterityError(
            "design_hold_excitation: the hold-and-wiggle routine is a "
            "fixed-base design (the hold loop has no balance controller). "
            "Use the fixed-base robot form (e.g. zoo.load('unitree_g1'))."
        )
    substeps = round(1.0 / (sample_rate_hz * sim_dt))
    if abs(substeps * sample_rate_hz * sim_dt - 1.0) > 1e-9 or substeps < 1:
        raise JaxterityError(
            f"design_hold_excitation: 1/sample_rate_hz must be an integer "
            f"multiple of sim_dt; got sample_rate_hz={sample_rate_hz}, "
            f"sim_dt={sim_dt}."
        )
    n = len(names)
    q_hold_arr = (
        default_hold_pose(robot) if q_hold is None else np.asarray(q_hold, float)
    )
    inertia = _mass_diagonal(robot, q_hold_arr)
    lower = np.asarray(
        [
            j.limits.lower
            for j in robot.joints
            if j.joint_type not in (JointType.FIXED, JointType.FLOATING)
        ]
    )
    upper = np.asarray(
        [
            j.limits.upper
            for j in robot.joints
            if j.joint_type not in (JointType.FIXED, JointType.FLOATING)
        ]
    )
    margin = np.minimum(q_hold_arr - lower, upper - q_hold_arr)
    accel = np.minimum(accel_target, displacement_fraction * margin * omega**2)
    amplitude = np.minimum(inertia * accel, torque_ceiling)
    index = np.arange(n)
    return HoldExcitation(
        joint_names=names,
        q_hold=tuple(float(v) for v in q_hold_arr),
        kp=tuple(float(v) for v in inertia * omega**2),
        kd=tuple(float(v) for v in 2.0 * zeta * omega * inertia),
        amplitude=tuple(float(v) for v in amplitude),
        frequency_hz=tuple(float(v) for v in 1.0 + 0.13 * index),
        phase=tuple(float(v) for v in 0.7 * index),
        duration_s=float(duration_s),
        sample_rate_hz=float(sample_rate_hz),
        sim_dt=float(sim_dt),
    )


def simulate_hold_window(
    excitation: HoldExcitation,
    plant: Robot,
    *,
    K_t_scale=None,
) -> JointTelemetry:
    """Run the routine on a (simulated) plant and collect the window.

    The plant is the *true* unit — typically a drifted or
    manufacturing-sampled variant of the robot the routine was designed
    for. Integration is fixed-step RK4 with the hold loop and multi-sine
    evaluated continuously; telemetry records the instantaneous total
    commanded torque at each sample instant.

    Args
    ----
    excitation : HoldExcitation
        The designed routine.
    plant : Robot
        The true unit's body model (fixed base, same joint set).
    K_t_scale : array-like | None
        The plant's true per-servo actuation scale: delivered torque =
        ``K_t_scale · commanded``. ``None`` means nominal.

    Returns
    -------
    JointTelemetry
        ``(time, q, qd, tau_commanded)`` named by joint, ready for
        :func:`jaxterity.calibrate.refit_drift_parameters`.

    Raises
    ------
    JaxterityError
        If the plant's joints don't match the design, the rollout goes
        non-finite, or any joint comes within the safety margin of a limit
        (limit impacts contaminate equation-error windows — ADR-017).
    """
    plant_names = _actuated_joint_names(plant)
    if plant_names != excitation.joint_names:
        raise JaxterityError(
            "simulate_hold_window: the plant's actuated joints do not match "
            "the excitation design (same robot family required). Designed "
            f"for {len(excitation.joint_names)} joints, plant has "
            f"{len(plant_names)}."
        )
    n = len(plant_names)
    f = plant.functional_dynamics()
    dt = excitation.sim_dt
    substeps = round(1.0 / (excitation.sample_rate_hz * dt))
    n_samples = excitation.n_samples
    t_samples = np.arange(n_samples) / excitation.sample_rate_hz

    q_hold = jnp.asarray(excitation.q_hold)
    kp = jnp.asarray(excitation.kp)
    kd = jnp.asarray(excitation.kd)
    amplitude = jnp.asarray(excitation.amplitude)
    omega_exc = 2.0 * jnp.pi * jnp.asarray(excitation.frequency_hz)
    phase = jnp.asarray(excitation.phase)
    kt = None if K_t_scale is None else jnp.asarray(K_t_scale, dtype=float)

    def command(s, t):
        wave = amplitude * jnp.sin(omega_exc * t + phase)
        return kp * (q_hold - s[:n]) - kd * s[n:] + wave

    def rk4(state, t):
        def dstate(s, t_):
            tau = command(s, t_)
            return f(s, tau if kt is None else kt * tau)

        k1 = dstate(state, t)
        k2 = dstate(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dstate(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dstate(state + dt * k3, t + dt)
        return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    @jax.jit
    def run(state0):
        def sample_step(s, t0):
            tau_c = command(s, t0)

            def sub(carry, _):
                s_, t_ = carry
                return (rk4(s_, t_), t_ + dt), None

            (s_next, _), _ = jax.lax.scan(sub, (s, t0), None, length=substeps)
            return s_next, (s, tau_c)

        _, out = jax.lax.scan(sample_step, state0, jnp.asarray(t_samples))
        return out

    state0 = jnp.concatenate([q_hold, jnp.zeros(n)])
    states, tau_cmd = (np.asarray(a) for a in run(state0))
    if not np.all(np.isfinite(states)):
        raise JaxterityError(
            "simulate_hold_window: the rollout went non-finite — the plant "
            "is too far from the design model for the hold loop (re-design "
            "the excitation on a model closer to this unit)."
        )
    lower = np.asarray(
        [
            j.limits.lower
            for j in plant.joints
            if j.joint_type not in (JointType.FIXED, JointType.FLOATING)
        ]
    )
    upper = np.asarray(
        [
            j.limits.upper
            for j in plant.joints
            if j.joint_type not in (JointType.FIXED, JointType.FLOATING)
        ]
    )
    q_all = states[:, :n]
    if np.any(q_all <= lower[None, :] + _LIMIT_MARGIN) or np.any(
        q_all >= upper[None, :] - _LIMIT_MARGIN
    ):
        worst = int(
            np.argmin(
                np.minimum(
                    (q_all - lower[None, :]).min(axis=0),
                    (upper[None, :] - q_all).min(axis=0),
                )
            )
        )
        raise JaxterityError(
            f"simulate_hold_window: joint {plant_names[worst]!r} came within "
            f"{_LIMIT_MARGIN} rad of a limit during the window — equation "
            "error through limit impacts is non-smooth (ADR-017). Reduce the "
            "excitation (accel_target / displacement_fraction) or move the "
            "hold pose."
        )
    return JointTelemetry(
        time=t_samples,
        sample_rate_hz=excitation.sample_rate_hz,
        joint_names=plant_names,
        q=q_all,
        qd=states[:, n:],
        tau=tau_cmd,
    )


__all__ = ["HoldExcitation", "design_hold_excitation", "simulate_hold_window"]
