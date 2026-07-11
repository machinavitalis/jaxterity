# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Feedforward-dominant precision tracking.

The task where *model accuracy is the outcome*: track a smooth joint
trajectory with inverse-dynamics feedforward torque computed from the
unit's **self-model**, plus only weak position feedback. With high-gain
feedback any model tracks; with weak feedback the feedforward has to be
right — stale friction or a drooped torque constant shows up directly as
tracking error. This is the scoring task for calibration/drift studies: the
same trajectory, the same plant, two models — the accurate model passes the
tolerance band, the stale one misses it.

Mechanics:

- The trajectory is analytic (multi-sine on selected *task joints* around a
  hold pose, amplitudes capped by limit margins), so the desired state and
  acceleration are exact at any time — no differencing.
- Feedforward is exact inverse dynamics from the model: with MJX forward
  dynamics affine in the applied torque, ``τ_ff = M(q_des) · (qdd_des −
  qacc(q_des, qd_des, τ=0))`` captures gravity, Coriolis, damping, *and the
  model's friction* in two operations. The commanded torque divides by the
  model's actuation scale (``K_t_scale``); the plant multiplies by its true
  one — an accurate scale cancels, a stale one injects a gain error.
- Task joints get weak feedback (default ω = 3 rad/s); the remaining joints
  get a stiff hold (the humanoid's waist is gravitationally unstable — they
  are scaffolding, not part of the score).

Task-joint selection rule (learned the hard way): pick joints whose
*dynamic* torque (inertial + gravity) dominates their Coulomb friction. A
light distal joint (elbow, wrist) is friction-dominated — its feedforward is
at the mercy of the stiction discontinuity at every velocity reversal, so
tracking error ratchets there for *any* model, accurate or not, and the
task stops discriminating. Shoulder-class joints on the G1 discriminate
cleanly.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..errors import JaxterityError
from ..robot import JointType, Robot
from ..twin import Twin

#: Default closed-loop frequencies (rad/s): weak on task joints (the point),
#: stiff on the scaffolding joints.
_TASK_OMEGA_DEFAULT = 3.0
_HOLD_OMEGA_DEFAULT = 12.0
_ZETA = 0.7


@dataclasses.dataclass(frozen=True)
class TrackingTrajectory:
    """An analytic multi-sine joint trajectory (static design data).

    Build with :func:`design_tracking_trajectory`. Desired position,
    velocity, and acceleration are exact closed forms at any time.

    Args
    ----
    joint_names : tuple[str, ...]
        All actuated joints, DOF order.
    task_joints : tuple[str, ...]
        The scored joints (the multi-sine rides on these).
    q_rest : tuple[float, ...]
        The rest/hold pose, rad.
    amplitude, frequency_hz, phase : tuple[float, ...]
        Per-joint multi-sine parameters (zero amplitude off-task).
    duration_s : float
        Trajectory length, s.
    """

    joint_names: tuple[str, ...]
    task_joints: tuple[str, ...]
    q_rest: tuple[float, ...]
    amplitude: tuple[float, ...]
    frequency_hz: tuple[float, ...]
    phase: tuple[float, ...]
    duration_s: float

    def desired(self, t):
        """``(q_des, qd_des, qdd_des)`` at time ``t`` (closed form).

        The sine envelope starts at zero displacement *and* zero velocity
        (``sin²``-ramped onset over the first quarter period is avoided in
        favour of a pure ``1 − cos`` start, which has both q and qd zero at
        t = 0).
        """
        amplitude = jnp.asarray(self.amplitude)
        omega = 2.0 * jnp.pi * jnp.asarray(self.frequency_hz)
        phase = jnp.asarray(self.phase)
        rest = jnp.asarray(self.q_rest)
        # x(t) = A·(1 − cos(ωt + φ·0)) … keep φ for de-phasing but apply it
        # inside the cosine *difference* so x(0) = ẋ(0) = 0 for every joint.
        angle = omega * t
        q = rest + amplitude * (1.0 - jnp.cos(angle)) * jnp.cos(phase)
        qd = amplitude * omega * jnp.sin(angle) * jnp.cos(phase)
        qdd = amplitude * omega**2 * jnp.cos(angle) * jnp.cos(phase)
        return q, qd, qdd


class TrackingReport(NamedTuple):
    """The precision-task verdict.

    Fields
    ------
    time : numpy.ndarray, shape ``(n,)``
        Sample times, s.
    q : numpy.ndarray, shape ``(n, nv)``
        Realized joint positions, rad.
    q_des : numpy.ndarray, shape ``(n, nv)``
        Desired joint positions, rad.
    error_rms, error_max : float
        Tracking error over the *task joints* (settle time excluded), rad.
    tolerance : float
        The pass band, rad.
    passed : bool
        ``error_max < tolerance``.
    """

    time: np.ndarray
    q: np.ndarray
    q_des: np.ndarray
    error_rms: float
    error_max: float
    tolerance: float
    passed: bool


def _actuated(robot: Robot):
    return tuple(
        j
        for j in robot.joints
        if j.joint_type not in (JointType.FIXED, JointType.FLOATING)
    )


def design_tracking_trajectory(
    robot: Robot,
    task_joints: tuple[str, ...],
    *,
    duration_s: float = 2.0,
    amplitude_fraction: float = 0.25,
    frequency_hz: float = 0.7,
    q_rest=None,
) -> TrackingTrajectory:
    """Design the precision trajectory on a robot's task joints.

    Args
    ----
    robot : Robot
        The (fixed-base) robot.
    task_joints : tuple[str, ...]
        The joints to swing and score (e.g. one arm).
    duration_s : float
        Length, s.
    amplitude_fraction : float
        Swing amplitude as a fraction of each task joint's limit margin
        from the rest pose. (The peak excursion is 2× the amplitude — the
        ``1 − cos`` waveform — so keep this ≤ 0.4.)
    frequency_hz : float
        Base swing frequency; successive task joints get small
        incommensurate offsets. Raise it for a feedforward-harder task (the
        "fast swing" stress test).
    q_rest : array-like | None
        Rest pose; :func:`jaxterity.calibrate.default_hold_pose` by default.

    Returns
    -------
    TrackingTrajectory

    Raises
    ------
    JaxterityError
        On an unknown task joint or a floating-base robot.
    """
    from ..calibrate.excitation import default_hold_pose

    joints = _actuated(robot)
    names = tuple(j.name for j in joints)
    if len(names) != len(set(names)) or any(
        j.joint_type is JointType.FLOATING for j in robot.joints
    ):
        raise JaxterityError(
            "design_tracking_trajectory needs the fixed-base robot form."
        )
    unknown = sorted(set(task_joints) - set(names))
    if unknown:
        raise JaxterityError(
            f"design_tracking_trajectory: unknown task joints {unknown}. "
            f"Actuated joints: {sorted(names)}."
        )
    rest = default_hold_pose(robot) if q_rest is None else np.asarray(q_rest, float)
    amplitude = np.zeros(len(names))
    frequency = np.full(len(names), frequency_hz)
    phase = np.zeros(len(names))
    for rank, task in enumerate(task_joints):
        i = names.index(task)
        joint = joints[i]
        margin = min(rest[i] - joint.limits.lower, joint.limits.upper - rest[i])
        # Peak excursion of the 1−cos waveform is 2·A.
        amplitude[i] = amplitude_fraction * margin / 2.0
        frequency[i] = frequency_hz * (1.0 + 0.11 * rank)
        phase[i] = 0.0
    return TrackingTrajectory(
        joint_names=names,
        task_joints=tuple(task_joints),
        q_rest=tuple(float(v) for v in rest),
        amplitude=tuple(float(v) for v in amplitude),
        frequency_hz=tuple(float(v) for v in frequency),
        phase=tuple(float(v) for v in phase),
        duration_s=float(duration_s),
    )


def feedforward_tracking(
    model: Twin,
    plant: Robot,
    trajectory: TrackingTrajectory,
    *,
    plant_K_t_scale=None,
    tolerance: float = 0.05,
    task_omega: float = _TASK_OMEGA_DEFAULT,
    hold_omega: float = _HOLD_OMEGA_DEFAULT,
    sim_dt: float = 0.002,
    sample_rate_hz: float = 100.0,
    settle_s: float = 0.2,
) -> TrackingReport:
    """Run the precision task: the model's feedforward against the plant.

    Args
    ----
    model : Twin
        The unit's *self-model* — its robot parameters and ``K_t_scale``
        produce the feedforward. Accurate model ⇒ pass; stale ⇒ fail.
    plant : Robot
        The true unit's body (drifted / manufacturing-sampled).
    trajectory : TrackingTrajectory
        The designed task (:func:`design_tracking_trajectory`).
    plant_K_t_scale : array-like | None
        The plant's true actuation scale (delivered = scale × commanded).
    tolerance : float
        Pass band on the task joints' max error, rad.
    task_omega, hold_omega : float
        Feedback frequencies (rad/s): weak on task joints, stiff hold on
        the scaffolding joints.
    sim_dt : float
        Plant integration step, s.
    sample_rate_hz : float
        Report sampling rate, Hz.
    settle_s : float
        Initial transient excluded from the score, s.

    Returns
    -------
    TrackingReport

    Raises
    ------
    JaxterityError
        If model/plant/trajectory joint sets disagree, or the rollout goes
        non-finite (the model was wrong enough to destabilize even the
        held joints — report that loudly, not as a score).
    """
    import mujoco
    from mujoco import mjx

    names = trajectory.joint_names
    if model.servo_names != names or tuple(j.name for j in _actuated(plant)) != names:
        raise JaxterityError(
            "feedforward_tracking: model twin, plant, and trajectory must "
            "share the same actuated joint set, in the same DOF order."
        )
    n = len(names)
    task_index = np.asarray([names.index(j) for j in trajectory.task_joints])

    # --- the model's inverse dynamics (exact, two operations) -------------
    model_mj = mujoco.MjModel.from_xml_string(model.robot.to_mjcf())
    model_mjx = mjx.put_model(model_mj)
    kt_model = jnp.ones(n) if model.K_t_scale is None else jnp.asarray(model.K_t_scale)

    def tau_feedforward(t):
        q_des, qd_des, qdd_des = trajectory.desired(t)
        data = mjx.make_data(model_mjx).replace(qpos=q_des, qvel=qd_des)
        data = mjx.forward(model_mjx, data)
        mass = mjx.full_m(model_mjx, data)
        return mass @ (qdd_des - data.qacc)

    # --- gains: weak on task joints, stiff scaffolding elsewhere ----------
    inertia_data = mujoco.MjData(model_mj)
    inertia_data.qpos[:] = np.asarray(trajectory.q_rest)
    mujoco.mj_forward(model_mj, inertia_data)
    full = np.zeros((n, n))
    mujoco.mj_fullM(model_mj, full, inertia_data.qM)
    inertia = np.diag(full)
    omega_vec = np.full(n, hold_omega)
    omega_vec[task_index] = task_omega
    kp = jnp.asarray(inertia * omega_vec**2)
    kd = jnp.asarray(2.0 * _ZETA * omega_vec * inertia)

    # --- the plant rollout -------------------------------------------------
    f_plant = plant.functional_dynamics()
    kt_plant = None if plant_K_t_scale is None else jnp.asarray(plant_K_t_scale, float)
    substeps = round(1.0 / (sample_rate_hz * sim_dt))
    n_samples = int(round(trajectory.duration_s * sample_rate_hz))
    t_samples = np.arange(n_samples) / sample_rate_hz

    def command(s, t):
        q_des, qd_des, _ = trajectory.desired(t)
        feedback = kp * (q_des - s[:n]) + kd * (qd_des - s[n:])
        return tau_feedforward(t) / kt_model + feedback

    def rk4(state, t):
        def dstate(s, t_):
            tau = command(s, t_)
            return f_plant(s, tau if kt_plant is None else kt_plant * tau)

        k1 = dstate(state, t)
        k2 = dstate(state + 0.5 * sim_dt * k1, t + 0.5 * sim_dt)
        k3 = dstate(state + 0.5 * sim_dt * k2, t + 0.5 * sim_dt)
        k4 = dstate(state + sim_dt * k3, t + sim_dt)
        return state + sim_dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    @jax.jit
    def run(state0):
        def sample_step(s, t0):
            def sub(carry, _):
                s_, t_ = carry
                return (rk4(s_, t_), t_ + sim_dt), None

            (s_next, _), _ = jax.lax.scan(sub, (s, t0), None, length=substeps)
            return s_next, s_next

        return jax.lax.scan(sample_step, state0, jnp.asarray(t_samples))[1]

    state0 = jnp.concatenate([jnp.asarray(trajectory.q_rest), jnp.zeros(n)])
    states = np.asarray(run(state0))
    if not np.all(np.isfinite(states)):
        raise JaxterityError(
            "feedforward_tracking: the rollout went non-finite — the model "
            "is too far from the plant for even the held joints. This is a "
            "diagnostic failure, not a score."
        )
    # Desired positions at the *post-step* sample instants.
    q_des = np.stack(
        [np.asarray(trajectory.desired(t)[0]) for t in t_samples + 1.0 / sample_rate_hz]
    )
    q = states[:, :n]
    keep = t_samples >= settle_s
    error = (q - q_des)[np.ix_(keep, task_index)]
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    error_max = float(np.max(np.abs(error)))
    return TrackingReport(
        time=t_samples + 1.0 / sample_rate_hz,
        q=q,
        q_des=q_des,
        error_rms=error_rms,
        error_max=error_max,
        tolerance=float(tolerance),
        passed=bool(error_max < tolerance),
    )


__all__ = [
    "TrackingTrajectory",
    "TrackingReport",
    "design_tracking_trajectory",
    "feedforward_tracking",
]
