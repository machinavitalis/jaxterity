# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Unitree G1 joystick walking controller.

A pretrained MuJoCo Playground locomotion policy, vendored and reimplemented
in **pure JAX** so it composes with the rest of the stack (``jit``/``vmap``,
parameter overrides for drift studies). Three vendored artifacts (all in
``jaxterity/zoo/unitree_g1/``, produced by ``scripts/vendor_g1_walk_scene.py``
at a pinned upstream commit, Apache-2.0):

- ``g1_walk.xml`` — the mesh-free locomotion scene the policy was trained
  against: Euler @ 2 ms, position actuators (kp 75 / 20 / 2 by joint class),
  per-class armature/damping/frictionloss, two foot collision boxes + floor +
  explicit contact pairs, the pelvis IMU velocimeter/gyro, and the ``home`` /
  ``knees_bent`` keyframes.
- ``g1_policy.npz`` — the policy MLP: observation normalization followed by
  three swish hidden layers (512/256/128) and a tanh action head (29 outputs).
- ``g1_policy_golden.npz`` — onnxruntime input/output pairs for parity tests.

The control contract (matching upstream ``play_g1_joystick.py``): every 20 ms
build the 103-dim observation

``[linvel(3), gyro(3), gravity(3), command(3), Δjoint_pos(29), joint_vel(29),
last_action(29), phase(4)]``

(IMU-frame linear velocity and angular rate; gravity direction in the IMU
frame; joint positions relative to the ``knees_bent`` pose; the previous
action; gait phase as cos/sin of two legs' clocks at 1.5 Hz), run the MLP,
and apply ``ctrl = action * 0.5 + knees_bent_pose`` to the position actuators,
then step the physics 10 × 2 ms.

``command`` is the joystick: ``[v_x, v_y, ω_z]`` in m/s and rad/s (the policy
was trained over roughly ±1.5 m/s forward, ±0.8 m/s lateral, ±2π rad/s yaw).

This controller is the *task layer* for the runtime-demo work: the
inverse-dynamics / sysid stack monitors the same rollouts it produces, and
``rollout(model=...)`` lets a caller run the identical controller on a
perturbed (drifted) plant.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..errors import JaxterityError

if TYPE_CHECKING:  # pragma: no cover - import-time only for annotations
    import mujoco

#: Vendored locomotion scene + policy (see module docstring).
WALK_MJCF_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "zoo/unitree_g1/g1_walk.xml"
)
POLICY_PATH = WALK_MJCF_PATH.with_name("g1_policy.npz")
GOLDEN_PATH = WALK_MJCF_PATH.with_name("g1_policy_golden.npz")
SOURCE = (
    "https://github.com/google-deepmind/mujoco_playground "
    "(g1_mjx_feetonly.xml + experimental/sim2sim/onnx/g1_policy.onnx)"
)
LICENSE = "Apache-2.0"

#: Control contract constants (upstream ``play_g1_joystick.py``).
OBS_DIM = 103
N_ACTUATED = 29
CTRL_DT = 0.02
SIM_DT = 0.002
N_SUBSTEPS = 10
ACTION_SCALE = 0.5
GAIT_FREQUENCY_HZ = 1.5


def _lazy_mjx():
    """Import mujoco/mjx lazily (heavy, optional at import time)."""
    import mujoco
    from mujoco import mjx

    return mujoco, mjx


def load_walk_model() -> "mujoco.MjModel":
    """Load the vendored G1 locomotion scene as a ``mujoco.MjModel``.

    Returns
    -------
    mujoco.MjModel
        The mesh-free scene the vendored policy was trained against
        (``nq=36``, ``nv=35``, ``nu=29``; floor + two foot boxes).
    """
    mujoco, _ = _lazy_mjx()
    return mujoco.MjModel.from_xml_path(str(WALK_MJCF_PATH))


class G1JoystickPolicy(eqx.Module):
    """The vendored walking policy MLP (pure JAX, float32).

    Layers: input normalization ``(obs - mean) * recip_std``, three
    swish-activated hidden layers, and a tanh head producing 29 actions in
    ``[-1, 1]`` (position-target offsets, scaled by :data:`ACTION_SCALE` at
    the controller).
    """

    obs_mean: jax.Array
    obs_recip_std: jax.Array
    w0: jax.Array
    b0: jax.Array
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w_out: jax.Array
    b_out: jax.Array

    @classmethod
    def load(cls, path: pathlib.Path | str = POLICY_PATH) -> "G1JoystickPolicy":
        """Load the vendored weights.

        Args
        ----
        path : pathlib.Path | str
            ``.npz`` produced by ``scripts/vendor_g1_walk_scene.py``.

        Returns
        -------
        G1JoystickPolicy
            The policy, weights as float32 JAX arrays.
        """
        with np.load(path) as archive:
            leaves = {key: jnp.asarray(archive[key]) for key in archive.files}
        return cls(**leaves)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Map one observation to one action (deterministic).

        Args
        ----
        obs : jax.Array, shape ``(103,)``
            The observation (see the module docstring for the layout). Cast
            to float32 internally — the network is float32.

        Returns
        -------
        jax.Array, shape ``(29,)``
            Actions in ``[-1, 1]`` (float32).
        """
        x = (obs.astype(jnp.float32) - self.obs_mean) * self.obs_recip_std
        x = jax.nn.swish(x @ self.w0 + self.b0)
        x = jax.nn.swish(x @ self.w1 + self.b1)
        x = jax.nn.swish(x @ self.w2 + self.b2)
        return jnp.tanh(x @ self.w_out + self.b_out)


class WalkCarry(NamedTuple):
    """The controller's per-tick state (a scan carry).

    Fields
    ------
    data : mjx.Data
        Physics state (sensors fresh — ``mjx.forward`` has run on it).
    phase : jax.Array, shape ``(2,)``
        Per-leg gait clocks, rad, wrapped to ``(-π, π]``.
    last_action : jax.Array, shape ``(29,)``
        The previous policy output (part of the observation).
    """

    data: Any
    phase: jax.Array
    last_action: jax.Array


class WalkRollout(NamedTuple):
    """Trajectories returned by :meth:`G1JoystickController.rollout`.

    Fields
    ------
    qpos : jax.Array, shape ``(n_steps, nq)``
        Joint configuration after each control tick (``nq = 36``:
        ``[x y z, quat(4), 29 joints]``).
    qvel : jax.Array, shape ``(n_steps, nv)``
        Generalized velocity after each tick (``nv = 35``).
    action : jax.Array, shape ``(n_steps, 29)``
        The policy action applied over each tick.
    ctrl : jax.Array, shape ``(n_steps, 29)``
        The position-actuator targets applied over each tick, rad.
    actuator_force : jax.Array, shape ``(n_steps, 29)``
        Actuator torques at the end of each tick, N·m — the telemetry channel
        the residual-monitor work consumes.
    """

    qpos: jax.Array
    qvel: jax.Array
    action: jax.Array
    ctrl: jax.Array
    actuator_force: jax.Array


class G1JoystickController:
    """The pretrained joystick walking controller, bound to the vendored scene.

    Builds the MJX model once at construction; :meth:`rollout` JIT-compiles a
    scan over control ticks (policy + 10 physics substeps each). Pass
    ``model=`` to run the same controller on a perturbed plant (e.g. drifted
    friction / actuator gains) without touching the policy.

    Args
    ----
    policy : G1JoystickPolicy | None
        The walking policy; the vendored weights by default.
    model : mujoco.MjModel | None
        The scene; the vendored ``g1_walk.xml`` by default. Must keep the
        trained topology (``nu = 29``, the pelvis IMU sensors, the
        ``knees_bent`` keyframe).

    Raises
    ------
    JaxterityError
        If the model is missing the sensors/keyframe the control contract
        needs.
    """

    def __init__(
        self,
        policy: G1JoystickPolicy | None = None,
        model: "mujoco.MjModel | None" = None,
    ):
        mujoco, mjx = _lazy_mjx()
        self.policy = policy if policy is not None else G1JoystickPolicy.load()
        self.mj_model = model if model is not None else load_walk_model()
        if int(self.mj_model.nu) != N_ACTUATED:
            raise JaxterityError(
                f"G1JoystickController needs the vendored locomotion scene "
                f"(nu == {N_ACTUATED} position actuators), got nu == "
                f"{int(self.mj_model.nu)}. Load it via load_walk_model()."
            )
        try:
            key = self.mj_model.keyframe("knees_bent")
            self._linvel_adr = int(self.mj_model.sensor("local_linvel_pelvis").adr[0])
            self._gyro_adr = int(self.mj_model.sensor("gyro_pelvis").adr[0])
            self._imu_site = int(self.mj_model.site("imu_in_pelvis").id)
        except KeyError as e:
            raise JaxterityError(
                "G1JoystickController model is missing the control contract "
                "(knees_bent keyframe, pelvis IMU velocimeter/gyro, "
                "imu_in_pelvis site). Use the vendored g1_walk.xml or a "
                "parameter-perturbed copy of it."
            ) from e
        self._init_qpos = jnp.asarray(key.qpos)
        self._init_ctrl = jnp.asarray(key.ctrl)
        self._default_angles = jnp.asarray(key.qpos[7:])
        self._phase_dt = 2.0 * jnp.pi * GAIT_FREQUENCY_HZ * CTRL_DT
        self.mjx_model = mjx.put_model(self.mj_model)
        self._mjx = mjx
        # Compiled rollout scans keyed by n_steps. The command and the plant
        # model are *traced arguments*, so re-rolling with a new joystick
        # command — or a parameter-perturbed (drifted) model of the same
        # structure — reuses the compiled kernel instead of recompiling.
        self._compiled_rollouts: dict[int, Any] = {}

    # --- control contract -------------------------------------------------

    def observation(
        self, data: Any, phase: jax.Array, last_action: jax.Array, command: jax.Array
    ) -> jax.Array:
        """Build the 103-dim observation from a (forwarded) physics state.

        Args
        ----
        data : mjx.Data
            Physics state with fresh sensors/site frames.
        phase : jax.Array, shape ``(2,)``
            Gait clocks, rad.
        last_action : jax.Array, shape ``(29,)``
            Previous policy output.
        command : jax.Array, shape ``(3,)``
            Joystick ``[v_x, v_y, ω_z]``, m/s and rad/s.

        Returns
        -------
        jax.Array, shape ``(103,)``
            The observation, float32.
        """
        linvel = jax.lax.dynamic_slice_in_dim(data.sensordata, self._linvel_adr, 3)
        gyro = jax.lax.dynamic_slice_in_dim(data.sensordata, self._gyro_adr, 3)
        imu_rotation = jnp.reshape(data.site_xmat[self._imu_site], (3, 3))
        gravity = imu_rotation.T @ jnp.array([0.0, 0.0, -1.0])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]
        gait = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])
        return jnp.concatenate(
            [
                linvel,
                gyro,
                gravity,
                jnp.asarray(command, dtype=float),
                joint_angles,
                joint_velocities,
                last_action,
                gait,
            ]
        ).astype(jnp.float32)

    def init_carry(self, model: Any | None = None) -> WalkCarry:
        """The initial carry: the ``knees_bent`` keyframe, sensors forwarded.

        Args
        ----
        model : mjx.Model | None
            Plant override (defaults to the vendored scene's MJX model).

        Returns
        -------
        WalkCarry
            Ready for :meth:`step` / :meth:`rollout`.
        """
        model = self.mjx_model if model is None else model
        data = self._mjx.make_data(model).replace(
            qpos=self._init_qpos, ctrl=self._init_ctrl
        )
        data = self._mjx.forward(model, data)
        return WalkCarry(
            data=data,
            phase=jnp.array([0.0, jnp.pi]),
            last_action=jnp.zeros(N_ACTUATED, dtype=jnp.float32),
        )

    def step(
        self, carry: WalkCarry, command: jax.Array, model: Any | None = None
    ) -> tuple[WalkCarry, jax.Array]:
        """One control tick: observe → policy → apply ctrl → 10 physics steps.

        Pure (``jit``/``vmap``/``scan``-safe).

        Args
        ----
        carry : WalkCarry
            Controller state.
        command : jax.Array, shape ``(3,)``
            Joystick ``[v_x, v_y, ω_z]``.
        model : mjx.Model | None
            Plant override (defaults to the vendored scene's MJX model).

        Returns
        -------
        (WalkCarry, jax.Array)
            The next carry and the action applied this tick.
        """
        model = self.mjx_model if model is None else model
        obs = self.observation(carry.data, carry.phase, carry.last_action, command)
        action = self.policy(obs)
        ctrl = action * ACTION_SCALE + self._default_angles
        data = carry.data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(
            0, N_SUBSTEPS, lambda _, d: self._mjx.step(model, d), data
        )
        phase = jnp.fmod(carry.phase + self._phase_dt + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return WalkCarry(data=data, phase=phase, last_action=action), action

    def rollout(
        self,
        command,
        n_steps: int,
        model: Any | None = None,
    ) -> WalkRollout:
        """Walk for ``n_steps`` control ticks under a constant joystick command.

        JIT-compiles the full scan once per ``n_steps`` (cached on the
        controller); the command and the plant model are traced arguments, so
        new commands and parameter-perturbed plants reuse the compiled
        kernel. 50 ticks = 1 simulated second.

        Args
        ----
        command : array-like, shape ``(3,)``
            Joystick ``[v_x, v_y, ω_z]``, m/s and rad/s.
        n_steps : int
            Number of 20 ms control ticks.
        model : mjx.Model | None
            Plant override — e.g. the vendored model with drifted
            friction/gain parameters — while the controller (policy,
            keyframe pose, sensors) stays fixed.

        Returns
        -------
        WalkRollout
            Per-tick trajectories (see :class:`WalkRollout`).
        """
        command = jnp.asarray(command, dtype=float)
        mjx_model = self.mjx_model if model is None else model

        if n_steps not in self._compiled_rollouts:

            def run(carry, cmd, plant):
                def tick(c: WalkCarry, _) -> tuple[WalkCarry, WalkRollout]:
                    c, action = self.step(c, cmd, plant)
                    frame = WalkRollout(
                        qpos=c.data.qpos,
                        qvel=c.data.qvel,
                        action=action,
                        ctrl=c.data.ctrl,
                        actuator_force=c.data.actuator_force,
                    )
                    return c, frame

                return jax.lax.scan(tick, carry, None, length=n_steps)

            self._compiled_rollouts[n_steps] = jax.jit(run)

        _, trajectory = self._compiled_rollouts[n_steps](
            self.init_carry(mjx_model), command, mjx_model
        )
        return trajectory


__all__ = [
    "G1JoystickController",
    "G1JoystickPolicy",
    "WalkCarry",
    "WalkRollout",
    "load_walk_model",
    "WALK_MJCF_PATH",
    "POLICY_PATH",
    "GOLDEN_PATH",
]
