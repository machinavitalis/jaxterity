# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Raw actuator interface — ``policies.S4``.

The lowest layer: read joint state, command joint torques, by name. Wraps a
``Robot`` + its compiled ``_RobotSystem``. :meth:`step` integrates the robot's
real MJX dynamics; that is the spec-correct path for end-to-end policy
evaluation. **Be aware** that the S2/S3/S1 layers (and the eval
harness) do *not* call :meth:`step` today — they integrate the WBC's joint
accelerations with their own Euler step and mirror the result into S4 via
:meth:`set_state` (the "math facade" mode documented below). So the
"same code path the simulator uses" property holds for a caller who drives
:meth:`apply_torque` + :meth:`step` directly, not for the layered stack as it
currently stands. The interface is intentionally minimal and LLM-friendly:
each function has a single clear purpose and a docstring an LLM can read cold.

This is the "thin layer over the Robot diagram" the higher S3/S2/S1 layers
build on. It does no kinematics, no planning, no control synthesis — those
are S3+.

S4 supports three use modes:

* **Torque + step** — :meth:`apply_torque` caches the commanded torque and
  :meth:`step` integrates the underlying MJX dynamics for ``dt`` seconds via
  ``jaxonomy.simulate`` (RK4). Spec-correct, but rebuilds its context each call
  (~seconds) — fine for a one-off rollout, not a tight loop.
* **Acceleration step** — :meth:`step_with_acceleration` realises a
  desired ``qdd`` through **real MJX**: inverse dynamics → generalized force →
  a cached, jitted RK4 forward (≈µs/call). This is what the S2 skills now use,
  so the layered stack integrates real physics rather than Euler-on-WBC-qdd.
* **Math facade** — :meth:`set_state` overwrites the cached state without
  running MJX. Now used only to *seed* an initial condition (e.g. the eval
  harness deflecting the cartpole pole before a policy runs); earlier the
  S2/S3 stack integrated WBC qdd by Euler and pushed the result here.

Prior versions built the diagram in ``__init__`` but never called it. The
:meth:`step` method makes the diagram live.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..actuators import IdealTorqueActuator
from ..robot import Robot

#: Process-level cache of compiled MJX steppers, keyed by ``attestation_handle``
#: Two S4s of the same robot have identical dynamics, so they share
#: one jitted ``step`` instead of each paying the ~2 s XLA compile. Bounded by the
#: number of distinct robots used in a process.
_STEPPER_CACHE: dict[str, object] = {}


class S4:
    """Raw actuator + encoder interface for a :class:`Robot`.

    Construction wraps a Robot in a diagram with an :class:`IdealTorqueActuator`
    wired to the robot's ``tau`` input port, plus a per-call mutable cache of
    ``(t, q, qd, tau)``. The class is *not* an equinox.Module — it holds
    runtime state the higher layers thread between ticks. For sim-free
    composition use the underlying Robot.

    Args
    ----
    robot : Robot
        The robot to drive.
    q0, qd0 : array-like or None
        Initial joint position / velocity. Default zeros.
    """

    def __init__(self, robot: Robot, *, q0=None, qd0=None):
        import jaxonomy
        import mujoco

        self.robot = robot
        # Joint → DOF index from the compiled MuJoCo model, in MJX DOF order
        # FIXED URDF joints carry no MuJoCo joint and are excluded, and
        # ``robot.joints`` is in leaf→root order while MJX numbers DOFs root→leaf
        # — so the model is the only authoritative source. (Assumes 1-DOF joints
        # on a fixed base: nq == nv == n; floating base is a follow-up.)
        mj_model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
        names_in_dof_order = [""] * int(mj_model.nv)
        for jid in range(int(mj_model.njnt)):
            dofadr = int(mj_model.jnt_dofadr[jid])
            names_in_dof_order[dofadr] = mj_model.joint(jid).name
        self._joint_names = tuple(names_in_dof_order)
        self._dof_index = {name: i for i, name in enumerate(self._joint_names)}
        n = int(mj_model.nv)
        self._n = n
        self._q = jnp.zeros(n) if q0 is None else jnp.asarray(q0, dtype=float)
        self._qd = jnp.zeros(n) if qd0 is None else jnp.asarray(qd0, dtype=float)
        self._tau = jnp.zeros(n)
        self._t = 0.0

        # Build the diagram once; reuse the context per step() call. The
        # IdealTorqueActuator exposes ``tau`` as a mutable dynamic parameter
        # so we can update the commanded torque between integration steps
        # without rebuilding the diagram.
        system = robot.build_system(q0=q0, qd0=qd0, actuation=True)
        self._tau_source = IdealTorqueActuator(n_dof=n, name="_s4_tau")
        builder = jaxonomy.DiagramBuilder()
        builder.add(self._tau_source)
        builder.add(system)
        builder.connect(self._tau_source.output_ports[0], system.input_ports[0])
        self._diagram = builder.build()
        self._system = system

    # --- reads ---------------------------------------------------------

    def read_encoder(self, joint_name: str) -> float:
        """Return the current joint position (rad) for ``joint_name``."""
        return float(self._q[self.dof_index(joint_name)])

    def read_velocity(self, joint_name: str) -> float:
        """Return the current joint velocity (rad/s) for ``joint_name``."""
        return float(self._qd[self.dof_index(joint_name)])

    def state(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the current ``(q, qd)`` for all joints."""
        return self._q, self._qd

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Names of every joint, in the order they appear in ``q`` / ``qd``."""
        return self._joint_names

    def dof_index(self, joint_name: str) -> int:
        """Return the ``q`` / ``qd`` array index for ``joint_name``.

        Public accessor for the higher policy layers (S2 / S3 / MCP tools)
        that need to address a specific joint by name. Raises ``KeyError``
        with the available names if the lookup fails.
        """
        if joint_name not in self._dof_index:
            raise KeyError(
                f"S4.dof_index: unknown joint {joint_name!r}. "
                f"Known joints: {list(self._dof_index)}."
            )
        return self._dof_index[joint_name]

    @property
    def time(self) -> float:
        """Current simulation time (seconds since construction or last reset)."""
        return self._t

    # --- writes --------------------------------------------------------

    def apply_torque(self, joint_name: str, tau: float) -> None:
        """Cache the commanded torque (N·m) on ``joint_name`` for the next step."""
        idx = self.dof_index(joint_name)
        self._tau = self._tau.at[idx].set(float(tau))

    def step(self, dt: float) -> None:
        """Integrate the robot's MJX dynamics by ``dt`` seconds.

        Uses the cached torque vector (per :meth:`apply_torque`) as the
        actuator input and updates the cached ``(t, q, qd)`` from the
        simulation result. The integrator is the same JAX-traceable
        forward-mode-friendly RK4 used elsewhere in the package.

        ``dt`` should be modest (1-10 ms is typical); for longer horizons
        call ``step`` repeatedly so the cached torque can be re-commanded
        between sub-steps.
        """
        import jaxonomy
        from jaxonomy import SimulatorOptions

        opts = SimulatorOptions(
            math_backend="jax",
            ode_solver_method="rk4",
            max_minor_step_size=min(float(dt), 5e-3),
            enable_autodiff=False,
        )
        ctx = self._diagram.create_context()
        # Push the cached torque into the actuator block's dynamic parameter.
        tau_ctx = ctx[self._tau_source.system_id]
        tau_ctx = tau_ctx.with_parameter("tau", self._tau)
        ctx = ctx.with_subcontext(self._tau_source.system_id, tau_ctx)
        # Push the cached robot state in.
        robot_ctx = ctx[self._system.system_id]
        robot_ctx = robot_ctx.with_continuous_state(
            jnp.concatenate([self._q, self._qd])
        )
        ctx = ctx.with_subcontext(self._system.system_id, robot_ctx)
        res = jaxonomy.simulate(
            self._diagram, ctx, (self._t, self._t + float(dt)), options=opts
        )
        new_state = res.context[self._system.system_id].continuous_state
        # The state is [q (nq,), qd (nv,)]; for fixed-base robots nq == nv ==
        # _n. Floating-base support is a follow-up.
        nq = self._n
        self._q = new_state[:nq]
        self._qd = new_state[nq : 2 * nq]
        self._t = self._t + float(dt)

    def reset(self, *, q=None, qd=None, t: float = 0.0) -> None:
        """Reset the cached joint state and clock. Useful between episodes."""
        n = self._n
        self._q = jnp.zeros(n) if q is None else jnp.asarray(q, dtype=float)
        self._qd = jnp.zeros(n) if qd is None else jnp.asarray(qd, dtype=float)
        self._tau = jnp.zeros(n)
        self._t = float(t)

    def set_state(self, q, qd) -> None:
        """Set the cached state directly (no MJX integration).

        Use when a higher layer is integrating its own dynamics (e.g. the
        S2 / S3 stack runs Euler on the WBC qdd) and just needs S4
        to mirror the result so its encoders / velocities are consistent.
        For the spec-correct path that runs the Robot's MJX dynamics, use
        :meth:`step` instead.
        """
        self._q = jnp.asarray(q, dtype=float)
        self._qd = jnp.asarray(qd, dtype=float)

    def inverse_dynamics(self, qdd) -> jnp.ndarray:
        """Generalized force ``tau`` (nv,) that achieves acceleration ``qdd`` at
        the current ``(q, qd)``.

        Uses the Robot's MJX inverse dynamics, so ``tau`` accounts for the full
        bias (Coriolis + gravity) **and** passive forces (joint damping).
        Commanding ``tau`` and calling :meth:`step` then reproduces ``qdd`` to
        machine precision (verified ~1e-14). The robot models carry no
        ``<actuator>`` definitions, so every generalized coordinate is
        force-controllable via ``qfrc_applied``; a robot with explicitly limited
        actuators would instead need a WBC dynamics-consistency constraint on the
        unactuated rows (deferred).
        """
        mjx = self._system._mjx
        model = self._system._mjx_model
        data = mjx.make_data(model).replace(
            qpos=self._q, qvel=self._qd, qacc=jnp.asarray(qdd, dtype=float)
        )
        data = mjx.inverse(model, data)
        return data.qfrc_inverse

    def _fast_stepper(self):
        """A cached, jitted ``(q, qd, qdd_des, dt) -> (q, qd)`` step over real MJX
        dynamics: inverse dynamics → generalized force → RK4 forward.

        Built once per *robot* and reused (≈µs/call after the first trace).
        Compiled steppers are cached process-wide by ``attestation_handle``
        (:data:`_STEPPER_CACHE`), so a second S4 of the same robot — a fresh one
        in every test, say — skips the ~2 s XLA compile. This is the tight-loop
        path; :meth:`step` goes through ``jaxonomy.simulate`` and rebuilds its
        context each call (~seconds), fine for a one-off rollout but unusable
        inside the S2 control loop.
        """
        key = self.robot.attestation_handle
        cached = _STEPPER_CACHE.get(key)
        if cached is not None:
            return cached

        import jax

        mjx = self._system._mjx
        model = self._system._mjx_model
        nq, nv = self._system._nq, self._system._nv
        rhs = self.robot.functional_dynamics()  # (state, tau) -> dstate over MJX

        def step(q, qd, qdd_des, dt):
            data = mjx.make_data(model).replace(qpos=q, qvel=qd, qacc=qdd_des)
            tau = mjx.inverse(model, data).qfrc_inverse
            state = jnp.concatenate([q, qd])
            k1 = rhs(state, tau)
            k2 = rhs(state + 0.5 * dt * k1, tau)
            k3 = rhs(state + 0.5 * dt * k2, tau)
            k4 = rhs(state + dt * k3, tau)
            nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return nxt[:nq], nxt[nq : nq + nv]

        stepper = jax.jit(step)
        _STEPPER_CACHE[key] = stepper
        return stepper

    def step_with_acceleration(self, qdd, dt: float) -> None:
        """Realize desired acceleration ``qdd`` over ``dt`` through **real MJX**
        dynamics.

        Converts ``qdd`` to a generalized force (inverse dynamics) and
        RK4-integrates the Robot's MJX dynamics with it. This is the spec-correct
        replacement for the :meth:`set_state` math facade in the S2/S3 stack: the
        layered policies now integrate real physics instead of Euler-on-WBC-qdd.
        Uses the cached fast stepper (:meth:`_fast_stepper`), not the per-call
        ``jaxonomy.simulate`` of :meth:`step`.
        """
        q, qd = self._fast_stepper()(
            self._q, self._qd, jnp.asarray(qdd, dtype=float), float(dt)
        )
        self._q = q
        self._qd = qd
        self._t = self._t + float(dt)


__all__ = ["S4"]
