# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The ``Robot`` abstraction and ``to_diagram``.

``Robot`` is the single source of truth (ADR-004): one object flows from
construction/import through sysid through simulation through Jaxility export.
It is an ``equinox.Module`` PyTree — identifiable parameters are JAX array leaves
on its ``Link``/``Joint`` children (ADR-007); names, joint types, and provenance
are static.

``to_diagram()`` lowers the kinematic tree to MJCF, builds a differentiable MJX
model (ADR-016), and wraps MJX forward dynamics as a Jaxonomy ``LeafSystem`` so
the robot composes with the rest of the engine (``simulate``, ``fit_parameters``,
autodiff). The identifiable leaves bind to Jaxonomy dynamic parameters at this
boundary, so ``jax.grad`` flows into them through ``simulate`` (use
``SimulatorOptions(enable_autodiff=True)`` for reverse mode).

Runtime-threaded dynamic parameters (forward-mode autodiff sees them through
``parameter_jacobian``): link ``mass`` / ``com`` (3-vec → ``body_ipos``) /
``inertia_diag`` (3-vec → ``body_inertia``, the principal moments), and joint
``damping`` / ``friction`` / ``armature``. The off-diagonal inertia tensor and
the inertia-frame rotation (``body_iquat``) still live in the MJCF structure
and are refit by structural rebuild via :meth:`Robot.with_parameters`.
"""

from __future__ import annotations

import enum
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jaxonomy
from jaxonomy.framework.leaf_system import LeafSystem

from ..diagnostics import attestation
from ..errors import JaxterityError
from . import implicit
from .mjcf import to_mjcf
from .structure import Joint, JointType, Link


class CalibrationState(str, enum.Enum):
    """How fully a robot's parameters have been identified from telemetry."""

    UNCALIBRATED = "uncalibrated"
    PARTIAL = "partial"
    CALIBRATED = "calibrated"


def _lazy_mjx():
    """Import mujoco/mjx lazily (heavy, optional at import time)."""
    import mujoco
    from mujoco import mjx

    return mujoco, mjx


def _quat_mul(q, r):
    """Hamilton product of two ``(w, x, y, z)`` quaternions (MuJoCo convention)."""
    w0, x0, y0, z0 = q[0], q[1], q[2], q[3]
    w1, x1, y1, z1 = r[0], r[1], r[2], r[3]
    return jnp.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


class Robot(eqx.Module):
    """A robot model: a kinematic tree of links and joints (ADR-004).

    Args
    ----
    links : tuple[Link, ...]
        The rigid bodies. Exactly one must be the root (no joint's child).
    joints : tuple[Joint, ...]
        The articulations connecting links.
    name : str
        Model name (static).
    source : str
        Provenance — where the model came from (e.g. a URDF path) (static).
    calibration_state : CalibrationState
        Calibration status (static).
    gravity : float
        Gravity magnitude (m/s²), applied along -z (static).
    ground_plane : bool
        Lower an infinite floor plane at ``z = 0`` so links carrying
        :class:`~jaxterity.robot.CollisionGeom` entries make ground contact.
        Static structure — it changes the attestation handle through
        the lowered MJCF. Default False.
    """

    links: tuple[Link, ...]
    joints: tuple[Joint, ...]
    name: str = eqx.field(static=True)
    source: str = eqx.field(static=True, default="constructed")
    calibration_state: CalibrationState = eqx.field(
        static=True, default=CalibrationState.UNCALIBRATED
    )
    gravity: float = eqx.field(static=True, default=9.81)
    ground_plane: bool = eqx.field(static=True, default=False)
    #: Sysid provenance: a tuple of ``(recipe_name, recipe_version,
    #: telemetry_hash)`` records, appended by ``SysidResult.apply_to``. Part of
    #: the attestation handle (ADR-008) so a calibrated robot's identity reflects
    #: how it was fitted.
    provenance: tuple[tuple[str, str, str], ...] = eqx.field(static=True, default=())

    # --- lookup ---------------------------------------------------------

    @property
    def link_map(self) -> dict[str, Link]:
        """Links keyed by name."""
        return {link.name: link for link in self.links}

    @property
    def joint_map(self) -> dict[str, Joint]:
        """Joints keyed by name."""
        return {joint.name: joint for joint in self.joints}

    # --- identifiable parameters ---------------------------------------

    def parameters(self) -> dict[str, object]:
        """All identifiable parameter leaves, keyed by ``"<name>.<field>"``.

        The canonical parameter set the attestation handle hashes. Every leaf is
        fittable via :meth:`with_parameters` (a structural rebuild). The subset
        differentiable at runtime via :func:`jaxterity.sysid.parameter_jacobian`
        is link ``mass`` / ``com`` / ``inertia_diag`` (the principal moments)
        and joint ``damping`` / ``friction`` / ``armature``; the
        off-diagonal inertia tensor still requires a structural rebuild.
        """
        params: dict[str, object] = {}
        for link in self.links:
            for field in link.parameter_spec():
                params[f"{link.name}.{field}"] = getattr(link, field)
        for joint in self.joints:
            for field in joint.parameter_spec():
                params[f"{joint.name}.{field}"] = getattr(joint, field)
        return params

    def with_parameters(self, updates: dict[str, object]) -> "Robot":
        """Return a new ``Robot`` with the given identifiable leaves replaced.

        Args
        ----
        updates : dict[str, object]
            ``{"<link_or_joint_name>.<field>": value}``, e.g.
            ``{"bob.mass": 1.2, "pivot.damping": 0.3}``.

        Returns
        -------
        Robot
            A new robot (the input is unchanged — ADR-004: sysid never mutates).

        Raises
        ------
        JaxterityError
            If a key does not name a known component or identifiable field.
        """
        link_map, joint_map = self.link_map, self.joint_map
        new_links = {name: link for name, link in link_map.items()}
        new_joints = {name: joint for name, joint in joint_map.items()}
        for key, value in updates.items():
            comp, _, field = key.partition(".")
            obj: Link | Joint
            if comp in new_links:
                obj = new_links[comp]
            elif comp in new_joints:
                obj = new_joints[comp]
            else:
                raise JaxterityError(
                    f"with_parameters: {comp!r} is not a known link or joint. "
                    f"Known: links {sorted(link_map)}, joints {sorted(joint_map)}."
                )
            if field not in obj.parameter_spec():
                raise JaxterityError(
                    f"with_parameters: {field!r} is not an identifiable parameter "
                    f"of {comp!r} (identifiable: {tuple(obj.parameter_spec())})."
                )
            replaced = eqx.tree_at(
                lambda o, _f=field: getattr(o, _f),  # type: ignore[misc]
                obj,
                jnp.asarray(value),
            )
            if comp in new_links:
                new_links[comp] = replaced
            else:
                new_joints[comp] = replaced
        return Robot(
            links=tuple(new_links[name] for name in link_map),
            joints=tuple(new_joints[name] for name in joint_map),
            name=self.name,
            source=self.source,
            calibration_state=self.calibration_state,
            gravity=self.gravity,
            ground_plane=self.ground_plane,
            provenance=self.provenance,
        )

    def with_welded_joints(self, names: tuple[str, ...]) -> "Robot":
        """Return a new robot with the named joints welded (``FIXED``).

        The standard way to carve a subsystem out of a full robot — e.g. a
        single arm of a humanoid for an arm-only training task: weld
        everything else and ``nv`` shrinks to the remaining articulation.
        Welded joints keep their kinematic origin (the body stays where it
        is at the zero configuration) and drop out of the actuated DOFs,
        the dynamic parameters, and the MJCF ``<joint>`` emission.

        Args
        ----
        names : tuple[str, ...]
            Joints to weld. Welding a ``FLOATING`` base is refused (drop it
            instead — a welded root is the fixed-base ``load()`` form).

        Returns
        -------
        Robot
            A new robot (the input is unchanged). Its attestation handle
            differs — the articulation is part of the model description.

        Raises
        ------
        JaxterityError
            On an unknown joint name or an attempt to weld a floating base.
        """
        import dataclasses

        joint_map = self.joint_map
        unknown = sorted(set(names) - set(joint_map))
        if unknown:
            raise JaxterityError(
                f"with_welded_joints: unknown joints {unknown}. Known: "
                f"{sorted(joint_map)}."
            )
        for name in names:
            if joint_map[name].joint_type is JointType.FLOATING:
                raise JaxterityError(
                    f"with_welded_joints: {name!r} is the floating base — "
                    "load the fixed-base form instead of welding it."
                )
        joints = tuple(
            dataclasses.replace(joint, joint_type=JointType.FIXED)
            if joint.name in names
            else joint
            for joint in self.joints
        )
        return Robot(
            links=self.links,
            joints=joints,
            name=self.name,
            source=self.source,
            calibration_state=self.calibration_state,
            gravity=self.gravity,
            ground_plane=self.ground_plane,
            provenance=self.provenance,
        )

    def with_provenance(
        self, record: tuple[str, str, str], *, calibrated: bool = True
    ) -> "Robot":
        """Return a new robot with a sysid provenance record appended.

        Args
        ----
        record : tuple[str, str, str]
            ``(recipe_name, recipe_version, telemetry_hash)``.
        calibrated : bool
            Set the calibration state to ``CALIBRATED`` (default True).

        Returns
        -------
        Robot
            A new robot; its ``attestation_handle`` now reflects the record.
        """
        return Robot(
            links=self.links,
            joints=self.joints,
            name=self.name,
            source=self.source,
            calibration_state=(
                CalibrationState.CALIBRATED if calibrated else self.calibration_state
            ),
            gravity=self.gravity,
            ground_plane=self.ground_plane,
            provenance=(*self.provenance, record),
        )

    # --- lowering -------------------------------------------------------

    def to_mjcf(self) -> str:
        """Lower the kinematic tree to an MJCF string (ADR-016 backend)."""
        return to_mjcf(
            self.link_map,
            self.joint_map,
            name=self.name,
            gravity=self.gravity,
            ground_plane=self.ground_plane,
        )

    # --- attestation (ADR-008) -------------------------------------------
    # attestation: changes-handle — version 3 digests one canonical JSON
    # document instead of separator-joined parts; version 2 is kept for
    # verifying handles recorded before the bump.

    def _attestation_extra(self, version: int) -> dict[str, object]:
        """The handle's ``extra`` under ``version``: JSON values from 3 on,
        the ``repr`` strings version 2 hashed."""
        state = self.calibration_state.value
        if version == 2:
            return {"calibration_state": state, "provenance": repr(self.provenance)}
        return {
            "calibration_state": state,
            "provenance": [list(record) for record in self.provenance],
        }

    def attestation_document(self) -> dict[str, object]:
        """The JSON document the current handle digests (for inspection or an
        external verifier): the lowered MJCF, every identifiable parameter as a
        float64 array, the calibration state and the provenance records."""
        return attestation.handle_document(
            model_description=self.to_mjcf(),
            parameters=self.parameters(),
            extra=self._attestation_extra(attestation.ALGORITHM_VERSION),
        )

    def attestation_handle_for(self, version: int) -> str:
        """The handle under algorithm ``version`` (``2`` recomputes the old form)."""
        return attestation.compute_handle(
            model_description=self.to_mjcf(),
            parameters=self.parameters(),
            extra=self._attestation_extra(version),
            version=version,
        )

    @property
    def attestation_handle(self) -> str:
        """The versioned BLAKE3 content hash (ADR-008), the Jaxility key."""
        return self.attestation_handle_for(attestation.ALGORITHM_VERSION)

    def verify_attestation_handle(self, expected: str) -> int | None:
        """The algorithm version under which ``expected`` is this robot's
        handle, or ``None``. A handle recorded under version 2 verifies as 2."""
        return attestation.matching_version(expected, self.attestation_handle_for)

    def to_diagram(
        self,
        q0=None,
        qd0=None,
        *,
        return_system: bool = False,
        implicit_vjp: bool = False,
    ):
        """Build a Jaxonomy ``Diagram`` simulating this robot via MJX.

        Args
        ----
        q0, qd0 : array-like | None
            Initial joint positions / velocities. Default zeros.
        return_system : bool
            If True, return ``(diagram, system)`` so callers can address the
            block's parameters/state via ``system.system_id``. Default False.
        implicit_vjp : bool
            Route the MJX constraint solve through the implicit-function-theorem
            wrapper so ``jax.grad`` works through ``simulate`` even with active
            joint limits (see :mod:`jaxterity.robot.implicit`). Forecloses
            forward-mode; contact-capable models are rejected. Default False.

        Returns
        -------
        jaxonomy.Diagram | tuple[jaxonomy.Diagram, LeafSystem]
            A single-block diagram whose continuous state is ``[q, qd]`` and
            whose dynamics come from MJX forward dynamics, with link ``mass`` and
            joint ``damping``/``friction``/``armature`` exposed as
            differentiable parameters.
        """
        system = self.build_system(q0=q0, qd0=qd0, implicit_vjp=implicit_vjp)
        builder = jaxonomy.DiagramBuilder()
        builder.add(system)
        diagram = builder.build()
        return (diagram, system) if return_system else diagram

    def build_system(
        self,
        q0=None,
        qd0=None,
        *,
        actuation: bool = False,
        implicit_vjp: bool = False,
    ):
        """Build the robot's MJX ``LeafSystem`` (for composing in a larger diagram).

        Use this to wire actuators/sensors around the robot in a custom
        ``DiagramBuilder``. With ``actuation=True`` the system exposes a ``tau``
        input port (length ``nv``, applied as ``qfrc_applied``).

        Args
        ----
        q0, qd0 : array-like | None
            Initial joint positions / velocities. Default zeros.
        actuation : bool
            Expose a joint-torque input port. Default False (free dynamics).
        implicit_vjp : bool
            Make the constraint solve reverse-mode differentiable via the
            implicit function theorem (see :mod:`jaxterity.robot.implicit`).
            Forecloses forward-mode through the block; contact-capable models
            are rejected at construction. Default False.

        Returns
        -------
        LeafSystem
            The robot block.
        """
        return _RobotSystem(
            self, q0=q0, qd0=qd0, actuation=actuation, implicit_vjp=implicit_vjp
        )

    def functional_dynamics(
        self, *, implicit_vjp: bool = False
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return a pure, ``jit``/``vmap``-able continuous-time dynamics function.

        The returned ``f(state, tau) -> dstate`` evaluates this robot's MJX
        forward dynamics with the robot's *current* parameters baked in (so a
        calibrated robot gives calibrated dynamics). It is the fast path for
        control and reinforcement learning:

        * ``state`` — ``concatenate([q, qd])``, lengths ``nq`` + ``nv``.
        * ``tau`` — generalized force, length ``nv``, applied as MJX
          ``qfrc_applied`` (for a prismatic joint this is a linear force in N;
          for a revolute joint, a torque in N·m).
        * ``dstate`` — ``concatenate([q̇, q̈])`` where ``q̇`` is the
          configuration derivative (``= qd`` for hinge/slide joints; the
          quaternion map for a free joint) and ``q̈`` is ``mjx.forward(...).qacc``.

        The model is compiled once when this method is called; the returned
        closure does no Python-level setup, so it ``jit``/``vmap``-composes and
        runs in microseconds. Contrast :meth:`jaxterity.policies.S4.step`, which
        rebuilds the simulator on every call and is ~10⁶× slower — use that for
        a faithful one-off rollout, this for training loops over many steps.

        Integrate it yourself (e.g. fixed-step RK4); the integrator choice is
        the caller's. Build the function once and reuse it::

            f = robot.functional_dynamics()
            def rk4(state, tau, dt):
                k1 = f(state, tau); k2 = f(state + 0.5 * dt * k1, tau)
                k3 = f(state + 0.5 * dt * k2, tau); k4 = f(state + dt * k3, tau)
                return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        Args
        ----
        implicit_vjp : bool
            Make the constraint solve reverse-mode differentiable via the
            implicit function theorem, so ``jax.grad`` works through rollouts
            even with active joint limits (see
            :mod:`jaxterity.robot.implicit`). Forecloses forward-mode
            (``jacfwd``) through the returned function; contact-capable
            models are rejected. Default False (raw MJX solve — forward-mode
            differentiable, reverse-mode raises on constrained models).

        Returns
        -------
        Callable[[Array, Array], Array]
            The dynamics ``f(state, tau) -> dstate``.
        """
        system = _RobotSystem(self, actuation=True, implicit_vjp=implicit_vjp)
        mjx = system._mjx
        model = system._mjx_model
        nq, nv = system._nq, system._nv
        config_derivative = system._config_derivative
        qacc_fn = system._qacc_fn

        if qacc_fn is not None:
            implicit_fn = qacc_fn  # non-Optional local for the closure

            def f_implicit(state: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
                q, qd = state[:nq], state[nq : nq + nv]
                qacc = implicit_fn(model, q, qd, jnp.asarray(tau))
                return jnp.concatenate([config_derivative(q, qd), qacc])

            return f_implicit

        def f(state: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            q, qd = state[:nq], state[nq : nq + nv]
            data = mjx.make_data(model).replace(
                qpos=q, qvel=qd, qfrc_applied=jnp.asarray(tau)
            )
            data = mjx.forward(model, data)
            return jnp.concatenate([config_derivative(q, qd), data.qacc])

        return f


#: Identifiable joint field → the per-DOF MJX model array it is threaded into.
_JOINT_FIELD_TO_MJX = {
    "damping": "dof_damping",
    "friction": "dof_frictionloss",
    "armature": "dof_armature",
}


class _RobotSystem(LeafSystem):
    """Jaxonomy ``LeafSystem`` wrapping MJX forward dynamics for a ``Robot``."""

    def __init__(
        self,
        robot: Robot,
        q0=None,
        qd0=None,
        actuation: bool = False,
        implicit_vjp: bool = False,
    ):
        super().__init__(name=robot.name)
        self._actuation = bool(actuation)
        mujoco, mjx = _lazy_mjx()

        mjcf = robot.to_mjcf()
        mj_model = mujoco.MjModel.from_xml_string(mjcf)
        self._mjx_model = mjx.put_model(mj_model)
        self._mjx = mjx
        self._nq = int(mj_model.nq)
        self._nv = int(mj_model.nv)
        # Reverse-mode differentiable constraint solve (IFT wrap); None → raw
        # MJX forward (forward-mode differentiable, reverse-mode raises on
        # constrained models). Built once — the wrap validates contact scope.
        self._qacc_fn = (
            implicit.implicit_qacc(mjx, self._mjx_model) if implicit_vjp else None
        )

        # index maps: link name -> body id; joint name -> dof index.
        # ``mass``, ``com`` (3-vec → body_ipos), and ``inertia_diag``
        # (3-vec → body_inertia, MuJoCo's *principal* moments) are all
        # runtime-threaded so forward-mode autodiff sees them. The off-diagonal
        # inertia tensor and the inertia-frame rotation (body_iquat) are not
        # yet threaded — fit by structural rebuild via Robot.with_parameters.
        # IMPORTANT: ``com`` and ``inertia_diag`` are seeded from the compiled
        # MJX model rather than directly from ``Link.com`` / ``jnp.diag(Link.inertia)``
        # — MuJoCo's MJCF compiler stores positions in the parent-body frame
        # and diagonalizes the inertia tensor into principal moments (via
        # ``body_iquat``). Seeding from the compiled values keeps the
        # dynamic-parameter defaults bit-equal to the un-overridden simulation.
        self._mass_params: list[tuple[str, int]] = []
        self._com_params: list[tuple[str, int]] = []
        self._inertia_params: list[tuple[str, int]] = []
        for link in robot.links:
            bid = int(mj_model.body(link.name).id)
            mname = f"{link.name}.mass"
            self.declare_dynamic_parameter(mname, jnp.asarray(link.mass))
            self._mass_params.append((mname, bid))
            cname = f"{link.name}.com"
            self.declare_dynamic_parameter(cname, jnp.asarray(mj_model.body_ipos[bid]))
            self._com_params.append((cname, bid))
            iname = f"{link.name}.inertia_diag"
            self.declare_dynamic_parameter(
                iname, jnp.asarray(mj_model.body_inertia[bid])
            )
            self._inertia_params.append((iname, bid))

        self._joint_params: list[tuple[str, str, int]] = []
        for joint in robot.joints:
            if joint.joint_type in (JointType.FIXED, JointType.FLOATING):
                continue  # FIXED is welded; FLOATING has no identifiable dynamics
            jid = int(mj_model.joint(joint.name).id)
            dof = int(mj_model.jnt_dofadr[jid])
            for field in ("damping", "friction", "armature"):
                pname = f"{joint.name}.{field}"
                self.declare_dynamic_parameter(
                    pname, jnp.asarray(getattr(joint, field))
                )
                self._joint_params.append((pname, field, dof))

        # qpos→q̇ map: for a free joint nq≠nv and the quaternion part of q̇ is not
        # the angular velocity (q̇_quat = ½·quat⊗ω), so the configuration
        # derivative is not simply q̇=qd. Record each MuJoCo joint's type and its
        # qpos/dof offsets (static ints), ordered by qpos address, to rebuild q̇.
        self._qmap = tuple(
            sorted(
                (
                    (
                        int(mj_model.jnt_type[jid]),
                        int(mj_model.jnt_qposadr[jid]),
                        int(mj_model.jnt_dofadr[jid]),
                    )
                    for jid in range(int(mj_model.njnt))
                ),
                key=lambda seg: seg[1],
            )
        )
        # mjtJoint: FREE=0, BALL=1. Only then does nq≠nv / the quaternion map apply.
        self._has_quat_dof = any(seg[0] in (0, 1) for seg in self._qmap)
        if any(seg[0] == 1 for seg in self._qmap):
            raise JaxterityError(
                "Ball joints are not supported yet (free/hinge/slide/fixed only)."
            )

        # Default to MuJoCo's reference configuration (qpos0): zeros for
        # hinge/slide, but the identity quaternion (not zeros) for a free joint.
        q0 = jnp.asarray(mj_model.qpos0) if q0 is None else jnp.asarray(q0, dtype=float)
        qd0 = jnp.zeros(self._nv) if qd0 is None else jnp.asarray(qd0, dtype=float)
        self.declare_continuous_state(
            default_value=jnp.concatenate([q0, qd0]), ode=self.ode
        )
        self.declare_continuous_state_output(name="state")
        # Optional joint-torque input (length nv), applied as qfrc_applied, so an
        # actuator block can drive the robot. Omitted by default → free dynamics.
        if self._actuation:
            self.declare_input_port(name="tau")

    def ode(self, time, state, *inputs, **params):
        mjx = self._mjx
        x = state.continuous_state
        q, qd = x[: self._nq], x[self._nq : self._nq + self._nv]

        model = self._mjx_model
        for pname, bid in self._mass_params:
            model = model.replace(body_mass=model.body_mass.at[bid].set(params[pname]))
        for pname, bid in self._com_params:
            model = model.replace(body_ipos=model.body_ipos.at[bid].set(params[pname]))
        for pname, bid in self._inertia_params:
            model = model.replace(
                body_inertia=model.body_inertia.at[bid].set(params[pname])
            )
        for pname, field, dof in self._joint_params:
            attr = _JOINT_FIELD_TO_MJX[field]
            model = model.replace(
                **{attr: getattr(model, attr).at[dof].set(params[pname])}
            )

        tau = (
            jnp.asarray(inputs[0])
            if (self._actuation and inputs)
            else jnp.zeros(self._nv)
        )
        if self._qacc_fn is not None:
            qacc = self._qacc_fn(model, q, qd, tau)
        else:
            data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
            qacc = mjx.forward(model, data).qacc
        return jnp.concatenate([self._config_derivative(q, qd), qacc])

    def _config_derivative(self, q, qd):
        """Map joint velocity ``qd`` (length nv) to the configuration derivative
        ``q̇`` (length nq).

        For hinge/slide joints ``q̇ = qd``. For a free joint the linear part is the
        world-frame velocity and the orientation part is the quaternion rate
        ``½·quat⊗[0, ω_body]`` — *not* the angular velocity — so nq≠nv there.
        """
        if not self._has_quat_dof:
            return qd  # nq == nv: q̇ = qd (the common articulated/fixed-base case)
        blocks = []
        for jtype, qadr, dofadr in self._qmap:
            if jtype == 0:  # free joint: 3 translation + 4 quaternion (qpos), 6 dof
                v_lin = qd[dofadr : dofadr + 3]
                omega = qd[dofadr + 3 : dofadr + 6]
                quat = q[qadr + 3 : qadr + 7]
                d_quat = 0.5 * _quat_mul(quat, jnp.concatenate([jnp.zeros(1), omega]))
                blocks.append(jnp.concatenate([v_lin, d_quat]))
            else:  # slide / hinge: a single dof, q̇ = qd
                blocks.append(qd[dofadr : dofadr + 1])
        return jnp.concatenate(blocks)


__all__ = ["Robot", "CalibrationState"]
