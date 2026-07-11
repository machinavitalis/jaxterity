# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The per-unit attested twin.

A calibrated :class:`~jaxterity.robot.Robot` is a *model of a design*; a
``Twin`` is a model of a *specific physical unit*. It composes the calibrated
body with the off-body physics that unit actually exhibits — per-servo heat
(:class:`~jaxterity.physics.thermal.ServoThermal`) and, for a mobile base, a
shared battery (:class:`~jaxterity.physics.battery.BatterySag`) — plus the
unit's identity: a static ``serial`` and a ``provenance`` chain (the off-body
calibration recipe versions + telemetry hashes, on top of the robot's own).

``Twin`` is an ``equinox.Module`` PyTree. Its differentiable leaves are the
robot's identifiable parameters **and** the off-body parameters (`C_th`,
`R_th`, `k_pwr`, the optional per-servo actuation scale `K_t_scale`, and
`V_oc`/`R_int` when a battery is present); everything else (serial,
provenance, ambient, the torque→current map, servo labels) is static.
This makes a fleet of units a single batched PyTree — `jax.vmap` over the
stacked per-unit leaves evaluates the whole fleet at once — and makes the unit
end-to-end differentiable for identification.

Forward rollout (:meth:`Twin.rollout`) uses the robot's
:meth:`~jaxterity.robot.Robot.functional_dynamics` fast path (never
``S4.step`` in a loop), integrating body, thermal, and (battery)
electrical state together. :meth:`Twin.attestation_handle` extends the robot's
handle (ADR-008) to be sensitive to the off-body parameters, the serial, and
the provenance, while staying stable under the cosmetic set: whitespace
in the model and parameter byte order do not change it, but two units with
identical parameters and *different serials* get distinct handles.

Honest scope: reuses the existing calibration; no new joint types. The
thermal model is per-DOF lumped single-node; the servo↔DOF map is
positional (thermal leaf ``i`` is driven by generalized-force component ``i``,
in the robot's MJX DOF order). :meth:`Twin.to_diagram` snapshots the current
leaf values into the composed Jaxonomy diagram (it is the simulation/composition
surface; the differentiable path is :meth:`Twin.rollout`).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
from jaxonomy.library import Constant, Demux

from ..diagnostics import attestation
from ..physics import bus_voltage
from ..physics.battery import _I_EPS_DEFAULT, BatterySag
from ..physics.thermal import _T_AMB_DEFAULT, ServoThermal
from ..robot import Robot


class TwinRollout(NamedTuple):
    """Trajectories returned by :meth:`Twin.rollout`.

    Fields
    ------
    q : jax.Array, shape ``(n_steps + 1, nq)``
        Joint positions over the rollout (including the initial state), rad.
    qd : jax.Array, shape ``(n_steps + 1, nv)``
        Joint velocities, rad/s.
    temperature : jax.Array, shape ``(n_steps + 1, n_servos)``
        Per-servo winding/case temperature, K.
    bus_voltage : jax.Array | None, shape ``(n_steps,)``
        Shared-bus voltage at each commanded torque, V — ``None`` for a
        battery-less (bench) twin.
    """

    q: jax.Array
    qd: jax.Array
    temperature: jax.Array
    bus_voltage: jax.Array | None


class Twin(eqx.Module):
    """A calibrated robot + its off-body physics + per-unit identity.

    See the module docstring for the model and the differentiable/static split.

    Args
    ----
    robot : Robot
        The calibrated body. Its identifiable leaves remain differentiable.
    serial : str
        The physical unit's serial number (static identity).
    C_th, R_th, k_pwr : array-like, shape ``(n_servos,)``
        Per-servo thermal capacitance (J/K), resistance (K/W), and torque²→power
        gain (W/(N·m)²). All differentiable leaves. ``n_servos`` must equal the
        robot's DOF count ``nv`` at rollout time.
    K_t_scale : array-like or None, shape ``(n_servos,)``
        Per-servo actuation (torque-constant) scale, dimensionless —
        delivered torque per commanded torque, nominal 1. A differentiable
        leaf modelling motor-constant drift / per-unit motor variance: the
        rollout applies ``tau_delivered = K_t_scale · tau_commanded`` to the
        body, the thermal heating, and the battery current draw. ``None``
        (default) means an exactly-nominal actuation path (identical to the
        prior twin, handle included).
    T_amb : float
        Ambient temperature, K (static). Default 298.15 K.
    V_oc, R_int : array-like or None
        Battery open-circuit voltage (V) and internal resistance (Ω),
        differentiable leaves — both ``None`` (default) for a bench unit, both
        set for a mobile (LeKiwi) unit.
    k_i : array-like or None, shape ``(n_servos,)``
        Per-servo torque→current map (A per N·m), static. Required iff a battery
        is present.
    i_eps : float
        Smoothing width for the battery current-draw ``|tau|`` (N·m, static).
    servo_names : tuple[str, ...] or None
        Per-servo labels in DOF order (static). Defaults to ``servo_0…``.
    provenance : tuple[tuple[str, str, str], ...]
        Off-body calibration records ``(recipe_name, recipe_version,
        telemetry_hash)`` (static), on top of the robot's own provenance.
    param_uncertainty : tuple[tuple[str, tuple[float, ...]], ...]
        Per-leaf marginal variance of the fitted off-body parameters,
        keyed by canonical path (e.g. ``"offbody.R_th"``), each a per-servo
        tuple. A **derived** quantity — it does **not** enter the attestation
        handle. Empty when no fit uncertainty has been attached. Read it through
        :meth:`uncertainty`.

    Raises
    ------
    ValueError
        If the thermal leaf shapes disagree, or a battery is half-specified, or
        ``k_i`` is missing/mismatched for a battery unit.
    """

    robot: Robot
    C_th: jax.Array
    R_th: jax.Array
    k_pwr: jax.Array
    K_t_scale: jax.Array | None
    V_oc: jax.Array | None
    R_int: jax.Array | None
    serial: str = eqx.field(static=True)
    T_amb: float = eqx.field(static=True)
    k_i: tuple[float, ...] | None = eqx.field(static=True)
    i_eps: float = eqx.field(static=True)
    servo_names: tuple[str, ...] = eqx.field(static=True)
    provenance: tuple[tuple[str, str, str], ...] = eqx.field(static=True)
    # attestation: invariant — uncertainty is a derived quantity, kept out of
    # the handle (a fit's covariance must not change the unit's identity).
    param_uncertainty: tuple[tuple[str, tuple[float, ...]], ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        robot: Robot,
        *,
        serial: str,
        C_th,
        R_th,
        k_pwr,
        K_t_scale=None,
        T_amb: float = _T_AMB_DEFAULT,
        V_oc=None,
        R_int=None,
        k_i=None,
        i_eps: float = _I_EPS_DEFAULT,
        servo_names: tuple[str, ...] | None = None,
        provenance: tuple[tuple[str, str, str], ...] = (),
        param_uncertainty: tuple[tuple[str, tuple[float, ...]], ...] = (),
    ):
        self.robot = robot
        self.serial = str(serial)
        self.C_th = jnp.asarray(C_th, dtype=float)
        self.R_th = jnp.asarray(R_th, dtype=float)
        self.k_pwr = jnp.asarray(k_pwr, dtype=float)
        n = self.C_th.shape[0]
        if self.R_th.shape != (n,) or self.k_pwr.shape != (n,):
            raise ValueError(
                f"Twin thermal leaves must all be shape ({n},); got C_th "
                f"{self.C_th.shape}, R_th {self.R_th.shape}, k_pwr "
                f"{self.k_pwr.shape}. One per servo (robot DOF)."
            )
        self.K_t_scale = (
            None if K_t_scale is None else jnp.asarray(K_t_scale, dtype=float)
        )
        if self.K_t_scale is not None and self.K_t_scale.shape != (n,):
            raise ValueError(
                f"Twin K_t_scale must be shape ({n},) — one actuation scale "
                f"per servo (robot DOF) — got {self.K_t_scale.shape}."
            )
        if (V_oc is None) != (R_int is None):
            raise ValueError(
                "Twin battery is half-specified: pass both V_oc and R_int "
                "(mobile/LeKiwi unit) or neither (bench unit)."
            )
        has_battery = V_oc is not None
        self.V_oc = None if V_oc is None else jnp.asarray(V_oc, dtype=float)
        self.R_int = None if R_int is None else jnp.asarray(R_int, dtype=float)
        if has_battery:
            if k_i is None or len(k_i) != n:
                raise ValueError(
                    f"Twin battery needs a per-servo k_i of length {n} (A per "
                    f"N·m); got {k_i!r}."
                )
            self.k_i = tuple(float(x) for x in k_i)
        else:
            self.k_i = None
        self.T_amb = float(T_amb)
        self.i_eps = float(i_eps)
        self.servo_names = (
            tuple(servo_names)
            if servo_names is not None
            else tuple(f"servo_{i}" for i in range(n))
        )
        if len(self.servo_names) != n:
            raise ValueError(
                f"servo_names has {len(self.servo_names)} entries but there are "
                f"{n} servos."
            )
        self.provenance = tuple(provenance)
        self.param_uncertainty = tuple(
            (str(key), tuple(float(v) for v in vals)) for key, vals in param_uncertainty
        )

    # --- identity -------------------------------------------------------

    @property
    def n_servos(self) -> int:
        """Number of servos (thermal nodes) — one per robot DOF."""
        return self.C_th.shape[0]

    @property
    def has_battery(self) -> bool:
        """True for a mobile (LeKiwi) unit with a shared-battery model."""
        return self.V_oc is not None

    def identifiable_params(self) -> dict[str, object]:
        """All differentiable leaves, keyed by canonical path.

        The robot's leaves under ``robot.<name>.<field>`` plus the off-body
        leaves under ``offbody.<field>``. This is the set the attestation handle
        hashes and the natural target for ``jax.grad``.

        Returns
        -------
        dict[str, object]
            ``{path: leaf}`` for every identifiable leaf.
        """
        params: dict[str, object] = {
            f"robot.{k}": v for k, v in self.robot.parameters().items()
        }
        params["offbody.C_th"] = self.C_th
        params["offbody.R_th"] = self.R_th
        params["offbody.k_pwr"] = self.k_pwr
        if self.K_t_scale is not None:
            params["offbody.K_t_scale"] = self.K_t_scale
        if self.has_battery:
            params["offbody.V_oc"] = self.V_oc
            params["offbody.R_int"] = self.R_int
        return params

    @property
    def attestation_handle(self) -> str:
        """The per-unit handle: the robot's handle extended off-body (ADR-008).

        Sensitive to the robot model + parameters, the off-body parameters, the
        ``serial``, and the ``provenance``. Stable under the cosmetic set
        whitespace in the lowered model and parameter byte order do not
        change it. Two units with identical parameters but different serials get
        distinct handles.

        Returns
        -------
        str
            The BLAKE3 hex digest.
        """
        params = self.identifiable_params()
        extra = {
            "serial": self.serial,
            "twin_provenance": repr(self.provenance),
            # Folds in the robot's full identity (calibration state + the robot's
            # own provenance), which the robot handle already canonicalizes.
            "robot_handle": self.robot.attestation_handle,
            "T_amb": repr(self.T_amb),
            "i_eps": repr(self.i_eps),
            "k_i": repr(self.k_i),
            "servo_names": repr(self.servo_names),
            "has_battery": repr(self.has_battery),
        }
        return attestation.compute_handle(
            model_description=self.robot.to_mjcf(),
            parameters=params,
            extra=extra,
        )

    def uncertainty(self) -> dict[str, np.ndarray]:
        """Per-leaf marginal variance of the fitted off-body parameters.

        Returns
        -------
        dict[str, numpy.ndarray]
            ``{path: per-servo variance}`` (e.g. ``"offbody.R_th"``); empty if no
            fit uncertainty has been attached. The standard error of a leaf is
            the elementwise square root.
        """
        return {
            key: np.asarray(vals, dtype=float) for key, vals in self.param_uncertainty
        }

    def with_serial(self, serial: str) -> "Twin":
        """Return a copy of this twin bound to a different serial (same params)."""
        return Twin(
            self.robot,
            serial=serial,
            C_th=self.C_th,
            R_th=self.R_th,
            k_pwr=self.k_pwr,
            K_t_scale=self.K_t_scale,
            T_amb=self.T_amb,
            V_oc=self.V_oc,
            R_int=self.R_int,
            k_i=self.k_i,
            i_eps=self.i_eps,
            servo_names=self.servo_names,
            provenance=self.provenance,
            param_uncertainty=self.param_uncertainty,
        )

    def with_thermal(
        self,
        *,
        C_th=None,
        R_th=None,
        k_pwr=None,
        provenance_record: tuple[str, str, str] | None = None,
        param_uncertainty: tuple[tuple[str, tuple[float, ...]], ...] | None = None,
    ) -> "Twin":
        """Return a copy with off-body thermal leaves replaced (e.g. from a fit).

        Updates only the thermal leaves given (the rest are kept), optionally
        appends a ``(recipe_name, recipe_version, telemetry_hash)`` provenance
        record and attaches the fit's per-leaf uncertainty. The returned twin has
        a different attestation handle — a thermal re-fit is a new unit version —
        but the uncertainty itself does **not** affect the handle.

        Args
        ----
        C_th, R_th, k_pwr : array-like or None
            New per-servo thermal leaves; ``None`` keeps the current value.
        provenance_record : tuple[str, str, str] or None
            A fit-provenance record to append.
        param_uncertainty : tuple or None
            The fit's per-leaf marginal variance; ``None`` keeps the
            current uncertainty.

        Returns
        -------
        Twin
            A new twin with the updated thermal parameters and provenance.
        """
        prov = self.provenance
        if provenance_record is not None:
            prov = (*prov, provenance_record)
        return Twin(
            self.robot,
            serial=self.serial,
            C_th=self.C_th if C_th is None else C_th,
            R_th=self.R_th if R_th is None else R_th,
            k_pwr=self.k_pwr if k_pwr is None else k_pwr,
            K_t_scale=self.K_t_scale,
            T_amb=self.T_amb,
            V_oc=self.V_oc,
            R_int=self.R_int,
            k_i=self.k_i,
            i_eps=self.i_eps,
            servo_names=self.servo_names,
            provenance=prov,
            param_uncertainty=(
                self.param_uncertainty
                if param_uncertainty is None
                else param_uncertainty
            ),
        )

    def with_actuation(
        self,
        K_t_scale,
        *,
        param_uncertainty: tuple[tuple[str, tuple[float, ...]], ...] | None = None,
        provenance_record: tuple[str, str, str] | None = None,
    ) -> "Twin":
        """Return a copy with the per-servo actuation scale replaced (or set).

        The actuation-drift counterpart of :meth:`with_thermal`: an actuation
        re-fit is a new unit version, so the returned twin has a different
        attestation handle.

        Args
        ----
        K_t_scale : array-like, shape ``(n_servos,)``
            New per-servo actuation (torque-constant) scale, dimensionless.
        provenance_record : tuple[str, str, str] or None
            A fit-provenance record ``(recipe_name, recipe_version,
            telemetry_hash)`` to append.

        Returns
        -------
        Twin
            A new twin with the updated actuation scale and provenance.
        """
        prov = self.provenance
        if provenance_record is not None:
            prov = (*prov, provenance_record)
        return Twin(
            self.robot,
            serial=self.serial,
            C_th=self.C_th,
            R_th=self.R_th,
            k_pwr=self.k_pwr,
            K_t_scale=K_t_scale,
            T_amb=self.T_amb,
            V_oc=self.V_oc,
            R_int=self.R_int,
            k_i=self.k_i,
            i_eps=self.i_eps,
            servo_names=self.servo_names,
            provenance=prov,
            param_uncertainty=(
                self.param_uncertainty
                if param_uncertainty is None
                else param_uncertainty
            ),
        )

    # --- dynamics -------------------------------------------------------

    def _thermal_rhs(self, T: jax.Array, tau: jax.Array) -> jax.Array:
        """Per-servo lumped thermal derivative (vectorized over servos)."""
        power_in = self.k_pwr * tau * tau
        power_out = (T - self.T_amb) / self.R_th
        return (power_in - power_out) / self.C_th

    def rollout(
        self,
        q0,
        qd0,
        T0,
        tau_seq,
        dt: float,
    ) -> TwinRollout:
        """Roll the unit forward under a torque sequence (the fast path).

        Integrates the body (RK4 over the robot's
        :meth:`~jaxterity.robot.Robot.functional_dynamics`) and the per-servo
        thermal ODE (RK4) in lockstep, and evaluates the (optional) bus voltage
        algebraically at each commanded torque. Pure and ``jit``/``vmap``-able.

        Args
        ----
        q0 : array-like, shape ``(nq,)``
            Initial joint positions, rad.
        qd0 : array-like, shape ``(nv,)``
            Initial joint velocities, rad/s. ``nv`` must equal ``n_servos``.
        T0 : array-like, shape ``(n_servos,)``
            Initial per-servo temperature, K.
        tau_seq : array-like, shape ``(n_steps, nv)``
            Commanded generalized force per step, N·m (applied as
            ``qfrc_applied``), held constant over each step (zero-order hold).
        dt : float
            Step size, s.

        Returns
        -------
        TwinRollout
            Body + thermal (+ bus-voltage) trajectories.

        Raises
        ------
        ValueError
            If ``qd0`` / ``T0`` / ``tau_seq`` widths disagree with ``n_servos``.

        Units
        -----
        q: rad; qd: rad/s; T: K; tau: N·m; dt: s; bus voltage: V.
        """
        q0 = jnp.asarray(q0, dtype=float)
        qd0 = jnp.asarray(qd0, dtype=float)
        T0 = jnp.asarray(T0, dtype=float)
        tau_seq = jnp.asarray(tau_seq, dtype=float)
        nv, n = qd0.shape[0], self.n_servos
        if nv != n or T0.shape[0] != n or tau_seq.shape[1] != n:
            raise ValueError(
                f"Twin.rollout shape mismatch: n_servos={n}, but qd0 width "
                f"{qd0.shape[0]}, T0 width {T0.shape[0]}, tau_seq width "
                f"{tau_seq.shape[1]}. The thermal nodes map 1:1 onto robot DOFs."
            )
        f = self.robot.functional_dynamics()
        return self._rollout_core(f, q0, qd0, T0, tau_seq, dt)

    def _rollout_core(self, f, q0, qd0, T0, tau_seq, dt):
        """The pure body+thermal+battery scan (shared by ``rollout`` and the
        compiled co-processor path). ``f`` is a prebuilt
        :meth:`~jaxterity.robot.Robot.functional_dynamics` closure.

        ``tau_seq`` is the *commanded* torque; when an actuation scale is
        present the *delivered* torque ``K_t_scale · tau`` drives the body,
        the thermal heating, and the battery current alike.
        """
        nq, nv = q0.shape[0], qd0.shape[0]
        delivered = (
            tau_seq if self.K_t_scale is None else tau_seq * self.K_t_scale[None, :]
        )

        def step(carry, tau):
            state, T = carry
            # body RK4
            b1 = f(state, tau)
            b2 = f(state + 0.5 * dt * b1, tau)
            b3 = f(state + 0.5 * dt * b2, tau)
            b4 = f(state + dt * b3, tau)
            state = state + (dt / 6.0) * (b1 + 2 * b2 + 2 * b3 + b4)
            # thermal RK4 (constant tau over the step)
            t1 = self._thermal_rhs(T, tau)
            t2 = self._thermal_rhs(T + 0.5 * dt * t1, tau)
            t3 = self._thermal_rhs(T + 0.5 * dt * t2, tau)
            t4 = self._thermal_rhs(T + dt * t3, tau)
            T = T + (dt / 6.0) * (t1 + 2 * t2 + 2 * t3 + t4)
            return (state, T), (state, T)

        state0 = jnp.concatenate([q0, qd0])
        _, (states, Ts) = jax.lax.scan(step, (state0, T0), delivered)
        states = jnp.concatenate([state0[None, :], states], axis=0)
        Ts = jnp.concatenate([T0[None, :], Ts], axis=0)

        bus = None
        if self.has_battery:
            k_i = jnp.asarray(self.k_i, dtype=float)
            bus = jax.vmap(
                lambda u: bus_voltage(
                    u, V_oc=self.V_oc, R_int=self.R_int, k_i=k_i, i_eps=self.i_eps
                )
            )(delivered)

        return TwinRollout(
            q=states[:, :nq],
            qd=states[:, nq : nq + nv],
            temperature=Ts,
            bus_voltage=bus,
        )

    def compiled_rollout(self, dt: float):
        """Return a ``jit``-compiled ``rollout(q0, qd0, T0, tau_seq) -> TwinRollout``.

        Builds the robot's :meth:`~jaxterity.robot.Robot.functional_dynamics`
        **once** and JIT-compiles the scan, so repeated calls (e.g. a self-check
        co-processor ticking at tens–hundreds of Hz) reuse the compiled
        kernel and run in microseconds — never rebuilding the MJX model per tick
        (which :meth:`rollout` does once per call). The twin's parameters are
        baked into the closure; build a new one if they change.

        Args
        ----
        dt : float
            Fixed integration step, s.

        Returns
        -------
        Callable[[Array, Array, Array, Array], TwinRollout]
            The compiled rollout. Inputs as in :meth:`rollout`.
        """
        f = self.robot.functional_dynamics()

        @jax.jit
        def rollout(q0, qd0, T0, tau_seq):
            return self._rollout_core(f, q0, qd0, T0, tau_seq, dt)

        return rollout

    def to_diagram(self, q0=None, qd0=None, *, torque_source=None):
        """Compose body + thermal (+ battery) into one Jaxonomy diagram.

        Wires a single commanded-torque source to the robot's ``tau`` input, to
        one :class:`~jaxterity.physics.thermal.ServoThermal` per DOF (via a
        ``Demux``), and (for a mobile unit) to the
        :class:`~jaxterity.physics.battery.BatterySag` block. This is the
        composition/simulation surface; it **snapshots** the current leaf values
        into the constructed blocks (the differentiable path is
        :meth:`rollout`).

        Args
        ----
        q0, qd0 : array-like | None
            Initial joint positions / velocities. Default zeros.
        torque_source : LeafSystem | None
            A block whose single output is the length-``nv`` commanded torque.
            Defaults to a zero :class:`~jaxonomy.library.Constant`.

        Returns
        -------
        jaxonomy.Diagram
            The composed diagram.
        """
        robot_sys = self.robot.build_system(q0=q0, qd0=qd0, actuation=True)
        nv = robot_sys._nv
        builder = jaxonomy.DiagramBuilder()
        builder.add(robot_sys)
        src = torque_source or Constant(jnp.zeros(nv), name="tau_cmd")
        builder.add(src)
        builder.connect(src.output_ports[0], robot_sys.input_ports[0])
        demux = Demux(nv, name="tau_demux")
        builder.add(demux)
        builder.connect(src.output_ports[0], demux.input_ports[0])
        for i in range(self.n_servos):
            thermal = ServoThermal(
                C_th=float(self.C_th[i]),
                R_th=float(self.R_th[i]),
                k_pwr=float(self.k_pwr[i]),
                T_amb=self.T_amb,
                name=f"thermal_{self.servo_names[i]}",
            )
            builder.add(thermal)
            builder.connect(demux.output_ports[i], thermal.input_ports[0])
        if self.has_battery:
            battery = BatterySag(
                k_i=jnp.asarray(self.k_i, dtype=float),
                V_oc=float(self.V_oc),
                R_int=float(self.R_int),
                i_eps=self.i_eps,
                name="battery",
            )
            builder.add(battery)
            builder.connect(src.output_ports[0], battery.input_ports[0])
        return builder.build()


def _rk4_body(f: Callable, state, tau, dt: float):
    """Standalone body RK4 step (the reference the composed rollout matches)."""
    k1 = f(state, tau)
    k2 = f(state + 0.5 * dt * k1, tau)
    k3 = f(state + 0.5 * dt * k2, tau)
    k4 = f(state + dt * k3, tau)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


__all__ = ["Twin", "TwinRollout"]
