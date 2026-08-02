# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The certified reflex: a per-unit recovery controller, not just a veto.

Predict-and-veto answers "should this command run?" and stops there. A veto is
only half an organ: something has to hold the unit while the vetoed command is
not running, and "command nothing" is not safe on a machine with gravity, heat,
and momentum. This module is the other half — a **safety filter plus recovery
controller** built from the unit's *measured* twin:

- :class:`ReflexController` clamps an arbitrary policy's torque to the largest
  nearby torque that keeps every barrier satisfied. It runs at execution time,
  needs no retraining of whatever proposed the command, and is agnostic about
  where the command came from (VLA, MPC, teleop, script).
- :meth:`ReflexController.recover` is the reflex's *own* action — brake to rest
  through the same filter — for when the gate vetoes.
- :class:`SimplexSwitch` is the arbitration: pass, derate, or hand over.

What makes this ours rather than a textbook CBF-QP: **the constraints come from
the measured twin, not from kinematics alone.** A joint-limit filter is
standard; a filter that also refuses to cook a servo (the thermal ceiling of
*this* unit's fitted ``R_th``/``k_pwr``) or to brown out a shared bus (this
unit's fitted ``V_oc``/``R_int``) is what a nominal-spec model cannot express.
The reflex takes its ceilings from the same
:class:`~jaxterity.selfcheck.SafetyEnvelope` the monitor vetoes on, so the gate
and the reflex cannot disagree about where the edge is.

Arithmetic profile (the part that matters for putting this on a microcontroller
later):

- Each barrier's **value and derivative come from one forward pass** —
  ``jax.jvp`` of ``h(x)`` along ``ẋ = f(x, τ)``, i.e. dual-number arithmetic,
  not a second evaluation and not a hand-derived Lie derivative.
- Barriers with relative degree 2 in torque (joint position) are expressed with
  a **braking-distance** term, ``h = (limit − q) − q̇²/(2 a_brake)``, which is
  relative degree 1 — so every row is one linear inequality in τ and there is
  no high-order CBF cascade to tune.
- The QP is solved by a **fixed** number of relinearizations, each running a
  **fixed** number of dual projected-gradient iterations: no data-dependent
  branches, no adaptive termination, no dynamic allocation. Cost per tick is
  fixed by ``(n_dof, n_rows, outer_iters, iters)`` alone — ~0.2 ms on a two-DOF
  twin on a workstation CPU, which is a measurement, not a bound.

Honest scope, stated rather than implied:

- **No WCET number.** The shape is WCET-boundable; nobody has measured a bound.
  That needs the lowered build (Jaxility T-130), not this module.
- **Not certified.** "Certified reflex" is the organ's name in the architecture,
  not a claim of third-party certification. Publicly this is envelope
  enforcement.
- The thermal and bus rows are **quadratic** in torque, so each tick
  relinearizes them a fixed number of times rather than solving them exactly.
  The joint rows are exact; the others are first-order per pass, and the
  returned decision reports the residual violation instead of hiding it.
- The fixed budget means the solution is approximate. If the barrier set is
  empty (the unit is *already* outside), no torque satisfies the rows; the
  decision says so rather than returning a confident number.
- When the rows **conflict** — stopping before a joint limit can need more
  braking torque than the servo can sustain — the resolution is a *declared
  priority*, not a proof: the mechanical rows are unconditional, the thermal and
  bus rows yield above ``thermal_priority``, and the decision names the row that
  yielded.

Units
-----
rad, rad/s, rad/s², N·m, K, V, s.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ..calibrate.active import dof_joint_names
from ..errors import JaxterityError
from ..physics import bus_voltage
from ..twin import Twin, TwinRollout
from .monitor import SafetyAction, SafetyDecision, SafetyEnvelope, SafetyMonitor

#: Default declared braking capability used by the joint-position barriers,
#: rad/s². Deliberately conservative: the barrier trusts that the unit can shed
#: this much velocity, so an *over*-stated value makes the barrier optimistic.
DEFAULT_BRAKE_DECEL = 20.0

#: Default class-K gain on the barrier condition ``ḣ + α h ≥ 0`` for the
#: mechanical rows, 1/s. Larger intervenes later and harder; smaller intervenes
#: earlier and more smoothly.
#:
#: A single α across rows would be wrong here, and the mistake is instructive:
#: α bounds how fast a barrier may be closed (``ḣ ≥ −α h``), so it has to match
#: the row's own timescale. At α = 5 1/s a servo 2 K below its ceiling would be
#: allowed to heat at 10 K/s — no clamp at all on a state whose time constant is
#: a minute. The thermal rows therefore take their gain from the unit's *fitted*
#: thermal time constant instead (``1 / (R_th·C_th)``), which is the measured
#: quantity the barrier is actually about.
#:
#: That choice has an exact interpretation worth knowing, because it is why the
#: reflex and the monitor cannot disagree. For the first-order thermal model
#: ``Ṫ = (T_ss(τ) − T)/τ_th``, the condition ``Ṫ ≤ (T_ceiling − T)/τ_th`` is
#: identically ``T_ss(τ) ≤ T_ceiling``: with α = 1/τ_th the thermal barrier
#: *is* the sustained-hold criterion
#: :class:`~jaxterity.selfcheck.SafetyMonitor` vetoes on — "never command a
#: torque you could not hold forever". Pass a larger ``alpha_thermal`` to allow
#: transient bursts above the sustainable torque (α = k/τ_th permits a steady
#: state k× further into the remaining headroom), at the cost of letting the
#: ceiling be approached faster.
DEFAULT_ALPHA = 5.0

#: Default joint-limit standoff when the envelope declares none, rad.
DEFAULT_JOINT_MARGIN_RAD = 0.05

#: Fixed dual projected-gradient iteration budget per linearization. Chosen by
#: measurement rather than taste: at 24 the binding row was still short of
#: convergence on the bench case (a residual violation the decision then had to
#: report), while the per-tick cost is dominated by the constraint Jacobian, not
#: the inner loop — 24 → 100 iterations cost ~0.19 → ~0.22 ms per tick on a
#: two-DOF twin (workstation CPU, not a WCET bound).
DEFAULT_ITERS = 100

#: Fixed number of relinearizations of the torque-quadratic rows per tick. Six
#: passes converge the bench cases (three left the braking case short).
DEFAULT_OUTER_ITERS = 6

#: Torque difference above which a command counts as clamped, N·m.
_ENGAGED_TOL = 1e-6

#: Row residual below which the fixed-budget solve counts as feasible. The
#: budget is finite, so a converged solve still leaves numerical residue;
#: reporting that as an envelope breach would cry wolf on every clamped tick.
_FEASIBLE_TOL = 1e-4


@dataclasses.dataclass(frozen=True)
class ReflexDecision:
    """What the reflex did to one commanded torque.

    Args
    ----
    tau : numpy.ndarray, shape ``(nv,)``
        The torque to actually apply, N·m.
    tau_desired : numpy.ndarray, shape ``(nv,)``
        What was proposed, N·m.
    engaged : bool
        True iff the filter changed the command.
    binding_constraint : str
        Name of the barrier row with the least slack (e.g.
        ``"servo_temperature[2]"``, ``"joint_upper[1]"``).
    slack : float
        That row's slack after filtering — negative means the row is still
        violated at the returned torque.
    violation : float
        Worst residual violation over all rows after the fixed iteration
        budget, in the rows' own (mixed) units. Compared against
        :attr:`tolerance` — a fixed budget leaves numerical residue, and
        calling that an envelope breach would cry wolf every tick.
    tolerance : float
        The residual below which the solve counts as feasible.
    barriers : dict[str, float]
        Per-row barrier value ``h`` at the current state — negative means the
        unit is already outside that barrier's safe set.
    detail : str
        One-line human/agent explanation.

    Units
    -----
    N·m; barrier values in each row's own unit (rad, (rad/s)², K, V).
    """

    tau: np.ndarray
    tau_desired: np.ndarray
    engaged: bool
    binding_constraint: str
    slack: float
    violation: float
    barriers: dict[str, float]
    detail: str
    tolerance: float = 1e-4

    @property
    def feasible(self) -> bool:
        """True iff every barrier row is satisfied at the returned torque."""
        return self.violation <= self.tolerance


@dataclasses.dataclass(frozen=True)
class SimplexDecision:
    """The arbitration between the userland command and the reflex.

    Args
    ----
    mode : str
        ``"pass"`` (command clamped only by the barriers), ``"derate"`` (scaled
        by the monitor's factor, then clamped), or ``"reflex"`` (the monitor
        vetoed; the reflex's own recovery action is applied).
    tau : numpy.ndarray
        The torque to apply, N·m.
    safety : SafetyDecision
        The monitor's verdict on the predicted horizon.
    reflex : ReflexDecision
        What the filter did to whatever was handed to it.
    detail : str
        One-line explanation joining the two.
    """

    mode: str
    tau: np.ndarray
    safety: SafetyDecision
    reflex: ReflexDecision

    @property
    def detail(self) -> str:
        """One line: the monitor's verdict and the filter's action."""
        return f"{self.mode}: {self.safety.detail} | {self.reflex.detail}"


class ReflexController:
    """Clamp any policy's torque to this unit's measured envelope.

    See the module docstring for the contract and the honest scope.

    Args
    ----
    twin : Twin
        The unit. Its robot supplies joint position/velocity/effort limits; its
        fitted off-body leaves supply the thermal and bus barriers.
    envelope : SafetyEnvelope
        The same declared limits the monitor vetoes on: ``temp_ceiling_k``
        (always enforced), ``bus_floor_v`` and ``joint_margin_rad`` when set,
        and ``torque_margin_frac`` as the fraction of each effort limit the
        filter may command.
    brake_decel : float | array-like
        Declared braking capability per joint, rad/s² (scalar broadcasts).
        Used by the joint-position barriers' stopping-distance term.
    alpha : float
        Class-K gain on ``ḣ + α h ≥ 0`` for the mechanical rows, 1/s.
    alpha_thermal : float | array-like | None
        Class-K gain for the thermal rows, 1/s. ``None`` (default) derives it
        per servo from the twin's fitted thermal time constant,
        ``1 / (R_th·C_th)`` — the measured number the row is about (see
        :data:`DEFAULT_ALPHA`).
    joint_margin_rad : float | None
        Standoff from the joint limits, rad. ``None`` takes the envelope's
        ``joint_margin_rad``, or :data:`DEFAULT_JOINT_MARGIN_RAD`.
    brake_gain : float
        Recovery damping, N·m per rad/s, used by :meth:`recover`.
    thermal_priority : float
        How hard the thermal and bus rows push when they conflict with a
        mechanical row, as a multiple of the torque box. The mechanical rows are
        unconditional; this one yields above the cap, so a stop that needs more
        braking torque than the thermal budget allows still happens — and is
        reported. Raise it to make the thermal budget harder to override.
    iters : int
        Fixed dual projected-gradient iterations per linearization.
    outer_iters : int
        Fixed number of relinearizations (the thermal and bus rows are
        quadratic in torque, so one tangent is optimistic — see
        :meth:`_solve_impl`).

    Raises
    ------
    JaxterityError
        If the robot's actuated-DOF count and the twin's thermal-node count
        disagree (the reflex maps servo *i* onto DOF *i*), or if a declared
        braking capability is not positive.

    Units
    -----
    rad, rad/s, rad/s², N·m, K, V.
    """

    def __init__(
        self,
        twin: Twin,
        envelope: SafetyEnvelope,
        *,
        brake_decel=DEFAULT_BRAKE_DECEL,
        alpha: float = DEFAULT_ALPHA,
        alpha_thermal=None,
        joint_margin_rad: float | None = None,
        brake_gain: float = 2.0,
        thermal_priority: float = 1.0,
        iters: int = DEFAULT_ITERS,
        outer_iters: int = DEFAULT_OUTER_ITERS,
    ):
        self.twin = twin
        self.envelope = envelope
        self.alpha = float(alpha)
        self.brake_gain = float(brake_gain)
        self.iters = int(iters)
        self.outer_iters = int(outer_iters)
        self._names = dof_joint_names(twin.robot)
        jmap = twin.robot.joint_map
        nv = len(self._names)
        if nv != twin.n_servos:
            raise JaxterityError(
                f"ReflexController: the robot has {nv} actuated DOFs but the "
                f"twin carries {twin.n_servos} thermal nodes. The reflex maps "
                "servo i onto DOF i, so the two must agree."
            )
        self._lower = jnp.asarray([jmap[n].limits.lower for n in self._names])
        self._upper = jnp.asarray([jmap[n].limits.upper for n in self._names])
        effort = np.array([float(jmap[n].limits.effort) for n in self._names])
        velocity = np.array([float(jmap[n].limits.velocity) for n in self._names])
        # A zero limit in a model means "unspecified", not "immovable".
        self._effort = jnp.asarray(np.where(effort > 0, effort, np.inf))
        self._qd_max = jnp.asarray(np.where(velocity > 0, velocity, np.inf))
        self._brake = jnp.broadcast_to(
            jnp.asarray(brake_decel, dtype=float), (nv,)
        ).astype(float)
        if not bool(jnp.all(self._brake > 0)):
            raise JaxterityError(
                f"ReflexController: brake_decel must be positive per joint "
                f"(rad/s²); got {brake_decel!r}. It is the deceleration the "
                "stopping-distance barrier assumes the unit can achieve."
            )
        self._margin = float(
            joint_margin_rad
            if joint_margin_rad is not None
            else (envelope.joint_margin_rad or DEFAULT_JOINT_MARGIN_RAD)
        )
        frac = envelope.torque_margin_frac
        self._tau_limit = self._effort * (1.0 if frac is None else float(frac))
        # Per-row class-K gains: the mechanical rows share ``alpha``; the
        # thermal rows take the unit's own fitted time constant.
        thermal_alpha = (
            1.0 / (jnp.asarray(twin.R_th) * jnp.asarray(twin.C_th))
            if alpha_thermal is None
            else jnp.broadcast_to(
                jnp.asarray(alpha_thermal, dtype=float), (twin.n_servos,)
            )
        )
        self._alpha_rows = jnp.concatenate(
            [jnp.full(3 * nv, self.alpha), jnp.asarray(thermal_alpha, dtype=float)]
        )
        self._dynamics = twin.robot.functional_dynamics()
        self._nv = nv
        self._nq = twin.robot.build_system(actuation=True)._nq
        self.row_names = self._row_names()
        # Declared row priority, for the states where the rows genuinely
        # conflict — stopping before an end stop can need more braking torque
        # than the thermal budget allows, and something has to yield. The
        # mechanical rows are hard (uncapped multiplier); the thermal and bus
        # rows yield above ``thermal_priority`` × the torque box, because
        # slamming into a joint limit at speed is an immediate hazard and
        # exceeding a thermal budget is a slow one. The decision reports which
        # row yielded rather than leaving the trade implicit.
        box = float(np.max(np.where(np.isfinite(effort), effort, 0.0)))
        cap = float(thermal_priority) * max(box, 1.0)
        soft = self.n_rows - 3 * nv
        self._lam_max = jnp.concatenate(
            [jnp.full(3 * nv, jnp.inf), jnp.full(soft, cap)]
        )
        self.thermal_priority = float(thermal_priority)
        self._solve = jax.jit(self._solve_impl)

    # --- structure ------------------------------------------------------

    def _row_names(self) -> tuple[str, ...]:
        rows = [f"joint_upper[{n}]" for n in self._names]
        rows += [f"joint_lower[{n}]" for n in self._names]
        rows += [f"joint_velocity[{n}]" for n in self._names]
        rows += [f"servo_temperature[{n}]" for n in self.twin.servo_names]
        if self.envelope.bus_floor_v is not None and self.twin.has_battery:
            rows.append("bus_voltage")
        return tuple(rows)

    @property
    def n_rows(self) -> int:
        """Number of barrier rows the filter enforces."""
        return len(self.row_names)

    def _barriers(self, x: jax.Array) -> jax.Array:
        """Barrier values ``h(x)`` (safe where positive), stacked.

        ``x = [q, qd, T]``. The joint rows carry a stopping-distance term so
        every row is relative degree 1 in torque; the thermal row is the
        headroom to the ceiling.
        """
        nq, nv = self._nq, self._nv
        q = x[:nq]
        qd = x[nq : nq + nv]
        T = x[nq + nv :]
        # Squared approach speed toward each stop (smooth, one-sided).
        toward_upper = jnp.maximum(qd, 0.0) ** 2 / (2.0 * self._brake)
        toward_lower = jnp.maximum(-qd, 0.0) ** 2 / (2.0 * self._brake)
        h_upper = (self._upper - self._margin) - q[:nv] - toward_upper
        h_lower = q[:nv] - (self._lower + self._margin) - toward_lower
        # An unspecified velocity limit is a row that can never bind; a large
        # finite placeholder keeps the arithmetic (and the "binding row" report)
        # well-defined without an inf-times-zero.
        h_speed = jnp.where(
            jnp.isinf(self._qd_max),
            jnp.full_like(qd, 1e6),
            jnp.where(jnp.isinf(self._qd_max), 0.0, self._qd_max) ** 2 - qd**2,
        )
        h_temp = self.envelope.temp_ceiling_k - T
        return jnp.concatenate([h_upper, h_lower, h_speed, h_temp])

    def _constraints(self, x: jax.Array, tau: jax.Array) -> jax.Array:
        """The row-wise requirement ``c(τ) ≥ 0`` at state ``x``.

        Barrier rows use the CBF condition ``ḣ + α h``, with ``h`` and ``ḣ``
        obtained from a **single** forward pass (``jax.jvp`` of ``h`` along
        ``ẋ = f(x, τ)`` — dual numbers). The bus row is algebraic in torque and
        enters directly as ``V(τ) − floor``.
        """
        nq, nv = self._nq, self._nv
        q, qd, T = x[:nq], x[nq : nq + nv], x[nq + nv :]
        dq, dqd, dT = self.twin.state_derivative(q, qd, T, tau, dynamics=self._dynamics)
        dx = jnp.concatenate([dq, dqd, dT])
        h, hdot = jax.jvp(self._barriers, (x,), (dx,))
        rows = hdot + self._alpha_rows * h
        if self.envelope.bus_floor_v is not None and self.twin.has_battery:
            delivered = (
                tau if self.twin.K_t_scale is None else tau * self.twin.K_t_scale
            )
            volts = bus_voltage(
                delivered,
                V_oc=self.twin.V_oc,
                R_int=self.twin.R_int,
                k_i=jnp.asarray(self.twin.k_i, dtype=float),
                i_eps=self.twin.i_eps,
            )
            rows = jnp.concatenate(
                [rows, jnp.asarray([volts - self.envelope.bus_floor_v])]
            )
        return rows

    # --- the filter -----------------------------------------------------

    def _solve_impl(self, x: jax.Array, tau_desired: jax.Array):
        """Fixed-budget sequential-linearization solve for the clamp QP.

        ``min ½‖τ − τ_des‖²`` subject to ``c(τ) ≥ 0`` and the torque box. The
        thermal and bus rows are quadratic in torque, so each outer pass
        linearizes ``c`` about the current iterate and the inner loop runs dual
        projected gradient on ``A τ ≥ b``. Both loops have **static** trip
        counts and no data-dependent branches, so a tick costs the same
        arithmetic every time: ``outer × (one Jacobian + inner × two
        matrix-vector products)``.

        Relinearizing matters rather than being pedantic: the tangent of a
        convex quadratic row is optimistic below the linearization point, so a
        single pass would return a torque that still violates the true row (by
        ~8% of the thermal rate in the bench case). Two further passes close it.
        """
        lo = -self._tau_limit
        hi = self._tau_limit
        tau = tau_desired
        for _ in range(self.outer_iters):
            c0 = self._constraints(x, tau)
            A = jax.jacfwd(lambda t: self._constraints(x, t))(tau)
            b = A @ tau - c0
            # Row-normalize. The rows are in different units (rad/s² against
            # K/s against V), so their gradients differ by orders of magnitude;
            # un-normalized, the dual step size is set by the largest row and
            # the binding small one never converges inside a fixed budget.
            # Scaling an inequality by a positive constant leaves the feasible
            # set untouched.
            scale = jnp.linalg.norm(A, axis=1) + 1e-12
            A = A / scale[:, None]
            b = b / scale
            # Step from a Lipschitz bound on the dual gradient (‖AAᵀ‖₂ ≤ ‖A‖_F²).
            rho = 1.0 / (jnp.sum(A * A) + 1e-12)

            def step(lam, _, A=A, b=b, rho=rho):
                candidate = jnp.clip(tau_desired + A.T @ lam, lo, hi)
                updated = lam + rho * (b - A @ candidate)
                # Clipping the multiplier is an exact ℓ1 penalty with weight
                # ``lam_max``: it is how the declared row priority is enforced
                # when the rows genuinely conflict (see ``_lam_max``).
                return jnp.clip(updated, 0.0, self._lam_max), None

            lam, _ = jax.lax.scan(step, jnp.zeros(A.shape[0]), None, length=self.iters)
            tau = jnp.clip(tau_desired + A.T @ lam, lo, hi)
        slack = self._constraints(x, tau)
        return tau, slack, self._barriers(x)

    def _state(self, q, qd, T) -> jax.Array:
        q = jnp.asarray(q, dtype=float)
        qd = jnp.asarray(qd, dtype=float)
        T = jnp.asarray(T, dtype=float)
        if qd.shape[0] != self._nv or T.shape[0] != self.twin.n_servos:
            raise JaxterityError(
                f"ReflexController: state widths must be qd {self._nv} and T "
                f"{self.twin.n_servos}; got {qd.shape[0]} and {T.shape[0]}."
            )
        return jnp.concatenate([q, qd, T])

    def barrier_values(self, q, qd, T) -> dict[str, float]:
        """Per-row barrier value ``h`` at a state (negative = already outside).

        Args
        ----
        q, qd, T : array-like
            Joint positions (rad), velocities (rad/s), per-servo temperature
            (K).

        Returns
        -------
        dict[str, float]
            ``{row_name: h}``. The bus row is omitted — it is a function of
            torque, not of state.

        Units
        -----
        rad (joint rows), (rad/s)² (velocity rows), K (thermal rows).
        """
        values = np.asarray(self._barriers(self._state(q, qd, T)))
        return {name: float(v) for name, v in zip(self.row_names, values)}

    def filter(self, q, qd, T, tau_desired) -> ReflexDecision:
        """Clamp a proposed torque to the nearest one this unit can survive.

        Args
        ----
        q, qd, T : array-like
            Measured joint positions (rad), velocities (rad/s), per-servo
            temperature (K).
        tau_desired : array-like, shape ``(nv,)``
            The policy's proposed generalized force, N·m. Its origin is
            irrelevant — this is the syscall boundary.

        Returns
        -------
        ReflexDecision
            The torque to apply, whether the filter engaged, the binding row,
            and the residual violation (nonzero only when no torque satisfies
            the rows, i.e. the unit is already outside).

        Raises
        ------
        JaxterityError
            If the state widths disagree with the twin.

        Units
        -----
        rad, rad/s, K, N·m.

        Examples
        --------
        >>> reflex = ReflexController(twin, envelope)  # doctest: +SKIP
        >>> decision = reflex.filter(q, qd, T, policy_tau)  # doctest: +SKIP
        >>> decision.engaged  # doctest: +SKIP
        True
        """
        x = self._state(q, qd, T)
        desired = jnp.asarray(tau_desired, dtype=float)
        if desired.shape[0] != self._nv:
            raise JaxterityError(
                f"ReflexController.filter: tau_desired has {desired.shape[0]} "
                f"entries but the robot has {self._nv} actuated DOFs."
            )
        tau, slack, barriers = self._solve(x, desired)
        tau_np = np.asarray(tau, dtype=float)
        slack_np = np.asarray(slack, dtype=float)
        desired_np = np.asarray(desired, dtype=float)
        idx = int(np.argmin(slack_np))
        binding = self.row_names[idx]
        violation = float(max(0.0, -slack_np.min()))
        engaged = bool(np.max(np.abs(tau_np - desired_np)) > _ENGAGED_TOL)
        if not engaged:
            detail = f"pass-through: every barrier clear (binding {binding})"
        elif violation > _FEASIBLE_TOL:
            yielded = binding.startswith(("servo_temperature", "bus_voltage"))
            why = (
                " — a mechanical row needed more torque than the thermal/bus "
                "budget allows, and by declared priority (thermal_priority) the "
                "stop wins; the budget is being spent, not respected"
                if yielded
                else " — the unit is already outside this barrier, so no torque "
                "satisfies it; recovery, not filtering, is the remaining move"
            )
            detail = f"clamped, {binding} still violated by {violation:.3g}{why}"
        else:
            detail = (
                f"clamped by {binding} (slack {slack_np[idx]:.3g}); "
                f"|Δτ| = {float(np.max(np.abs(tau_np - desired_np))):.3g} N·m"
            )
        return ReflexDecision(
            tau=tau_np,
            tau_desired=desired_np,
            engaged=engaged,
            binding_constraint=binding,
            slack=float(slack_np[idx]),
            violation=violation,
            barriers={
                name: float(v)
                for name, v in zip(self.row_names, np.asarray(barriers, dtype=float))
            },
            detail=detail,
            tolerance=_FEASIBLE_TOL,
        )

    def recover(self, q, qd, T) -> ReflexDecision:
        """The reflex's own action: brake to rest, through the same barriers.

        What runs when the gate vetoes. The braking torque
        ``−brake_gain · q̇`` is a *request* like any other — it goes through
        :meth:`filter`, so the recovery itself cannot cook a servo or brown out
        the bus on the way to standstill.

        Args
        ----
        q, qd, T : array-like
            Measured state.

        Returns
        -------
        ReflexDecision
            The recovery torque and what constrained it.

        Units
        -----
        rad, rad/s, K; torque N·m.
        """
        qd_arr = jnp.asarray(qd, dtype=float)
        brake = -self.brake_gain * qd_arr
        decision = self.filter(q, qd, T, brake)
        return dataclasses.replace(
            decision, detail=f"recovery brake — {decision.detail}"
        )


class SimplexSwitch:
    """Arbitrate between userland's command and the reflex.

    The classic simplex arrangement, with the twin as the decision surface: the
    monitor judges the predicted horizon, and the reflex is always in the path.

    - **pass** — the horizon is clear; the command still goes through the
      filter (a clear forecast is not a promise about *this* tick).
    - **derate** — the command is scaled by the monitor's factor, then
      filtered.
    - **reflex** — the monitor vetoed; userland's command is dropped and the
      reflex's recovery action runs instead.

    Args
    ----
    monitor : SafetyMonitor
        The gate. Its envelope should be the one the reflex was built with —
        otherwise the two disagree about where the edge is.
    reflex : ReflexController
        The filter / recovery controller.

    Raises
    ------
    JaxterityError
        If the monitor and the reflex are bound to different units.
    """

    def __init__(self, monitor: SafetyMonitor, reflex: ReflexController):
        if monitor.twin.attestation_handle != reflex.twin.attestation_handle:
            raise JaxterityError(
                "SimplexSwitch: the monitor and the reflex are bound to "
                "different unit versions (their twin attestation handles "
                "differ). Build both from the same Twin — a gate that vetoes "
                "against one model and a reflex that recovers against another "
                "is not a safety layer."
            )
        self.monitor = monitor
        self.reflex = reflex

    def step(self, prediction: TwinRollout, q, qd, T, tau_desired) -> SimplexDecision:
        """Decide and produce the torque to apply for one tick.

        Args
        ----
        prediction : TwinRollout
            The co-processor's forecast for this tick (see
            :class:`~jaxterity.selfcheck.CoProcessor`).
        q, qd, T : array-like
            Measured state.
        tau_desired : array-like
            Userland's proposed generalized force, N·m.

        Returns
        -------
        SimplexDecision
            The mode, the torque, and both underlying decisions.

        Units
        -----
        rad, rad/s, K, N·m.
        """
        desired = np.asarray(tau_desired, dtype=float)
        safety = self.monitor.evaluate(prediction, action=desired)
        if safety.action is SafetyAction.VETO:
            decision = self.reflex.recover(q, qd, T)
            mode = "reflex"
        elif safety.action is SafetyAction.DERATE:
            decision = self.reflex.filter(q, qd, T, safety.factor * desired)
            mode = "derate"
        else:
            decision = self.reflex.filter(q, qd, T, desired)
            mode = "pass"
        return SimplexDecision(
            mode=mode, tau=decision.tau, safety=safety, reflex=decision
        )


__all__ = [
    "ReflexController",
    "ReflexDecision",
    "SimplexSwitch",
    "SimplexDecision",
    "DEFAULT_ALPHA",
    "DEFAULT_BRAKE_DECEL",
    "DEFAULT_ITERS",
    "DEFAULT_JOINT_MARGIN_RAD",
]
