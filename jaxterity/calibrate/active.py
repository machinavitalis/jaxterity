# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Active system identification: excite a robot over its servo bus, then fit.

The offline ``examples/sysid_so101.py`` story is "generate telemetry by
simulating a *true* robot, then identify it." Active mode closes that loop on a
*physical* unit: it streams a designed, within-limit excitation to the arm over
its servo bus, captures the motion, and feeds the **same** forward-mode
Levenberg–Marquardt fit — all from one ``jaxterity calibrate --active --port …``
invocation, no manual log handoff.

The transport is abstracted behind :class:`ActiveLink` so the whole path is
gate-testable without hardware:

- :class:`LoopbackLink` — a **seeded** simulated arm. It runs the excitation
  through the robot's own free dynamics with *planted* parameters, so the fit
  must recover them. Deterministic: a fixed seed gives bit-identical telemetry
  and therefore a bit-identical fit.
- :class:`SerialBusLink` — the real Feetech STS-bus transport (lazy ``pyserial``).
  Validated on hardware separately (its test is ``hardware``-marked and skipped
  by ``scripts/check.sh``).

The excitation is a **coast probe**: each joint gets a small initial velocity and
the arm coasts under its own dynamics (exactly the regime
``examples/sysid_so101.py`` identifies damping from). It is *open-loop* and
*limit-bounded* — :func:`validate_excitation` simulates it first and rejects any
profile that would drive a joint past its position or velocity limit **before a
single command is sent** (honest scope: no closed-loop excitation that could
destabilize the arm).

Honest gap: the fit observer is the free-dynamics model the loopback also uses,
so loopback recovery is exact. A real position-controlled bus coasts via the
servos' velocity/torque mode; the fidelity of the free-dynamics observer to a
real coast (vs. a servo-in-the-loop model) is the piece validated on hardware.
"""

from __future__ import annotations

import abc
import dataclasses

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
from jaxonomy import SimulatorOptions

from ..data.schema import JointTelemetry
from ..errors import CalibrationError
from ..robot import Robot
from ..robot.structure import JointType
from ..sysid import SysidResult
from ..sysid.recipes import (
    _DEFAULT_COND_MAX,
    IdentifiabilityReport,
    identifiability_from_jacobian,
)
from ..sysid.uncertainty import (
    bootstrap_covariance,
    laplace_covariance,
    per_parameter_variance,
)

#: Forward-mode requires the reverse-mode adjoint off (so JAX traces the real ops,
#: including MJX's constraint solver); fixed-step RK4 keeps the run deterministic.
#: Identical to ``examples/sysid_so101.py`` so the active fit matches the offline one.
_DEFAULT_OPTIONS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="rk4",
    enable_autodiff=False,
    max_minor_step_size=0.005,
    max_major_steps=200,
)

#: Default coast-probe sample times (s) — multiple shooting conditions the fit.
#: Kept short so the gravity-driven coast of the *uncalibrated* SO-101 stays
#: comfortably within joint limits (peak ≈ 0.68× the limit at 0.2 s); a longer
#: horizon slams ``shoulder_lift`` into its stop (and is rejected by
#: :func:`validate_excitation`).
_DEFAULT_SAMPLE_TIMES = (0.1, 0.2)
#: Fraction of a joint's position half-range the excitation may use (safety margin).
_LIMIT_MARGIN = 0.9

#: Parameter fields that are physically non-negative. A fitted
#: ``"<joint>.<field>"`` whose field suffix appears here gets a lower bound of 0
#: in the projected LM fit (a negative damping/mass/inertia is not a fit, it is
#: an identifiability failure — see :func:`excitation_identifiability`).
_NONNEGATIVE_FIELDS = frozenset(
    {"damping", "armature", "frictionloss", "mass", "inertia"}
)


def _lower_bounds(target_params: tuple[str, ...]) -> np.ndarray:
    """Per-parameter lower bounds: 0 for non-negative fields, else unbounded."""
    return np.asarray(
        [
            0.0 if p.rsplit(".", 1)[-1] in _NONNEGATIVE_FIELDS else -np.inf
            for p in target_params
        ],
        dtype=float,
    )


def dof_joint_names(robot: Robot) -> tuple[str, ...]:
    """Movable-joint names in MJX DOF order (the order of simulation state).

    Args
    ----
    robot : Robot
        The robot whose lowered model defines the DOF order.

    Returns
    -------
    tuple[str, ...]
        Non-fixed, non-floating joint names ordered by their MJX DOF address —
        the column order of the ``q`` / ``qd`` telemetry channels.
    """
    import mujoco

    mj = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    pairs: list[tuple[int, str]] = []
    for joint in robot.joints:
        if joint.joint_type in (JointType.FIXED, JointType.FLOATING):
            continue
        jid = int(mj.joint(joint.name).id)
        pairs.append((int(mj.jnt_dofadr[jid]), joint.name))
    return tuple(name for _, name in sorted(pairs))


@dataclasses.dataclass(frozen=True)
class Excitation:
    """A designed, within-limit coast probe the bus will execute.

    Each joint starts at ``q0`` with velocity ``qd0`` and the arm coasts under
    its own dynamics; the state is captured at ``sample_times``. Stored as plain
    tuples (JAX-free, hashable, deterministic).

    Args
    ----
    q0 : tuple[float, ...]
        Initial joint positions, rad (length = robot DOF count).
    qd0 : tuple[float, ...]
        Initial joint velocities, rad/s (the excitation "kick").
    sample_times : tuple[float, ...]
        Strictly-increasing capture times, s.
    """

    q0: tuple[float, ...]
    qd0: tuple[float, ...]
    sample_times: tuple[float, ...]

    @property
    def n_dof(self) -> int:
        """Number of joints the excitation drives."""
        return len(self.q0)


def design_coast_excitation(
    robot: Robot,
    *,
    speed: float = 0.1,
    sample_times: tuple[float, ...] = _DEFAULT_SAMPLE_TIMES,
) -> Excitation:
    """Design a gentle, within-limit coast probe (uniform initial velocity).

    Starts every joint at zero with a small uniform velocity ``speed``. The
    result is checked by :func:`validate_excitation` before it is ever sent.

    Args
    ----
    robot : Robot
        The robot to excite.
    speed : float
        Initial joint velocity applied to every DOF, rad/s.
    sample_times : tuple[float, ...]
        Capture times, s.

    Returns
    -------
    Excitation
        A coast probe with ``q0 = 0`` and ``qd0 = speed`` on every DOF.
    """
    nv = robot.build_system(actuation=True)._nv
    return Excitation(
        q0=tuple(0.0 for _ in range(nv)),
        qd0=tuple(float(speed) for _ in range(nv)),
        sample_times=tuple(float(t) for t in sample_times),
    )


def _make_observer(robot, target_params, excitation, options):
    """Return a jitted ``observe(values) -> stacked states`` (sysid_so101 idiom).

    Builds the diagram once with the excitation's initial state; overrides the
    ``target_params`` in the context per call; simulates to each sample time and
    stacks the full states ``[q, qd]``. Forward-mode-friendly (adjoint off).
    """
    q0 = jnp.asarray(excitation.q0, dtype=float)
    qd0 = jnp.asarray(excitation.qd0, dtype=float)
    diagram, system = robot.to_diagram(q0=q0, qd0=qd0, return_system=True)
    sid = system.system_id

    @jax.jit
    def observe(values):
        ctx = diagram.create_context()
        sub = ctx[sid]
        for name, value in zip(target_params, values):
            sub = sub.with_parameter(name, value)
        ctx = ctx.with_subcontext(sid, sub)
        states = [
            jaxonomy.simulate(diagram, ctx, (0.0, t), options=options)
            .context[sid]
            .continuous_state
            for t in excitation.sample_times
        ]
        return jnp.concatenate(states)

    return observe, system._nq, system._nv


def _forward_mode_lm(
    observe, target, n_params, *, iterations, lam, max_step, lower=None, est0=None
):
    """Forward-mode Gauss–Newton / Levenberg–Marquardt (sysid_so101 idiom).

    Differentiates ``observe`` with :func:`jax.jacfwd` (traverses MJX's
    constraint solver). When ``lower`` is given (per-parameter lower bounds,
    ``-inf`` for unbounded), each iterate is projected onto ``est >= lower`` —
    projected Gauss–Newton, the ``windowed.py`` idiom — so physically
    non-negative parameters cannot converge below zero. ``est0`` sets the
    starting estimate (default zeros — the joint-damping lane's convention);
    a fit whose parameters gate the sensitivity of others (e.g. a payload
    mass gating its CoM columns) must start at a non-degenerate prior.

    Returns ``(estimate, history, jac0, hit_lower)``: the estimate, the
    per-iteration residual cost, the Jacobian at the *initial* estimate (reused
    by the identifiability report, so it costs nothing extra), and a boolean
    mask marking parameters whose final unprojected step fell below ``lower``
    (the fit is pinned at the bound — a red flag, not a fit).
    """
    observe_jac = jax.jit(jax.jacfwd(observe))
    est = jnp.zeros(n_params) if est0 is None else jnp.asarray(est0, dtype=float)
    history: list[float] = []
    jac0: np.ndarray | None = None
    hit_lower = np.zeros(n_params, dtype=bool)
    for _ in range(iterations):
        residual = np.asarray(observe(est)) - target
        jac = np.asarray(observe_jac(est))
        if jac0 is None:
            jac0 = jac
        history.append(float(residual @ residual))
        hessian = jac.T @ jac
        damped = hessian + lam * np.diag(np.diag(hessian) + 1e-12)
        step = np.clip(-np.linalg.solve(damped, jac.T @ residual), -max_step, max_step)
        raw = np.asarray(est) + step
        if lower is None:
            est = jnp.asarray(raw)
        else:
            hit_lower = raw < lower
            est = jnp.asarray(np.maximum(raw, lower))
    if jac0 is None:  # iterations == 0 — still report sensitivity at the start.
        jac0 = np.asarray(observe_jac(est))
    return est, history, jac0, hit_lower


def validate_excitation(
    robot: Robot,
    excitation: Excitation,
    *,
    options: SimulatorOptions | None = None,
    n_check: int = 24,
) -> None:
    """Reject an excitation that would exceed a joint's position or velocity limit.

    Called **before** any command reaches the bus. Checks each DOF's initial
    velocity against its velocity limit, then simulates the coast (on the
    *uncalibrated* robot) over a dense grid and checks the peak |position|
    against the joint's position limits (with a safety margin).

    Args
    ----
    robot : Robot
        The robot to excite.
    excitation : Excitation
        The candidate coast probe.
    options : SimulatorOptions | None
        Integration options (defaults to the module's fixed-step RK4).
    n_check : int
        Number of dense grid points over the horizon to check positions at.

    Raises
    ------
    CalibrationError
        If any joint's velocity or simulated position would exceed its limit.

    Units
    -----
    Positions in rad, velocities in rad/s.
    """
    options = options or _DEFAULT_OPTIONS
    names = dof_joint_names(robot)
    jmap = robot.joint_map
    qd0 = np.asarray(excitation.qd0, dtype=float)

    # 1. velocity limits — cheap, no simulation needed.
    for i, name in enumerate(names):
        vlim = jmap[name].limits.velocity
        if vlim > 0 and abs(float(qd0[i])) > vlim:
            raise CalibrationError(
                f"active sysid excitation would exceed the velocity limit of "
                f"joint {name!r}: |qd0|={abs(float(qd0[i])):.3g} rad/s > "
                f"{vlim:.3g} rad/s. Fix: lower the excitation speed."
            )

    # 2. position limits — simulate the coast on a dense grid and check peaks.
    horizon = max(excitation.sample_times)
    grid = tuple(float(t) for t in np.linspace(horizon / n_check, horizon, n_check))
    probe = dataclasses.replace(excitation, sample_times=grid)
    observe, nq, nv = _make_observer(robot, (), probe, options)
    states = np.asarray(observe(jnp.zeros(0))).reshape(len(grid), nq + nv)
    q_traj = states[:, :nv]  # fixed-base arm: nq == nv, DOF-aligned
    for i, name in enumerate(names):
        lim = jmap[name].limits
        peak = float(np.max(np.abs(q_traj[:, i])))
        bound = _LIMIT_MARGIN * max(abs(lim.lower), abs(lim.upper))
        if bound > 0 and peak > bound:
            raise CalibrationError(
                f"active sysid excitation would drive joint {name!r} to "
                f"|q|={peak:.3g} rad, past {_LIMIT_MARGIN:.0%} of its limit "
                f"({bound:.3g} rad of [{lim.lower:.3g}, {lim.upper:.3g}]). "
                "Fix: shorten the horizon or lower the excitation speed."
            )


def excitation_identifiability(
    robot: Robot,
    target_params: tuple[str, ...],
    excitation: Excitation | None = None,
    *,
    at_params: dict[str, float] | None = None,
    cond_max: float = _DEFAULT_COND_MAX,
    options: SimulatorOptions | None = None,
) -> IdentifiabilityReport:
    """Report whether a designed excitation can identify the target parameters.

    Simulates the coast probe once at the prior parameter values — no hardware,
    no noise, no fit — and differentiates the stacked ``[q, qd]`` observation
    w.r.t. ``target_params`` by forward-mode autodiff. That sensitivity
    Jacobian ``J`` is exactly what the Levenberg–Marquardt fit in
    :func:`active_identify` iterates on, and under iid Gaussian sensor noise of
    variance ``σ²`` the Gram matrix ``JᵀJ/σ²`` is the excitation's **Fisher
    information** about the parameters: a near-zero singular value of the
    column-normalized ``J`` means the probe carries (almost) no information
    along that parameter direction, so no amount of LM iteration — or hardware
    time — will pin it down; the noise decides the answer. Run this check
    *before* spending minutes of fitting or moving a real arm. A ``rank``
    deficit or ``condition_number`` above ``cond_max`` means "redesign the
    excitation" (longer horizon, faster kick, more sample times), not "fit
    harder"; ``weak_params`` names the parameters with the largest normalized
    marginal variance ``[(JᵀJ)⁻¹]_ii`` — the first to go unstable as noise
    rises. Cost: one forward-mode Jacobian evaluation, i.e. roughly one LM
    iteration of :func:`active_identify` (which repeats this check for free
    and folds it into the result's warnings).

    Args
    ----
    robot : Robot
        The (uncalibrated) robot the probe will run on.
    target_params : tuple[str, ...]
        Identifiable leaves to be fitted, keyed ``"<joint>.<field>"`` (e.g.
        ``"shoulder_lift.damping"``) — the Jacobian's column order.
    excitation : Excitation | None
        The designed coast probe. Defaults to :func:`design_coast_excitation`.
    at_params : dict[str, float] | None
        Parameter values (keyed like ``target_params``) to linearize at.
        Defaults to zero for every target — the same initial estimate the LM
        fit starts from, so the report matches the fit's first iteration.
    cond_max : float
        Conditioning ceiling for the ``identifiable`` verdict (default 1e6).
    options : SimulatorOptions | None
        Integration options (defaults to the module's fixed-step RK4).

    Returns
    -------
    IdentifiabilityReport
        Singular values, rank, condition number, verdict, and ``weak_params``
        of the excitation's sensitivity Jacobian (see
        :class:`~jaxterity.sysid.IdentifiabilityReport`;
        :meth:`~jaxterity.sysid.IdentifiabilityReport.raise_if_unidentifiable`
        turns the verdict into a hard gate).

    Raises
    ------
    CalibrationError
        If ``target_params`` does not match the Jacobian's column count
        (malformed parameter keys).

    Units
    -----
    Observations stack joint positions (rad) and velocities (rad/s); Jacobian
    columns are per unit of each target parameter (e.g. rad per N·m·s/rad for
    a damping). Column normalization makes the report scale-free, so the
    verdict is unchanged by unit choices.
    """
    options = options or _DEFAULT_OPTIONS
    excitation = excitation or design_coast_excitation(robot)
    target = tuple(target_params)
    observe, _nq, _nv = _make_observer(robot, target, excitation, options)
    if at_params is None:
        values = jnp.zeros(len(target))
    else:
        values = jnp.asarray([float(at_params[p]) for p in target], dtype=float)
    jac = np.asarray(jax.jit(jax.jacfwd(observe))(values))
    return identifiability_from_jacobian(jac, target, cond_max=cond_max)


class ActiveLink(abc.ABC):
    """Transport that executes an :class:`Excitation` and returns captured telemetry."""

    @property
    @abc.abstractmethod
    def n_dof(self) -> int:
        """Number of joints on the bus."""

    @abc.abstractmethod
    def run_excitation(self, excitation: Excitation) -> JointTelemetry:
        """Execute the excitation and return per-joint ``(q, qd, tau)`` telemetry."""

    def close(self) -> None:
        """Release the transport (no-op by default)."""


class LoopbackLink(ActiveLink):
    """A seeded simulated arm: the robot's free dynamics with planted parameters.

    Runs the excitation through the *true* robot (the planted parameters bound
    on top of ``base_robot``) and returns the resulting motion as telemetry, so
    :func:`active_identify` must recover the planted values. Deterministic: a
    fixed ``seed`` (with optional Gaussian sensor noise) yields bit-identical
    telemetry.

    Args
    ----
    base_robot : Robot
        The uncalibrated robot the fit also uses (so recovery is exact).
    planted : dict[str, float]
        The *true* parameter values to plant, keyed ``"<joint>.<field>"``.
    seed : int
        PRNG seed for the optional sensor noise (determinism).
    noise_std : float
        Standard deviation of additive Gaussian position/velocity noise (rad,
        rad/s). Default 0.0 (noise-free → exact recovery).
    options : SimulatorOptions | None
        Integration options (defaults to the module's fixed-step RK4).
    """

    def __init__(
        self,
        base_robot: Robot,
        planted: dict[str, float],
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        options: SimulatorOptions | None = None,
    ):
        self._robot = base_robot
        self._planted = dict(planted)
        self._seed = int(seed)
        self._noise_std = float(noise_std)
        self._options = options or _DEFAULT_OPTIONS
        self._names = dof_joint_names(base_robot)

    @property
    def n_dof(self) -> int:
        return len(self._names)

    def run_excitation(self, excitation: Excitation) -> JointTelemetry:
        target = tuple(self._planted)
        values = jnp.asarray([self._planted[k] for k in target], dtype=float)
        observe, nq, nv = _make_observer(self._robot, target, excitation, self._options)
        n_t = len(excitation.sample_times)
        states = np.asarray(observe(values)).reshape(n_t, nq + nv)
        q = states[:, :nv].copy()
        qd = states[:, nq : nq + nv].copy()
        if self._noise_std > 0.0:
            rng = np.random.default_rng(self._seed)
            q += rng.normal(0.0, self._noise_std, size=q.shape)
            qd += rng.normal(0.0, self._noise_std, size=qd.shape)
        t = np.asarray(excitation.sample_times, dtype=np.float64)
        fs = 1.0 / float(np.min(np.diff(t))) if t.size >= 2 else 1.0
        return JointTelemetry(
            time=t,
            sample_rate_hz=fs,
            joint_names=self._names,
            q=q,
            qd=qd,
            tau=np.zeros_like(q),
        )


class SerialBusLink(ActiveLink):
    """Real Feetech STS servo-bus transport (lazy ``pyserial``).

    Drives the SO-101's serial-bus servos: enters velocity mode, applies the
    excitation kick, releases to coast, and reads back position/velocity/load/
    temperature/voltage at the sample times. Validated on hardware separately —
    constructing it needs a real port, so its test is ``hardware``-marked and
    skipped by ``scripts/check.sh``.

    Args
    ----
    port : str
        Serial device path (e.g. ``"/dev/ttyACM0"``).
    joint_names : tuple[str, ...]
        Joint names in DOF order (see :func:`dof_joint_names`).
    servo_ids : tuple[int, ...]
        Bus IDs aligned with ``joint_names``.
    ticks_per_rad : tuple[float, ...]
        Per-servo encoder resolution (counts/rad), aligned with ``joint_names``.
    baudrate : int
        Bus baud rate.

    Raises
    ------
    CalibrationError
        If ``pyserial`` is not installed.
    """

    def __init__(
        self,
        port: str,
        joint_names: tuple[str, ...],
        servo_ids: tuple[int, ...],
        ticks_per_rad: tuple[float, ...],
        *,
        baudrate: int = 1_000_000,
    ):
        try:
            import serial  # noqa: F401  (pyserial)
        except ImportError as e:  # pragma: no cover - hardware path
            raise CalibrationError(
                "SerialBusLink needs pyserial (the real servo-bus transport). "
                "Install the [hardware] extra, or use LoopbackLink for a "
                "hardware-free run."
            ) from e
        import serial

        if not (len(joint_names) == len(servo_ids) == len(ticks_per_rad)):
            raise CalibrationError(
                "SerialBusLink: joint_names, servo_ids, and ticks_per_rad must "
                f"be the same length; got {len(joint_names)}/{len(servo_ids)}/"
                f"{len(ticks_per_rad)}."
            )
        self._names = tuple(joint_names)
        self._ids = tuple(int(i) for i in servo_ids)
        self._ticks_per_rad = tuple(float(t) for t in ticks_per_rad)
        self._port = serial.Serial(port, baudrate=baudrate, timeout=0.05)

    @property
    def n_dof(self) -> int:
        return len(self._names)

    def run_excitation(  # pragma: no cover - hardware
        self, excitation: Excitation
    ) -> JointTelemetry:
        # Hardware path: enter velocity mode, apply qd0, release to coast, and
        # poll feedback at the sample times. The wire protocol (Feetech STS sync
        # read/write) lives behind these helpers; this orchestration is what the
        # hardware-marked test exercises against a real bus.
        raise CalibrationError(
            "SerialBusLink.run_excitation runs only against a real servo bus "
            "(hardware-marked, validated separately). Use "
            "LoopbackLink for a hardware-free run."
        )

    def close(self) -> None:  # pragma: no cover - hardware
        self._port.close()


class ReplayLink(ActiveLink):
    """Replay pre-captured telemetry through the fit (split capture/fit).

    The real bus and the simulation-heavy fit need not run on the same machine.
    A lightweight on-device script (no JAX/MJX) streams the excitation over the
    bus and records ``(q, qd, tau)`` telemetry; this link feeds that captured
    telemetry into the **same** :func:`active_identify` on a workstation with the
    full stack. So a Raspberry Pi (which has the servo bus but not the ML stack)
    captures, and the fit runs where the simulator already works — without a
    manual fit handoff, just a telemetry-file handoff.

    The captured telemetry must match the excitation :func:`active_identify`
    validates and fits against: same joint count, and one telemetry sample per
    excitation sample time (the fit's observer samples at exactly those times).

    Args
    ----
    telemetry : JointTelemetry
        The captured per-joint motion, ``q``/``qd`` shape ``(n_samples, n_dof)``
        in the robot's DOF order.
    """

    def __init__(self, telemetry: JointTelemetry):
        self._telemetry = telemetry

    @property
    def n_dof(self) -> int:
        return self._telemetry.q.shape[1]

    def run_excitation(self, excitation: Excitation) -> JointTelemetry:
        if self._telemetry.q.shape[1] != excitation.n_dof:
            raise CalibrationError(
                f"replayed telemetry drives {self._telemetry.q.shape[1]} joints "
                f"but the excitation has {excitation.n_dof}. They must match "
                "(DOF order)."
            )
        if self._telemetry.n_samples != len(excitation.sample_times):
            raise CalibrationError(
                f"replayed telemetry has {self._telemetry.n_samples} samples but "
                f"the excitation has {len(excitation.sample_times)} sample times. "
                "The capture must sample at exactly the excitation's times (the "
                "fit observer samples there)."
            )
        return self._telemetry


def active_identify(
    robot: Robot,
    link: ActiveLink,
    target_params: tuple[str, ...],
    *,
    excitation: Excitation | None = None,
    iterations: int = 8,
    lam: float = 1e-2,
    max_step: float = 0.08,
    options: SimulatorOptions | None = None,
    uncertainty: str | None = "laplace",
    n_boot: int = 64,
    seed: int = 0,
    bounds: bool = True,
    cond_max: float = _DEFAULT_COND_MAX,
    warn_rel_std: float = 1.0,
) -> SysidResult:
    """Run active system identification end to end and return a calibrated fit.

    Validates the excitation (rejecting out-of-limit profiles before any command
    is sent), streams it over ``link``, and recovers ``target_params`` from the
    captured telemetry with a forward-mode Levenberg–Marquardt fit — the same
    fit ``examples/sysid_so101.py`` uses offline. When ``uncertainty`` is set,
    also attaches a local per-parameter covariance, readable through
    :meth:`~jaxterity.sysid.SysidResult.interval`.

    The result is guarded three ways, so an under-informed fit fails loudly
    instead of returning plausible nonsense (silence was the T-219 bug):

    - **Identifiability** — the excitation's sensitivity Jacobian (the fit's
      own first iteration, so it costs nothing extra) is scored exactly as
      :func:`excitation_identifiability`; a rank deficit, a condition number
      above ``cond_max``, or named weak parameters become result warnings.
    - **Physical bounds** — with ``bounds`` on, physically non-negative
      parameters (``damping``/``armature``/``frictionloss``/``mass``/
      ``inertia`` fields) are projected to ``>= 0`` each LM iteration; a fit
      pinned at 0 by the projection is flagged by name.
    - **Uncertainty** — any parameter whose standard error exceeds
      ``warn_rel_std`` × its fitted magnitude is flagged as unresolved.

    Args
    ----
    robot : Robot
        The (uncalibrated) robot to identify.
    link : ActiveLink
        The transport (loopback fake or real serial bus).
    target_params : tuple[str, ...]
        Identifiable leaves to fit, keyed ``"<joint>.<field>"`` (e.g.
        ``"shoulder_lift.damping"``).
    excitation : Excitation | None
        The coast probe. Defaults to :func:`design_coast_excitation`.
    iterations, lam, max_step : int, float, float
        Levenberg–Marquardt controls.
    options : SimulatorOptions | None
        Integration options (defaults to the module's fixed-step RK4).
    uncertainty : str or None
        Fit-uncertainty estimator: ``"laplace"`` (default — the Gauss-Newton
        covariance at the optimum, one extra solve), ``"bootstrap"`` (a residual
        bootstrap, ``n_boot`` refits), or ``None`` to skip it.
    n_boot : int
        Number of bootstrap refits when ``uncertainty="bootstrap"``.
    seed : int
        PRNG seed for the bootstrap resampling (determinism).
    bounds : bool
        Project physically non-negative parameters to ``>= 0`` during the fit
        (default). ``False`` restores the unconstrained fit (the covariance
        and identifiability warnings still fire).
    cond_max : float
        Conditioning ceiling for the identifiability warning (default 1e6).
    warn_rel_std : float
        Relative standard-error threshold for the uncertainty warning:
        warn when ``std_error > warn_rel_std * |fitted|`` (default 1.0,
        i.e. a 100% relative error).

    Returns
    -------
    SysidResult
        The fitted ``target_params``, the final-state RMS residual (plus the
        excitation's condition number under ``residuals``), the telemetry
        hash, any identifiability / bound / uncertainty warnings, and (when
        requested) the per-parameter covariance — ready to
        :meth:`~jaxterity.sysid.SysidResult.apply_to`.

    Raises
    ------
    CalibrationError
        If the excitation is out of limits, its width disagrees with the link,
        or ``uncertainty`` is not a recognized estimator.
    """
    options = options or _DEFAULT_OPTIONS
    excitation = excitation or design_coast_excitation(robot)
    if excitation.n_dof != link.n_dof:
        raise CalibrationError(
            f"excitation drives {excitation.n_dof} joints but the link has "
            f"{link.n_dof}. They must match (DOF order)."
        )

    # 1. reject an out-of-limit excitation BEFORE any command reaches the bus.
    validate_excitation(robot, excitation, options=options)

    # 2. stream it and capture telemetry.
    telemetry = link.run_excitation(excitation)

    # 3. fit the captured motion (forward-mode LM, same observer as offline).
    target = np.concatenate(
        [
            np.concatenate([telemetry.q[k], telemetry.qd[k]])
            for k in range(telemetry.n_samples)
        ]
    )
    observe, _nq, _nv = _make_observer(robot, target_params, excitation, options)
    lower = _lower_bounds(target_params) if bounds else None
    estimate, history, jac0, hit_lower = _forward_mode_lm(
        observe,
        target,
        len(target_params),
        iterations=iterations,
        lam=lam,
        max_step=max_step,
        lower=lower,
    )
    estimate = np.asarray(estimate)
    fitted = {p: jnp.asarray(float(v)) for p, v in zip(target_params, estimate)}
    rms = float(np.sqrt(history[-1] / target.size))

    # 4. identifiability of the excitation — scored from the fit's own
    #    first-iteration Jacobian (same matrix excitation_identifiability builds).
    notes: list[str] = []
    report = identifiability_from_jacobian(jac0, target_params, cond_max=cond_max)
    if not report.identifiable:
        deficit = len(target_params) - report.rank
        notes.append(
            f"active_identify: the excitation is not identifiable — rank "
            f"{report.rank} of {len(target_params)} parameters (deficit "
            f"{deficit}), condition number {report.condition_number:.3g} > "
            f"{cond_max:.3g}. The fitted values are unconstrained along the "
            "missing directions; redesign the excitation (see "
            "excitation_identifiability)."
        )
    elif report.weak_params:
        notes.append(
            f"active_identify: weakly identified under this excitation: "
            f"{', '.join(report.weak_params)} (condition number "
            f"{report.condition_number:.3g}). Treat them as loosely constrained."
        )
    for name, hit in zip(target_params, hit_lower):
        if hit:
            notes.append(
                f"active_identify: {name} fit hit lower bound 0 (the "
                "unconstrained step wanted a negative, unphysical value); the "
                "parameter is likely unidentifiable under this excitation — "
                "see excitation_identifiability."
            )

    # 5. (optional) attach a local fit covariance + flag unresolved parameters.
    covariance = _estimate_covariance(
        observe,
        target,
        estimate,
        target_params,
        uncertainty=uncertainty,
        n_boot=n_boot,
        seed=seed,
        iterations=iterations,
        lam=lam,
        max_step=max_step,
        lower=lower,
    )
    if covariance is not None:
        for name in target_params:
            std = float(np.sqrt(max(float(covariance[name]), 0.0)))
            scale = abs(float(fitted[name]))
            if std > warn_rel_std * max(scale, np.finfo(float).tiny):
                notes.append(
                    f"active_identify: {name} is unresolved — standard error "
                    f"{std:.3g} exceeds {warn_rel_std:.0%} of the fitted value "
                    f"{scale:.3g}. Do not trust the point estimate; use a "
                    "richer excitation or lower-noise telemetry."
                )
    return SysidResult(
        fitted=fitted,
        covariance=covariance,
        residuals={
            "final_state_rms": rms,
            "excitation_condition_number": float(report.condition_number),
        },
        telemetry_hash=telemetry.hash().hex(),
        recipe_name="active.sysid",
        recipe_version="1.0",
        warnings=tuple(notes),
    )


def _estimate_covariance(
    observe,
    target,
    estimate,
    target_params,
    *,
    uncertainty,
    n_boot,
    seed,
    iterations,
    lam,
    max_step,
    lower=None,
) -> dict[str, float] | None:
    """Per-parameter marginal variance for an active fit, or ``None``.

    Shares the forward-mode ``observe`` of the fit: Laplace differentiates the
    residual at the optimum; bootstrap refits resampled-residual datasets with
    the same :func:`_forward_mode_lm`.
    """
    if uncertainty is None:
        return None
    target = np.asarray(target, dtype=float)
    names = tuple(target_params)

    def residual_fn(theta):
        return observe(jnp.asarray(theta, dtype=float)) - jnp.asarray(target)

    if uncertainty == "laplace":
        cov = laplace_covariance(residual_fn, estimate, n_obs=target.size)
    elif uncertainty == "bootstrap":
        predict_hat = np.asarray(observe(jnp.asarray(estimate, dtype=float)))
        residual_hat = target - predict_hat

        def refit(target_star):
            est_star, _, _, _ = _forward_mode_lm(
                observe,
                np.asarray(target_star, dtype=float),
                len(names),
                iterations=iterations,
                lam=lam,
                max_step=max_step,
                lower=lower,
            )
            return np.asarray(est_star)

        cov = bootstrap_covariance(
            predict_hat, residual_hat, refit, n_boot=n_boot, seed=seed
        )
    else:
        raise CalibrationError(
            f"active_identify: unknown uncertainty estimator {uncertainty!r}. "
            "Use 'laplace' (default), 'bootstrap', or None."
        )
    return per_parameter_variance(cov, names)


__all__ = [
    "ActiveLink",
    "LoopbackLink",
    "SerialBusLink",
    "ReplayLink",
    "Excitation",
    "design_coast_excitation",
    "validate_excitation",
    "excitation_identifiability",
    "active_identify",
    "dof_joint_names",
]
