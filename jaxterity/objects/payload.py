# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Grasped-payload identification: what the camera cannot see, the arm can feel.

A manipulated object's mass, centre of mass, and inertia are invisible in
pixels — two visually identical balls (dense vs. hollow) demand different
motions. But once the object is *held*, the robot's own joint telemetry carries
that information: the arm's calibrated dynamics are the force scale that makes
the payload's absolute parameters identifiable (the missing scale that makes
passive video-only estimation a ratio-only problem).

This module attaches a :class:`RigidPayload` to a grasp link as a **welded MJX
body** (:func:`attach_payload` — no new DOFs; MuJoCo composes the dynamics, no
manual parallel-axis bookkeeping) and reuses the active-sysid lane end to end:
the same coast excitation, the same forward-mode Levenberg–Marquardt fit, and
the same three honesty guards as ``active_identify`` (T-219/C-119) — a Fisher
identifiability report *before* any fit (:func:`payload_identifiability`),
non-negative bound projection during it, and named unresolved-uncertainty
warnings after it. An under-informed probe fails loudly, never silently.

The transport is the lane's :class:`~jaxterity.calibrate.active.ActiveLink`:
:class:`PayloadLoopbackLink` is the seeded planted-parameter fake (recovery
tests), and a real capture replays through
:class:`~jaxterity.calibrate.active.ReplayLink` unchanged.

Frame
-----
``RigidPayload.com`` and ``offset`` are expressed in the payload's own body
frame and the attach link's body frame respectively; the fitted CoM is
"where is the object's centre of mass relative to the grasp point."

Units
-----
SI throughout: mass kg, CoM/offset m, inertia kg·m².
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxonomy import SimulatorOptions

from ..calibrate.active import (
    _DEFAULT_OPTIONS,
    ActiveLink,
    Excitation,
    _forward_mode_lm,
    design_coast_excitation,
    dof_joint_names,
    validate_excitation,
)
from ..data.schema import JointTelemetry
from ..errors import CalibrationError, JaxterityError
from ..robot import Robot
from ..robot.frame import Frame, FramedQuantity, Position, Transform
from ..robot.structure import Joint, JointLimits, JointType, Link
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

#: The fittable payload parameter groups, in flat-vector order.
_FIT_GROUPS = ("mass", "com", "inertia_diag")
#: Default fit: mass + CoM (inertia needs a much richer probe — opt in).
_DEFAULT_FIT = ("mass", "com")


class RigidPayload(eqx.Module):
    """A rigid object held by (welded to) a robot link, with prior inertials.

    The prior is where the Levenberg–Marquardt fit starts *and* the point the
    identifiability report linearizes at — a zero-mass prior would zero the
    CoM sensitivity columns, so ``mass`` must be positive.

    Args
    ----
    attach_link : str
        Name of the robot link the object is rigidly held by (static).
    mass : float | jax.Array
        Prior mass, kg. Must be > 0 (see above). Default 0.1.
    com : tuple | jax.Array
        Prior centre of mass in the payload frame, m. Default the origin.
    inertia : jax.Array | None
        Prior 3×3 inertia tensor about the CoM, kg·m². Default: a solid
        sphere of radius ``0.03 m`` at the prior mass.
    name : str
        The payload body's name in the lowered model (static).
        Default ``"payload"``.
    offset : tuple[float, float, float]
        Grasp point: the payload frame's origin in the attach link's frame, m
        (static — part of the model description). Default the link origin.

    Frame
    -----
    ``com`` is in the payload body frame; ``offset`` is in ``attach_link``'s
    body frame.

    Units
    -----
    kg, m, kg·m².

    Raises
    ------
    JaxterityError
        If the prior mass is not positive.
    """

    mass: jax.Array
    com: jax.Array
    inertia: jax.Array
    attach_link: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    offset: tuple[float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        attach_link: str,
        *,
        mass=0.1,
        com=(0.0, 0.0, 0.0),
        inertia=None,
        name: str = "payload",
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.attach_link = str(attach_link)
        self.name = str(name)
        self.offset = (float(offset[0]), float(offset[1]), float(offset[2]))
        self.mass = jnp.asarray(mass, dtype=float)
        if float(self.mass) <= 0.0:
            raise JaxterityError(
                f"RigidPayload prior mass must be positive; got {float(self.mass)!r}."
                " The prior is the fit's linearization point — a zero mass zeroes"
                " the CoM sensitivity columns. Use a rough guess (any positive"
                " value); the fit refines it."
            )
        self.com = jnp.asarray(com, dtype=float)
        if inertia is None:
            # Solid sphere of radius 0.03 m at the prior mass: I = (2/5) m r².
            i = 0.4 * float(self.mass) * 0.03**2
            inertia = i * jnp.eye(3)
        self.inertia = jnp.asarray(inertia, dtype=float)


def attach_payload(robot: Robot, payload: RigidPayload) -> Robot:
    """Return a new robot with the payload welded to its grasp link.

    The payload becomes a real body in the lowered model (one new
    :class:`~jaxterity.robot.Link` + one ``FIXED``
    :class:`~jaxterity.robot.Joint`): MuJoCo composes the coupled dynamics and
    the payload's ``mass``/``com``/``inertia_diag`` become runtime-threaded
    dynamic parameters like any other link's — which is exactly what
    :func:`identify_payload` fits. No DOFs are added.

    # attestation: changes-handle — the payload body is part of the model
    # description (MJCF) and its inertials join ``Robot.parameters()``.

    Args
    ----
    robot : Robot
        The (typically calibrated) robot doing the holding.
    payload : RigidPayload
        The object and its priors.

    Returns
    -------
    Robot
        A new robot (the input is unchanged); its ``attestation_handle``
        differs from the bare robot's.

    Raises
    ------
    JaxterityError
        If the attach link is unknown, or the payload's name collides with an
        existing link or joint.
    """
    link_map, joint_map = robot.link_map, robot.joint_map
    if payload.attach_link not in link_map:
        raise JaxterityError(
            f"attach_payload: attach link {payload.attach_link!r} is not a "
            f"known link. Known: {sorted(link_map)}."
        )
    weld_name = f"{payload.name}_weld"
    for taken in (payload.name, weld_name):
        if taken in link_map or taken in joint_map:
            raise JaxterityError(
                f"attach_payload: name {taken!r} is already a link or joint of "
                f"{robot.name!r}. Pick a different payload name."
            )
    body = Link(
        name=payload.name,
        body_frame=Frame(payload.name),
        mass=payload.mass,
        com=payload.com,
        inertia=payload.inertia,
    )
    weld = Joint(
        name=weld_name,
        joint_type=JointType.FIXED,
        parent_link=payload.attach_link,
        child_link=payload.name,
        axis=FramedQuantity(
            Position(jnp.array([0.0, 0.0, 1.0])), Frame(payload.attach_link)
        ),
        limits=JointLimits(lower=0.0, upper=0.0, effort=0.0, velocity=0.0),
        origin=Transform(
            rotation=jnp.eye(3),
            translation=jnp.asarray(payload.offset, dtype=float),
            source=Frame(payload.name),
            target=Frame(payload.attach_link),
        ),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
        armature=jnp.asarray(0.0),
    )
    return Robot(
        links=(*robot.links, body),
        joints=(*robot.joints, weld),
        name=robot.name,
        source=robot.source,
        calibration_state=robot.calibration_state,
        gravity=robot.gravity,
        ground_plane=robot.ground_plane,
        provenance=robot.provenance,
    )


def _fit_layout(
    payload: RigidPayload, fit: tuple[str, ...]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Flat-vector layout for the selected fit groups.

    Returns ``(names, prior, lower)``: per-scalar report names (CoM and
    inertia components indexed ``[i]``), the prior starting vector, and the
    per-scalar lower bounds (0 for mass/inertia, unbounded for CoM).
    """
    unknown = tuple(g for g in fit if g not in _FIT_GROUPS)
    if unknown or not fit:
        raise CalibrationError(
            f"identify_payload: fit groups {unknown or fit!r} not understood. "
            f"Choose a non-empty subset of {_FIT_GROUPS}."
        )
    names: list[str] = []
    prior: list[float] = []
    lower: list[float] = []
    if "mass" in fit:
        names.append(f"{payload.name}.mass")
        prior.append(float(payload.mass))
        lower.append(0.0)
    if "com" in fit:
        for i in range(3):
            names.append(f"{payload.name}.com[{i}]")
            prior.append(float(payload.com[i]))
            lower.append(-np.inf)
    if "inertia_diag" in fit:
        moments = np.linalg.eigvalsh(np.asarray(payload.inertia))
        for i in range(3):
            names.append(f"{payload.name}.inertia_diag[{i}]")
            prior.append(float(moments[i]))
            lower.append(0.0)
    return tuple(names), np.asarray(prior, dtype=float), np.asarray(lower)


def _payload_observer(
    attached: Robot,
    payload: RigidPayload,
    fit: tuple[str, ...],
    excitation: Excitation,
    options: SimulatorOptions,
):
    """A jitted ``observe(flat) -> stacked [q, qd]`` binding the flat payload
    vector onto the payload body's runtime-threaded dynamic parameters.

    The scalar-parameter observer in ``calibrate.active`` zips one value per
    parameter key; the payload CoM/inertia are 3-vector leaves, so this
    observer owns the flat ↔ leaf mapping instead.
    """
    import jaxonomy

    q0 = jnp.asarray(excitation.q0, dtype=float)
    qd0 = jnp.asarray(excitation.qd0, dtype=float)
    diagram, system = attached.to_diagram(q0=q0, qd0=qd0, return_system=True)
    sid = system.system_id
    name = payload.name

    @jax.jit
    def observe(flat):
        ctx = diagram.create_context()
        sub = ctx[sid]
        k = 0
        if "mass" in fit:
            sub = sub.with_parameter(f"{name}.mass", flat[k])
            k += 1
        if "com" in fit:
            sub = sub.with_parameter(f"{name}.com", flat[k : k + 3])
            k += 3
        if "inertia_diag" in fit:
            sub = sub.with_parameter(f"{name}.inertia_diag", flat[k : k + 3])
            k += 3
        ctx = ctx.with_subcontext(sid, sub)
        states = [
            jaxonomy.simulate(diagram, ctx, (0.0, t), options=options)
            .context[sid]
            .continuous_state
            for t in excitation.sample_times
        ]
        return jnp.concatenate(states)

    return observe, system._nq, system._nv


class PayloadLoopbackLink(ActiveLink):
    """A seeded simulated arm holding a *planted* payload.

    Runs the excitation through the payload-attached robot with the true
    payload values bound (defaults: the payload's own priors), so
    :func:`identify_payload` must recover them. Deterministic: a fixed
    ``seed`` (with optional Gaussian sensor noise) yields bit-identical
    telemetry. True values are bound at runtime — not compiled in — so an
    unphysical plant (e.g. a negative mass, for bound-projection tests) is
    representable.

    Args
    ----
    base_robot : Robot
        The bare robot (no payload) the fit also starts from.
    payload : RigidPayload
        The payload structure/priors — the same object handed to
        :func:`identify_payload`.
    true_mass : float | None
        Planted mass, kg (default: the payload's prior).
    true_com : tuple | None
        Planted CoM, m (default: the prior).
    true_inertia_diag : tuple | None
        Planted principal moments, kg·m² (default: the prior's).
    seed : int
        PRNG seed for the optional sensor noise (determinism).
    noise_std : float
        Additive Gaussian position/velocity noise std (rad, rad/s). Default 0.
    options : SimulatorOptions | None
        Integration options (defaults to the lane's fixed-step RK4).

    Units
    -----
    kg, m, kg·m²; telemetry in rad, rad/s.
    """

    def __init__(
        self,
        base_robot: Robot,
        payload: RigidPayload,
        *,
        true_mass: float | None = None,
        true_com=None,
        true_inertia_diag=None,
        seed: int = 0,
        noise_std: float = 0.0,
        options: SimulatorOptions | None = None,
    ):
        self._attached = attach_payload(base_robot, payload)
        self._payload = payload
        self._true_mass = true_mass
        self._true_com = true_com
        self._true_inertia_diag = true_inertia_diag
        self._seed = int(seed)
        self._noise_std = float(noise_std)
        self._options = options or _DEFAULT_OPTIONS
        self._names = dof_joint_names(self._attached)

    @property
    def n_dof(self) -> int:
        return len(self._names)

    def run_excitation(self, excitation: Excitation) -> JointTelemetry:
        fit = ("mass", "com", "inertia_diag")
        _, prior, _ = _fit_layout(self._payload, fit)
        true = prior.copy()
        if self._true_mass is not None:
            true[0] = float(self._true_mass)
        if self._true_com is not None:
            true[1:4] = np.asarray(self._true_com, dtype=float)
        if self._true_inertia_diag is not None:
            true[4:7] = np.asarray(self._true_inertia_diag, dtype=float)
        observe, nq, nv = _payload_observer(
            self._attached, self._payload, fit, excitation, self._options
        )
        n_t = len(excitation.sample_times)
        states = np.asarray(observe(jnp.asarray(true))).reshape(n_t, nq + nv)
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


def payload_identifiability(
    robot: Robot,
    payload: RigidPayload,
    excitation: Excitation | None = None,
    *,
    fit: tuple[str, ...] = _DEFAULT_FIT,
    cond_max: float = _DEFAULT_COND_MAX,
    options: SimulatorOptions | None = None,
) -> IdentifiabilityReport:
    """Report whether a probe can identify the payload parameters — before
    moving anything.

    The payload sibling of
    :func:`~jaxterity.calibrate.active.excitation_identifiability`: simulates
    the probe once at the payload's *prior* values and scores the sensitivity
    Jacobian's Fisher information. A rank deficit or a condition number above
    ``cond_max`` means "redesign the probe" (or drop a fit group — a
    quasi-static probe cannot excite ``inertia_diag``), not "fit harder".

    Args
    ----
    robot : Robot
        The bare robot (the payload is attached internally).
    payload : RigidPayload
        The object and its priors (the linearization point).
    excitation : Excitation | None
        The candidate probe. Defaults to
        :func:`~jaxterity.calibrate.active.design_coast_excitation`.
    fit : tuple[str, ...]
        Which parameter groups to score: a subset of
        ``("mass", "com", "inertia_diag")``. Default mass + CoM.
    cond_max : float
        Conditioning ceiling for the ``identifiable`` verdict.
    options : SimulatorOptions | None
        Integration options (defaults to the lane's fixed-step RK4).

    Returns
    -------
    IdentifiabilityReport
        Per-scalar names (CoM/inertia components indexed ``[i]``), singular
        values, rank, condition number, verdict, and weak parameters.

    Units
    -----
    Observations stack joint positions (rad) and velocities (rad/s); the
    report is column-normalized, hence scale-free.
    """
    options = options or _DEFAULT_OPTIONS
    excitation = excitation or design_coast_excitation(robot)
    attached = attach_payload(robot, payload)
    names, prior, _ = _fit_layout(payload, tuple(fit))
    observe, _nq, _nv = _payload_observer(
        attached, payload, tuple(fit), excitation, options
    )
    jac = np.asarray(jax.jit(jax.jacfwd(observe))(jnp.asarray(prior)))
    return identifiability_from_jacobian(jac, names, cond_max=cond_max)


def identify_payload(
    robot: Robot,
    link: ActiveLink,
    payload: RigidPayload,
    *,
    fit: tuple[str, ...] = _DEFAULT_FIT,
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
    """Identify a held object's inertial parameters from the robot's own
    joint telemetry.

    Attaches the payload as a welded body, validates the excitation
    (rejecting out-of-limit probes before any command is sent), streams it
    over ``link``, and recovers the selected payload parameters with the same
    forward-mode Levenberg–Marquardt fit as
    :func:`~jaxterity.calibrate.active.active_identify` — started at the
    payload's *prior* (a zero start would zero the CoM sensitivities). The
    arm's own calibrated dynamics are the force scale: this is what makes the
    payload's absolute mass identifiable where passive observation gives
    ratios only.

    The result carries the lane's three honesty guards (T-219/C-119): the
    probe's identifiability (scored from the fit's own first-iteration
    Jacobian), physical non-negativity (``mass``/``inertia_diag`` projected to
    ``>= 0``; a pinned fit is named), and per-parameter uncertainty (a
    standard error exceeding ``warn_rel_std`` × the fitted magnitude is
    flagged as unresolved).

    Args
    ----
    robot : Robot
        The bare robot (typically calibrated — the fit inherits its dynamics
        as ground truth for the force scale).
    link : ActiveLink
        The transport: :class:`PayloadLoopbackLink` (seeded fake) or
        :class:`~jaxterity.calibrate.active.ReplayLink` (real capture).
    payload : RigidPayload
        The object: attach point, name, and priors.
    fit : tuple[str, ...]
        Parameter groups to fit — a subset of
        ``("mass", "com", "inertia_diag")``. Default mass + CoM;
        ``inertia_diag`` needs a much richer probe (check
        :func:`payload_identifiability` first).
    excitation : Excitation | None
        The probe. Defaults to
        :func:`~jaxterity.calibrate.active.design_coast_excitation`.
    iterations, lam, max_step : int, float, float
        Levenberg–Marquardt controls.
    options : SimulatorOptions | None
        Integration options (defaults to the lane's fixed-step RK4).
    uncertainty : str or None
        ``"laplace"`` (default), ``"bootstrap"``, or ``None``.
    n_boot : int
        Bootstrap refit count when ``uncertainty="bootstrap"``.
    seed : int
        PRNG seed for the bootstrap resampling.
    bounds : bool
        Project ``mass``/``inertia_diag`` to ``>= 0`` during the fit
        (default).
    cond_max : float
        Conditioning ceiling for the identifiability warning.
    warn_rel_std : float
        Relative standard-error threshold for the unresolved warning.

    Returns
    -------
    SysidResult
        ``fitted`` holds leaf-shaped values (``"<name>.mass"`` scalar,
        ``"<name>.com"`` / ``"<name>.inertia_diag"`` 3-vectors) so
        :meth:`~jaxterity.sysid.SysidResult.apply_to` calibrates the
        payload-attached robot directly; ``covariance`` and the warning set
        are per scalar component (``"<name>.com[i]"``).

    Raises
    ------
    CalibrationError
        If the excitation is out of limits or its width disagrees with the
        link, or the fit groups / uncertainty estimator are not recognized.

    Units
    -----
    kg, m, kg·m²; telemetry in rad, rad/s.
    """
    options = options or _DEFAULT_OPTIONS
    excitation = excitation or design_coast_excitation(robot)
    fit = tuple(fit)
    if excitation.n_dof != link.n_dof:
        raise CalibrationError(
            f"identify_payload: excitation drives {excitation.n_dof} joints but "
            f"the link has {link.n_dof}. They must match (DOF order)."
        )
    attached = attach_payload(robot, payload)
    names, prior, lower = _fit_layout(payload, fit)

    # 1. reject an out-of-limit probe BEFORE any command reaches the bus.
    #    (The welded payload adds no DOFs, so the attached robot shares the
    #    bare robot's limits; validate on the attached model — the one that
    #    will actually move.)
    validate_excitation(attached, excitation, options=options)

    # 2. stream it and capture telemetry.
    telemetry = link.run_excitation(excitation)

    # 3. fit, starting from the prior (forward-mode LM, projected bounds).
    target = np.concatenate(
        [
            np.concatenate([telemetry.q[k], telemetry.qd[k]])
            for k in range(telemetry.n_samples)
        ]
    )
    observe, _nq, _nv = _payload_observer(attached, payload, fit, excitation, options)
    estimate, history, jac0, hit_lower = _forward_mode_lm(
        observe,
        target,
        len(names),
        iterations=iterations,
        lam=lam,
        max_step=max_step,
        lower=lower if bounds else None,
        est0=prior,
    )
    estimate = np.asarray(estimate)
    rms = float(np.sqrt(history[-1] / target.size))

    # 4. identifiability of the probe — from the fit's own first-iteration
    #    Jacobian (evaluated at the prior, the same point
    #    payload_identifiability scores).
    notes: list[str] = []
    report = identifiability_from_jacobian(jac0, names, cond_max=cond_max)
    if not report.identifiable:
        deficit = len(names) - report.rank
        notes.append(
            f"identify_payload: the probe is not identifiable — rank "
            f"{report.rank} of {len(names)} parameters (deficit {deficit}), "
            f"condition number {report.condition_number:.3g} > {cond_max:.3g}. "
            "The fitted values are unconstrained along the missing directions; "
            "redesign the probe or drop a fit group (see "
            "payload_identifiability)."
        )
    elif report.weak_params:
        notes.append(
            f"identify_payload: weakly identified under this probe: "
            f"{', '.join(report.weak_params)} (condition number "
            f"{report.condition_number:.3g}). Treat them as loosely constrained."
        )
    for pname, hit in zip(names, hit_lower):
        if hit:
            notes.append(
                f"identify_payload: {pname} fit hit lower bound 0 (the "
                "unconstrained step wanted a negative, unphysical value); the "
                "parameter is likely unidentifiable under this probe — see "
                "payload_identifiability."
            )

    # 5. (optional) local fit covariance + unresolved-parameter flags.
    covariance = _payload_covariance(
        observe,
        target,
        estimate,
        names,
        uncertainty=uncertainty,
        n_boot=n_boot,
        seed=seed,
        iterations=iterations,
        lam=lam,
        max_step=max_step,
        lower=lower if bounds else None,
        prior=prior,
    )
    if covariance is not None:
        for pname, value in zip(names, estimate):
            std = float(np.sqrt(max(float(covariance[pname]), 0.0)))
            scale = abs(float(value))
            if std > warn_rel_std * max(scale, float(np.finfo(float).tiny)):
                notes.append(
                    f"identify_payload: {pname} is unresolved — standard error "
                    f"{std:.3g} exceeds {warn_rel_std:.0%} of the fitted value "
                    f"{scale:.3g}. Do not trust the point estimate; use a "
                    "richer probe or lower-noise telemetry."
                )

    # 6. fold the flat estimate back into leaf-shaped fitted values so
    #    SysidResult.apply_to binds them via Robot.with_parameters.
    fitted: dict[str, jax.Array] = {}
    k = 0
    if "mass" in fit:
        fitted[f"{payload.name}.mass"] = jnp.asarray(float(estimate[k]))
        k += 1
    if "com" in fit:
        fitted[f"{payload.name}.com"] = jnp.asarray(estimate[k : k + 3])
        k += 3
    if "inertia_diag" in fit:
        # Robot.with_parameters targets the full tensor leaf ``inertia``;
        # principal moments land on its diagonal (the prior's orientation is
        # kept — fitting the principal axes needs a structural rebuild).
        fitted[f"{payload.name}.inertia"] = jnp.diag(jnp.asarray(estimate[k : k + 3]))
        k += 3

    return SysidResult(
        fitted=fitted,
        covariance=covariance,
        residuals={
            "final_state_rms": rms,
            "excitation_condition_number": float(report.condition_number),
        },
        telemetry_hash=telemetry.hash().hex(),
        recipe_name="objects.payload",
        recipe_version="1.0",
        warnings=tuple(notes),
    )


def _payload_covariance(
    observe,
    target,
    estimate,
    names,
    *,
    uncertainty,
    n_boot,
    seed,
    iterations,
    lam,
    max_step,
    lower,
    prior,
) -> dict[str, float] | None:
    """Per-scalar marginal variance for a payload fit, or ``None``."""
    if uncertainty is None:
        return None
    target = np.asarray(target, dtype=float)

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
                est0=prior,
            )
            return np.asarray(est_star)

        cov = bootstrap_covariance(
            predict_hat, residual_hat, refit, n_boot=n_boot, seed=seed
        )
    else:
        raise CalibrationError(
            f"identify_payload: unknown uncertainty estimator {uncertainty!r}. "
            "Use 'laplace' (default), 'bootstrap', or None."
        )
    return per_parameter_variance(cov, tuple(names))


__all__ = [
    "RigidPayload",
    "attach_payload",
    "PayloadLoopbackLink",
    "payload_identifiability",
    "identify_payload",
]
