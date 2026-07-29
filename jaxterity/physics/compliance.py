# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Structure and environment: the flex and drag the body model leaves out.

The MJX body model is a rigid tree on rigid wheels. Three things it therefore
cannot show, each of which has already cost a real margin somewhere:

* **Suspension / wheel-roller compliance.** The tip-over gate
  (:mod:`jaxterity.checkup.stability`) computes a zero-moment point against a
  *rigid* base and then carries an inflated margin floor — 20 mm instead of the
  12 mm the geometry suggests — precisely because omni-roller squish and base
  rocking are unmodeled and somebody has to pay for them. :func:`compliant_margin`
  makes that payment explicit: given the support springs, it computes how far
  the base tilts under the load the motion transfers, how far that tilt walks
  the centre of mass, and subtracts the result from the rigid margin. What was
  a hand-tuned safety factor becomes a modeled term, and the floor can come
  back down to what it should be.
* **Joint / link torsional compliance.** Between a motor and the link it
  drives sits a shaft, a bracket, and a bolted joint, none of them rigid. The
  signature is a resonance — often the first mode of the whole arm — and a
  controller tuned against a rigid model walks straight into it.
  :class:`JointCompliance` is that spring, and :func:`torsional_resonance_hz`
  tells you where it will ring. This is *parasitic* compliance, distinct from
  :class:`~jaxterity.actuators.series_elastic.SeriesElasticActuator`'s
  *designed* compliance: the SEA's spring is the point of the mechanism and is
  soft enough to be a state you control; this one you would remove if you
  could, and its only job in the model is to be where the resonance is.
* **Rolling resistance.** A wheeled base does not coast forever. Rolling
  resistance is what sets how far it does, and a mobile-base plan built on a
  frictionless model over-predicts range and under-predicts the torque needed
  to hold a slope. :func:`rolling_resistance_force` is the standard
  Coulomb-plus-viscous law, identifiable from a coast-down in one regression
  (:func:`fit_coast_down`).

Units
-----
N, N/m, m, rad, kg, m/s, N·m/rad.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from ..data.schema import ValidationReport
from ..errors import CalibrationError, JaxterityError
from ..parameter import ParameterSpec

#: Standard gravity, m/s² (matching :mod:`jaxterity.checkup.stability`).
_GRAVITY = 9.81
#: Default smoothing width for the rolling-resistance sign (m/s).
_V_EPS_DEFAULT = 1e-3


# --------------------------------------------------------------------------
# Suspension / wheel compliance
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SuspensionProfile:
    """The vertical springs a mobile base actually stands on.

    One spring per ground-contact point, in the same base frame and the same
    order as the support polygon a :class:`~jaxterity.checkup.stability.BaseProfile`
    declares. For a LeKiwi-class base the "springs" are the omni-wheel rollers
    and the tyre/hub stack; for a suspended base they are the suspension
    proper. Either way what matters is the same number: how far the contact
    deflects per newton.

    Args
    ----
    contact_xy : tuple[tuple[float, float], ...]
        Ground-contact points in the base frame, m. At least three, and not
        collinear (a rigid plate on two springs is a hinge, not a base).
    stiffness_n_per_m : tuple[float, ...]
        Vertical stiffness at each contact, N/m. All > 0, one per contact.
    com_height_m : float
        Height of the whole vehicle's centre of mass above the contact plane,
        m. This is the lever that converts a tilt into a horizontal CoM shift,
        so a tall robot on soft wheels loses margin fast.
    damping_n_s_per_m : tuple[float, ...] | None
        Per-contact damping, N·s/m. Not used by the quasi-static margin
        correction (which assumes the tilt has settled); carried because it is
        what sets the *rocking* mode's decay, and a base whose rocking does not
        decay within a capture segment invalidates the quasi-static assumption
        entirely.

    Raises
    ------
    JaxterityError
        On fewer than three contacts, a length mismatch, a non-positive
        stiffness, or a non-positive CoM height.
    """

    contact_xy: tuple[tuple[float, float], ...]
    stiffness_n_per_m: tuple[float, ...]
    com_height_m: float
    damping_n_s_per_m: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.contact_xy) < 3:
            raise JaxterityError(
                f"SuspensionProfile needs at least 3 contact points to define a "
                f"tilt plane, got {len(self.contact_xy)}. Two springs make a "
                "hinge, and its tilt is unbounded — model the third support "
                "(even a stiff caster) explicitly."
            )
        if len(self.stiffness_n_per_m) != len(self.contact_xy):
            raise JaxterityError(
                f"SuspensionProfile has {len(self.contact_xy)} contact(s) but "
                f"{len(self.stiffness_n_per_m)} stiffness value(s); one per "
                "contact, in the same order."
            )
        if any(k <= 0 for k in self.stiffness_n_per_m):
            raise JaxterityError(
                f"SuspensionProfile needs strictly positive stiffness at every "
                f"contact, got {self.stiffness_n_per_m}. A rigid contact is the "
                "large-stiffness limit — use a big number, not zero."
            )
        if self.com_height_m <= 0:
            raise JaxterityError(
                f"SuspensionProfile.com_height_m must be > 0, got "
                f"{self.com_height_m}. It is the lever arm that turns base tilt "
                "into a horizontal CoM shift; at zero there is no coupling and "
                "the correction is trivially zero."
            )
        if self.damping_n_s_per_m is not None and len(self.damping_n_s_per_m) != len(
            self.contact_xy
        ):
            raise JaxterityError(
                f"SuspensionProfile has {len(self.contact_xy)} contact(s) but "
                f"{len(self.damping_n_s_per_m)} damping value(s)."
            )


#: REPRESENTATIVE LeKiwi suspension — omni-roller-class stiffness at the three
#: wheel contacts of :data:`jaxterity.checkup.stability.LEKIWI_BASE`, with the
#: arm-plus-base CoM about 12 cm up. Plausible, **not measured** (same contract
#: as the base profile it accompanies): press on a real unit with a scale and a
#: dial indicator before trusting an absolute margin correction.
LEKIWI_SUSPENSION = SuspensionProfile(
    contact_xy=((0.0575, 0.0996), (-0.115, 0.0), (0.0575, -0.0996)),
    stiffness_n_per_m=(4.0e4, 4.0e4, 4.0e4),
    com_height_m=0.12,
)


def suspension_tilt(
    suspension: SuspensionProfile, zmp_xy, vertical_force: float
) -> tuple[float, float]:
    """Steady tilt of a rigid base standing on vertical springs.

    A rigid plate on ``n`` springs has three degrees of freedom — heave and two
    tilts — so the deflection field is a plane, ``z(x, y) = z₀ + a·x + b·y``.
    Force and moment balance against a vertical resultant ``F_z`` acting at the
    zero-moment point give the 3×3 system

        Σ kᵢ·[1, xᵢ, yᵢ]ᵀ·[1, xᵢ, yᵢ] · [z₀, a, b]ᵀ
            = F_z·[1, x_zmp, y_zmp]ᵀ

    whose ``(a, b)`` are the tilt slopes. Statically determinate three-point
    bases and over-constrained four-point ones both fall out of the same
    solve, which is why this is done in stiffness form rather than by
    distributing reactions.

    Args
    ----
    suspension : SuspensionProfile
        The support springs.
    zmp_xy : array-like, shape ``(2,)``
        Zero-moment point in the base frame, m — where the resultant acts.
    vertical_force : float
        Total vertical reaction, N (weight plus the vertical inertial term).

    Returns
    -------
    tuple[float, float]
        Tilt slopes ``(dz/dx, dz/dy)``, dimensionless (≈ radians for small
        angles).

    Raises
    ------
    CalibrationError
        If the contacts are collinear, which leaves the tilt about that line
        undetermined.

    Units
    -----
    m, N → dimensionless slope.

    Examples
    --------
    >>> tilt = suspension_tilt(LEKIWI_SUSPENSION, (0.0, 0.0), 30.0)
    >>> bool(abs(tilt[0]) < 1e-3 and abs(tilt[1]) < 1e-12)
    True
    """
    contacts = np.asarray(suspension.contact_xy, dtype=float)
    stiffness = np.asarray(suspension.stiffness_n_per_m, dtype=float)
    basis = np.concatenate([np.ones((contacts.shape[0], 1)), contacts], axis=1)
    matrix = basis.T @ (stiffness[:, None] * basis)
    zmp = np.asarray(zmp_xy, dtype=float).reshape(2)
    load = float(vertical_force) * np.array([1.0, zmp[0], zmp[1]])
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CalibrationError(
            "suspension_tilt: the contact points are collinear (or nearly so), "
            "so the tilt about that line is undetermined. Fix: declare the "
            "actual contact geometry — a real base is not a hinge."
        )
    solution = np.linalg.solve(matrix, load)
    return float(solution[1]), float(solution[2])


def compliant_margin(
    rigid_margin,
    zmp_xy,
    suspension: SuspensionProfile,
    vertical_force: float,
):
    """Shrink a rigid-base ZMP margin by what the suspension gives away.

    The tilt from :func:`suspension_tilt` walks the centre of mass sideways by
    ``h_com × tilt``, and at quasi-static equilibrium the zero-moment point
    follows it. The margin available inside the support polygon shrinks by the
    magnitude of that walk::

        margin_compliant = margin_rigid − h_com·‖tilt‖

    Subtracting the *magnitude* (rather than the projection onto the nearest
    polygon edge) is deliberately conservative: it is correct when the tilt
    points at the closest edge and pessimistic otherwise, and a stability gate
    should err that way. It is also first-order — the tilt is computed at the
    rigid ZMP rather than iterated to the shifted one, which is accurate while
    the shift is small compared to the polygon, and a base where it is *not*
    small has already failed the gate.

    Args
    ----
    rigid_margin : float or numpy.ndarray
        Margin(s) from :func:`jaxterity.checkup.stability.zmp_margins`, m.
    zmp_xy : array-like, shape ``(2,)`` or ``(n, 2)``
        The corresponding zero-moment point(s) in the base frame, m.
    suspension : SuspensionProfile
        The support springs.
    vertical_force : float
        Total vertical reaction, N.

    Returns
    -------
    numpy.ndarray
        Corrected margin(s), m — same shape as ``rigid_margin``.

    Units
    -----
    m, N → m.
    """
    points = np.atleast_2d(np.asarray(zmp_xy, dtype=float))
    shifts = np.empty(points.shape[0])
    for i, point in enumerate(points):
        tilt_x, tilt_y = suspension_tilt(suspension, point, vertical_force)
        shifts[i] = suspension.com_height_m * float(np.hypot(tilt_x, tilt_y))
    corrected = np.asarray(rigid_margin, dtype=float) - shifts.reshape(
        np.shape(rigid_margin)
    )
    return corrected


def margin_floor_credit(
    suspension: SuspensionProfile,
    support_polygon_xy,
    vertical_force: float,
) -> float:
    """How much margin floor the compliance accounts for, m.

    The worst-case CoM walk over the support polygon's own vertices — i.e. how
    much of a hand-set margin floor is *explained* by suspension compliance
    rather than by unknown unknowns. Quote it when arguing a floor down: a
    20 mm floor covering an 8 mm modeled compliance leaves 12 mm of genuine
    unmodelled budget, and that is a defensible number in a way that "20 mm,
    because it tipped once" is not.

    Args
    ----
    suspension : SuspensionProfile
        The support springs.
    support_polygon_xy : array-like, shape ``(k, 2)``
        The support polygon's vertices in the base frame, m — the extreme
        positions the ZMP can reach while still (nominally) stable.
    vertical_force : float
        Total vertical reaction, N.

    Returns
    -------
    float
        Worst-case CoM shift over those vertices, m.

    Units
    -----
    m, N → m.
    """
    vertices = np.asarray(support_polygon_xy, dtype=float)
    worst = 0.0
    for vertex in vertices:
        tilt_x, tilt_y = suspension_tilt(suspension, vertex, vertical_force)
        worst = max(worst, suspension.com_height_m * float(np.hypot(tilt_x, tilt_y)))
    return worst


# --------------------------------------------------------------------------
# Joint / link torsional compliance
# --------------------------------------------------------------------------


def torsional_resonance_hz(stiffness, inertia):
    """Natural frequency of an inertia on a torsional spring, Hz.

    ``f = √(k/J)/2π`` — where the arm rings, and the ceiling on any controller
    bandwidth a rigid model would happily promise.

    Args
    ----
    stiffness : float or jax.Array
        Torsional stiffness, N·m/rad.
    inertia : float or jax.Array
        Inertia on the compliant side, kg·m².

    Returns
    -------
    jax.Array
        Undamped natural frequency, Hz.

    Units
    -----
    N·m/rad, kg·m² → Hz.

    Examples
    --------
    >>> round(float(torsional_resonance_hz(500.0, 0.01)), 1)
    35.6
    """
    return jnp.sqrt(stiffness / inertia) / (2.0 * jnp.pi)


def stiffness_from_resonance(frequency_hz, inertia):
    """Torsional stiffness implied by a measured resonance, N·m/rad.

    The inverse of :func:`torsional_resonance_hz`, and the practical way to get
    a stiffness: a hammer tap and an FFT beat any attempt to fit ``k`` from a
    quasi-static deflection you cannot measure.

    Args
    ----
    frequency_hz : float or jax.Array
        Measured resonant frequency, Hz.
    inertia : float or jax.Array
        Inertia on the compliant side, kg·m².

    Returns
    -------
    jax.Array
        Torsional stiffness, N·m/rad.

    Units
    -----
    Hz, kg·m² → N·m/rad.
    """
    return inertia * (2.0 * jnp.pi * frequency_hz) ** 2


class JointCompliance:
    """Parasitic torsional spring-damper between a driven side and a link.

    A stateless torque map, deliberately: like
    :class:`~jaxterity.actuators.backlash.PositionalBacklash`, both angles are
    states of the body model, so the compliance stays a pure function of them
    and composes anywhere. Not a ``LeafSystem`` subclass by itself —
    :meth:`torque` is the whole model, and
    :class:`~jaxterity.actuators.series_elastic.SeriesElasticActuator` already
    covers the case where the compliance is the *mechanism* rather than a
    defect in it.

    Args
    ----
    stiffness : float
        Torsional stiffness, N·m/rad. Must be > 0.
    damping : float
        Torsional damping, N·m·s/rad. Must be >= 0.
    inertia : float
        Inertia on the far side of the spring, kg·m² — only used to report the
        resonance. Must be > 0.

    Raises
    ------
    JaxterityError
        If stiffness or inertia is non-positive, or damping is negative.

    Examples
    --------
    >>> link = JointCompliance(stiffness=500.0, damping=0.5, inertia=0.01)
    >>> round(float(link.resonance_hz), 1)
    35.6
    """

    def __init__(self, *, stiffness: float, damping: float = 0.0, inertia: float = 1.0):
        if stiffness <= 0 or inertia <= 0 or damping < 0:
            raise JaxterityError(
                f"JointCompliance needs stiffness > 0, inertia > 0 and damping "
                f">= 0; got {stiffness}, {inertia}, {damping}."
            )
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.inertia = float(inertia)

    @property
    def resonance_hz(self):
        """Where this joint rings, Hz (see :func:`torsional_resonance_hz`)."""
        return torsional_resonance_hz(self.stiffness, self.inertia)

    @property
    def damping_ratio(self) -> float:
        """Dimensionless damping ratio ``c / (2√(k·J))``.

        Below ~0.1 the mode rings for many cycles and any controller crossing
        it will excite it; that is the number to look at, not the raw damping.
        """
        return self.damping / (2.0 * np.sqrt(self.stiffness * self.inertia))

    def torque(self, delta_theta, delta_velocity=0.0):
        """Transmitted torque across the compliance, N·m.

        Args
        ----
        delta_theta : float or jax.Array
            Twist ``θ_drive − θ_link``, rad.
        delta_velocity : float or jax.Array
            Twist rate, rad/s.

        Returns
        -------
        jax.Array
            Torque, N·m.

        Units
        -----
        rad, rad/s → N·m.
        """
        return self.stiffness * delta_theta + self.damping * delta_velocity

    def parameter_spec(self) -> ParameterSpec:
        """The identifiable leaves: stiffness and damping (inertia is measured)."""
        return ParameterSpec(("stiffness", "damping"))


# --------------------------------------------------------------------------
# Rolling resistance
# --------------------------------------------------------------------------


def rolling_resistance_force(
    normal_force, velocity, *, c_rr, c_v=0.0, v_eps: float = _V_EPS_DEFAULT
):
    """Resistive force on a rolling wheel, opposing motion.

    ``F = −(c_rr·N·tanh(v/v_eps) + c_v·v)``: a load-proportional Coulomb term
    (deformation loss at the contact patch, the dominant one at low speed) plus
    a viscous term (bearing drag and, on a fast base, everything speed-
    dependent lumped together). The sign is smoothed with the same ``tanh``
    idiom the actuator friction models use, so the force is exactly zero at
    rest and differentiable through it.

    Args
    ----
    normal_force : float or jax.Array
        Vertical load on the wheel, N (>= 0).
    velocity : float or jax.Array
        Ground speed at the contact, m/s.
    c_rr : float or jax.Array
        Rolling-resistance coefficient, dimensionless (≈0.01 for a hard wheel
        on a hard floor, ≈0.05 for an omni-roller on carpet).
    c_v : float or jax.Array
        Viscous coefficient, N·s/m.
    v_eps : float
        Smoothing width for the sign, m/s.

    Returns
    -------
    jax.Array
        Resistive force, N (negative when moving forward).

    Units
    -----
    N, m/s → N.

    Examples
    --------
    >>> float(rolling_resistance_force(30.0, 0.0, c_rr=0.02))
    -0.0
    >>> round(float(rolling_resistance_force(30.0, 1.0, c_rr=0.02)), 3)
    -0.6
    """
    return -(c_rr * normal_force * jnp.tanh(velocity / v_eps) + c_v * velocity)


class RollingResistance:
    """Coulomb-plus-viscous rolling resistance for a wheeled base.

    A stateless force map (see :func:`rolling_resistance_force`) with the two
    identifiable coefficients and a coast-down validator. Kept out of the
    ``LeafSystem`` hierarchy for the same reason as :class:`JointCompliance`:
    the wheel's state belongs to the body model, and a pure map composes into
    a diagram, a fit, and a plot alike.

    Args
    ----
    c_rr : float
        Rolling-resistance coefficient, dimensionless. Must be >= 0.
    c_v : float
        Viscous coefficient, N·s/m. Must be >= 0.
    v_eps : float
        Sign-smoothing width, m/s.

    Raises
    ------
    JaxterityError
        If either coefficient is negative (a wheel that *supplies* energy as
        it rolls is not a modeling choice).

    Examples
    --------
    >>> wheel = RollingResistance(c_rr=0.02)
    >>> round(float(wheel.deceleration(9.81 * 5.0, 5.0, 1.0)), 4)
    0.1962
    """

    def __init__(
        self, *, c_rr: float = 0.0, c_v: float = 0.0, v_eps: float = _V_EPS_DEFAULT
    ):
        if c_rr < 0 or c_v < 0:
            raise JaxterityError(
                f"RollingResistance needs c_rr >= 0 and c_v >= 0, got {c_rr} and {c_v}."
            )
        self.c_rr = float(c_rr)
        self.c_v = float(c_v)
        self.v_eps = float(v_eps)

    def force(self, normal_force, velocity):
        """Resistive force, N (see :func:`rolling_resistance_force`)."""
        return rolling_resistance_force(
            normal_force, velocity, c_rr=self.c_rr, c_v=self.c_v, v_eps=self.v_eps
        )

    def deceleration(self, normal_force, mass, velocity):
        """Magnitude of the deceleration this produces, m/s².

        Args
        ----
        normal_force : float or jax.Array
            Total vertical load, N.
        mass : float or jax.Array
            Vehicle mass, kg.
        velocity : float or jax.Array
            Ground speed, m/s.

        Returns
        -------
        jax.Array
            Deceleration magnitude, m/s².

        Units
        -----
        N, kg, m/s → m/s².
        """
        return -self.force(normal_force, velocity) / mass

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("c_rr", "c_v"))

    def validate(self, time, velocity, mass: float) -> ValidationReport:
        """Score these coefficients against a measured coast-down.

        Compares the trace's measured deceleration to the model's at each
        sample. A coast-down is the cleanest rolling-resistance experiment
        there is — no traction, no actuator model, no torque sensor — so a
        mismatch here is about the wheels and the floor, nothing else.

        Args
        ----
        time : numpy.ndarray, shape ``(n,)``
            Timestamps, s.
        velocity : numpy.ndarray, shape ``(n,)``
            Ground speed while coasting, m/s.
        mass : float
            Vehicle mass, kg.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS deceleration residual is within 10 % of the
            trace's own mean deceleration — a relative bound, because a
            coast-down's absolute scale varies by an order of magnitude with
            surface.

        Units
        -----
        s, m/s, kg.
        """
        t = np.asarray(time, dtype=np.float64)
        v = np.asarray(velocity, dtype=np.float64)
        measured = np.gradient(v, t)
        predicted = np.asarray(self.force(mass * _GRAVITY, jnp.asarray(v)) / mass)
        residual = predicted - measured
        rms = float(np.sqrt(np.mean(residual * residual)))
        scale = float(np.mean(np.abs(measured)))
        report = ValidationReport()
        if scale > 0 and rms > 0.1 * scale:
            report.warnings.append(
                f"coast-down deceleration residual RMS={rms:.3g} m/s² exceeds "
                f"10% of the trace's mean deceleration ({scale:.3g} m/s²) — "
                "c_rr / c_v may not match this surface."
            )
        return report


def fit_coast_down(time, velocity, mass: float) -> dict[str, float]:
    """Identify rolling-resistance coefficients from a coast-down.

    While coasting, ``m·v̇ = −c_rr·m·g − c_v·v``, which is linear in both
    coefficients: an OLS of the measured deceleration on ``[m·g, v]`` returns
    them directly. The two are separable only because they scale differently
    with speed, so the trace must span a *range* of speeds — a coast-down from
    a single speed to a stop does that automatically, which is why it is the
    right experiment.

    Args
    ----
    time : numpy.ndarray, shape ``(n,)``
        Timestamps, s.
    velocity : numpy.ndarray, shape ``(n,)``
        Ground speed while coasting (drive disengaged), m/s.
    mass : float
        Vehicle mass, kg. Must be > 0.

    Returns
    -------
    dict[str, float]
        ``{"c_rr": ..., "c_v": ...}``, dimensionless and N·s/m.

    Raises
    ------
    CalibrationError
        If the mass is non-positive, the trace is shorter than 3 samples, or
        the base never actually slows (nothing to regress).

    Units
    -----
    s, m/s, kg.
    """
    if mass <= 0:
        raise CalibrationError(
            f"fit_coast_down needs a positive vehicle mass, got {mass} kg."
        )
    t = np.asarray(time, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if t.size < 3:
        raise CalibrationError(
            f"fit_coast_down needs at least 3 samples, got {t.size}."
        )
    acceleration = np.gradient(v, t)
    if float(np.ptp(v)) <= 0.0:
        raise CalibrationError(
            "fit_coast_down: the base never slows in this trace, so neither "
            "coefficient is observable. Fix: disengage the drive and record "
            "until it stops."
        )
    design = np.stack([np.full_like(v, -mass * _GRAVITY), -v], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, mass * acceleration, rcond=None)
    return {"c_rr": float(coefficients[0]), "c_v": float(coefficients[1])}


__all__ = [
    "LEKIWI_SUSPENSION",
    "JointCompliance",
    "RollingResistance",
    "SuspensionProfile",
    "compliant_margin",
    "fit_coast_down",
    "margin_floor_credit",
    "rolling_resistance_force",
    "stiffness_from_resonance",
    "suspension_tilt",
    "torsional_resonance_hz",
]
