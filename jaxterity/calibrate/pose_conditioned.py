# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Friction parameters as a function of pose, because the scalar is wrong.

Two checkup sessions on LK-0001, under *identical* excitation recipes, moved
``shoulder_pan``'s fitted damping by **+229 %**. The fit change was monotone in
held-pose difference across all six joints (pan's held pose differed by 2094
ticks, the others by 40-518), which names the mechanism: pan is the base yaw
axis, so what it has to drive is the arm's inertia *about the vertical*, and that
is set entirely by how far the arm is extended. It also explains a July fit of
0.415 against 0.03-0.11 today.

So a scalar viscous/Coulomb pair reported as a property of the *unit* is a
category error on any joint whose load depends on configuration, and drift
tracking on such a joint is meaningless: re-measure at a different pose and the
"drift" you see is arm extension.

Two things here, in increasing order of ambition:

* **The canonical-pose contract** (:class:`CanonicalPose`,
  :func:`require_same_pose`) — the floor. A parameter is reported *at a declared
  pose*, and comparing two fits taken at different poses raises rather than
  returning a number, naming the joint, the servos and the tick deltas. Refusing
  is strictly better than a confident wrong comparison, and needs no extra data.
* **The parameter surface** (:class:`PoseSurface`, :func:`fit_pose_surface`) —
  regress the parameter on the pose variable that dominates it and report
  ``∂param/∂pose`` as a first-class output. Then a *value* is quoted at the
  canonical pose, a *sensitivity* says how much of any observed change is merely
  pose, and comparisons across poses become possible instead of forbidden.

Which pose variable dominates is per-joint physics, not a fit choice:

* **Yaw axes** (``shoulder_pan``) — gravity torque about a vertical axis is
  identically zero, so the load is *inertia*: use
  :func:`axis_inertia_feature`. This is the pan case above.
* **Pitch axes** (``shoulder_lift``, ``elbow_flex``) — the dominant term is the
  static gravity torque the joint holds: use :func:`gravity_torque_feature`.

Both features come from the robot model, evaluated at the pose the protocol
already pinned (:attr:`jaxterity.checkup.protocol.JointProtocol.pose_ticks`), so
the regression data exists in every stored run. :func:`joint_angle_feature` is
the model-free fallback — one servo's angle as a proxy — for when no model is at
hand.

Units
-----
Parameter values are whatever was fitted (N·m·s/rad for viscous damping, N·m for
Coulomb friction). Features carry their own units: kg·m² for axis inertia, N·m
for gravity torque, rad for a joint angle. A slope is therefore
*parameter-unit per feature-unit* and :attr:`PoseSurface.feature_units` records
which, because a slope whose units are unstated is not a measurement.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import CalibrationError
from .types import JointCalibration

#: Default pose agreement, raw ticks. Matches
#: :data:`jaxterity.checkup.protocol.POSE_TOLERANCE_TICKS` — servos never settle
#: on the same tick, so "the same pose" has to be a distance, not equality.
POSE_TOLERANCE_TICKS = 25

#: Minimum distinct feature values needed to fit a slope. Two points always fit a
#: line exactly, leaving no residual to judge it by, so two is accepted but
#: reported with ``r_squared`` NaN rather than a flattering 1.0.
_MIN_POSES = 2

#: A pose feature: held pose as ``(servo_id, ticks)`` pairs → one scalar.
PoseFeature = Callable[[tuple[tuple[int, int], ...]], float]


class CrossPoseComparisonError(CalibrationError):
    """Raised when two fits taken at different poses are compared as if equal.

    Carries the offending servos and their tick deltas, because "these are not
    the same experiment" is only useful if it says *where*.
    """


@dataclasses.dataclass(frozen=True)
class PoseObservation:
    """One fitted parameter value, and the pose it was measured at.

    Attributes
    ----------
    joint : str
        Joint the parameter belongs to.
    value : float
        The fitted parameter.
    pose_ticks : tuple[tuple[int, int], ...]
        Held pose as ``(servo_id, ticks)`` pairs — copy of the protocol's.
    ci95 : tuple[float, float] | None
        Wald interval, when the fit reported one. Used to weight the regression:
        a fit that barely constrained the parameter should not pull the surface
        as hard as one that did.
    label : str
        Free-text provenance (run id, session name) so a refusal can name which
        two things disagreed.
    """

    joint: str
    value: float
    pose_ticks: tuple[tuple[int, int], ...]
    ci95: tuple[float, float] | None = None
    label: str = ""

    @property
    def sigma(self) -> float:
        """Standard error implied by the 95 % interval (NaN if none)."""
        if self.ci95 is None:
            return float("nan")
        return float(self.ci95[1] - self.ci95[0]) / (2.0 * 1.959963984540054)


@dataclasses.dataclass(frozen=True)
class CanonicalPose:
    """A declared pose that parameters are reported *at*.

    Attributes
    ----------
    name : str
        Human-readable name, used verbatim in refusals ("home", "arm-extended").
    pose_ticks : tuple[tuple[int, int], ...]
        The pose, as ``(servo_id, ticks)`` pairs.
    tolerance_ticks : int
        Per-servo agreement required to call a pose this one.
    """

    name: str
    pose_ticks: tuple[tuple[int, int], ...]
    tolerance_ticks: int = POSE_TOLERANCE_TICKS

    def deltas(self, pose: Sequence[tuple[int, int]]) -> dict[int, int]:
        """Per-servo signed tick differences ``pose - canonical``.

        Only servos present in *both* poses are compared: a pose that carries
        extra servos is not thereby a different pose.
        """
        mine = dict(self.pose_ticks)
        return {
            sid: int(ticks) - int(mine[sid])
            for sid, ticks in pose
            if sid in mine and int(ticks) != int(mine[sid])
        }

    def distance(self, pose: Sequence[tuple[int, int]]) -> int:
        """Largest per-servo tick difference from this pose (0 if identical)."""
        deltas = self.deltas(pose)
        return max((abs(d) for d in deltas.values()), default=0)

    def contains(self, pose: Sequence[tuple[int, int]]) -> bool:
        """Whether *pose* is this pose, within :attr:`tolerance_ticks`."""
        return self.distance(pose) <= self.tolerance_ticks


def require_same_pose(
    a: PoseObservation,
    b: PoseObservation,
    tolerance_ticks: int = POSE_TOLERANCE_TICKS,
) -> None:
    """Refuse a cross-pose comparison, by name.

    Call this before differencing, ratioing, or drift-gating two fitted values.
    It is the whole canonical-pose contract in one function: if the two poses
    differ by more than *tolerance_ticks* on any shared servo, raise instead of
    letting a pose difference be read as wear.

    Args
    ----
    a, b : PoseObservation
        The two fits about to be compared.
    tolerance_ticks : int
        Per-servo agreement required.

    Raises
    ------
    CrossPoseComparisonError
        If the poses differ, naming the joint, each offending servo and its tick
        delta. Also if the two observations are not the same joint at all.
    """
    if a.joint != b.joint:
        raise CrossPoseComparisonError(
            f"require_same_pose: {a.joint!r} and {b.joint!r} are different "
            "joints; there is nothing to compare."
        )
    ref = CanonicalPose(
        name=a.label or "first observation",
        pose_ticks=a.pose_ticks,
        tolerance_ticks=tolerance_ticks,
    )
    deltas = {
        sid: d
        for sid, d in ref.deltas(b.pose_ticks).items()
        if abs(d) > tolerance_ticks
    }
    if not deltas:
        return
    where = ", ".join(f"servo {sid}: {d:+d} ticks" for sid, d in sorted(deltas.items()))
    raise CrossPoseComparisonError(
        f"require_same_pose: {a.joint}'s two fits were taken at different held "
        f"poses ({a.label or 'a'} vs {b.label or 'b'}): {where} — beyond the "
        f"{tolerance_ticks}-tick tolerance. Comparing them would read the pose "
        f"difference as a change in the joint: on LK-0001 that mechanism moved "
        f"shoulder_pan's damping +229 %. Either re-measure both at one declared "
        f"pose, or fit a pose surface (fit_pose_surface) and compare at a "
        f"canonical pose."
    )


@dataclasses.dataclass(frozen=True)
class PoseSurface:
    """A parameter fitted as an affine function of one pose variable.

    ``value(x) = intercept + slope * x``, where ``x`` is the feature.

    Attributes
    ----------
    joint : str
        Joint name.
    feature_name, feature_units : str
        What was regressed against, and in what units — so ``slope`` is
        interpretable.
    intercept, slope : float
        Affine coefficients. ``slope`` *is* the ``∂param/∂pose`` this exists to
        report.
    slope_ci95 : tuple[float, float] | None
        Wald interval on the slope, or ``None`` with fewer than three poses
        (two points fit a line with zero residual and no error estimate).
    canonical : CanonicalPose | None
        The pose values are quoted at, when one was declared.
    canonical_feature, canonical_value : float
        The feature at the canonical pose, and the parameter there. These are the
        numbers to report and to track over time.
    r_squared : float
        Fraction of the parameter's spread the pose variable explains. NaN with
        two poses. A high value is the finding: it means what looked like drift
        was pose.
    residual_rms : float
        RMS of the fit residual, parameter units.
    features : tuple[float, ...]
        The per-observation feature values, in input order (kept so a caller can
        see the lever arm the slope was fitted over).
    values : tuple[float, ...]
        The per-observation parameter values, in the same order.
    n_poses : int
        Number of observations.
    through_origin : bool
        Whether the fit was proportional (no intercept).
    """

    joint: str
    feature_name: str
    feature_units: str
    intercept: float
    slope: float
    slope_ci95: tuple[float, float] | None
    canonical: CanonicalPose | None
    canonical_feature: float
    canonical_value: float
    r_squared: float
    residual_rms: float
    features: tuple[float, ...]
    values: tuple[float, ...]
    n_poses: int
    through_origin: bool

    def at(self, feature: float) -> float:
        """The parameter at a given feature value."""
        return float(self.intercept + self.slope * float(feature))

    @property
    def spread_explained(self) -> float:
        """Parameter change the fitted poses' feature range accounts for.

        In parameter units: ``|slope| * (max feature - min feature)``. Compare it
        against the raw spread of the observed values — if it accounts for most
        of it, the joint's apparent instability is geometry.
        """
        if not self.features:
            return float("nan")
        return abs(self.slope) * (max(self.features) - min(self.features))

    @property
    def raw_spread_ratio(self) -> float:
        """Largest / smallest observed parameter value — the damage, unadjusted.

        3.29 for LK-0001's ``shoulder_pan`` across two sessions that ran the
        *same* recipe. Quote it next to :attr:`normalized_spread` to show how
        much of it was geometry.
        """
        lo, hi = min(self.values), max(self.values)
        return float(hi / lo) if lo > 0 else float("inf")

    @property
    def normalized_spread(self) -> float:
        """Relative spread of ``value / feature`` across the observations.

        The test of whether the parameter is *proportional* to the feature: if
        this is small while :attr:`raw_spread_ratio` is large, the parameter is
        not a property of the joint at all, it is that ratio — for LK-0001's pan,
        0.04 against a raw ratio of 3.29. Returns NaN if any feature is zero.
        """
        feats = np.asarray(self.features, dtype=float)
        vals = np.asarray(self.values, dtype=float)
        if np.any(feats == 0.0):
            return float("nan")
        ratio = vals / feats
        mean = float(np.mean(ratio))
        return float(np.ptp(ratio) / mean) if mean != 0 else float("nan")

    @property
    def pose_dominated(self) -> bool:
        """Whether pose explains most of this parameter's observed variation.

        True when :attr:`r_squared` is at least 0.8 over three or more poses. A
        two-pose fit cannot support the claim and returns False, even though its
        line passes through both points exactly — which is precisely why
        ``r_squared`` is NaN there rather than 1.0.
        """
        return self.n_poses >= 3 and self.r_squared >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        d = dataclasses.asdict(self)
        d["features"] = list(self.features)
        d["values"] = list(self.values)
        if self.slope_ci95 is not None:
            d["slope_ci95"] = list(self.slope_ci95)
        if self.canonical is not None:
            d["canonical"] = {
                "name": self.canonical.name,
                "pose_ticks": [[int(s), int(t)] for s, t in self.canonical.pose_ticks],
                "tolerance_ticks": int(self.canonical.tolerance_ticks),
            }
        return d


def fit_pose_surface(
    observations: Sequence[PoseObservation],
    feature: PoseFeature,
    *,
    feature_name: str = "pose",
    feature_units: str = "",
    canonical: CanonicalPose | None = None,
    through_origin: bool = False,
) -> PoseSurface:
    """Regress a fitted parameter on one pose variable.

    Weighted least squares on ``[1, x]``, weighting each observation by
    ``1/sigma²`` when its interval is known and equally otherwise — a fit that
    barely constrained the parameter should not pull the surface as hard as one
    that did.

    Args
    ----
    observations : Sequence[PoseObservation]
        Two or more fits of the *same* parameter on the *same* joint, at
        different poses. Same-pose duplicates are allowed and are what gives the
        residual something to measure; what is required is at least two
        *distinct* feature values.
    feature : PoseFeature
        Pose → scalar (see :func:`axis_inertia_feature`,
        :func:`gravity_torque_feature`, :func:`joint_angle_feature`).
    feature_name, feature_units : str
        Labels carried into the result so the slope is interpretable.
    canonical : CanonicalPose | None
        Pose to quote the value at. Defaults to the first observation's pose,
        named ``"first observation"`` — a default that at least makes the
        reference explicit rather than implied.
    through_origin : bool
        Fit ``value = slope * x`` with no intercept. Use it when the parameter is
        expected to *scale* with the feature rather than shift with it — and
        check :attr:`PoseSurface.normalized_spread` to see whether it does. On
        LK-0001's ``shoulder_pan`` the proportional form is the true one: its
        damping/inertia ratio is 14.5 against 13.9 across the two sessions
        (4 % apart) while the raw damping differs 3.3x, which says the fit was
        absorbing unmodelled inertia into the damping channel rather than
        measuring pose-modulated friction. An affine fit through two points
        cannot tell you that; the ratio can.

    Returns
    -------
    PoseSurface
        With ``slope`` = ``∂param/∂pose``.

    Raises
    ------
    CalibrationError
        If fewer than two observations are given, they are not all the same
        joint, or every pose maps to the same feature value (a slope through one
        x is not identifiable — say so rather than returning a huge number).
    """
    if len(observations) < _MIN_POSES:
        raise CalibrationError(
            f"fit_pose_surface: needs at least {_MIN_POSES} observations at "
            f"different poses, got {len(observations)}. One pose supports a "
            "value, not a surface."
        )
    joints = {o.joint for o in observations}
    if len(joints) != 1:
        raise CalibrationError(
            f"fit_pose_surface: observations span several joints {sorted(joints)}; "
            "fit one joint at a time — the dominating pose variable is per-joint."
        )
    joint = observations[0].joint

    x = np.array([float(feature(o.pose_ticks)) for o in observations])
    y = np.array([float(o.value) for o in observations])
    if float(np.ptp(x)) == 0.0:
        raise CalibrationError(
            f"fit_pose_surface: every pose gives {feature_name} = {x[0]:.6g} for "
            f"{joint}, so the slope is not identifiable. Either the poses really "
            "are the same (report a value, not a surface), or the chosen feature "
            "is not the one this joint's load depends on."
        )

    sigma = np.array([o.sigma for o in observations])
    weight = np.where(
        np.isfinite(sigma) & (sigma > 0), 1.0 / np.maximum(sigma, 1e-300) ** 2, np.nan
    )
    if not np.all(np.isfinite(weight)):
        weight = np.ones_like(y)  # mixed or absent intervals: weight equally

    design = x[:, None] if through_origin else np.column_stack([np.ones_like(x), x])
    scale = np.sqrt(weight)
    coef, *_ = np.linalg.lstsq(design * scale[:, None], y * scale, rcond=None)
    intercept = 0.0 if through_origin else float(coef[0])
    slope = float(coef[-1])

    predicted = intercept + slope * x
    residual = y - predicted
    dof = len(observations) - design.shape[1]
    if dof > 0:
        ss_tot = float(np.sum(weight * (y - np.average(y, weights=weight)) ** 2))
        ss_res = float(np.sum(weight * residual**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sigma2 = ss_res / dof
        cov = sigma2 * np.linalg.inv((design * weight[:, None]).T @ design)
        half = 1.959963984540054 * float(np.sqrt(max(cov[-1, -1], 0.0)))
        slope_ci95: tuple[float, float] | None = (slope - half, slope + half)
        residual_rms = float(np.sqrt(np.mean(residual**2)))
    else:
        r_squared, slope_ci95, residual_rms = float("nan"), None, 0.0

    reference = canonical or CanonicalPose(
        name="first observation", pose_ticks=observations[0].pose_ticks
    )
    canonical_feature = float(feature(reference.pose_ticks))
    return PoseSurface(
        joint=joint,
        feature_name=feature_name,
        feature_units=feature_units,
        intercept=intercept,
        slope=slope,
        slope_ci95=slope_ci95,
        canonical=reference,
        canonical_feature=canonical_feature,
        canonical_value=intercept + slope * canonical_feature,
        r_squared=r_squared,
        residual_rms=residual_rms,
        features=tuple(float(v) for v in x),
        values=tuple(float(v) for v in y),
        n_poses=len(observations),
        through_origin=through_origin,
    )


def _pose_to_qpos(
    pose_ticks: Sequence[tuple[int, int]],
    calibrations: Sequence[JointCalibration],
    mj_model,
) -> np.ndarray:
    """Held pose (servo ticks) → a full qpos vector, unlisted joints at qpos0."""
    q = np.array(mj_model.qpos0, dtype=float).copy()
    by_servo = {int(c.servo_id): c for c in calibrations}
    for sid, ticks in pose_ticks:
        cal = by_servo.get(int(sid))
        if cal is None:
            continue  # a servo with no calibration cannot be placed; leave qpos0
        try:
            jid = int(mj_model.joint(cal.joint).id)
        except KeyError:
            continue  # calibration names a joint this model does not have
        q[int(mj_model.jnt_qposadr[jid])] = cal.ticks_to_rad(float(ticks))
    return q


def _joint_dof(mj_model, joint: str) -> int:
    try:
        return int(mj_model.jnt_dofadr[int(mj_model.joint(joint).id)])
    except KeyError as exc:  # pragma: no cover - model/name mismatch
        raise CalibrationError(
            f"pose feature: the model has no joint named {joint!r}."
        ) from exc


def axis_inertia_feature(
    robot,
    calibrations: Sequence[JointCalibration],
    joint: str,
) -> PoseFeature:
    """Inertia about *joint*'s own axis, as a function of the held pose.

    ``M(q)[dof, dof]`` from the model's mass matrix — what the joint actually has
    to accelerate, in kg·m². This is the right feature for a **yaw** axis: the
    gravity torque about a vertical axis is identically zero, so pose does not
    change what the joint holds, it changes what the joint must swing. Arm
    extension is a proxy for this quantity; this is the quantity.

    Args
    ----
    robot : Robot
        Model the inertia is evaluated on.
    calibrations : Sequence[JointCalibration]
        Tick↔radian maps, used to place the held pose. Servos without one are
        left at the model's reference configuration.
    joint : str
        The joint whose axis inertia is wanted.

    Returns
    -------
    PoseFeature
        ``pose_ticks -> kg·m²``.

    Units
    -----
    kg·m².
    """
    import mujoco

    mj_model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    dof = _joint_dof(mj_model, joint)
    data = mujoco.MjData(mj_model)
    full = np.zeros((mj_model.nv, mj_model.nv))

    def evaluate(pose_ticks: tuple[tuple[int, int], ...]) -> float:
        data.qpos[:] = _pose_to_qpos(pose_ticks, calibrations, mj_model)
        data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, data)
        mujoco.mj_fullM(mj_model, data, full)
        return float(full[dof, dof])

    return evaluate


def gravity_torque_feature(
    robot,
    calibrations: Sequence[JointCalibration],
    joint: str,
) -> PoseFeature:
    """Static gravity torque *joint* holds at the held pose, N·m.

    ``qfrc_bias[dof]`` at zero velocity, so Coriolis vanishes and what is left is
    gravity. The right feature for a **pitch** axis (``shoulder_lift``,
    ``elbow_flex``): extending the arm multiplies the torque the joint holds,
    which loads the friction mechanisms directly.

    Args
    ----
    robot : Robot
        Model the torque is evaluated on.
    calibrations : Sequence[JointCalibration]
        Tick↔radian maps for placing the pose.
    joint : str
        Joint whose held torque is wanted.

    Returns
    -------
    PoseFeature
        ``pose_ticks -> N·m`` (signed: sign is the joint-axis convention).

    Units
    -----
    N·m.
    """
    import mujoco

    mj_model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    dof = _joint_dof(mj_model, joint)
    data = mujoco.MjData(mj_model)

    def evaluate(pose_ticks: tuple[tuple[int, int], ...]) -> float:
        data.qpos[:] = _pose_to_qpos(pose_ticks, calibrations, mj_model)
        data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, data)
        return float(data.qfrc_bias[dof])

    return evaluate


def joint_angle_feature(
    calibration: JointCalibration,
) -> PoseFeature:
    """One servo's angle as the pose variable, rad — the model-free fallback.

    Useful when no robot model is at hand, or as a sanity check that a
    model-based feature is picking up the pose change it should: on a serial arm
    the elbow angle is a decent stand-in for arm extension. It is a *proxy*, and
    a slope against it is not a physical derivative of anything — prefer
    :func:`axis_inertia_feature` or :func:`gravity_torque_feature` when a model
    exists.

    Args
    ----
    calibration : JointCalibration
        The servo whose angle to read out of the pose.

    Returns
    -------
    PoseFeature
        ``pose_ticks -> rad``. NaN when that servo is absent from the pose,
        which propagates into the fit rather than silently reading 0 rad.

    Units
    -----
    rad.
    """
    servo_id = int(calibration.servo_id)

    def evaluate(pose_ticks: tuple[tuple[int, int], ...]) -> float:
        for sid, ticks in pose_ticks:
            if int(sid) == servo_id:
                return float(calibration.ticks_to_rad(float(ticks)))
        return float("nan")

    return evaluate


__all__ = [
    "POSE_TOLERANCE_TICKS",
    "CanonicalPose",
    "CrossPoseComparisonError",
    "PoseFeature",
    "PoseObservation",
    "PoseSurface",
    "axis_inertia_feature",
    "fit_pose_surface",
    "gravity_torque_feature",
    "joint_angle_feature",
    "require_same_pose",
]
