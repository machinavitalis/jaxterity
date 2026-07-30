# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Pose-conditioned friction parameters.

The headline test here is on real hardware data: the two committed LK-0001
sessions ran the *same* excitation recipe and fitted ``shoulder_pan`` dampings
3.29x apart. Against the axis inertia at each held pose, the ratio is consistent
to 4 %. That is both the acceptance criterion for this work and a finding about
the model — see :func:`test_the_real_3x_discrepancy_is_explained_by_axis_inertia`.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate import (
    CanonicalPose,
    CrossPoseComparisonError,
    PoseObservation,
    axis_inertia_feature,
    fit_pose_surface,
    gravity_torque_feature,
    joint_angle_feature,
    require_same_pose,
)
from jaxterity.calibrate.types import JointCalibration
from jaxterity.errors import CalibrationError

_DATA = pathlib.Path(__file__).parents[1]
_CALIB = _DATA / "calibrate" / "data" / "lekiwi" / "calibration_final.json"
_SESSIONS = _DATA / "checkup" / "data"

_HOME = ((1, 1794), (2, 801), (3, 3077), (4, 959), (5, 2380), (6, 2048))


@pytest.fixture(scope="module")
def calibrations():
    doc = json.loads(_CALIB.read_text())
    fields = JointCalibration.__dataclass_fields__
    return [
        JointCalibration(**{k: v for k, v in c.items() if k in fields})
        for c in doc["calibrations"]
    ]


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def pan_sessions():
    """The two real sessions' shoulder_pan fits, with their held poses."""
    out = []
    for tag in "ab":
        doc = json.loads((_SESSIONS / f"lk0001_v2_session_{tag}.json").read_text())
        (protocol,) = [
            p for p in doc["protocol"]["joints"] if p["joint"] == "shoulder_pan"
        ]
        (fit,) = [j for j in doc["joints"] if j["joint"] == "shoulder_pan"]
        out.append(
            PoseObservation(
                joint="shoulder_pan",
                value=fit["damping"],
                pose_ticks=tuple((int(s), int(t)) for s, t in protocol["pose_ticks"]),
                ci95=tuple(fit["damping_ci95"]),
                label=f"session_{tag}",
            )
        )
    return out


# --- the canonical-pose contract ------------------------------------------


def test_same_pose_within_tolerance_is_allowed():
    a = PoseObservation("elbow_flex", 0.05, _HOME, label="run1")
    jittered = tuple((sid, ticks + 7) for sid, ticks in _HOME)
    b = PoseObservation("elbow_flex", 0.052, jittered, label="run2")
    require_same_pose(a, b)  # settling noise is not a different experiment


def test_a_different_pose_is_refused_and_names_the_servos():
    a = PoseObservation("shoulder_pan", 0.033, _HOME, label="session_a")
    moved = tuple((sid, ticks + (2094 if sid == 2 else 0)) for sid, ticks in _HOME)
    b = PoseObservation("shoulder_pan", 0.109, moved, label="session_b")
    with pytest.raises(CrossPoseComparisonError) as exc:
        require_same_pose(a, b)
    message = str(exc.value)
    assert "shoulder_pan" in message
    assert "servo 2" in message and "+2094 ticks" in message
    assert "session_a" in message and "session_b" in message
    # It must also say what to do instead, not merely refuse.
    assert "fit_pose_surface" in message


def test_the_real_sessions_are_refused_by_the_contract(pan_sessions):
    # This is the case that motivated the work: identical recipes, and the
    # contract is what stops the 3.29x being reported as wear.
    with pytest.raises(CrossPoseComparisonError):
        require_same_pose(*pan_sessions)


def test_comparing_two_different_joints_is_refused():
    a = PoseObservation("elbow_flex", 0.05, _HOME)
    b = PoseObservation("wrist_flex", 0.05, _HOME)
    with pytest.raises(CrossPoseComparisonError, match="different"):
        require_same_pose(a, b)


def test_canonical_pose_distance_ignores_servos_it_does_not_declare():
    canonical = CanonicalPose("home", _HOME[:3])
    extended = (*_HOME[:3], (9, 4000))
    assert canonical.distance(extended) == 0
    assert canonical.contains(extended)


def test_canonical_pose_reports_the_largest_departure():
    canonical = CanonicalPose("home", _HOME)
    moved = tuple(
        (sid, ticks + (30 if sid == 1 else 200 if sid == 3 else 0))
        for sid, ticks in _HOME
    )
    assert canonical.distance(moved) == 200
    assert canonical.deltas(moved) == {1: 30, 3: 200}
    assert not canonical.contains(moved)


# --- the parameter surface -------------------------------------------------


def test_the_real_3x_discrepancy_is_explained_by_axis_inertia(
    pan_sessions, robot, calibrations
):
    """T-235's acceptance criterion, on real hardware data.

    Two deliberately different poses on one unit whose raw fits differ ~3x must
    yield consistent pose-conditioned parameters. They do, and the *form* is
    informative: the consistent quantity is the ratio damping/inertia, not an
    affine offset, which says the damping channel was absorbing inertia the
    single-joint model does not carry.
    """
    feature = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    surface = fit_pose_surface(
        pan_sessions,
        feature,
        feature_name="axis_inertia",
        feature_units="kg*m^2",
        through_origin=True,
    )
    assert surface.raw_spread_ratio == pytest.approx(3.29, abs=0.05)
    assert surface.normalized_spread < 0.05, (
        f"damping/inertia should be consistent across the two poses, "
        f"got {surface.normalized_spread:.1%} spread"
    )
    # The inertia really does move by the ~3x the damping did.
    lo, hi = min(surface.features), max(surface.features)
    assert hi / lo == pytest.approx(3.4, abs=0.2)


def test_the_pose_with_more_inertia_is_the_pose_with_more_damping(
    pan_sessions, robot, calibrations
):
    # Falsifiable ordering check on the mechanism claim: if the inertia story is
    # wrong, nothing forces these two rankings to agree.
    feature = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    pairs = sorted((feature(o.pose_ticks), o.value) for o in pan_sessions)
    inertias, dampings = zip(*pairs)
    assert list(dampings) == sorted(dampings), (
        f"damping did not increase with axis inertia: {pairs}"
    )
    assert inertias[0] < inertias[1]


def test_a_surface_recovers_a_known_law_and_its_derivative():
    # value = 0.02 + 8.0 * x, six poses. The slope IS the deliverable.
    def feature(pose):
        return dict(pose)[2] / 1000.0

    observations = [
        PoseObservation(
            "elbow_flex", 0.02 + 8.0 * (ticks / 1000.0), ((1, 1000), (2, ticks))
        )
        for ticks in (500, 900, 1300, 1700, 2100, 2500)
    ]
    surface = fit_pose_surface(observations, feature, feature_name="x")
    assert surface.slope == pytest.approx(8.0, rel=1e-9)
    assert surface.intercept == pytest.approx(0.02, abs=1e-9)
    assert surface.r_squared == pytest.approx(1.0, abs=1e-9)
    assert surface.at(1.0) == pytest.approx(8.02, rel=1e-9)
    assert surface.pose_dominated


def test_a_surface_fitted_on_disjoint_pose_subsets_agrees_at_one_canonical_pose():
    # The consistency claim with enough poses to be non-trivial: fit on three
    # poses, fit on three others, quote both at the same canonical pose.
    rng = np.random.default_rng(20260729)

    def feature(pose):
        return dict(pose)[2] / 1000.0

    def observe(ticks):
        truth = 0.02 + 8.0 * (ticks / 1000.0)
        noisy = truth * (1.0 + 0.01 * rng.standard_normal())
        return PoseObservation("elbow_flex", noisy, ((1, 1000), (2, ticks)))

    canonical = CanonicalPose("home", ((1, 1000), (2, 1500)))
    low = fit_pose_surface(
        [observe(t) for t in (500, 800, 1100)], feature, canonical=canonical
    )
    high = fit_pose_surface(
        [observe(t) for t in (1900, 2200, 2500)], feature, canonical=canonical
    )
    # Raw values across the two groups differ by more than 3x...
    assert max(high.values) / min(low.values) > 3.0
    # ...yet both surfaces agree on the parameter at the canonical pose.
    assert high.canonical_value == pytest.approx(low.canonical_value, rel=0.05)
    assert low.canonical_feature == pytest.approx(1.5)


def test_intervals_weight_the_regression():
    def feature(pose):
        return float(dict(pose)[2])

    # A wildly off value with a huge interval must not drag the surface as hard
    # as three tight ones.
    tight = [
        PoseObservation(
            "elbow_flex", 1.0 * t, ((2, t),), ci95=(1.0 * t - 0.01, 1.0 * t + 0.01)
        )
        for t in (1.0, 2.0, 3.0)
    ]
    sloppy = PoseObservation("elbow_flex", 50.0, ((2, 4.0),), ci95=(0.0, 100.0))
    weighted = fit_pose_surface([*tight, sloppy], feature)
    unweighted = fit_pose_surface(
        [PoseObservation(o.joint, o.value, o.pose_ticks) for o in (*tight, sloppy)],
        feature,
    )
    assert abs(weighted.slope - 1.0) < abs(unweighted.slope - 1.0)


def test_two_poses_report_nan_r_squared_rather_than_a_flattering_one():
    def feature(pose):
        return float(dict(pose)[2])

    surface = fit_pose_surface(
        [
            PoseObservation("elbow_flex", 0.1, ((2, 1),)),
            PoseObservation("elbow_flex", 0.3, ((2, 2),)),
        ],
        feature,
    )
    assert np.isnan(surface.r_squared)
    assert surface.slope_ci95 is None
    assert not surface.pose_dominated  # cannot support the claim on two points
    assert surface.slope == pytest.approx(0.2)


def test_three_poses_give_a_slope_interval():
    def feature(pose):
        return float(dict(pose)[2])

    surface = fit_pose_surface(
        [
            PoseObservation("elbow_flex", v, ((2, t),))
            for t, v in ((1.0, 0.10), (2.0, 0.31), (3.0, 0.49))
        ],
        feature,
    )
    assert surface.slope_ci95 is not None
    lo, hi = surface.slope_ci95
    assert lo < surface.slope < hi


def test_one_observation_is_refused():
    with pytest.raises(CalibrationError, match="at least 2"):
        fit_pose_surface([PoseObservation("elbow_flex", 0.1, ((2, 1),))], lambda p: 1.0)


def test_mixed_joints_are_refused():
    with pytest.raises(CalibrationError, match="several joints"):
        fit_pose_surface(
            [
                PoseObservation("elbow_flex", 0.1, ((2, 1),)),
                PoseObservation("wrist_flex", 0.2, ((2, 2),)),
            ],
            lambda p: float(dict(p)[2]),
        )


def test_a_constant_feature_is_refused_by_name():
    # The failure a caller most needs told about: the chosen feature does not
    # move with the poses they captured, so no slope exists.
    with pytest.raises(CalibrationError, match="not identifiable"):
        fit_pose_surface(
            [
                PoseObservation("elbow_flex", 0.1, ((2, 1),)),
                PoseObservation("elbow_flex", 0.2, ((2, 2),)),
            ],
            lambda p: 3.0,
            feature_name="constant",
        )


def test_spread_explained_is_in_parameter_units():
    def feature(pose):
        return float(dict(pose)[2])

    surface = fit_pose_surface(
        [PoseObservation("elbow_flex", 2.0 * t, ((2, t),)) for t in (1.0, 2.0, 3.0)],
        feature,
    )
    assert surface.spread_explained == pytest.approx(4.0)  # slope 2 over range 2


def test_to_dict_is_json_round_trippable(pan_sessions, robot, calibrations):
    feature = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    surface = fit_pose_surface(
        pan_sessions, feature, canonical=CanonicalPose("home", _HOME)
    )
    d = surface.to_dict()
    round_tripped = json.loads(json.dumps(d))
    assert set(round_tripped) == set(d)
    assert round_tripped["features"] == d["features"]
    assert round_tripped["slope"] == d["slope"]
    assert d["canonical"]["name"] == "home"
    assert d["canonical"]["pose_ticks"][0] == [1, 1794]


# --- the features ----------------------------------------------------------


def test_gravity_torque_is_zero_for_the_yaw_axis_which_is_why_inertia_is_used(
    robot, calibrations
):
    """The physics behind choosing inertia over gravity torque for pan.

    Gravity torque about a vertical axis is identically zero, so on a yaw joint
    the gravity feature carries no pose information at all — and a surface fitted
    against it would be refused as unidentifiable rather than quietly wrong.
    """
    yaw = gravity_torque_feature(robot, calibrations, "shoulder_pan")
    pitch = gravity_torque_feature(robot, calibrations, "shoulder_lift")
    extended = tuple((sid, ticks + (1200 if sid == 2 else 0)) for sid, ticks in _HOME)
    # Zero to solver noise, and five-plus orders below the pitch axis at the very
    # same pose — so a surface fitted against it would be refused, not wrong.
    assert abs(yaw(_HOME)) < 1e-5
    assert abs(yaw(extended)) < 1e-5
    assert abs(yaw(_HOME)) < 1e-4 * abs(pitch(_HOME))

    inertia = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    assert inertia(extended) != pytest.approx(inertia(_HOME), rel=1e-3)


def test_gravity_torque_does_move_with_pose_on_a_pitch_axis(robot, calibrations):
    gravity = gravity_torque_feature(robot, calibrations, "shoulder_lift")
    extended = tuple((sid, ticks + (900 if sid == 3 else 0)) for sid, ticks in _HOME)
    assert abs(gravity(extended) - gravity(_HOME)) > 1e-3


def test_axis_inertia_is_positive_and_monotone_in_the_lift_angle(robot, calibrations):
    # Monotone, not "grows": whether raising servo 2's ticks extends or folds the
    # arm is a sign convention of this unit's calibration, and asserting a
    # direction would be asserting the convention. On this calibration the sweep
    # *folds* the arm, so inertia falls; what matters is that the feature tracks
    # the pose smoothly and over a wide enough range to fit a slope against.
    inertia = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    values = [
        inertia(tuple((sid, ticks + (d if sid == 2 else 0)) for sid, ticks in _HOME))
        for d in (0, 400, 800, 1200)
    ]
    assert all(v > 0 for v in values)
    assert values == sorted(values, reverse=True), f"not monotone: {values}"
    assert max(values) / min(values) > 2.0


def test_joint_angle_feature_is_nan_for_a_servo_absent_from_the_pose(calibrations):
    (pan,) = [c for c in calibrations if c.joint == "shoulder_pan"]
    feature = joint_angle_feature(pan)
    assert np.isnan(feature(((9, 2000),)))
    assert feature(((pan.servo_id, pan.offset_ticks),)) == pytest.approx(0.0)


def test_unknown_joint_names_are_reported_not_ignored(robot, calibrations):
    with pytest.raises(CalibrationError, match="no joint named"):
        axis_inertia_feature(robot, calibrations, "not_a_joint")


def test_a_pose_naming_an_uncalibrated_servo_falls_back_to_the_reference_pose(
    robot, calibrations
):
    # A servo with no calibration cannot be placed; it must not be silently
    # read as 0 rad, which would be a *different* pose rather than an unknown one.
    inertia = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    assert inertia(((99, 1234),)) == pytest.approx(inertia(()))
