# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Structure and environment — suspension, torsional compliance, rolling drag.

The suspension tests are about the *coupling* the tip-over gate was missing:
a load off-centre tilts the base, the tilt walks the centre of mass, and the
margin shrinks. The torsional tests pin the resonance round-trip (the only
practical way to measure a structural stiffness), and the rolling-resistance
tests recover both coefficients from a coast-down, which is the experiment the
model exists to be fitted from.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import CalibrationError, JaxterityError
from jaxterity.physics.compliance import (
    LEKIWI_SUSPENSION,
    JointCompliance,
    RollingResistance,
    SuspensionProfile,
    compliant_margin,
    fit_coast_down,
    margin_floor_credit,
    rolling_resistance_force,
    stiffness_from_resonance,
    suspension_tilt,
    torsional_resonance_hz,
)

_GRAVITY = 9.81


def _square_suspension(stiffness=4.0e4, com_height=0.12) -> SuspensionProfile:
    return SuspensionProfile(
        contact_xy=((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)),
        stiffness_n_per_m=(stiffness,) * 4,
        com_height_m=com_height,
    )


# --- suspension ------------------------------------------------------------


def test_a_centred_load_does_not_tilt_a_symmetric_base():
    tilt = suspension_tilt(_square_suspension(), (0.0, 0.0), 100.0)
    assert tilt == pytest.approx((0.0, 0.0), abs=1e-12)


def test_an_offset_load_tilts_toward_it():
    tilt_x, tilt_y = suspension_tilt(_square_suspension(), (0.05, 0.0), 100.0)
    # z(x) = z0 + a*x with the plate pressed down where the load is: the slope
    # rises away from the load.
    assert tilt_x > 0.0
    assert tilt_y == pytest.approx(0.0, abs=1e-12)


def test_softer_springs_tilt_more():
    stiff = suspension_tilt(_square_suspension(stiffness=1e5), (0.05, 0.0), 100.0)
    soft = suspension_tilt(_square_suspension(stiffness=1e4), (0.05, 0.0), 100.0)
    assert abs(soft[0]) > abs(stiff[0])
    # Linear model: 10x softer is exactly 10x the tilt.
    assert abs(soft[0]) == pytest.approx(10.0 * abs(stiff[0]), rel=1e-9)


def test_tilt_scales_with_the_load():
    light = suspension_tilt(_square_suspension(), (0.05, 0.0), 50.0)
    heavy = suspension_tilt(_square_suspension(), (0.05, 0.0), 100.0)
    assert abs(heavy[0]) == pytest.approx(2 * abs(light[0]), rel=1e-9)


def test_collinear_contacts_are_rejected_at_construction_or_solve():
    hinge = SuspensionProfile(
        contact_xy=((0.0, 0.0), (0.1, 0.0), (0.2, 0.0)),
        stiffness_n_per_m=(1e4,) * 3,
        com_height_m=0.1,
    )
    with pytest.raises(CalibrationError, match="collinear"):
        suspension_tilt(hinge, (0.0, 0.0), 100.0)


def test_two_contacts_are_rejected():
    with pytest.raises(JaxterityError, match="at least 3 contact"):
        SuspensionProfile(
            contact_xy=((0.0, 0.0), (0.1, 0.0)),
            stiffness_n_per_m=(1e4, 1e4),
            com_height_m=0.1,
        )


def test_zero_stiffness_is_rejected_with_the_rigid_limit_spelled_out():
    with pytest.raises(JaxterityError, match="large-stiffness limit"):
        SuspensionProfile(
            contact_xy=((0.1, 0.1), (-0.1, 0.1), (0.0, -0.1)),
            stiffness_n_per_m=(1e4, 0.0, 1e4),
            com_height_m=0.1,
        )


def test_compliance_only_ever_shrinks_the_margin():
    suspension = _square_suspension()
    rigid = np.array([0.03, 0.02, 0.05])
    zmp = np.array([[0.05, 0.0], [0.0, 0.04], [0.0, 0.0]])
    corrected = compliant_margin(rigid, zmp, suspension, 100.0)
    assert np.all(corrected <= rigid + 1e-15)
    # A centred ZMP costs nothing; an offset one does.
    assert corrected[2] == pytest.approx(rigid[2])
    assert corrected[0] < rigid[0]


def test_a_taller_robot_loses_more_margin():
    zmp = np.array([[0.05, 0.0]])
    rigid = np.array([0.03])
    short = compliant_margin(rigid, zmp, _square_suspension(com_height=0.1), 100.0)
    tall = compliant_margin(rigid, zmp, _square_suspension(com_height=0.4), 100.0)
    assert tall[0] < short[0]


def test_margin_floor_credit_is_positive_and_grows_as_springs_soften():
    polygon = ((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1))
    stiff = margin_floor_credit(_square_suspension(stiffness=1e5), polygon, 100.0)
    soft = margin_floor_credit(_square_suspension(stiffness=1e4), polygon, 100.0)
    assert 0.0 < stiff < soft


def test_lekiwi_suspension_credit_is_a_fraction_of_the_declared_floor():
    from jaxterity.checkup.stability import LEKIWI_BASE

    credit = margin_floor_credit(
        LEKIWI_SUSPENSION, LEKIWI_BASE.support_polygon_xy, 30.0
    )
    # The representative springs explain part of the 20 mm floor, not all of
    # it — which is exactly what the floor's comment claims is left over.
    assert 0.0 < credit < LEKIWI_BASE.margin_min_m


def test_zmp_margins_honours_a_declared_suspension():
    """The wiring test: the tip-over gate must actually consume the springs.

    Same trajectory, same base, twice — once rigid (the historical behaviour)
    and once with support compliance. The compliant margin must be strictly
    smaller everywhere, which is what lets a floor that was carrying this
    effect as slack come down honestly.
    """
    pytest.importorskip("mujoco")
    from jaxterity import zoo
    from jaxterity.calibrate.active import dof_joint_names
    from jaxterity.checkup.stability import LEKIWI_BASE, zmp_margins

    robot = zoo.load("so100")
    dof_names = dof_joint_names(robot)
    t = np.linspace(0.0, 2.0, 50)
    q_t = np.zeros((t.size, len(dof_names)))
    q_t[:, 0] = 0.4 * np.sin(2 * np.pi * t / 2.0)  # a real, accelerating motion

    rigid = zmp_margins(robot, q_t, float(t[1] - t[0]), dof_names, LEKIWI_BASE)
    compliant = zmp_margins(
        robot,
        q_t,
        float(t[1] - t[0]),
        dof_names,
        LEKIWI_BASE,
        suspension=_square_suspension(stiffness=5.0e3),
    )
    assert np.all(compliant < rigid)


# --- torsional compliance --------------------------------------------------


def test_resonance_round_trips_through_stiffness():
    stiffness = 500.0
    inertia = 0.01
    f = float(torsional_resonance_hz(stiffness, inertia))
    assert float(stiffness_from_resonance(f, inertia)) == pytest.approx(stiffness)


def test_stiffer_joint_rings_higher():
    assert float(torsional_resonance_hz(2000.0, 0.01)) > float(
        torsional_resonance_hz(500.0, 0.01)
    )


def test_torque_is_linear_in_twist_and_rate():
    joint = JointCompliance(stiffness=500.0, damping=2.0, inertia=0.01)
    assert float(joint.torque(0.01)) == pytest.approx(5.0)
    assert float(joint.torque(0.01, 1.0)) == pytest.approx(7.0)


def test_damping_ratio_flags_a_ringing_mode():
    ringing = JointCompliance(stiffness=500.0, damping=0.05, inertia=0.01)
    damped = JointCompliance(stiffness=500.0, damping=3.0, inertia=0.01)
    assert ringing.damping_ratio < 0.1 < damped.damping_ratio


def test_non_positive_stiffness_is_rejected():
    with pytest.raises(JaxterityError, match="stiffness > 0"):
        JointCompliance(stiffness=0.0)


# --- rolling resistance ----------------------------------------------------


def test_resistance_opposes_motion_and_vanishes_at_rest():
    assert float(rolling_resistance_force(30.0, 0.0, c_rr=0.02)) == pytest.approx(0.0)
    assert float(rolling_resistance_force(30.0, 1.0, c_rr=0.02)) < 0.0
    assert float(rolling_resistance_force(30.0, -1.0, c_rr=0.02)) > 0.0


def test_coulomb_term_scales_with_load_and_viscous_with_speed():
    light = float(rolling_resistance_force(10.0, 1.0, c_rr=0.02))
    heavy = float(rolling_resistance_force(30.0, 1.0, c_rr=0.02))
    assert heavy == pytest.approx(3 * light, rel=1e-6)
    slow = float(rolling_resistance_force(0.0, 1.0, c_rr=0.0, c_v=5.0))
    fast = float(rolling_resistance_force(0.0, 2.0, c_rr=0.0, c_v=5.0))
    assert fast == pytest.approx(2 * slow)


def test_negative_coefficients_are_rejected():
    with pytest.raises(JaxterityError, match="c_rr >= 0"):
        RollingResistance(c_rr=-0.01)


def _coast_down(c_rr=0.02, c_v=1.5, mass=6.0, v0=1.5, dt=0.02):
    """Integrate the model's own coast-down, stopping when the base does."""
    times = [0.0]
    speeds = [v0]
    v = v0
    t = 0.0
    while v > 0.05 and t < 60.0:
        acceleration = -(c_rr * mass * _GRAVITY + c_v * v) / mass
        v = v + dt * acceleration
        t = t + dt
        times.append(t)
        speeds.append(v)
    return np.asarray(times), np.asarray(speeds)


def test_fit_recovers_both_coefficients_from_a_coast_down():
    time, velocity = _coast_down(c_rr=0.02, c_v=1.5)
    fitted = fit_coast_down(time, velocity, mass=6.0)
    assert fitted["c_rr"] == pytest.approx(0.02, rel=0.05)
    assert fitted["c_v"] == pytest.approx(1.5, rel=0.05)


def test_fit_refuses_a_base_that_never_slows():
    n = 50
    with pytest.raises(CalibrationError, match="never slows"):
        fit_coast_down(np.arange(n) * 0.02, np.full(n, 1.0), mass=6.0)


def test_fit_refuses_a_non_positive_mass():
    time, velocity = _coast_down()
    with pytest.raises(CalibrationError, match="positive vehicle mass"):
        fit_coast_down(time, velocity, mass=0.0)


def test_validate_passes_on_its_own_coast_down_and_flags_a_wrong_surface():
    time, velocity = _coast_down(c_rr=0.02, c_v=1.5)
    assert RollingResistance(c_rr=0.02, c_v=1.5).validate(time, velocity, 6.0).ok
    carpet = RollingResistance(c_rr=0.08, c_v=1.5)
    assert not carpet.validate(time, velocity, 6.0).ok
