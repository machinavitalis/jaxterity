# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The LeKiwi calibration planner must derive the floor from the base's lowest
*surface* point, not its lowest geom centre.

Regression for the wheel-sinking bug: ``data.geom_xpos`` is a geom's centre, so
snapping the floor plane there put it at wheel-axle height — the bottom half of
every wheel penetrated the floor in the replay render, and the arm's clearance
check was stricter than reality by one wheel radius.

These cases run on a synthetic MuJoCo model so they need none of the Ekumen
LeKiwi assets (which the planner fetches out-of-tree).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from calib_planner_lekiwi import _geom_lowest_z  # noqa: E402

# Each geom is positioned so that its true lowest surface point is exactly z=0.
# Tilted cases matter: they are the ones a centre-based or axis-aligned
# approximation gets wrong.
_SCENE = """
<mujoco><worldbody>
  <body pos="0 0 0.05"><geom name="sphere" type="sphere" size="0.05"/></body>
  <body pos="0.3 0 0.09">
    <geom name="capsule_up" type="capsule" size="0.02 0.07"/></body>
  <body pos="0.6 0 0.02" euler="90 0 0">
    <geom name="capsule_flat" type="capsule" size="0.02 0.07"/></body>
  <body pos="0.9 0 0.03"><geom name="cyl_up" type="cylinder" size="0.04 0.03"/></body>
  <body pos="1.2 0 0.04" euler="90 0 0">
    <geom name="cyl_flat" type="cylinder" size="0.04 0.03"/></body>
  <body pos="1.5 0 0.06"><geom name="box" type="box" size="0.03 0.04 0.06"/></body>
  <body pos="1.8 0 0.070710678118654752" euler="0 45 0">
    <geom name="box_tilt" type="box" size="0.05 0.05 0.05"/></body>
  <body pos="2.1 0 0.07">
    <geom name="ellipsoid" type="ellipsoid" size="0.03 0.05 0.07"/></body>
</worldbody></mujoco>
"""

_TOL_M = 1e-7  # analytic support point; bounded by the scene's decimal literals


@pytest.fixture(scope="module")
def scene():
    m = mujoco.MjModel.from_xml_string(_SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize(
    "name",
    [
        "sphere",
        "capsule_up",
        "capsule_flat",
        "cyl_up",
        "cyl_flat",
        "box",
        "box_tilt",
        "ellipsoid",
    ],
)
def test_lowest_point_is_the_contact_point(scene, name):
    """Every geom rests on z=0, so its computed lowest point must be 0."""
    m, d = scene
    g = m.geom(name).id
    assert _geom_lowest_z(m, d, g) == pytest.approx(0.0, abs=_TOL_M)


def test_centre_would_sink_the_geom(scene):
    """The old centre-based derivation sits strictly above the contact point.

    This is the bug: for a wheel the gap is one radius, and the floor plane snaps
    to the axle.
    """
    m, d = scene
    for name in ("sphere", "cyl_up", "box_tilt"):
        g = m.geom(name).id
        centre_z = float(d.geom_xpos[g, 2])
        assert centre_z > _geom_lowest_z(m, d, g) + 1e-4


# A wide, flat plate — the shape whose bounding sphere is wildly wrong. This is
# the LeKiwi chassis case: 0.30 x 0.30 x 0.02, bottom resting at z = 0.
_PLATE = """
<mujoco><asset>
  <mesh name="plate" vertex="-0.15 -0.15 -0.01  0.15 -0.15 -0.01  0.15 0.15 -0.01
                             -0.15 0.15 -0.01  -0.15 -0.15 0.01  0.15 -0.15 0.01
                              0.15 0.15 0.01  -0.15 0.15 0.01"/>
</asset><worldbody>
  <body pos="0 0 0.01"><geom name="plate" type="mesh" mesh="plate"/></body>
  <body pos="1 0 0.01" euler="0 30 0">
    <geom name="plate_tilt" type="mesh" mesh="plate"/></body>
</worldbody></mujoco>
"""


@pytest.fixture(scope="module")
def plate():
    m = mujoco.MjModel.from_xml_string(_PLATE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_mesh_uses_vertices_not_the_bounding_sphere(plate):
    """A flat plate's bounding sphere is half its diagonal — unusable as a floor.

    Falling back to ``geom_rbound`` here would drop the derived floor ~20 cm
    below the real one and make the arm's clearance check permissive, which is
    the dangerous direction.
    """
    m, d = plate
    g = m.geom("plate").id
    assert _geom_lowest_z(m, d, g) == pytest.approx(0.0, abs=_TOL_M)

    rbound_floor = float(d.geom_xpos[g, 2]) - float(m.geom_rbound[g])
    assert rbound_floor < -0.15  # would have been ~20 cm too low


def test_mesh_accounts_for_rotation(plate):
    """A tilted plate reaches lower than a flat one; vertices must show that."""
    m, d = plate
    flat = _geom_lowest_z(m, d, m.geom("plate").id)
    tilt = _geom_lowest_z(m, d, m.geom("plate_tilt").id)
    assert tilt < flat
    # 30 deg about y on a 0.30 x 0.02 section: half-extent = 0.15*sin30 + 0.01*cos30
    expected = float(d.geom_xpos[m.geom("plate_tilt").id, 2]) - (
        0.15 * np.sin(np.pi / 6) + 0.01 * np.cos(np.pi / 6)
    )
    assert tilt == pytest.approx(expected, abs=1e-6)


def test_floor_is_the_min_over_base_geoms(scene):
    """The planner takes a min over base geoms; that min is the true ground."""
    m, d = scene
    floor = min(_geom_lowest_z(m, d, g) for g in range(m.ngeom))
    assert floor == pytest.approx(0.0, abs=_TOL_M)
    # and it is strictly below the old centre-based answer
    assert floor < min(float(d.geom_xpos[g, 2]) for g in range(m.ngeom))


def test_rotation_is_accounted_for(scene):
    """A tilted box dips lower than an axis-aligned one of the same size."""
    m, d = scene
    half = 0.05
    g = m.geom("box_tilt").id
    extent = float(d.geom_xpos[g, 2]) - _geom_lowest_z(m, d, g)
    # 45 deg about y: extent = half*(|cos|+|sin|) = half*sqrt(2)
    assert extent == pytest.approx(half * np.sqrt(2.0), abs=1e-6)
    assert extent > half  # an axis-aligned computation would under-read
