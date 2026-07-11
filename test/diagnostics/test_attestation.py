# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Attestation-handle stability tests (ADR-008, CONTEXT.md invariant 5).

The handle is the provenance key Jaxility consumes, so it must be:
- **sensitive** to anything that changes deployment behaviour (model bytes, bound
  parameters, calibration provenance), and
- **stable** against everything that does not (cosmetic model edits, parameter
  byte order / endianness, the host process).

Cross-*device* reproducibility of *fitted* parameter values (CPU vs GPU last-ULP)
is out of scope here — it is OQ-4. The handle itself is a pure function of its
inputs (verified across a fresh process), so given identical float values it is
identical across machines of the same architecture; the v2 algorithm hashes
parameters in fixed little-endian order so endianness does not matter either.

(The actuator-block-version sensitivity in the spec is not yet exercised:
actuators are not bound into the robot handle — that lands with the actuator
calibration recipe planned for a later release.)
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from jaxterity.diagnostics import attestation
from jaxterity.diagnostics.attestation import compute_handle

# --- unit level: compute_handle (no MuJoCo needed) -------------------------


def test_handle_is_deterministic_64_hex():
    h1 = compute_handle(model_description="m", parameters={"a.mass": [1.5]})
    h2 = compute_handle(model_description="m", parameters={"a.mass": [1.5]})
    assert h1 == h2
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)


def test_handle_canonicalizes_whitespace():
    """Cosmetic whitespace in the model string must not change the handle."""
    a = compute_handle(
        model_description="<body name='x'>  <joint/></body>", parameters={}
    )
    b = compute_handle(
        model_description="<body name='x'>\n   <joint/></body>\n", parameters={}
    )
    assert a == b


def test_handle_independent_of_parameter_byte_order():
    """Endianness of the input arrays must not change the handle (v2)."""
    little = compute_handle(
        model_description="m", parameters={"a.mass": np.array([1.5, 2.5], dtype="<f8")}
    )
    big = compute_handle(
        model_description="m", parameters={"a.mass": np.array([1.5, 2.5], dtype=">f8")}
    )
    assert little == big


def test_handle_sensitive_to_model_params_and_extra():
    base = compute_handle(
        model_description="m", parameters={"a.mass": [1.5]}, extra={"recipe": "1"}
    )
    assert (
        compute_handle(  # model bytes
            model_description="m2", parameters={"a.mass": [1.5]}, extra={"recipe": "1"}
        )
        != base
    )
    assert (
        compute_handle(  # parameter value
            model_description="m", parameters={"a.mass": [1.6]}, extra={"recipe": "1"}
        )
        != base
    )
    assert (
        compute_handle(  # parameter key
            model_description="m",
            parameters={"a.damping": [1.5]},
            extra={"recipe": "1"},
        )
        != base
    )
    assert (
        compute_handle(  # extra (recipe/telemetry/version) value
            model_description="m", parameters={"a.mass": [1.5]}, extra={"recipe": "2"}
        )
        != base
    )


def test_handle_folds_algorithm_version(monkeypatch):
    """A change to the algorithm version must change every handle (no collision)."""
    base = compute_handle(model_description="m", parameters={"a.mass": [1.5]})
    monkeypatch.setattr(
        attestation, "ALGORITHM_VERSION", attestation.ALGORITHM_VERSION + 1
    )
    assert compute_handle(model_description="m", parameters={"a.mass": [1.5]}) != base


def test_handle_reproducible_across_processes():
    """A fresh process yields the same handle (pure function — cross-machine proxy)."""
    kwargs = dict(
        model_description="body 0 0 0",
        parameters={"a.mass": [1.5], "b.d": [0.3]},
        extra={"v": "1"},
    )
    code = (
        "from jaxterity.diagnostics.attestation import compute_handle;"
        f"print(compute_handle(**{kwargs!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == compute_handle(**kwargs)


# --- robot level: through the Robot pipeline --------------------------------
# (The handle path — to_mjcf + compute_handle — is pure string/hash work, and
# from_urdf is stdlib XML, so none of this needs MuJoCo.)

_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base">
    <inertial><mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="bob">
    <inertial><origin xyz="0 0 -1"/><mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="pivot" type="continuous">
    <parent link="base"/><child link="bob"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
</robot>
"""


def _load(tmp_path, text, name="arm.urdf"):
    from jaxterity.robot import from_urdf

    path = tmp_path / name
    path.write_text(text)
    return from_urdf(path)


def test_robot_handle_sensitive_to_model_and_params(tmp_path):
    from jaxterity.robot import Robot

    robot = _load(tmp_path, _URDF)
    h0 = robot.attestation_handle
    # bound parameter change
    assert robot.with_parameters({"bob.mass": 2.0}).attestation_handle != h0
    # model-structure change (gravity is part of the lowered MJCF)
    spun = Robot(
        links=robot.links,
        joints=robot.joints,
        name=robot.name,
        gravity=3.71,  # Mars
    )
    assert spun.attestation_handle != h0
    # provenance / calibration state
    assert robot.with_provenance(("r", "1", "tel")).attestation_handle != h0
    assert (
        robot.with_provenance(("r", "1", "tel")).attestation_handle
        != robot.with_provenance(("r", "2", "tel")).attestation_handle  # recipe version
    )


def test_robot_handle_sensitive_to_armature(tmp_path):
    """Joint armature joins parameters() and the lowered MJCF, so it
    must move the handle (operating principle 6: attestation chain at every
    change)."""
    robot = _load(tmp_path, _URDF)
    assert "pivot.armature" in robot.parameters()
    h0 = robot.attestation_handle
    assert robot.with_parameters({"pivot.armature": 0.007}).attestation_handle != h0


def test_robot_handle_stable_under_irrelevant_urdf_edits(tmp_path):
    """Whitespace and comments in the source URDF must not change the handle."""
    h0 = _load(tmp_path, _URDF, "a.urdf").attestation_handle
    noisy = _URDF.replace(
        '<robot name="arm">', '<robot name="arm">\n  <!-- a comment -->\n\n'
    ).replace("<joint", "   <joint")
    assert _load(tmp_path, noisy, "b.urdf").attestation_handle == h0
