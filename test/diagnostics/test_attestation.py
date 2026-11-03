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
identical across machines of the same architecture; parameters are hashed as
little-endian float64 arrays so endianness does not matter either.

Version 3 digests the canonical bytes of one JSON document; version 2 joined
the parts with separator bytes. Handles recorded under version 2 are pinned
below and must keep verifying through the kept v2 code path.

(The actuator-block-version sensitivity in the spec is not yet exercised:
actuators are not bound into the robot handle — that lands with the actuator
calibration recipe planned for a later release.)
"""

from __future__ import annotations

import json
import subprocess
import sys

import blake3
import numpy as np
import pytest

from jaxterity.diagnostics import attestation
from jaxterity.diagnostics.attestation import (
    compute_handle,
    handle_document,
    verify_handle,
)
from jaxterity.diagnostics.canonical import canonical_dumps, coerce

# Handles computed by the version-2 implementation before version 3 landed.
# The v2 path must keep reproducing them byte for byte (ADR-008: no silent
# collision, and a recorded handle keeps verifying).
V2_UNIT = "c605bbfb5058e5e394b198d8439b6aa2109cd92ff6ab64e56cef30215dd3b1ff"
V2_UNIT_KWARGS = dict(
    model_description="body 0 0 0",
    parameters={"a.mass": [1.5], "b.d": [0.3]},
    extra={"v": "1"},
)
V2_UNIT_NO_EXTRA = "421d6c5b2c3e1772136d77eb2a809dd4eab28c164279c7fd6bcd3bac45699db5"
V2_ROBOT = "b0adeb17001aebb5f71566ba8daef2fa8f9e1d2f94baee534d597720ca54afbc"
V2_ROBOT_WITH_PROVENANCE = (
    "400f1f5b7f89f6f4f86ffea5ce03528a2a4961a0cd61941368bbd25a69684633"
)

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


# --- version 3 / version 2 ----------------------------------------------------


def test_version_3_is_current_and_differs_from_version_2():
    assert attestation.ALGORITHM_VERSION == 3
    assert attestation.SUPPORTED_VERSIONS == (3, 2)
    current = compute_handle(**V2_UNIT_KWARGS)
    assert current == compute_handle(**V2_UNIT_KWARGS, version=3)
    assert current != V2_UNIT
    assert len(current) == 64 and all(c in "0123456789abcdef" for c in current)


def test_version_2_path_reproduces_recorded_handles():
    assert compute_handle(**V2_UNIT_KWARGS, version=2) == V2_UNIT
    assert (
        compute_handle(model_description="m", parameters={"a.mass": [1.5]}, version=2)
        == V2_UNIT_NO_EXTRA
    )


def test_verify_handle_names_the_matching_version():
    assert verify_handle(V2_UNIT, **V2_UNIT_KWARGS) == 2
    assert verify_handle(compute_handle(**V2_UNIT_KWARGS), **V2_UNIT_KWARGS) == 3
    assert verify_handle("0" * 64, **V2_UNIT_KWARGS) is None
    # A v3 extra that v2 stringified differently is passed as legacy_extra.
    extra = {"provenance": [["r", "1", "tel"]]}
    legacy = {"provenance": repr((("r", "1", "tel"),))}
    old = compute_handle(model_description="m", parameters={}, extra=legacy, version=2)
    assert (
        verify_handle(
            old, model_description="m", parameters={}, extra=extra, legacy_extra=legacy
        )
        == 2
    )


def test_handle_document_is_canonical_json_without_repr():
    """The v3 handle is BLAKE3 over the canonical bytes of a printable document."""
    document = handle_document(
        model_description="<a>\n   <b/> </a>\n",
        parameters={"a.mass": [1.5]},
        extra={"provenance": [["r", "1", "tel"]], "calibration_state": "calibrated"},
    )
    assert document["version"] == 3
    assert document["mjcf"] == "<a> <b/> </a>"
    assert document["params"]["a.mass"].dtype == np.dtype("<f8")
    json.dumps(coerce(document))  # plain JSON once arrays are in their form
    expected = (
        b'{"calibration_state":"calibrated","mjcf":"<a> <b/> </a>",'
        b'"params":{"a.mass":{"$nd":{"dtype":"<f8","shape":[1],"values":["1.5"]}}},'
        b'"provenance":[["r","1","tel"]],"version":3}'
    )
    assert canonical_dumps(document) == expected
    assert (
        compute_handle(
            model_description="<a>\n   <b/> </a>\n",
            parameters={"a.mass": [1.5]},
            extra={
                "provenance": [["r", "1", "tel"]],
                "calibration_state": "calibrated",
            },
        )
        == blake3.blake3(expected).hexdigest()
    )


def test_handle_document_rejects_reserved_extra_keys():
    with pytest.raises(ValueError, match="reserved"):
        handle_document(model_description="m", parameters={}, extra={"mjcf": "x"})
    with pytest.raises(ValueError, match="no handle document"):
        handle_document(model_description="m", parameters={}, version=2)


def test_robot_version_2_handles_still_verify(tmp_path):
    robot = _load(tmp_path, _URDF)
    assert robot.attestation_handle != V2_ROBOT
    assert robot.attestation_handle_for(2) == V2_ROBOT
    assert robot.verify_attestation_handle(V2_ROBOT) == 2
    assert robot.verify_attestation_handle(robot.attestation_handle) == 3
    assert robot.verify_attestation_handle("0" * 64) is None
    with_prov = robot.with_provenance(("r", "1", "tel"))
    assert with_prov.attestation_handle_for(2) == V2_ROBOT_WITH_PROVENANCE
    assert with_prov.verify_attestation_handle(V2_ROBOT_WITH_PROVENANCE) == 2


def test_robot_attestation_document_is_plain_json(tmp_path):
    robot = _load(tmp_path, _URDF).with_provenance(("r", "1", "tel"))
    document = robot.attestation_document()
    assert document["calibration_state"] == "calibrated"
    assert document["provenance"] == [["r", "1", "tel"]]
    assert set(document["params"]) == set(robot.parameters())
    encoded = canonical_dumps(document)
    assert b"object at 0x" not in encoded and b"(('r'" not in encoded
    assert robot.attestation_handle == blake3.blake3(encoded).hexdigest()
