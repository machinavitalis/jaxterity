# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin handles across algorithm versions: v3 is current, v2 still verifies."""

from __future__ import annotations

import json

import blake3
import jax.numpy as jnp
import pytest

pytest.importorskip("mujoco")

from jaxterity import zoo  # noqa: E402
from jaxterity.diagnostics.canonical import canonical_dumps, coerce  # noqa: E402
from jaxterity.twin import Twin  # noqa: E402

# ``Twin(zoo.load("cartpole"), serial="UNIT-7", ...)`` as hashed by the
# version-2 implementation before version 3 landed.
V2_TWIN = "e422e27812374452a3af26eeac6c5afaf382fcd0a164fb5fe194243f6d24c380"


@pytest.fixture(scope="module")
def twin() -> Twin:
    return Twin(
        zoo.load("cartpole"),
        serial="UNIT-7",
        C_th=jnp.full(2, 15.0),
        R_th=jnp.full(2, 4.0),
        k_pwr=jnp.full(2, 6.0),
        K_t_scale=jnp.ones(2),
    )


def test_version_2_twin_handle_still_verifies(twin):
    assert twin.attestation_handle != V2_TWIN
    assert twin.attestation_handle_for(2) == V2_TWIN
    assert twin.verify_attestation_handle(V2_TWIN) == 2
    assert twin.verify_attestation_handle(twin.attestation_handle) == 3
    assert twin.verify_attestation_handle("0" * 64) is None


def test_twin_attestation_document_is_plain_json(twin):
    document = twin.attestation_document()
    assert document["version"] == 3
    assert document["serial"] == "UNIT-7"
    assert document["servo_names"] == list(twin.servo_names)
    assert document["has_battery"] is False
    assert document["k_i"] is None
    assert document["robot_handle"] == twin.robot.attestation_handle
    assert set(document["params"]) == set(twin.identifiable_params())
    json.dumps(coerce(document))
    encoded = canonical_dumps(document)
    assert b"object at 0x" not in encoded
    assert twin.attestation_handle == blake3.blake3(encoded).hexdigest()
