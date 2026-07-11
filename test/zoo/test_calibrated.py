# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""zoo.calibrated() — load + apply per-robot calibration artifacts."""

from __future__ import annotations

import pytest

from jaxterity import zoo
from jaxterity.errors import ZooError


@pytest.mark.parametrize("name", ["cartpole", "so100"])
def test_calibrated_loads_and_applies_provenance(name):
    """A calibrated zoo robot carries its recipe + telemetry hash on the chain."""
    r = zoo.calibrated(name)
    flat = [entry for tup in r.provenance for entry in tup]
    # The recipe name comes from one of the recipe families: the inertial/
    # full_body/contact recipes ("sysid.*") or the active recipes
    # ("active.*", e.g. so100's real closed-loop friction calibration).
    assert any(("sysid." in p or "active." in p) for p in flat), r.provenance
    assert any("1.0" == p for p in flat), r.provenance


def test_calibrated_unknown_robot_raises():
    with pytest.raises(ZooError, match="Unknown zoo robot"):
        zoo.calibrated("nonexistent")


def test_calibrated_robot_handle_differs_from_uncalibrated():
    """Applying the calibration changes the attestation handle (ADR-008)."""
    uncal = zoo.load("so100").attestation_handle
    cal = zoo.calibrated("so100").attestation_handle
    assert uncal != cal


# --- recalibrate runner --------------------------------------------------


def test_recalibrate_dry_run_returns_payload_with_synthetic_false(tmp_path):
    """recalibrate(write=False) doesn't touch disk and returns a real-fit payload."""
    pytest.importorskip("mujoco")
    import jax.numpy as jnp
    import jaxonomy
    import numpy as np
    from jaxonomy import SimulatorOptions

    from jaxterity.data import JointTelemetry

    # Use a tiny synthetic trajectory just to exercise the runner.
    robot = zoo.load("cartpole")
    diagram, system = robot.to_diagram(
        q0=jnp.zeros(2),
        qd0=jnp.asarray([0.0, 0.5]),
        return_system=True,
    )
    sid = system.system_id
    t = np.linspace(0.0, 0.2, 6, dtype=np.float64)
    qs, qds = [], []
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        enable_autodiff=False,
        max_minor_step_size=0.005,
    )
    for ti in t:
        res = jaxonomy.simulate(
            diagram,
            diagram.create_context(),
            (0.0, float(ti) + 1e-9),
            options=opts,
        )
        state = np.asarray(res.context[sid].continuous_state)
        nq = state.size // 2
        qs.append(state[:nq])
        qds.append(state[nq:])
    tel = JointTelemetry(
        time=t,
        sample_rate_hz=1.0 / (t[1] - t[0]),
        joint_names=("slider", "pole_hinge"),
        q=np.asarray(qs),
        qd=np.asarray(qds),
        tau=np.zeros_like(np.asarray(qs)),
    )
    new_robot, payload = zoo.recalibrate("cartpole", tel, write=False, note="unit test")
    assert payload["synthetic"] is False
    assert payload["recipe_name"] == "sysid.inertial"
    assert "fitted" in payload and len(payload["fitted"]) > 0
    assert payload["note"] == "unit test"


def test_recalibrate_unknown_robot_raises():
    from jaxterity.errors import ZooError

    with pytest.raises(ZooError, match="Unknown zoo robot"):
        zoo.recalibrate("nonexistent", None, write=False)


def test_recalibrate_unknown_recipe_raises():
    from jaxterity.errors import ZooError

    with pytest.raises(ZooError, match="unknown recipe"):
        zoo.recalibrate("cartpole", None, recipe="invalid", write=False)
