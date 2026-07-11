# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Rerun adapter — graceful absence + signature presence."""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.diagnostics import ParityReport
from jaxterity.diagnostics.rerun_adapter import _require_rerun, log_parity_report
from jaxterity.errors import JaxterityError


def test_require_rerun_raises_when_missing():
    """When rerun-sdk isn't installed, the adapter raises with install instructions."""
    try:
        import rerun  # noqa: F401
    except ImportError:
        with pytest.raises(JaxterityError, match="pip install"):
            _require_rerun()
    else:
        pytest.skip("rerun-sdk is installed; can't test the missing-dep path here")


def test_log_parity_report_with_rerun_installed():
    """When rerun is present, the adapter accepts well-formed inputs and runs."""
    pytest.importorskip("rerun")
    rng = np.random.default_rng(0)
    sim = {"q": rng.standard_normal((10, 2)), "qd": rng.standard_normal((10, 2))}
    real = {"q": sim["q"] + 0.01, "qd": sim["qd"] + 0.005}
    report = ParityReport(
        quantities=("q", "qd"),
        rms={"q": 0.01, "qd": 0.005},
        peak={"q": 0.02, "qd": 0.01},
        n_samples=10,
    )
    log_parity_report(sim, real, report, namespace="unit-test")  # raises on failure
