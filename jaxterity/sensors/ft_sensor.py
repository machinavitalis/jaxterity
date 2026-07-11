# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Six-axis force/torque sensor with cross-axis coupling.

``FTSensor`` models a 6-axis F/T transducer (the kind ATI's mini40 or Bota's
SensONE ship) attached between a manipulator's end-effector and its load.
Inputs are the true 3-force and 3-torque applied to the sensor; outputs are
the measured 6-vector with bias and a 6×6 cross-axis coupling matrix applied.

Model — stateless::

    [F, τ]_read = C @ [F, τ]_true + bias

where ``C`` is the 6×6 calibration matrix (identity for a perfect sensor;
off-diagonal entries capture cross-axis coupling) and ``bias`` is a 6-vector.

Identifiable parameters (42 scalars in 3 array leaves):

* ``bias`` (6-vec, [N×3, N·m×3]): per-channel offset.
* ``coupling`` (6×6, dimensionless): cross-axis coupling matrix. Identity at
  default; the off-diagonal entries are the leakage of each true-axis load
  into other axes' readings. Typical magnitudes are ~1% for a well-calibrated
  ATI head.
* ``noise_density`` (6-vec, [N/√Hz×3, N·m/√Hz×3]): per-channel white-noise
  density. Sysid metadata only — same convention as IMU; not in forward sim.

Identification — :attr:`FTSensor.sysid_protocol` documents a known-weight
sweep at multiple poses to identify bias (zero-load offset), then a series of
single-axis force / torque inputs to fill in the coupling matrix.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import FTSensorTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor

_SYSID_PROTOCOL = SysidProtocol(
    name="ft-known-load-sweep",
    excitation="static-pose",
    required_columns=("force", "torque"),
    duration_s=120.0,
    sample_rate_hz=500.0,
    setup_notes=(
        "Bias: hold zero load through a sweep of orientations (so gravity acts "
        "on each axis); the per-axis average of the load-free reading is the "
        "bias. Coupling: apply a known load along each principal axis one at "
        "a time and record the full 6-vector response; the columns of C "
        "(after bias subtraction) are the reading per unit input. 500 Hz "
        "captures the strain-gauge bandwidth typical of an analog FT head."
    ),
)


class FTSensor(Sensor):
    """6-axis force/torque sensor with bias and a 6×6 coupling matrix.

    Args
    ----
    bias : array-like length 6 or None
        Per-channel offset, [N×3, N·m×3]. Default zeros.
    coupling : array-like (6, 6) or None
        Cross-axis coupling matrix. Default identity (perfect sensor).
    noise_density : array-like length 6 or None
        Per-channel white-noise density. Default zeros (metadata only).
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        bias=None,
        coupling=None,
        noise_density=None,
        name: str = "ft_sensor",
    ):
        super().__init__(name=name)
        self.declare_dynamic_parameter(
            "bias",
            jnp.zeros(6) if bias is None else jnp.asarray(bias, dtype=float),
        )
        self.declare_dynamic_parameter(
            "coupling",
            jnp.eye(6) if coupling is None else jnp.asarray(coupling, dtype=float),
        )
        self.declare_dynamic_parameter(
            "noise_density",
            jnp.zeros(6)
            if noise_density is None
            else jnp.asarray(noise_density, dtype=float),
        )
        self.declare_input_port(name="force_true")
        self.declare_input_port(name="torque_true")
        self.declare_output_port(self._force_out, name="force_reading")
        self.declare_output_port(self._torque_out, name="torque_reading")

    @property
    def bias(self):
        return self.dynamic_parameters["bias"].get()

    @property
    def coupling(self):
        return self.dynamic_parameters["coupling"].get()

    @property
    def noise_density(self):
        return self.dynamic_parameters["noise_density"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "ft_sensor") -> "FTSensor":
        """Return an instance with bias=0, coupling=I, noise_density=0."""
        return cls(name=name)

    def _wrench(self, *, force_true, torque_true, params):
        w_true = jnp.concatenate([force_true, torque_true])
        return params["coupling"] @ w_true + params["bias"]

    def _force_out(self, time, state, *inputs, **params):
        return self._wrench(force_true=inputs[0], torque_true=inputs[1], params=params)[
            :3
        ]

    def _torque_out(self, time, state, *inputs, **params):
        return self._wrench(force_true=inputs[0], torque_true=inputs[1], params=params)[
            3:
        ]

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("bias", "coupling", "noise_density"))

    def validate(self, telemetry: FTSensorTelemetry) -> ValidationReport:
        """Noise-density check on a static F/T trace (per-axis de-meaned RMS).

        **Scope:** this tests the
        **noise density only**. Unlike :meth:`IMU.validate` — where the static
        truth is pinned (gyro=0, accel magnitude=g) — an F/T sensor's static
        reading is ``bias + (unknown) ambient load``. The bias is therefore
        confounded with whatever constant load is on the sensor and is **not**
        identifiable from a single unknown-load static trace; the per-axis mean
        is removed for exactly that reason. **Coupling** likewise needs
        *known-load* telemetry (the ``ft-known-load-sweep`` protocol), which the
        :class:`~jaxterity.data.schema.FTSensorTelemetry` schema does not yet
        carry — so coupling is out of scope here too. The earlier warning
        claimed "bias / coupling / noise_density may not match"; that overstated
        what the de-meaned residual can see, and is corrected below. To validate
        bias, record an *unloaded* trace and compare the per-axis mean to
        ``bias`` directly.
        """
        from ..testing import tolerances as tol

        force = np.asarray(telemetry.force, dtype=np.float64)  # (n, 3)
        torque = np.asarray(telemetry.torque, dtype=np.float64)
        nd = np.asarray(self.noise_density, dtype=np.float64)  # (6,)
        fs = telemetry.sample_rate_hz
        f_res = force - force.mean(axis=0)[None, :]
        t_res = torque - torque.mean(axis=0)[None, :]
        rms = np.sqrt(np.mean(np.concatenate([f_res, t_res], axis=1) ** 2, axis=0))
        expected = nd * np.sqrt(fs / 2.0)
        floor = tol.atol("sensor_reading", "rk4", "float64") * 1e3
        bound = np.maximum(3.0 * expected, floor)
        report = ValidationReport()
        if np.any(rms > bound):
            report.warnings.append(
                f"FT noise RMS={rms.tolist()} exceeds bound {bound.tolist()} "
                "(per-axis) — noise_density may not match this trace. (Bias and "
                "coupling are not testable from an unknown-load static trace; see "
                "the method docstring.)"
            )
        return report


__all__ = ["FTSensor"]
