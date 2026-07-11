# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""3-axis magnetometer with bias / noise / misalignment.

Same model shape as :class:`~jaxterity.sensors.imu.IMU`'s accelerometer half —
the physics is identical: a 3-axis transducer in a fixed-mount frame with
constant bias, white noise, and a misalignment rotation from body to sensor::

    mag_read = R(misalignment) @ mag_body + bias

Three array leaves (bias, noise_density, misalignment), 9 identifiable scalars.
Noise is sysid metadata, not in the forward sim (the rationale from IMU
applies — see :class:`IMU`'s module docstring).

Why a separate class from `IMU`? Real IMU silicon ships in 6-DOF (accel + gyro)
and 9-DOF (+ magnetometer) variants — they're sold separately. Conflating them
forces every IMU user to choose between three identifiable parameters they don't
have or a partial calibration; the cleaner separation is one class per
transducer family. A future composed `MARG(IMU + Magnetometer)` block is the
natural sensor-fusion entry point.

The forward model and the validate residual reuse the IMU's
:func:`~jaxterity.sensors.imu._rodrigues` exp map. Identification needs a
known global-magnetic-field reference (Earth's field at the recording site,
typically ~50 µT at 60° dip in the Northern hemisphere); the recipe assumes
a static-pose trace so the per-axis de-mean removes the unknown ambient field.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import MagnetometerTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor
from .imu import _rodrigues

_SYSID_PROTOCOL = SysidProtocol(
    name="mag-static-pose",
    excitation="static-pose",
    required_columns=("mag",),
    duration_s=60.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Place the magnetometer through a sweep of static poses (a sphere fit "
        "needs ~12 well-spread orientations). Bias from the per-axis mean of "
        "the still recording; misalignment from aligning the recovered field "
        "vector to the known reference (Earth field at the recording site). "
        "noise_density from PSD of a 30 s still recording at one pose."
    ),
)


class Magnetometer(Sensor):
    """3-axis magnetometer with bias, noise density, and axis-angle misalignment.

    Args
    ----
    bias : array-like length 3 or None
        Per-axis offset, T. Default zeros.
    noise_density : array-like length 3 or None
        Per-axis white-noise density, T/√Hz. Default zeros (sysid metadata).
    misalignment : array-like length 3 or None
        Axis-angle rotation from body to sensor frame, rad. Default identity.
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        bias=None,
        noise_density=None,
        misalignment=None,
        name: str = "magnetometer",
    ):
        super().__init__(name=name)
        for n, x in (
            ("bias", bias),
            ("noise_density", noise_density),
            ("misalignment", misalignment),
        ):
            val = jnp.zeros(3) if x is None else jnp.asarray(x, dtype=float)
            self.declare_dynamic_parameter(n, val)
        self.declare_input_port(name="mag_body")
        self.declare_output_port(self._mag_out, name="mag_reading")

    @property
    def bias(self):
        return self.dynamic_parameters["bias"].get()

    @property
    def noise_density(self):
        return self.dynamic_parameters["noise_density"].get()

    @property
    def misalignment(self):
        return self.dynamic_parameters["misalignment"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "magnetometer") -> "Magnetometer":
        return cls(name=name)

    def _mag_out(self, time, state, *inputs, **params):
        return _rodrigues(params["misalignment"]) @ inputs[0] + params["bias"]

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("bias", "noise_density", "misalignment"))

    def validate(self, telemetry: MagnetometerTelemetry) -> ValidationReport:
        """De-meaned per-axis residual against 3·noise_density·√(fs/2)."""
        from ..testing import tolerances as tol

        mag = np.asarray(telemetry.mag, dtype=np.float64)
        nd = np.asarray(self.noise_density, dtype=np.float64)
        fs = telemetry.sample_rate_hz
        res = mag - mag.mean(axis=0)[None, :]
        rms = np.sqrt(np.mean(res * res, axis=0))
        expected = nd * np.sqrt(fs / 2.0)
        # Earth-field magnetometers operate in the 1e-8..1e-4 T range — use the
        # bare ``sensor_reading`` atol without the ×1e3 IMU/FT floor; magnetic
        # signals are far smaller than acceleration / wrench signals.
        floor = tol.atol("sensor_reading", "rk4", "float64")
        bound = np.maximum(3.0 * expected, floor)
        report = ValidationReport()
        if np.any(rms > bound):
            report.warnings.append(
                f"magnetometer residual RMS={rms.tolist()} T exceeds bound "
                f"{bound.tolist()} — bias / misalignment / noise_density may not match."
            )
        return report


__all__ = ["Magnetometer"]
