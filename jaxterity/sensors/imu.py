# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""IMU sensor — accelerometer + gyroscope with bias, noise, misalignment.

``IMU`` models a strapdown inertial measurement unit attached to a body frame.
Inputs are the body-frame linear acceleration and angular velocity. Outputs are
the IMU-frame readings with bias, misalignment, and (optionally; see "noise"
below) noise applied.

Model — stateless::

    R           = exp_map(misalignment_axis_angle)               (3×3 rotation)
    accel_read  = R · accel_body + bias_accel
    gyro_read   = R · gyro_body  + bias_gyro

Identifiable parameters (15 scalars in 5 array leaves):

* ``bias_accel`` (3-vec, m/s²): constant offset on each accelerometer axis.
* ``bias_gyro`` (3-vec, rad/s): constant offset on each gyro axis.
* ``noise_density_accel`` (3-vec, m/s²/√Hz): per-axis white-noise density.
* ``noise_density_gyro`` (3-vec, rad/s/√Hz): per-axis white-noise density.
* ``misalignment`` (3-vec, rad): axis-angle rotation from the body frame to
  the IMU frame. Identity at ``[0, 0, 0]``.

**Noise is sysid metadata, not part of the forward model.** The forward sim is
deterministic; ``noise_density_*`` exist so that a sysid recipe can
predict the bound on the residual under matched parameters and so that a
domain-randomization layer can apply matched-density noise externally.
Bringing noise into the forward sim would couple the IMU to an RNG key and
break ``LeafSystem``'s deterministic-port contract; the right place for noise
injection is a downstream ``WhiteNoise`` block.

The misalignment is parameterized as a 3-vector axis-angle (Rodrigues vector)
to keep the leaf JAX-array-shaped and ``vmap``-clean. The exponential map is
applied inline (small-angle stable at the origin via the standard Taylor
expansion).

Identification — :attr:`IMU.sysid_protocol` documents a turntable / static-pose
sweep that's enough to identify bias (static-pose averages) and misalignment
(against gravity), plus a still recording for noise density (PSD of static
readings). The fitter is the ``sysid.actuator`` recipe (which handles
sensors symmetrically — the procedural shape is the same).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import IMUTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor

#: Standard gravity (m/s²) — the magnitude a static accelerometer reads after
#: bias removal, used by :meth:`IMU.validate` to test the accel bias.
_STANDARD_GRAVITY = 9.80665
#: Tolerance (m/s²) on the bias-removed static accel magnitude vs gravity. Wide
#: enough to absorb local-gravity variation (~9.78–9.83) and the mean's noise,
#: tight enough to catch a bias error a fraction of g in size.
_ACCEL_BIAS_TOL = 0.25

_SYSID_PROTOCOL = SysidProtocol(
    name="imu-turntable",
    excitation="static-pose",
    required_columns=("accel", "gyro"),
    duration_s=60.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Place the IMU through a sweep of static poses (12 axis-aligned + 4 "
        "off-axis is the FRC-style minimum). Bias is identified from the "
        "static-pose averages; misalignment is identified by aligning gravity. "
        "Append a 30 s still recording at one pose for noise-density "
        "estimation via Welch PSD. The fitter is closed-form for bias "
        "(simple averages), linear-least-squares for misalignment (Wahba's "
        "problem), Welch-PSD for noise density."
    ),
)


def _rodrigues(axis_angle):
    """Axis-angle → 3×3 rotation matrix (Rodrigues' formula, small-angle stable)."""
    theta_sq = jnp.sum(axis_angle * axis_angle)
    theta = jnp.sqrt(jnp.maximum(theta_sq, 1e-30))
    # Small-angle Taylor: for very small θ, sin(θ)/θ ≈ 1, (1−cos(θ))/θ² ≈ 1/2.
    sin_term = jnp.where(theta < 1e-6, 1.0, jnp.sin(theta) / theta)
    cos_term = jnp.where(theta < 1e-6, 0.5, (1.0 - jnp.cos(theta)) / theta_sq)
    K = jnp.array(
        [
            [0.0, -axis_angle[2], axis_angle[1]],
            [axis_angle[2], 0.0, -axis_angle[0]],
            [-axis_angle[1], axis_angle[0], 0.0],
        ]
    )
    return jnp.eye(3) + sin_term * K + cos_term * (K @ K)


class IMU(Sensor):
    """6-axis IMU (3 accel + 3 gyro) with bias, noise, and misalignment.

    See the module docstring for the model, the parameter set, the noise
    convention, and the misalignment parameterization (axis-angle, Rodrigues).

    Args
    ----
    bias_accel, bias_gyro : array-like length 3 or None
        Per-axis biases. Default zeros.
    noise_density_accel, noise_density_gyro : array-like length 3 or None
        Per-axis white-noise densities. Default zeros (metadata only — not in
        forward model; see module docstring).
    misalignment : array-like length 3 or None
        Rodrigues axis-angle rotation, rad. Default zeros (identity).
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        bias_accel=None,
        bias_gyro=None,
        noise_density_accel=None,
        noise_density_gyro=None,
        misalignment=None,
        name: str = "imu",
    ):
        super().__init__(name=name)
        for n, x in (
            ("bias_accel", bias_accel),
            ("bias_gyro", bias_gyro),
            ("noise_density_accel", noise_density_accel),
            ("noise_density_gyro", noise_density_gyro),
            ("misalignment", misalignment),
        ):
            val = jnp.zeros(3) if x is None else jnp.asarray(x, dtype=float)
            self.declare_dynamic_parameter(n, val)
        self.declare_input_port(name="accel_body")
        self.declare_input_port(name="gyro_body")
        self.declare_output_port(self._accel_out, name="accel_reading")
        self.declare_output_port(self._gyro_out, name="gyro_reading")

    @property
    def bias_accel(self):
        return self.dynamic_parameters["bias_accel"].get()

    @property
    def bias_gyro(self):
        return self.dynamic_parameters["bias_gyro"].get()

    @property
    def noise_density_accel(self):
        return self.dynamic_parameters["noise_density_accel"].get()

    @property
    def noise_density_gyro(self):
        return self.dynamic_parameters["noise_density_gyro"].get()

    @property
    def misalignment(self):
        return self.dynamic_parameters["misalignment"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "imu") -> "IMU":
        """Return an instance with identifiability-neutral defaults: all zeros."""
        return cls(name=name)

    def _accel_out(self, time, state, *inputs, **params):
        R = _rodrigues(params["misalignment"])
        return R @ inputs[0] + params["bias_accel"]

    def _gyro_out(self, time, state, *inputs, **params):
        R = _rodrigues(params["misalignment"])
        return R @ inputs[1] + params["bias_gyro"]

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(
            (
                "bias_accel",
                "bias_gyro",
                "noise_density_accel",
                "noise_density_gyro",
                "misalignment",
            )
        )

    def validate(self, telemetry: IMUTelemetry) -> ValidationReport:
        """Residual report against a *static* IMUTelemetry trace.

        Assumes the IMU was static for the trace — the standard turntable
        posture. Under that assumption the truth is physically pinned: the gyro
        reads zero (so its per-axis *mean* is the gyro bias) and the
        bias-removed accel mean must have the magnitude of gravity (whatever
        the unknown pose orientation). The check therefore tests three things:

        1. **gyro bias** — ``mean(gyro) − bias_gyro`` must be ~0;
        2. **accel bias** — ``‖mean(accel) − bias_accel‖`` must equal ``g``;
        3. **noise density** — the de-meaned per-axis RMS must sit under the
           declared noise-density bound.

        **Not checked: the misalignment rotation.** A pure rotation is
        norm-preserving and maps one static gravity vector to another valid
        one, so misalignment is *not* identifiable from a single static pose —
        it needs the multi-pose sweep (see :attr:`sysid_protocol`). The
        previous implementation de-meaned every axis, discarding the bias
        signal in (1)/(2), so a wrong bias passed validation; that is the bug
        this method now fixes.
        """
        from ..testing import tolerances as tol

        accel = np.asarray(telemetry.accel, dtype=np.float64)  # (n, 3)
        gyro = np.asarray(telemetry.gyro, dtype=np.float64)
        n = accel.shape[0]
        b_a = np.asarray(self.bias_accel, dtype=np.float64)
        b_g = np.asarray(self.bias_gyro, dtype=np.float64)
        nd_a = np.asarray(self.noise_density_accel, dtype=np.float64)
        nd_g = np.asarray(self.noise_density_gyro, dtype=np.float64)
        fs = telemetry.sample_rate_hz
        floor = tol.atol("sensor_reading", "rk4", "float64") * 1e3
        report = ValidationReport()

        # (1) Gyro bias: static gyro truth is exactly zero, so the per-axis mean
        # reading IS the bias. Threshold scales with the observed noise's
        # standard-error-of-the-mean (independent of the declared density).
        mean_resid_g = gyro.mean(axis=0) - b_g
        sem_g = gyro.std(axis=0) / np.sqrt(max(n, 1))
        gyro_bias_bound = np.maximum(5.0 * sem_g, floor)
        if np.any(np.abs(mean_resid_g) > gyro_bias_bound):
            report.warnings.append(
                f"gyro mean residual {mean_resid_g.tolist()} rad/s exceeds "
                f"{gyro_bias_bound.tolist()} — bias_gyro does not match this "
                "static trace (static gyro truth is zero, so the mean reading "
                "is the bias)."
            )

        # (2) Accel bias: bias-removed static accel must carry gravity's
        # magnitude, whatever the (unknown) pose orientation.
        g_meas = float(np.linalg.norm(accel.mean(axis=0) - b_a))
        if abs(g_meas - _STANDARD_GRAVITY) > _ACCEL_BIAS_TOL:
            report.warnings.append(
                f"bias-removed static accel magnitude {g_meas:.4f} m/s² differs "
                f"from gravity {_STANDARD_GRAVITY:.4f} by more than "
                f"{_ACCEL_BIAS_TOL} — bias_accel does not match this trace. (A "
                "pure misalignment rotation is norm-preserving and is not "
                "identifiable from a single static pose; use the multi-pose "
                "turntable protocol to identify misalignment.)"
            )

        # (3) Noise density: de-mean to isolate the noise, bound by the model.
        res_a = accel - accel.mean(axis=0)[None, :]
        res_g = gyro - gyro.mean(axis=0)[None, :]
        expected_a = nd_a * np.sqrt(fs / 2.0)
        expected_g = nd_g * np.sqrt(fs / 2.0)
        rms_a = np.sqrt(np.mean(res_a * res_a, axis=0))
        rms_g = np.sqrt(np.mean(res_g * res_g, axis=0))
        a_bound = np.maximum(expected_a * 3.0, floor)
        g_bound = np.maximum(expected_g * 3.0, floor)
        if np.any(rms_a > a_bound):
            report.warnings.append(
                f"accelerometer noise RMS={rms_a.tolist()} m/s² exceeds "
                f"3 · noise_density · √(fs/2) = {a_bound.tolist()} — "
                "noise_density_accel may not match this trace."
            )
        if np.any(rms_g > g_bound):
            report.warnings.append(
                f"gyro noise RMS={rms_g.tolist()} rad/s exceeds "
                f"3 · noise_density · √(fs/2) = {g_bound.tolist()} — "
                "noise_density_gyro may not match this trace."
            )
        return report


__all__ = ["IMU"]
