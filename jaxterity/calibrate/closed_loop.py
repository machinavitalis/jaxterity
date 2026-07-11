# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

r"""Closed-loop active sysid for non-backdrivable geared servos.

WHY THIS EXISTS.  The :class:`~jaxterity.calibrate.Excitation` "coast probe"
identifies a joint from its *free* dynamics: kick it and watch it coast.
That regime does not exist on a geared Feetech STS3215. A real torque-off coast
decelerates to rest in ~170 ms (hardware finding) — nothing like the
frictionless MJX observer — and gravity-loaded joints cannot torque-off safely at
all. So the free-coast observer is the *wrong identifiability source* for these
non-backdrivable servos.

THE FIX — excite THROUGH the servo's own inner loop, identify by EQUATION ERROR.
The unit is driven by a band-limited position **multisine** in the servo's native
position mode (torque held on the whole time); the on-device capture records the
response (position, the inner loop's PWM-duty ``load``, bus voltage). This module
runs the **inverse-dynamics equation-error** fit on a workstation:

    M(q) q'' + C(q,q') q' + g(q)  +  b q'  +  tau_c sign(q')   =   K_t * load + c0
    \_________ tau_rb (MJX) _________/   \___ joint friction __/    \_ servo torque _/

i.e. the rigid-body joint torque (one MJX ``mj_inverse`` per sample, over the full
measured arm pose) plus identified viscous/Coulomb friction must equal the servo's
delivered torque. The delivered torque is read from the duty-cycle ``load`` proxy
(current sense is not exposed) scaled by ``K_t`` and a constant offset ``c0`` (the
holding-load deadband / gravity baseline). Because the equation is **linear** in
``[K_t, c0, b, tau_c]`` given ``tau_rb``, the fit is a single least-squares solve —
robust to the sensor noise that swamps a forward q''-residual fit (double-
differencing quantized encoder ticks). Derivatives are low-pass filtered below the
multisine band; ``tau_rb`` carries the inertial term, so reflected inertia is not a
separate (collinear) regressor.

WHAT IS IDENTIFIABLE (declared honestly, per CONTEXT invariant 4).  Viscous damping
``b`` is the dominant identifiable channel (the duty load tracks velocity strongly).
Coulomb friction ``tau_c`` has a near-zero gradient and is reported but typically
small; it is *held* (left at the prior) when its fit is not positive. ``K_t``/``c0``
are measurement-side calibrations of the duty proxy, not plant parameters, and are
returned in :attr:`SysidResult.residuals`, not bound onto the robot. The fit quality
is reported as the held-out torque-prediction error reduction versus the incoming
(URDF/datasheet) model — the number the validation check uses.

SUPPORTED ROUTE.  The Pi has the servo bus but not JAX/MJX, so the capture is a
split handoff: ``scripts/lekiwi_commanded_capture.py`` runs on the LeKiwi and writes
a telemetry JSON; :func:`load_commanded_capture` + :func:`identify_closed_loop` run
where the simulator already works. This generalizes the ReplayLink seam
(telemetry-file handoff, not a fit handoff) to commanded excitation + equation error.
``SerialBusLink.run_excitation`` stays a stub: an in-process drive would need the
full stack on-device, which the Pi does not carry.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np

from ..errors import CalibrationError
from ..robot import Robot
from ..sysid import SysidResult
from .active import dof_joint_names

#: Servo-id -> joint name for the SO-101 / LeKiwi arm (wheels 7..9 excluded).
LEKIWI_ARM_JOINT_BY_ID = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}

#: Default Coulomb-friction smoothing velocity (rad/s) for ``tanh(q'/eps)``.
_DEFAULT_EPS = 0.05
#: Default derivative low-pass cutoff (Hz) — above the multisine band, below noise.
_DEFAULT_FC = 6.0
#: Fraction of the capture used to fit; the remainder is the held-out test split.
_DEFAULT_TRAIN_FRAC = 0.7


@dataclasses.dataclass(frozen=True)
class CommandedCapture:
    """Parsed commanded-excitation telemetry from one LeKiwi/SO-101 arm joint.

    Built by :func:`load_commanded_capture` from the on-device capture JSON. Holds
    the excited joint's response in SI units plus the full arm pose held during the
    capture (the other joints are static, but every joint's gravity torque and the
    excited joint's effective inertia depend on it).

    Fields
    ------
    joint : str
        The excited joint's name (a DOF of the robot).
    servo_id : int
        Its Feetech bus id.
    t : np.ndarray
        Sample times, s, shape ``(n,)``.
    q : np.ndarray
        Excited-joint position, rad, shape ``(n,)`` (calibration-signed).
    load_proxy : np.ndarray
        Servo torque proxy = ``(load/1000) * bus_voltage`` (duty-fraction times
        volts), shape ``(n,)``; scaled to N·m by the fitted ``K_t``.
    pose_rad : np.ndarray
        Full arm pose in DOF order, rad, shape ``(n_dof,)`` — the held
        configuration of the non-excited joints (excited joint's value is
        overwritten per-sample by :func:`identify_closed_loop`).
    dof_names : tuple[str, ...]
        Robot DOF order for ``pose_rad``.
    source_hash : str
        BLAKE2b hash of the raw capture bytes (telemetry provenance).
    meta : dict
        The capture's excitation metadata (kind, amplitude, multisine freqs, …).
    """

    joint: str
    servo_id: int
    t: np.ndarray
    q: np.ndarray
    load_proxy: np.ndarray
    pose_rad: np.ndarray
    dof_names: tuple[str, ...]
    source_hash: str
    meta: dict

    @property
    def n_samples(self) -> int:
        """Number of telemetry samples."""
        return int(self.t.shape[0])


def _ticks_to_rad(ticks: np.ndarray, cal: dict) -> np.ndarray:
    """Convert raw encoder ticks to radians via a per-joint calibration row."""
    return (
        cal["sign"]
        * (np.asarray(ticks, float) - cal["offset_ticks"])
        / cal["ticks_per_rad"]
    )


def load_commanded_capture(
    source: str | dict,
    robot: Robot,
    *,
    calibration: str | dict | None = None,
) -> CommandedCapture:
    """Load an on-device commanded-excitation capture into SI units.

    Reads the JSON written by ``scripts/lekiwi_commanded_capture.py`` (schema
    ``"lekiwi-commanded-capture-v1"``), converts encoder ticks to radians, builds
    the full arm pose held during the capture, and forms the duty-cycle torque
    proxy. The result feeds :func:`identify_closed_loop`.

    Args
    ----
    source : str or dict
        Path to the capture JSON, or the already-parsed dict.
    robot : Robot
        The robot whose DOF order defines ``pose_rad`` (e.g.
        ``zoo.calibrated("so100")`` for the LeKiwi arm).
    calibration : str or dict or None
        The full per-joint calibration (path or parsed) used to convert the
        *other* joints' held pose ticks. Defaults to the single-joint calibration
        embedded in the capture (sufficient when no pose is recorded).

    Returns
    -------
    CommandedCapture
        Parsed telemetry in SI units with the full pose.

    Raises
    ------
    CalibrationError
        If the schema is unrecognized, the capture aborted, or the excited joint
        is not a DOF of ``robot``.
    """
    raw_bytes: bytes
    if isinstance(source, dict):
        cap = source
        raw_bytes = json.dumps(cap, sort_keys=True).encode()
    else:
        with open(source, "rb") as f:
            raw_bytes = f.read()
        cap = json.loads(raw_bytes)

    if cap.get("schema") != "lekiwi-commanded-capture-v1":
        raise CalibrationError(
            f"load_commanded_capture: unrecognized schema {cap.get('schema')!r}; "
            "expected 'lekiwi-commanded-capture-v1' (see "
            "scripts/lekiwi_commanded_capture.py)."
        )
    if cap.get("aborted"):
        raise CalibrationError(
            f"load_commanded_capture: capture of {cap.get('joint')!r} aborted "
            "(a safety guard tripped); re-run with a smaller amplitude or a "
            "center further from the joint limit."
        )

    joint = cap["joint"]
    dof_names = dof_joint_names(robot)
    if joint not in dof_names:
        raise CalibrationError(
            f"load_commanded_capture: excited joint {joint!r} is not a DOF of the "
            f"robot (DOFs: {dof_names}). Pass the matching robot (the LeKiwi arm "
            "is zoo.calibrated('so100'))."
        )

    cal = cap["calibration"]
    t = np.asarray(cap["t"], float)
    q = _ticks_to_rad(np.asarray(cap["pos_ticks"]), cal)
    load = np.asarray(cap["load_raw"], float)
    volt = np.asarray(cap["volt_dV"], float) / 10.0
    load_proxy = (load / 1000.0) * volt

    # Full arm pose held during the capture (other joints static). Convert each
    # recorded servo position with its own calibration row when available.
    pose = np.zeros(len(dof_names))
    cal_by_id: dict[int, dict] = {}
    if calibration is not None:
        cdoc = (
            calibration
            if isinstance(calibration, dict)
            else json.load(open(calibration))
        )
        cal_by_id = {int(c["servo_id"]): c for c in cdoc["calibrations"]}
    idx = {n: i for i, n in enumerate(dof_names)}
    for sid_str, ticks in cap.get("pose_ticks", {}).items():
        sid = int(sid_str)
        name = LEKIWI_ARM_JOINT_BY_ID.get(sid)
        if name is None or name not in idx:
            continue
        row = cal_by_id.get(sid, cal if sid == cap["servo_id"] else None)
        if row is None:
            continue
        pose[idx[name]] = _ticks_to_rad(ticks, row)

    return CommandedCapture(
        joint=joint,
        servo_id=int(cap["servo_id"]),
        t=t,
        q=q,
        load_proxy=load_proxy,
        pose_rad=pose,
        dof_names=dof_names,
        source_hash=hashlib.blake2b(raw_bytes, digest_size=16).hexdigest(),
        meta=dict(cap.get("excitation", {})),
    )


def _lowpass(x: np.ndarray, t: np.ndarray, fc: float) -> np.ndarray:
    """Zero-phase FFT low-pass (deterministic), cutoff ``fc`` Hz."""
    n = len(x)
    dt = float(np.mean(np.diff(t)))
    freqs = np.fft.rfftfreq(n, dt)
    spec = np.fft.rfft(x - x.mean())
    spec[freqs > fc] = 0.0
    return np.fft.irfft(spec, n) + x.mean()


def _filtered_derivatives(
    capture: CommandedCapture, fc: float
) -> tuple[np.ndarray, np.ndarray]:
    """Low-pass the excited joint's position, then differentiate to q'/q''."""
    qf = _lowpass(capture.q, capture.t, fc)
    qd = np.gradient(qf, capture.t)
    qdd = np.gradient(qd, capture.t)
    return qd, qdd


def _rigid_body_torque(
    robot: Robot, capture: CommandedCapture, qd: np.ndarray, qdd: np.ndarray
) -> np.ndarray:
    """Per-sample rigid-body joint torque on the excited DOF via MJX inverse dynamics.

    One ``mujoco.mj_inverse`` per sample over the full measured pose (only the
    excited joint moves). Returns ``tau_rb`` (M q'' + C q' + g) on the excited DOF
    with the model's *current* friction/damping — the incoming-model baseline.
    """
    import mujoco

    mj = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    data = mujoco.MjData(mj)
    j = int(mj.joint(capture.joint).dofadr[0])
    dofadr = {n: int(mj.joint(n).dofadr[0]) for n in capture.dof_names}
    qpos0 = np.zeros(mj.nq)
    for n, val in zip(capture.dof_names, capture.pose_rad):
        qpos0[dofadr[n]] = val
    tau_rb = np.empty(len(qd))
    for i in range(len(qd)):
        data.qpos[:] = qpos0
        data.qpos[j] = capture.q[i]
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qvel[j] = qd[i]
        data.qacc[j] = qdd[i]
        mujoco.mj_inverse(mj, data)
        tau_rb[i] = data.qfrc_inverse[j]
    return tau_rb


def identify_closed_loop(
    robot: Robot,
    capture: CommandedCapture,
    *,
    fc: float = _DEFAULT_FC,
    eps: float = _DEFAULT_EPS,
    train_frac: float = _DEFAULT_TRAIN_FRAC,
) -> SysidResult:
    """Identify a joint's viscous/Coulomb friction from a commanded-excitation capture.

    Runs the inverse-dynamics equation-error fit (module docstring): low-pass the
    excited joint's position to get ``q'``/``q''``, compute the rigid-body torque
    ``tau_rb`` per sample with MJX inverse dynamics over the full pose, and solve the
    **linear** least squares

        ``tau_rb = K_t * load_proxy + c0 - b * q' - tau_c * tanh(q'/eps)``

    for ``[K_t, c0, b, tau_c]`` on the first ``train_frac`` of the capture. Viscous
    damping ``b`` and Coulomb ``tau_c`` are bound onto the robot (clamped to >= 0;
    a non-positive raw fit is *held* at the prior and flagged in
    :attr:`SysidResult.residuals`). ``K_t``/``c0`` are measurement calibrations and
    are reported, not bound. Fit quality is the held-out torque-prediction error
    reduction versus the incoming model — the validation metric.

    Args
    ----
    robot : Robot
        The robot to identify (the incoming/URDF model; its current
        friction/damping is the baseline the held-out reduction is measured against).
    capture : CommandedCapture
        Parsed commanded-excitation telemetry (:func:`load_commanded_capture`).
    fc : float
        Derivative low-pass cutoff, Hz (above the multisine band).
    eps : float
        Coulomb smoothing velocity, rad/s.
    train_frac : float
        Fraction of samples used to fit; the rest is the held-out test split.

    Returns
    -------
    SysidResult
        ``fitted`` = identified ``{joint}.damping`` / ``{joint}.friction`` (only the
        channels that came out positive); ``covariance`` from the LS standard errors
        (readable via :meth:`SysidResult.interval`); ``residuals`` carrying
        the held-out RMS baseline/fitted, the reduction fraction, ``K_t``, ``c0``,
        and which channels were held.

    Raises
    ------
    CalibrationError
        If the capture is too short to split, or the regression is rank-deficient.
    """
    n = capture.n_samples
    if n < 20:
        raise CalibrationError(
            f"identify_closed_loop: capture has only {n} samples; need >= 20 to "
            "fit and hold out a test split."
        )
    qd, qdd = _filtered_derivatives(capture, fc)
    tau_rb = _rigid_body_torque(robot, capture, qd, qdd)
    load_proxy = capture.load_proxy

    ntr = max(10, int(train_frac * n))
    tr = slice(0, ntr)
    te = slice(ntr, n)
    if n - ntr < 5:
        raise CalibrationError(
            "identify_closed_loop: not enough held-out samples; lower train_frac "
            "or capture a longer excitation."
        )

    # Design matrix: tau_rb = [load, 1, -qd, -tanh] @ [K_t, c0, b, tau_c].
    design = np.column_stack([load_proxy, np.ones(n), -qd, -np.tanh(qd / eps)])
    coef, *_ = np.linalg.lstsq(design[tr], tau_rb[tr], rcond=None)
    if not np.all(np.isfinite(coef)):
        raise CalibrationError(
            "identify_closed_loop: the regression did not converge (non-finite "
            "coefficients); the capture may have too little motion."
        )
    K_t, c0, b, tau_c = (float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3]))

    # LS covariance: sigma^2 (X'X)^-1 on the training residual.
    resid_tr = tau_rb[tr] - design[tr] @ coef
    dof = max(1, ntr - design.shape[1])
    sigma2 = float(resid_tr @ resid_tr) / dof
    xtx_inv = np.linalg.pinv(design[tr].T @ design[tr])
    var = sigma2 * np.diag(xtx_inv)  # [K_t, c0, b, tau_c]

    # Held-out fit-quality vs the incoming model (baseline: load+offset only, i.e.
    # the robot's current friction — no per-unit viscous/Coulomb identified).
    base = np.column_stack([load_proxy, np.ones(n)])
    base_coef, *_ = np.linalg.lstsq(base[tr], tau_rb[tr], rcond=None)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
    rms_base = rms((tau_rb - base @ base_coef)[te])
    rms_fit = rms((tau_rb - design @ coef)[te])
    reduction = 1.0 - rms_fit / rms_base if rms_base > 0 else 0.0

    # Bind only physically-valid (positive) friction channels; hold the rest.
    fitted: dict[str, Any] = {}
    covariance: dict[str, float] = {}
    held: list[str] = []
    for name, value, v, key in (
        ("damping", b, var[2], f"{capture.joint}.damping"),
        ("friction", tau_c, var[3], f"{capture.joint}.friction"),
    ):
        if value > 0.0:
            fitted[key] = np.asarray(value)
            covariance[key] = float(v)
        else:
            held.append(name)

    residuals = {
        "held_out_rms_baseline": rms_base,
        "held_out_rms_fitted": rms_fit,
        "held_out_reduction": reduction,
        "train_rms": rms(resid_tr),
        "K_t": K_t,
        "load_offset": c0,
        "viscous_damping": b,
        "coulomb_friction": tau_c,
        "n_held_channels": float(len(held)),
    }
    return SysidResult(
        fitted=fitted,
        covariance=covariance or None,
        residuals=residuals,
        telemetry_hash=capture.source_hash,
        recipe_name="active.closed_loop",
        recipe_version="1.0",
    )


def cross_trajectory_validation(
    robot: Robot,
    fit: SysidResult,
    validation: CommandedCapture,
    *,
    fc: float = _DEFAULT_FC,
    eps: float = _DEFAULT_EPS,
) -> dict[str, float]:
    """Test whether an identified fit generalizes to a *different* real trajectory.

    The held-out reduction inside :func:`identify_closed_loop` is a tail split of
    the *same* excitation. This is the stronger check the validation asks for: take
    the **plant** parameters identified on one capture (viscous/Coulomb friction —
    the leaves bound onto the robot) and ask whether they predict the joint torque
    of a *separately captured* trajectory better than the frictionless model.

    The measurement-side calibration (``K_t``, ``c0`` — the duty-proxy→torque scale
    and offset) is **not** transferred: it is a per-capture sensor calibration (it
    shifts with the operating point on a nonlinear geared servo, especially under
    gravity), so it is refit on the validation capture for *both* the baseline and
    the friction-augmented model. Only the friction (the plant truth) is carried
    over. This isolates "did we identify the plant?" from "is the duty proxy linear?".

    Args
    ----
    robot : Robot
        The incoming (frictionless/URDF) model — the baseline.
    fit : SysidResult
        A fit from :func:`identify_closed_loop` (its ``{joint}.damping`` /
        ``{joint}.friction`` are the transferred plant parameters).
    validation : CommandedCapture
        A separately captured trajectory of the *same* joint.
    fc, eps : float
        Derivative low-pass cutoff (Hz) and Coulomb smoothing velocity (rad/s);
        match the fit's settings.

    Returns
    -------
    dict[str, float]
        ``rms_baseline`` / ``rms_fitted`` (N·m, on the validation capture) and the
        ``reduction`` fraction (> 0 ⇒ the plant fit generalizes to new motion).
    """
    joint = validation.joint
    b = float(fit.fitted.get(f"{joint}.damping", 0.0))
    tau_c = float(fit.fitted.get(f"{joint}.friction", 0.0))
    qd, qdd = _filtered_derivatives(validation, fc)
    tau_rb = _rigid_body_torque(robot, validation, qd, qdd)
    friction = b * qd + tau_c * np.tanh(qd / eps)  # transferred plant torque

    design = np.column_stack([validation.load_proxy, np.ones(len(qd))])
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
    # Frictionless baseline: best K_t, c0 on the validation capture.
    base = design @ np.linalg.lstsq(design, tau_rb, rcond=None)[0]
    # Friction-augmented: refit K_t, c0 with the transferred friction folded in.
    fitp = design @ np.linalg.lstsq(design, tau_rb + friction, rcond=None)[0] - friction
    rms_base, rms_fit = rms(tau_rb - base), rms(tau_rb - fitp)
    return {
        "rms_baseline": rms_base,
        "rms_fitted": rms_fit,
        "reduction": 1.0 - rms_fit / rms_base if rms_base > 0 else 0.0,
    }


__all__ = [
    "CommandedCapture",
    "load_commanded_capture",
    "identify_closed_loop",
    "cross_trajectory_validation",
    "LEKIWI_ARM_JOINT_BY_ID",
]
