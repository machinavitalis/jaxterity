# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Windowed online re-fit of declared drift parameters.

The slow recalibration loop: every so often (each simulated/real "hour"),
take a window of recent joint telemetry and re-fit the unit's **declared
drift parameters** — joint Coulomb friction / viscous damping (robot leaves)
and the per-servo actuation scale ``K_t_scale`` (a :class:`~jaxterity.twin.Twin`
leaf) — so the per-unit self-model tracks the aging plant instead of going
stale. Each accepted re-fit appends provenance and produces a **new
attestation handle**: the rolling chain of handles *is* the "model kept
current, with proof" record.

Method — **equation error**, not simulation error: for each interior sample
the residual is the *torque prediction error*

``r_t = M_diag ⊙ (qacc_pred(q_t, qd_t, K_t_scale · tau_t; θ) − qacc_meas_t)``

with ``qacc_meas`` from a central difference of the measured velocities,
``qacc_pred`` one MJX forward-dynamics evaluation, and ``M_diag`` the
mass-matrix diagonal at the window's mean configuration — a fixed per-DOF
weight that converts the acceleration error to torque units (N·m). Weighting
matters physically and numerically: an unweighted acceleration residual is
dominated by near-massless distal joints (wrists, ankle roll), where the
central-difference noise is huge in rad/s² but negligible in N·m. The
reported RMS is the "predicted minus measured torque" stream a residual
monitor consumes. No integration through time means no error accumulation,
per-sample parallelism (``vmap``), and a window cost linear in its length.
The fit is a bounded Gauss-Newton / Levenberg-Marquardt loop with
**forward-mode** Jacobians (reverse-mode cannot traverse MJX's constraint
solver — ADR-016/017; friction loss lives exactly there), projected onto the
schema's bounds after each step.

Scope split: this recipe fits the *dynamics* channels of the drift schema.
Thermal drift (``offbody.R_th`` …) needs temperature telemetry and has its
own recipe — :func:`jaxterity.calibrate.fit_servo_thermal` /
:func:`jaxterity.calibrate.calibrate_twin_thermal`. Passing a thermal
path here raises with that pointer.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..data.schema import JointTelemetry
from ..errors import JaxterityError
from ..sysid.result import SysidResult
from ..twin import DriftParameter, Twin

RECIPE_NAME = "sysid.windowed_drift"
RECIPE_VERSION = "1.0"

#: Robot-leaf joint fields this recipe can fit (runtime-threaded in MJX).
_ROBOT_FIELDS = ("friction", "damping")
_FIELD_TO_MJX = {"friction": "dof_frictionloss", "damping": "dof_damping"}


class WindowedFitResult(eqx.Module):
    """The outcome of one windowed drift re-fit.

    Fields
    ------
    twin : Twin
        The re-fit unit: robot leaves updated + calibration provenance
        (robot chain), ``K_t_scale`` updated + provenance (twin chain). Its
        ``attestation_handle`` differs from the input twin's — one handle per
        re-fit commit is the rolling attestation record.
    sysid_result : SysidResult
        The recipe-standard result (ADR-010) carrying **all** fitted values
        (robot paths and ``offbody.K_t_scale`` alike), residual statistics,
        and the telemetry hash.
    residual_rms_before, residual_rms_after : float
        Window RMS of the torque prediction residual (N·m) at the incoming
        and the fitted parameters — the "predicted minus measured" number a
        residual monitor consumes.
    """

    twin: Twin
    sysid_result: SysidResult
    residual_rms_before: float = eqx.field(static=True)
    residual_rms_after: float = eqx.field(static=True)


def _lazy_mjx():
    import mujoco
    from mujoco import mjx

    return mujoco, mjx


def _partition_parameters(
    parameters: tuple[DriftParameter, ...],
) -> tuple[tuple[DriftParameter, ...], DriftParameter | None]:
    """Split schema entries into robot-leaf scalars and the K_t_scale vector."""
    robot_entries: list[DriftParameter] = []
    kt_entry: DriftParameter | None = None
    for entry in parameters:
        head, _, rest = entry.path.partition(".")
        if head == "robot":
            comp, _, field = rest.rpartition(".")
            if field not in _ROBOT_FIELDS or not comp:
                raise JaxterityError(
                    f"refit_drift_parameters: {entry.path!r} is not a fittable "
                    f"robot leaf. This recipe fits joint {_ROBOT_FIELDS} "
                    "(runtime-threaded in MJX); inertial leaves go through the "
                    "structural sysid recipes."
                )
            robot_entries.append(entry)
        elif entry.path == "offbody.K_t_scale":
            kt_entry = entry
        elif head == "offbody":
            raise JaxterityError(
                f"refit_drift_parameters: {entry.path!r} is a thermal/electrical "
                "channel — it needs temperature/bus telemetry, not joint "
                "telemetry. Use jaxterity.calibrate.fit_servo_thermal / "
                "calibrate_twin_thermal for it."
            )
        else:
            raise JaxterityError(
                f"refit_drift_parameters: {entry.path!r} is not a "
                "Twin.identifiable_params path (expected 'robot.<joint>.<field>' "
                "or 'offbody.K_t_scale')."
            )
    return tuple(robot_entries), kt_entry


def _column_permutation(telemetry: JointTelemetry, servo_names: tuple[str, ...]):
    """Map telemetry columns onto the twin's DOF order (by joint name)."""
    if set(telemetry.joint_names) != set(servo_names):
        missing = sorted(set(servo_names) - set(telemetry.joint_names))
        extra = sorted(set(telemetry.joint_names) - set(servo_names))
        raise JaxterityError(
            f"refit_drift_parameters: telemetry joints do not match the twin's "
            f"servos (missing {missing}, unexpected {extra}). The window must "
            "cover every twin DOF, named by joint."
        )
    order = [telemetry.joint_names.index(name) for name in servo_names]
    return np.asarray(order, dtype=int)


def _mass_diagonal(mj_model: Any, q_mean: np.ndarray) -> np.ndarray:
    """Mass-matrix diagonal at a configuration — the fixed residual weights."""
    import mujoco

    data = mujoco.MjData(mj_model)
    data.qpos[:] = q_mean
    mujoco.mj_forward(mj_model, data)
    full = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, data, full)
    return np.diag(full).copy()


def _make_predictor(
    twin: Twin, robot_entries: tuple[DriftParameter, ...], fit_kt: bool
) -> tuple[Callable[[jnp.ndarray, Any, Any, Any], jnp.ndarray], Any]:
    """Build ``predict(theta, q, qd, tau) -> qacc`` (vmapped over samples).

    ``theta`` packs the robot scalars first, then (if ``fit_kt``) the
    per-servo actuation scale. The MJX model is built once; each evaluation
    threads the parameters in via ``model.replace`` (the same mechanism as
    ``Robot.to_diagram``'s runtime parameters) and runs one forward dynamics.
    Also returns the compiled ``mujoco.MjModel`` (for the residual weights).
    """
    mujoco, mjx = _lazy_mjx()
    mj_model = mujoco.MjModel.from_xml_string(twin.robot.to_mjcf())
    if int(mj_model.nv) != twin.n_servos:
        raise JaxterityError(
            f"refit_drift_parameters: the twin's robot has nv == {mj_model.nv} "
            f"but {twin.n_servos} servos. The windowed re-fit needs the "
            "fixed-base form whose DOFs map 1:1 onto servos (for the G1: "
            "zoo.load('unitree_g1'), not the floating-base form)."
        )
    mjx_model = mjx.put_model(mj_model)
    n_robot = len(robot_entries)
    targets = []
    for entry in robot_entries:
        comp, _, field = entry.path[len("robot.") :].rpartition(".")
        dof = int(np.asarray(mj_model.joint(comp).dofadr)[0])
        targets.append((_FIELD_TO_MJX[field], dof))

    def qacc_one(theta, q, qd, tau):
        model = mjx_model
        for k, (field, dof) in enumerate(targets):
            model = model.replace(
                **{field: getattr(model, field).at[dof].set(theta[k])}
            )
        delivered = tau * theta[n_robot:] if fit_kt else tau
        data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=delivered)
        return mjx.forward(model, data).qacc

    return jax.vmap(qacc_one, in_axes=(None, 0, 0, 0)), mj_model


def refit_drift_parameters(
    twin: Twin,
    telemetry: JointTelemetry,
    parameters: tuple[DriftParameter, ...],
    *,
    iterations: int = 8,
    levenberg: float = 1e-2,
    prior_sigma_fraction: float = 0.25,
) -> WindowedFitResult:
    """Re-fit the declared drift parameters from one telemetry window.

    See the module docstring for the method. The input twin is unchanged
    (ADR-004: sysid never mutates); the returned twin carries the fitted
    leaves, the appended provenance, and a new attestation handle.

    Args
    ----
    twin : Twin
        The unit's current self-model. Its robot must be the fixed-base form
        (DOFs map 1:1 onto servos).
    telemetry : JointTelemetry
        The window: joint positions (rad), velocities (rad/s), and
        **commanded** torques (N·m), named by joint. Needs ≥ 8 samples on a
        steady clock; columns may be in any order (aligned by name).
    parameters : tuple[DriftParameter, ...]
        The declared drift schema to fit — e.g.
        :func:`jaxterity.twin.g1_drift_parameters` minus its thermal entries.
        Robot joint friction/damping scalars and ``offbody.K_t_scale`` are
        fittable here; thermal paths raise (see the module docstring).
    iterations : int
        Levenberg-Marquardt iterations. Default 8.
    levenberg : float
        LM damping factor on the Gauss-Newton normal equations. Default 1e-2.
    prior_sigma_fraction : float
        Prior anchoring: each parameter carries a Gaussian prior at its
        *incoming* value with σ = this fraction of its schema bound range.
        Channels the window actually informs override the prior; channels
        the excitation never reaches (an idle wrist during a leg window)
        stay put instead of wandering to the bounds — wandering
        unidentified parameters would later read as drift anomalies.
        Default 0.25.

    Returns
    -------
    WindowedFitResult
        The re-fit twin + the recipe-standard :class:`SysidResult` + window
        residual RMS before/after.

    Raises
    ------
    JaxterityError
        On a non-fittable schema path, a joint-name mismatch, a robot whose
        DOFs don't map onto servos, or too short a window.
    """
    robot_entries, kt_entry = _partition_parameters(parameters)
    fit_kt = kt_entry is not None
    if not robot_entries and not fit_kt:
        raise JaxterityError(
            "refit_drift_parameters: the parameter set is empty after "
            "partitioning — nothing to fit."
        )
    if telemetry.n_samples < 8:
        raise JaxterityError(
            f"refit_drift_parameters: the window has {telemetry.n_samples} "
            "samples; need ≥ 8 (the residual uses interior samples with a "
            "central-difference acceleration)."
        )

    order = _column_permutation(telemetry, twin.servo_names)
    t = np.asarray(telemetry.time, dtype=np.float64)
    q = np.asarray(telemetry.q, dtype=np.float64)[:, order]
    qd = np.asarray(telemetry.qd, dtype=np.float64)[:, order]
    tau = np.asarray(telemetry.tau, dtype=np.float64)[:, order]
    # Central-difference acceleration on the interior samples.
    qacc_meas = (qd[2:] - qd[:-2]) / (t[2:] - t[:-2])[:, None]
    q_in, qd_in, tau_in = q[1:-1], qd[1:-1], tau[1:-1]

    # θ packing: robot scalars first, then the K_t_scale vector.
    n = twin.n_servos
    robot_params = twin.robot.parameters()
    theta0_parts = [
        np.atleast_1d(np.asarray(robot_params[e.path[len("robot.") :]], float))
        for e in robot_entries
    ]
    lo_parts = [np.full(1, e.lower) for e in robot_entries]
    hi_parts = [np.full(1, e.upper) for e in robot_entries]
    if kt_entry is not None:
        kt0 = np.ones(n) if twin.K_t_scale is None else np.asarray(twin.K_t_scale)
        theta0_parts.append(kt0)
        lo_parts.append(np.full(n, kt_entry.lower))
        hi_parts.append(np.full(n, kt_entry.upper))
    theta = np.concatenate(theta0_parts)
    lower = np.concatenate(lo_parts)
    upper = np.concatenate(hi_parts)

    predict, mj_model = _make_predictor(twin, robot_entries, fit_kt)
    # Torque-unit residual weights (see the module docstring).
    weights = jnp.asarray(_mass_diagonal(mj_model, q.mean(axis=0)))

    @jax.jit
    def residuals(theta_):
        pred = predict(
            theta_, jnp.asarray(q_in), jnp.asarray(qd_in), jnp.asarray(tau_in)
        )
        return ((pred - jnp.asarray(qacc_meas)) * weights[None, :]).ravel()

    residuals_jac = jax.jit(jax.jacfwd(residuals))

    def rms(r: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(r))))

    residual_before = np.asarray(residuals(jnp.asarray(theta)))
    cost_rms = rms(residual_before)
    residual_rms_before = cost_rms
    # Prior anchoring (see Args): Gaussian prior at the incoming values with
    # per-channel σ from the schema bound range.
    theta_prior = theta.copy()
    prior_precision = 1.0 / np.square(prior_sigma_fraction * (upper - lower))
    best_theta, best_rms = theta, cost_rms
    for _ in range(iterations):
        r = np.asarray(residuals(jnp.asarray(theta)))
        jac = np.asarray(residuals_jac(jnp.asarray(theta)))
        hessian = jac.T @ jac
        damped = (
            hessian
            + levenberg * np.diag(np.diag(hessian) + 1e-12)
            + np.diag(prior_precision)
        )
        gradient = jac.T @ r + prior_precision * (theta - theta_prior)
        step = -np.linalg.solve(damped, gradient)
        theta = np.clip(theta + step, lower, upper)  # projected GN (schema bounds)
        cost_rms = rms(np.asarray(residuals(jnp.asarray(theta))))
        if cost_rms < best_rms:
            best_theta, best_rms = theta, cost_rms
    theta, residual_rms_after = best_theta, best_rms

    # Unpack + apply: robot leaves via the recipe-standard SysidResult path,
    # the actuation vector via Twin.with_actuation — both append provenance.
    telemetry_hash = telemetry.hash().hex()
    fitted_robot = {
        e.path[len("robot.") :]: jnp.asarray(theta[k])
        for k, e in enumerate(robot_entries)
    }
    fitted_all: dict[str, Any] = {f"robot.{k}": v for k, v in fitted_robot.items()}
    sysid_result = SysidResult(
        fitted=fitted_robot,
        residuals={
            "torque_rms_before": residual_rms_before,
            "torque_rms_after": residual_rms_after,
        },
        telemetry_hash=telemetry_hash,
        recipe_name=RECIPE_NAME,
        recipe_version=RECIPE_VERSION,
    )
    new_robot = sysid_result.apply_to(twin.robot) if robot_entries else twin.robot
    new_kt = jnp.asarray(theta[len(robot_entries) :]) if fit_kt else twin.K_t_scale
    if fit_kt:
        fitted_all["offbody.K_t_scale"] = new_kt
    new_twin = Twin(
        new_robot,
        serial=twin.serial,
        C_th=twin.C_th,
        R_th=twin.R_th,
        k_pwr=twin.k_pwr,
        K_t_scale=new_kt,
        T_amb=twin.T_amb,
        V_oc=twin.V_oc,
        R_int=twin.R_int,
        k_i=twin.k_i,
        i_eps=twin.i_eps,
        servo_names=twin.servo_names,
        provenance=(
            *twin.provenance,
            (RECIPE_NAME, RECIPE_VERSION, telemetry_hash),
        ),
    )
    # The recipe-standard result reports everything fitted, robot + offbody.
    sysid_result = SysidResult(
        fitted=fitted_all,
        residuals=sysid_result.residuals,
        telemetry_hash=telemetry_hash,
        recipe_name=RECIPE_NAME,
        recipe_version=RECIPE_VERSION,
    )
    return WindowedFitResult(
        twin=new_twin,
        sysid_result=sysid_result,
        residual_rms_before=residual_rms_before,
        residual_rms_after=residual_rms_after,
    )


__all__ = ["refit_drift_parameters", "WindowedFitResult", "RECIPE_NAME"]
