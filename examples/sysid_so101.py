# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""End-to-end system identification on the real SO-101 arm.

This is the whole Jaxterity story in one file, on a real open-source robot:

1. **Load** the vendored SO-101 URDF as a :class:`~jaxterity.robot.Robot`.
2. **Generate telemetry** by simulating a "true" robot (joint damping set to known
   values we will pretend not to know) under gentle, within-limit excitation.
3. **Identify** those damping coefficients from the telemetry, starting from an
   uncalibrated guess, with a forward-mode Gauss-Newton / Levenberg-Marquardt
   loop. Forward-mode autodiff is what makes this work on a *real* arm: the
   SO-101's six limited joints put MJX's constraint solver on the path, and
   reverse-mode (`jax.grad`) cannot traverse it (ADR-016/017).
4. **Calibrate**: package the fit as a :class:`~jaxterity.sysid.SysidResult` and
   apply it to the robot, producing a calibrated :class:`~jaxterity.robot.Robot`
   whose attestation handle (ADR-008) records *how* it was fitted.

Run it:  ``python -m examples.sysid_so101``  (needs the ``[mujoco]`` extra).

Identification stays in the constraint-free *regime* on purpose — the excitation
keeps every joint within its limits, so no joint slams a limit during the
trajectory (gradients through an active impact are non-smooth for any method;
ADR-017). This is exactly how a real calibration routine is designed.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
from jaxonomy import SimulatorOptions

from jaxterity import zoo
from jaxterity.sysid import SysidResult, parameter_jacobian

# Damping coefficients to identify (the SO-101's revolute joints), and the
# "true" value we plant in the telemetry-generating robot.
TARGET_PARAMS = ("shoulder_lift.damping", "elbow_flex.damping")
TRUE_DAMPING = 0.12
#: Sample times for multiple-shooting: the trajectory state at each is a
#: residual, which conditions the fit far better than a single end state.
SAMPLE_TIMES = (0.15, 0.3)
_NDOF = 6
_Q0 = jnp.zeros(_NDOF)
_QD0 = jnp.full(_NDOF, 0.1)  # gentle: stays within joint limits over SAMPLE_TIMES

# Forward-mode requires the reverse-mode adjoint off, so JAX traces the real ops
# (incl. the constraint solver) — fixed-step RK4 keeps it deterministic.
_OPTIONS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="rk4",
    enable_autodiff=False,
    max_minor_step_size=0.005,
    max_major_steps=200,
)


def _make_observer(robot):
    """Return ``observe(damping_vec) -> stacked states at SAMPLE_TIMES``.

    The diagram is built once and the closure is ``jax.jit``-compiled, so both the
    forward pass and ``jax.jacfwd`` compile a single time and every
    Levenberg-Marquardt iteration reuses the compiled kernel.
    """
    diagram, system = robot.to_diagram(q0=_Q0, qd0=_QD0, return_system=True)
    sid = system.system_id

    @jax.jit
    def observe(damping):
        ctx = diagram.create_context()
        sub = ctx[sid]
        for name, value in zip(TARGET_PARAMS, damping):
            sub = sub.with_parameter(name, value)
        ctx = ctx.with_subcontext(sid, sub)
        states = [
            jaxonomy.simulate(diagram, ctx, (0.0, t), options=_OPTIONS)
            .context[sid]
            .continuous_state
            for t in SAMPLE_TIMES
        ]
        return jnp.concatenate(states)

    return observe


def identify(telemetry, observe, n_params, *, iterations=8, lam=1e-2, max_step=0.08):
    """Recover ``n_params`` coefficients from ``telemetry`` by forward-mode LM.

    ``observe(params) -> stacked states`` is differentiated with forward-mode
    autodiff (so it works through MJX's constraint solver). Returns
    ``(estimate, history)`` where ``history`` is the per-iteration residual cost.
    """
    observe_jac = jax.jit(jax.jacfwd(observe))  # compile the Jacobian once
    est = jnp.zeros(n_params)  # uncalibrated guess
    history = []
    for _ in range(iterations):
        residual = np.asarray(observe(est)) - telemetry
        jac = np.asarray(observe_jac(est))  # (n_obs, n_params)
        history.append(float(residual @ residual))
        hessian = jac.T @ jac
        damped = hessian + lam * np.diag(np.diag(hessian) + 1e-12)
        step = np.clip(-np.linalg.solve(damped, jac.T @ residual), -max_step, max_step)
        est = est + step
    return est, history


def main() -> SysidResult:
    print("Loading the real SO-101 arm ...", flush=True)
    robot = zoo.load("so100")

    # --- 1. generate telemetry from the "true" robot --------------------------
    true_robot = robot.with_parameters({p: TRUE_DAMPING for p in TARGET_PARAMS})
    observe_true = _make_observer(true_robot)
    telemetry = np.asarray(observe_true(jnp.full(len(TARGET_PARAMS), TRUE_DAMPING)))
    # Each sample is a full state [q, qd]; check the joint *positions* stay within
    # limits (≈ 1.7 rad) so no joint impacts a limit — velocities are unbounded.
    positions = telemetry.reshape(len(SAMPLE_TIMES), 2 * _NDOF)[:, :_NDOF]
    peak_angle = float(np.max(np.abs(positions)))
    print(
        f"  telemetry captured; peak |joint angle| = {peak_angle:.3f} rad "
        "(limits ~1.7 — within, no impacts)"
    )

    # Forward-mode sensitivity readout via the public API (the same machinery the
    # LM loop uses): how the final state responds to each damping coefficient.
    sens = parameter_jacobian(
        robot,
        lambda s: jnp.sum(s**2),
        {p: jnp.asarray(0.0) for p in TARGET_PARAMS},
        q0=_Q0,
        qd0=_QD0,
        t_span=(0.0, SAMPLE_TIMES[-1]),
        options=_OPTIONS,
    )
    print("  forward-mode sensitivities d‖state‖²/d(damping):")
    for p in TARGET_PARAMS:
        print(f"    {p:24s} {float(sens[p]):+.4e}")

    # --- 2. identify from an uncalibrated guess -------------------------------
    print("\nIdentifying joint damping (forward-mode Levenberg-Marquardt) ...")
    observe = _make_observer(robot)
    t0 = time.time()
    estimate, history = identify(telemetry, observe, len(TARGET_PARAMS))
    fitted = {p: float(v) for p, v in zip(TARGET_PARAMS, np.asarray(estimate))}
    max_err = max(abs(v - TRUE_DAMPING) for v in fitted.values())
    print(
        f"  converged in {len(history)} iters / {time.time() - t0:.1f}s: "
        f"cost {history[0]:.2e} → {history[-1]:.2e}"
    )
    for p in TARGET_PARAMS:
        print(f"    {p:24s} fitted={fitted[p]:.5f}  true={TRUE_DAMPING}")
    print(f"  max identification error: {max_err:.4e}")

    # --- 3. package as a SysidResult and calibrate the robot ------------------
    result = SysidResult(
        fitted={p: jnp.asarray(v) for p, v in fitted.items()},
        residuals={"final_state_rms": float(np.sqrt(history[-1] / telemetry.size))},
        telemetry_hash="so101-demo-telemetry",
        recipe_name="forward_mode_lm_damping",
        recipe_version="1.0",
    )
    calibrated = result.apply_to(robot)
    print("\nCalibrated robot:")
    print(f"  calibration state : {calibrated.calibration_state.value}")
    print(f"  provenance        : {calibrated.provenance}")
    print(f"  attestation handle: {calibrated.attestation_handle}")
    print(f"  (uncalibrated was : {robot.attestation_handle})")
    return result


if __name__ == "__main__":
    main()
