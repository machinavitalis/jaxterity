# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Regenerate the independent-integrator sysid oracle traces.

The ``sysid.actuator`` LM fitters (Servo / BLDC / SeriesElastic) are otherwise
only exercised by round-trip tests that generate their truth telemetry with the
*same* fixed-step RK4 the fitter uses (see
``test/sysid/test_actuator_recipe.py``). A shared discretization or sign error
in that RK4 would pass such a test silently — the fit recovers the bug, not the
physics.

This script produces the *independent* half: it integrates a hand-written,
standalone transcription of each device's documented ODE with SciPy's implicit,
adaptive ``Radau`` integrator (5th-order, L-stable) at a far tighter tolerance
than the fitter's fixed 8-substep *explicit* RK4, then freezes the resulting
traces to ``.npz``. The recovery test
(``test_actuator_recipe_independent.py``) fits those frozen traces through the
recipe and asserts it recovers the known parameters. Because the data and the
fitter no longer share an integrator, a discretization/sign bug in the fitter's
rollout can no longer hide behind an identical bug in the data generator.

Scope of what this closes (and what it does not):

* Closes: self-consistency with the fitter's own discretization, and silent
  drift (the oracle is frozen on disk, not regenerated from the model each run).
* Does NOT close: a physics *model-form* error copied into both the fitter and
  this transcription. Catching that needs a trace whose ground truth is external
  to jaxterity entirely (a published third-party reference log, or real actuator
  telemetry). That remains a tracked follow-up.

SciPy is required to (re)generate; it is NOT required to run the test, which
only loads the frozen ``.npz``. Regenerate with::

    python test/sysid/data/independent/generate.py

Bump nothing automatically: if a device model form changes, regenerate here and
review the diff. The recipe version guards fitted-value provenance separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

_HERE = Path(__file__).resolve().parent

# Integrator settings — deliberately unlike the fitter's fixed 8-substep
# *explicit* RK4: an implicit adaptive Radau (5th-order, L-stable) at tight
# tolerance, so the trace is effectively the exact ODE solution at the sample
# grid and robust to the sharp smoothed-deadband / saturation curvature.
_METHOD = "Radau"
_RTOL = 1e-10
_ATOL = 1e-13
_V_EPS = 1e-3  # Coulomb sign-smoothing width; matches the device default.


def _integrate_zoh(rhs, y0, t_grid, u_samples):
    """Integrate ``y' = rhs(t, y, u)`` on ``t_grid`` with a ZOH input.

    ``u_samples[i]`` is held over ``[t_grid[i], t_grid[i+1])`` — the same
    zero-order-hold convention the fitter and the device ``validate`` use
    (setpoint applied on the leading edge of each interval). Integration is
    performed one interval at a time with the input held constant, so each
    ``solve_ivp`` call sees a smooth vector field with no in-flight input
    discontinuity; within the interval the implicit adaptive stepper picks its
    own substeps, independent of the fitter's fixed count.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    u_samples = np.asarray(u_samples, dtype=np.float64)
    n = t_grid.size
    dim = np.asarray(y0, dtype=np.float64).size
    out = np.empty((dim, n), dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64)
    out[:, 0] = y
    for i in range(n - 1):
        u_i = u_samples[i]
        sol = solve_ivp(
            lambda t, yy, u=u_i: rhs(t, yy, u),
            (float(t_grid[i]), float(t_grid[i + 1])),
            y,
            method=_METHOD,
            rtol=_RTOL,
            atol=_ATOL,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at interval {i}: {sol.message}")
        y = sol.y[:, -1]
        out[:, i + 1] = y
    return out  # shape (state_dim, n)


# --- device ODEs (standalone transcription of the module docstrings) --------


def _servo_rhs(t, y, q_set, *, Kp, Kd, T_max, deadband, J, b, Fc):
    q, v = y
    err = q_set - q
    err_eff = err - deadband * np.tanh(err / deadband)
    tau_cmd = Kp * err_eff - Kd * v
    tau_act = T_max * np.tanh(tau_cmd / T_max)
    dv = (tau_act - b * v - Fc * np.tanh(v / _V_EPS)) / J
    return np.array([v, dv])


def _bldc_rhs(t, y, i_ref, *, R, L, Kt, Ke, Kp_i, J, b, Fc):
    q, v, i = y
    V_app = Kp_i * (i_ref - i)
    e_back = Ke * v
    di = (V_app - R * i - e_back) / L
    tau = Kt * i
    dv = (tau - b * v - Fc * np.tanh(v / _V_EPS)) / J
    return np.array([v, dv, di])


def _sea_rhs(t, y, tau_in, *, k, c, J_m, J_l, b_m, b_l):
    qm, vm, ql, vl = y
    tau_spring = k * (qm - ql) + c * (vm - vl)
    dvm = (tau_in - tau_spring - b_m * vm) / J_m
    dvl = (tau_spring - b_l * vl) / J_l
    return np.array([vm, dvm, vl, dvl])


# --- excitations + truth parameters -----------------------------------------


def _gen_servo():
    truth = dict(Kp=20.0, Kd=0.3, T_max=5.0, deadband=1e-4, J=1e-3, b=0.0, Fc=0.0)
    # Warm start the recovery test hands the recipe (off truth, ODE well-posed).
    warm = dict(Kp=15.0, Kd=0.4, T_max=5.0, deadband=1e-4, J=1.2e-3, b=0.0, Fc=0.0)
    fs, samples_per_step = 200.0, 200
    setpoints = [0.0, 0.5, 0.5, -0.2, 0.0]
    t = np.arange(len(setpoints) * samples_per_step, dtype=np.float64) / fs
    q_set = np.repeat(np.asarray(setpoints, dtype=np.float64), samples_per_step)
    y = _integrate_zoh(
        lambda tt, yy, u: _servo_rhs(tt, yy, u, **truth), [0.0, 0.0], t, q_set
    )
    return {
        "device": "Servo",
        "telemetry": "ActuatorStepTelemetry",
        "arrays": {"time": t, "setpoint": q_set, "position": y[0]},
        "sample_rate_hz": fs,
        "truth": truth,
        "warm": warm,
    }


def _gen_bldc():
    truth = dict(R=1.2, L=1.5e-3, Kt=0.05, Ke=0.05, Kp_i=12.0, J=1.2e-4, b=1e-4, Fc=0.0)
    warm = dict(R=1.0, L=1e-3, Kt=0.04, Ke=0.04, Kp_i=10.0, J=1e-4, b=0.0, Fc=0.0)
    fs, T = 2000.0, 1.0
    n = int(fs * T) + 1
    t = np.arange(n, dtype=np.float64) / fs
    f0, rate = 0.5, (50.0 - 0.5) / T  # linear chirp 0.5 -> 50 Hz on i_ref
    iref = np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t * t))
    y = _integrate_zoh(
        lambda tt, yy, u: _bldc_rhs(tt, yy, u, **truth), [0.0, 0.0, 0.0], t, iref
    )
    return {
        "device": "BLDC",
        "telemetry": "ActuatorChirpTelemetry",
        "arrays": {"time": t, "setpoint": iref, "velocity": y[1], "current": y[2]},
        "sample_rate_hz": fs,
        "truth": truth,
        "warm": warm,
    }


def _gen_sea():
    truth = dict(k=120.0, c=0.15, J_m=1.2e-3, J_l=1.1e-2, b_m=1e-3, b_l=1e-3)
    warm = dict(k=100.0, c=0.1, J_m=1e-3, J_l=1e-2, b_m=0.0, b_l=0.0)
    fs, T = 2000.0, 1.0
    n = int(fs * T) + 1
    t = np.arange(n, dtype=np.float64) / fs
    f0, rate = 0.5, (30.0 - 0.5) / T
    tau = 0.5 * np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t * t))
    y = _integrate_zoh(
        lambda tt, yy, u: _sea_rhs(tt, yy, u, **truth), [0.0, 0.0, 0.0, 0.0], t, tau
    )
    return {
        "device": "SeriesElasticActuator",
        "telemetry": "SeriesElasticTelemetry",
        "arrays": {"time": t, "tau_input": tau, "q_motor": y[0], "q_load": y[2]},
        "sample_rate_hz": fs,
        "truth": truth,
        "warm": warm,
    }


def _write(name: str, spec: dict) -> None:
    meta = {
        "device": spec["device"],
        "telemetry": spec["telemetry"],
        "sample_rate_hz": spec["sample_rate_hz"],
        "integrator": {"method": _METHOD, "rtol": _RTOL, "atol": _ATOL},
        "truth": spec["truth"],
        "warm": spec["warm"],
        "provenance": (
            "Independent DOP853 integration of a standalone transcription of the "
            "device ODE; NOT produced by the fitter's RK4. See generate.py."
        ),
    }
    out = _HERE / f"{name}.npz"
    np.savez(
        out,
        meta_json=np.asarray(json.dumps(meta)),
        **{k: np.asarray(v, dtype=np.float64) for k, v in spec["arrays"].items()},
    )
    cols = ", ".join(spec["arrays"])
    print(f"wrote {out.name}: {cols}  (n={spec['arrays']['time'].size})")


def main() -> None:
    _write("servo_step", _gen_servo())
    _write("bldc_chirp", _gen_bldc())
    _write("sea_chirp", _gen_sea())


if __name__ == "__main__":
    main()
