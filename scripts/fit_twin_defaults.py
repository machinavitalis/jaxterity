#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
"""Fit the twin-builder off-body defaults from the committed real LeKiwi captures.

This script is the provenance for the numbers baked into
``jaxterity/twin/__init__.py`` (T-220). It loads the real per-unit captures
committed under ``test/calibrate/data/lekiwi/`` and runs the existing fit
recipes — nothing here invents a new model:

- **Battery** (``_LEKIWI_V_OC`` / ``_LEKIWI_R_INT``): ``fit_battery_sag`` on
  ``aggregate_load_trace.json`` (real shared-bus voltage under a synchronized
  three-servo multisine). The Thévenin fit is linear LS; the fitted ``R_int``
  is in the duty-current frame (``I = i_stall * duty``, see
  ``jaxterity/calibrate/battery.py``), so the per-servo torque→current gain
  ``k_i`` is *not* identified by this trace and stays representative.
- **Thermal** (``_SO101_R_TH`` / ``_SO101_K_PWR``): ``fit_servo_thermal`` on
  the real ``temp_C`` channel of each ``commanded_<joint>.json`` capture, with
  the driving torque taken from the closed-loop equation-error calibration of
  the same capture (``tau = K_t * load_proxy + c0``, the documented duty→torque
  conversion from ``jaxterity/calibrate/closed_loop.py``) and ``C_th`` held at
  the twin default. The fit is run twice per joint: on the raw trace and after
  a rolling-median glitch rejection (the Feetech temperature register shows
  obvious single-sample bus-read glitches, e.g. one-tick jumps to 150 °C).

Run from the repo root::

    python scripts/fit_twin_defaults.py

The script prints every fitted value with its fit-quality statistics and a
per-parameter verdict (BAKED IN vs NOT CREDIBLE). A thermal "NOT CREDIBLE"
verdict is an expected, honest outcome for these captures: the temperature
register has 1 °C quantization and the excitations last only 14–16 s — far
shorter than the servo thermal time constant (R_th·C_th ≈ minutes) — so the
trace carries essentially no thermal information.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "test" / "calibrate" / "data" / "lekiwi"

#: The commanded-excitation captures carrying a real per-servo temp_C channel.
THERMAL_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: Held-known lumped thermal capacitance (J/K) — the twin default; only
#: k_pwr/C_th and R_th·C_th are identifiable from a temperature trace.
C_TH_KNOWN = 15.0
#: Ambient during the captures, K (bench indoor).
T_AMB = 298.15

#: Credibility gates for baking a thermal fit into the builder defaults:
#: both parameters physical (positive), the regression explaining a
#: non-negligible share of the Ṫ variance, and the standard error smaller
#: than the estimate itself.
R2_MIN = 0.2


def despike(temp: np.ndarray, *, window: int = 7, jump_c: float = 5.0) -> np.ndarray:
    """Reject single-sample telemetry glitches with a rolling median.

    The Feetech present-temperature register occasionally returns a wild
    single-sample value (a bus read glitch — e.g. 34 → 150 → 34 °C in three
    consecutive samples). Samples deviating more than ``jump_c`` °C from the
    rolling ``window``-sample median are replaced by that median. Genuine
    thermal transients are orders of magnitude slower than one sample, so
    this cannot remove real dynamics.

    Args
    ----
    temp : numpy.ndarray, shape ``(n,)``, units °C
        Raw temperature channel.
    window : int
        Rolling-median window, samples (odd).
    jump_c : float
        Deviation threshold, °C.

    Returns
    -------
    numpy.ndarray
        Cleaned copy, same shape and units.
    """
    out = temp.astype(float).copy()
    half = window // 2
    for i in range(len(out)):
        lo, hi = max(0, i - half), min(len(out), i + half + 1)
        med = float(np.median(temp[lo:hi]))
        if abs(temp[i] - med) > jump_c:
            out[i] = med
    return out


def thermal_r_squared(
    t: np.ndarray, T: np.ndarray, tau: np.ndarray, R_th: float, k_pwr: float
) -> float:
    """R² of the fitted lumped-ODE regression on the Ṫ samples.

    Reconstructs the same central-difference regression ``fit_servo_thermal``
    solves and reports the share of Ṫ variance the fitted (R_th, k_pwr)
    explain. Units: t in s, T in K, tau in N·m.
    """
    Tdot = (T[2:] - T[:-2]) / (t[2:] - t[:-2])
    T_c, tau_c = T[1:-1], tau[1:-1]
    pred = (k_pwr / C_TH_KNOWN) * tau_c**2 - (T_c - T_AMB) / (R_th * C_TH_KNOWN)
    ss_tot = float(np.sum((Tdot - Tdot.mean()) ** 2))
    resid = Tdot - pred
    return 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0


def fit_battery() -> None:
    """Fit and report the LeKiwi Thévenin battery from the aggregate trace."""
    from jaxterity.calibrate import fit_battery_sag, load_bus_load_trace

    trace = DATA / "aggregate_load_trace.json"
    capture = load_bus_load_trace(str(trace))
    fit = fit_battery_sag(capture)
    v_oc = float(fit.fitted["V_oc"])
    r_int = float(fit.fitted["R_int"])
    r = fit.residuals
    print("== Battery (aggregate_load_trace.json) ==")
    print(f"  servos {capture.servo_ids}, {capture.n_samples} samples")
    print(f"  V_oc  = {v_oc:.4f} V")
    print(
        f"  R_int = {r_int:.5f} Ohm   "
        f"(duty-current frame, i_stall = {r['i_stall_a']} A)"
    )
    print(
        f"  r_squared = {r['r_squared']:.3f}   sag = {r['sag_v']:.2f} V   "
        f"peak I = {r['peak_current_a']:.2f} A"
    )
    print(f"  telemetry hash: {fit.telemetry_hash}")
    credible = 0.0 < r_int and r["r_squared"] > 0.5 and r["sag_v"] > 0.3
    print(
        f"  verdict: {'BAKED IN' if credible else 'NOT CREDIBLE'} "
        "(V_oc, R_int); k_i NOT identified by this trace (duty-current "
        "frame) — stays representative."
    )
    print()


def fit_thermal() -> None:
    """Fit and report the per-servo thermal model from the real temp_C traces."""
    from jaxterity import zoo
    from jaxterity.calibrate import identify_closed_loop, load_commanded_capture
    from jaxterity.calibrate.thermal import fit_servo_thermal
    from jaxterity.data import ThermalTelemetry

    robot = zoo.calibrated("so100")
    calibration = str(DATA / "calibration_final.json")
    print(
        f"== Thermal (commanded_<joint>.json temp_C, C_th held at {C_TH_KNOWN} J/K) =="
    )
    credible: list[tuple[str, float, float]] = []
    for joint in THERMAL_JOINTS:
        path = DATA / f"commanded_{joint}.json"
        capture = load_commanded_capture(str(path), robot, calibration=calibration)
        raw = json.loads(path.read_text())
        temp_c = np.asarray(raw["temp_C"], dtype=float)
        clean_c = despike(temp_c)
        # Duty→torque via the closed-loop calibration of this same capture.
        cl = identify_closed_loop(robot, capture)
        K_t = float(cl.residuals["K_t"])
        c0 = float(cl.residuals["load_offset"])
        tau = K_t * capture.load_proxy + c0
        duration = float(capture.t[-1] - capture.t[0])
        for label, channel in (("raw", temp_c), ("despiked", clean_c)):
            T_k = channel + 273.15
            telemetry = ThermalTelemetry(
                time=capture.t,
                sample_rate_hz=float(1.0 / np.mean(np.diff(capture.t))),
                servo_names=(joint,),
                temperature=T_k[:, None],
            )
            fit = fit_servo_thermal(
                telemetry, tau[:, None], C_th=C_TH_KNOWN, T_amb=T_AMB
            )
            R_th = float(fit["R_th"][0])
            k_pwr = float(fit["k_pwr"][0])
            R_sd = float(np.sqrt(fit["R_th_var"][0]))
            k_sd = float(np.sqrt(fit["k_pwr_var"][0]))
            r2 = thermal_r_squared(capture.t, T_k, tau, R_th, k_pwr)
            ok = (
                R_th > 0
                and k_pwr > 0
                and r2 > R2_MIN
                and R_sd < abs(R_th)
                and k_sd < abs(k_pwr)
            )
            print(
                f"  {joint:14s} {label:8s} dT={np.ptp(channel):5.1f} C over "
                f"{duration:4.1f} s  K_t={K_t:+.3f}  "
                f"R_th={R_th:+9.3g} +/- {R_sd:8.2g} K/W  "
                f"k_pwr={k_pwr:+9.3g} +/- {k_sd:8.2g} W/(N.m)^2  "
                f"r2={r2:+.3f}  {'credible' if ok else 'not credible'}"
            )
            if label == "despiked" and ok:
                credible.append((joint, R_th, k_pwr))
    print()
    if credible:
        R_med = float(np.median([r for _, r, _ in credible]))
        k_med = float(np.median([k for _, _, k in credible]))
        print(
            f"  verdict: BAKED IN — median over credible joints "
            f"{[j for j, _, _ in credible]}: R_th={R_med:.3g} K/W, "
            f"k_pwr={k_med:.3g} W/(N.m)^2"
        )
    else:
        print(
            "  verdict: NOT CREDIBLE — no joint yields a physical, "
            "significant fit. The temp_C register is quantized to 1 C and "
            "the 14-16 s excitations are far shorter than the servo thermal "
            "time constant, so the trace carries no usable thermal signal. "
            "The representative _SO101_{C_TH,R_TH,K_PWR} defaults stay in "
            "place; a per-unit thermal fit needs a dedicated long-duration "
            "(several-minute) heating capture."
        )
    print()


def main() -> None:
    """Run both fits and print the provenance report."""
    fit_battery()
    fit_thermal()


if __name__ == "__main__":
    main()
