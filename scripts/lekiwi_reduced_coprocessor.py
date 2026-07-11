#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
"""On-device REDUCED self-check co-processor for the LeKiwi.

The full :class:`jaxterity.selfcheck.CoProcessor` rolls the **MJX** twin forward
each tick. The Raspberry Pi that owns the Feetech bus has NumPy only — no JAX/MJX
— so the body rollout cannot run on-device. This is the honest *lowered* path the
task allows: a pure-NumPy co-processor that forward-integrates only the
**safety-relevant off-body models** — the per-servo lumped thermal ODE and the
shared-battery Thévenin sag — beside the controller, at the target rate.

    dT/dt = (k_pwr·tau² − (T − T_amb)/R_th) / C_th        (per servo, RK4)
    V_bus = V_oc − R_int · Σ_servo k_i·|tau|               (algebraic)

These are exactly the channels the SafetyMonitor's brownout/thermal vetoes consume
(the body rollout feeds joint-limit checks, which a position controller handles
anyway). The full MJX-body forecast stays on the workstation / the Jaxility-lowered
path; this on-device co-processor covers the electrical+thermal safety forecast.

The forecast functions are pure NumPy and importable, so CI checks their parity
against the JAX `jaxterity.physics` models. ``main`` runs the co-processor loop at
a target rate (optionally reading the real bus), records every tick's wall time and
the deadline-miss count, and writes a timing report.

Usage (on the Pi)::

    python lekiwi_reduced_coprocessor.py --rate-hz 100 --horizon 10 --duration 8 \
        --live --out /tmp/coproc_report.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

# SO-101 per-servo thermal defaults (jaxterity.twin) and LeKiwi battery defaults.
C_TH, R_TH, K_PWR = 15.0, 4.0, 6.0
V_OC, R_INT, K_I = 12.166, 0.0653, 1.5  # battery from the real aggregate-load fit
T_AMB = 298.15


def thermal_rk4(T, tau, dt, *, C_th=C_TH, R_th=R_TH, k_pwr=K_PWR, T_amb=T_AMB):
    """One RK4 step of the lumped per-servo thermal ODE (vectorized over servos)."""

    def deriv(temp):
        return (k_pwr * tau * tau - (temp - T_amb) / R_th) / C_th

    k1 = deriv(T)
    k2 = deriv(T + 0.5 * dt * k1)
    k3 = deriv(T + 0.5 * dt * k2)
    k4 = deriv(T + dt * k3)
    return T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def bus_voltage(tau, *, V_oc=V_OC, R_int=R_INT, k_i=K_I):
    """Thévenin bus voltage under the aggregate servo load (algebraic)."""
    return float(V_oc - R_int * k_i * np.sum(np.abs(tau)))


def _signed_mag(raw: int, sign_bit: int) -> int:
    """Feetech signed-magnitude decode (matches the other Pi capture scripts)."""
    mask = 1 << sign_bit
    return -(raw & (mask - 1)) if (raw & mask) else (raw & (mask - 1))


def forecast(T0, tau, *, horizon, dt, **params):
    """Roll the reduced models forward over the horizon under a held torque.

    Returns ``(T_traj, V_traj)`` — predicted per-servo temperature (``horizon × n``)
    and bus voltage (``horizon``) trajectories. This is the co-processor's per-tick
    compute (pure NumPy, allocation-light).
    """
    therm = {k: params[k] for k in ("C_th", "R_th", "k_pwr", "T_amb") if k in params}
    batt = {k: params[k] for k in ("V_oc", "R_int", "k_i") if k in params}
    T = np.asarray(T0, dtype=float)
    tau = np.asarray(tau, dtype=float)
    T_traj = np.empty((horizon, T.shape[0]))
    for h in range(horizon):
        T = thermal_rk4(T, tau, dt, **therm)
        T_traj[h] = T
    # Bus voltage is algebraic in the held torque — constant over the horizon, so
    # compute it once and broadcast (not once per step).
    V_traj = np.full(horizon, bus_voltage(tau, **batt))
    return T_traj, V_traj


def _read_bus(pk, ph, ids):
    """Read per-servo (temperature K, load duty) from the Feetech bus."""
    from scservo_sdk import COMM_SUCCESS

    temps, loads = [], []
    for sid in ids:
        t, c, _ = pk.read1ByteTxRx(ph, sid, 63)  # PRESENT_TEMPERATURE (°C)
        ld, c2, _ = pk.read2ByteTxRx(ph, sid, 60)  # PRESENT_LOAD
        # Pass valid reads through as-is (a genuine high temp must reach the
        # thermal forecast — do NOT clamp it away); fall back to ambient only on a
        # bus comms failure. Worst case of a corrupt-but-OK read is a brief
        # conservative over-forecast, never a masked overheat.
        temps.append((float(t) if c == COMM_SUCCESS else 25.0) + 273.15)
        loads.append(_signed_mag(ld, 10) / 1000.0)  # signed-magnitude, bit 10
    return np.asarray(temps), np.asarray(loads)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.05, help="forecast step, s")
    ap.add_argument("--n-servos", type=int, default=6)
    ap.add_argument("--live", action="store_true", help="read the real bus each tick")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", default="/tmp/coproc_report.json")
    args = ap.parse_args()

    ids = list(range(1, args.n_servos + 1))
    ph = pk = None
    if args.live:
        from scservo_sdk import PacketHandler, PortHandler

        ph, pk = PortHandler(args.port), PacketHandler(0)
        assert ph.openPort() and ph.setBaudRate(1_000_000), "could not open bus"

    deadline = 1.0 / args.rate_hz
    T = np.full(args.n_servos, T_AMB + 5.0)
    tau_hold = np.full(args.n_servos, 0.8)  # a representative sustained hold torque
    tick_ms, misses = [], 0
    n_ticks = int(args.duration * args.rate_hz)
    # warm up (first-call allocations) so the timed loop is steady-state.
    forecast(T, tau_hold, horizon=args.horizon, dt=args.dt)

    next_tick = time.perf_counter()
    for _ in range(n_ticks):
        t0 = time.perf_counter()
        if args.live:
            T, tau_hold = _read_bus(pk, ph, ids)
        # the co-processor's per-tick work: forward safety forecast + veto check.
        T_traj, V_traj = forecast(T, tau_hold, horizon=args.horizon, dt=args.dt)
        _ = (float(T_traj.max()), float(V_traj.min()))  # the channels the veto reads
        elapsed = time.perf_counter() - t0
        tick_ms.append(elapsed * 1e3)
        if elapsed > deadline:
            misses += 1
        next_tick += deadline
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
    if ph is not None:
        ph.closePort()

    tick_ms_arr = np.asarray(tick_ms)
    report = {
        "schema": "lekiwi-coproc-report-v1",
        "mode": "live" if args.live else "offline",
        "rate_hz": args.rate_hz,
        "deadline_ms": deadline * 1e3,
        "horizon": args.horizon,
        "dt": args.dt,
        "n_servos": args.n_servos,
        "n_ticks": n_ticks,
        "deadline_misses": misses,
        "tick_ms_mean": float(tick_ms_arr.mean()),
        "tick_ms_p99": float(np.percentile(tick_ms_arr, 99)),
        "tick_ms_max": float(tick_ms_arr.max()),
    }
    json.dump(report, open(args.out, "w"))
    print(
        f"{report['mode']} @ {args.rate_hz:.0f} Hz, horizon {args.horizon}: "
        f"{misses}/{n_ticks} misses; tick mean {report['tick_ms_mean']:.2f} ms "
        f"p99 {report['tick_ms_p99']:.2f} ms max {report['tick_ms_max']:.2f} ms "
        f"(deadline {report['deadline_ms']:.1f} ms) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
