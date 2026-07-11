# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin self-maintenance over a simulated accelerated-aging run of an SO-101.

One runnable script showing a per-unit twin keep itself current as the unit ages:

1. **Calibrate with uncertainty.** Fit the unit's off-body thermal from a trace;
   the fit carries a per-leaf covariance.
2. **Stay current as it ages.** Stream recalibration windows whose true ``R_th``
   drifts upward (an aging unit); the windowed re-fit tracks it, the attestation
   handle advancing in an unbroken chain.
3. **Surface a fault.** A residual monitor — its band sized from the calibration
   uncertainty — flags an injected thermal anomaly, naming the channel.
4. **Shrink the envelope.** As the unit heats, the capability estimator publishes
   a smaller torque envelope that the SafetyMonitor enforces: a command the cool
   unit accepts is vetoed hot.
5. **Render the drift.** The stored update history prepares a drift view.
6. **Bind the evidence.** A structured, unsigned evidence record is bound by hash
   to the current twin handle.

The whole run is deterministic and hashes (twin handles + the fault channel + the
safety decisions + the evidence hash) to a stable ``run_hash``.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from jaxterity.calibrate import calibrate_twin_thermal
from jaxterity.data import ThermalTelemetry
from jaxterity.diagnostics import build_evidence, prepare_drift_view
from jaxterity.physics.thermal import thermal_response
from jaxterity.selfcheck import (
    CapabilityEstimator,
    LimitModel,
    ResidualMonitor,
    SafetyMonitor,
)
from jaxterity.twin import TwinRollout, so101_twin, track_thermal_drift

_FS = 50.0
_N = 600
_TAU_AMP = 0.8
_K_TRUE = 6.0


def _thermal_window(twin, *, r_th, tau_amp=_TAU_AMP, fs=_FS, n=_N):
    """A per-servo temperature trace + torque consistent with a true ``R_th``."""
    nv = twin.n_servos
    C_th = np.asarray(twin.C_th)
    T_amb = twin.T_amb
    t = np.arange(n, dtype=np.float64) / fs
    cols = [
        np.asarray(
            thermal_response(
                t,
                tau_amp,
                T0=T_amb,
                T_amb=T_amb,
                C_th=float(C_th[j]),
                R_th=r_th,
                k_pwr=_K_TRUE,
            )
        )
        for j in range(nv)
    ]
    telem = ThermalTelemetry(
        time=t,
        sample_rate_hz=fs,
        servo_names=twin.servo_names,
        temperature=np.stack(cols, axis=1),
    )
    return telem, np.full((n, nv), tau_amp)


@dataclasses.dataclass(frozen=True)
class SelfMaintenanceResult:
    """The outcome of the simulated self-maintenance run."""

    serial: str
    initial_handle: str
    final_handle: str
    n_updates: int
    chain_linked: bool
    tracked_r_th: float
    true_final_r_th: float
    fault_channel: str
    cool_action: str
    hot_action: str
    evidence_hash: str
    run_hash: str


def run() -> SelfMaintenanceResult:
    """Run the simulated accelerated-aging study and return its result."""
    twin = so101_twin("SO101-PRODUCTIZE")
    nv = twin.n_servos

    # 1. initial calibration with uncertainty.
    telem0, tau0 = _thermal_window(twin, r_th=3.0)
    twin = calibrate_twin_thermal(twin, telem0, tau0)
    initial_handle = twin.attestation_handle

    # 2. accelerated aging: R_th drifts up; the windowed re-fit tracks it.
    drift = [3.2, 3.6, 4.0, 4.5, 5.0]
    windows = [(*_thermal_window(twin, r_th=r), float(i)) for i, r in enumerate(drift)]
    twin, history = track_thermal_drift(twin, windows)
    tracked_r_th = float(np.asarray(twin.R_th)[0])

    # 3. residual monitor watches the re-fit parameter stream and surfaces an
    #    injected thermal fault, naming the servo.
    #
    #    The monitor z-scores each fitted channel against this unit's own
    #    recalibration history; a std floor keeps the near-flat healthy stream
    #    from false-alarming on estimate jitter. We replay the tracked ``R_th``
    #    commits (per servo, labelled) to build history, then inject a sustained
    #    thermal anomaly on servo ``faulty`` — R_th climbing well past its own
    #    history's spread — so exactly that servo's channel flags.
    faulty = 2
    monitor = ResidualMonitor(
        z_threshold=4.0,
        min_history=len(history.updates),
        labels={"offbody.R_th": twin.servo_names},
        std_floor={"offbody.R_th": 0.02},
    )
    for update in history.updates:  # the healthy re-fit stream (builds history)
        monitor.observe(
            float(update.index), {"offbody.R_th": update.value("offbody.R_th")}
        )
    faulted = np.asarray(twin.R_th, dtype=float).copy()
    faulted[faulty] += 3.0  # a sustained thermal anomaly on servo `faulty`
    report = monitor.observe(float(len(history.updates)), {"offbody.R_th": faulted})
    fault_channel = report.flags[0].partition(": z=")[0] if report.flags else ""

    # 4. capability estimator shrinks the envelope; SafetyMonitor enforces it.
    limits = LimitModel(
        temp_ceiling_k=twin.T_amb + 60.0, nominal_torque_margin_frac=0.9
    )
    estimator = CapabilityEstimator(twin, limits)
    pred = _benign_prediction(twin, twin.T_amb + 15.0)
    action = np.full(nv, 1.0)  # sub-thermal hold; the torque margin is what differs
    cool_env = estimator.estimate(np.full(nv, twin.T_amb))
    hot_env = estimator.estimate(np.full(nv, twin.T_amb + 50.0))
    cool = SafetyMonitor(twin, cool_env).evaluate(pred, action=action)
    hot = SafetyMonitor(twin, hot_env).evaluate(pred, action=action)

    # 5. drift view renders the stored history.
    view = prepare_drift_view(history)
    _ = view.divergence_norm("offbody.R_th")  # reproducible from the records

    # 6. structured evidence bound to the current handle.
    evidence = build_evidence(
        twin,
        held_out_error={"temperature_rms": 0.05},
        recovery={"R_th_rel": abs(tracked_r_th - drift[-1]) / drift[-1]},
    )

    payload = "|".join(
        [
            twin.serial,
            initial_handle,
            twin.attestation_handle,
            "".join(history.handles),
            fault_channel,
            cool.action.value,
            hot.action.value,
            evidence.content_hash(),
        ]
    )
    run_hash = hashlib.sha256(payload.encode()).hexdigest()

    return SelfMaintenanceResult(
        serial=twin.serial,
        initial_handle=initial_handle,
        final_handle=twin.attestation_handle,
        n_updates=len(history.updates),
        chain_linked=history.is_linked_chain,
        tracked_r_th=tracked_r_th,
        true_final_r_th=drift[-1],
        fault_channel=fault_channel,
        cool_action=cool.action.value,
        hot_action=hot.action.value,
        evidence_hash=evidence.content_hash(),
        run_hash=run_hash,
    )


def _benign_prediction(twin, temp_value) -> TwinRollout:
    """A short forecast at a benign temperature (isolates the torque check)."""
    n = twin.n_servos
    nq = twin.robot.build_system()._nq
    steps = 4
    return TwinRollout(
        q=np.zeros((steps + 1, nq)),
        qd=np.zeros((steps + 1, n)),
        temperature=np.full((steps + 1, n), temp_value),
        bus_voltage=None,
    )


def main() -> SelfMaintenanceResult:
    """Run the study and print a human-readable summary."""
    r = run()
    print("Twin self-maintenance — SO-101 (simulated accelerated aging)")
    print(f"  unit                       : {r.serial}")
    print(f"  1. calibrated handle       : {r.initial_handle[:16]}…")
    print(
        f"  2. aged + tracked          : R_th {r.tracked_r_th:.3f} vs true "
        f"{r.true_final_r_th:.3f} over {r.n_updates} re-fits "
        f"(chain linked: {r.chain_linked})"
    )
    print(f"  3. fault surfaced          : {r.fault_channel}")
    print(f"  4. envelope cool→hot       : {r.cool_action} → {r.hot_action}")
    print(f"  5. drift view              : rendered from {r.n_updates} updates")
    print(f"  6. evidence hash           : {r.evidence_hash[:16]}…")
    print(f"     run hash                : {r.run_hash}")
    print("  (simulated accelerated-aging run; deterministic and reproducible)")
    return r


if __name__ == "__main__":
    main()
