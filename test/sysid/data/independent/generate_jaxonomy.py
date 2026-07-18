# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Regenerate the jaxonomy-sourced sysid oracles (SEA, BLDC, Servo).

The `Radau` oracles in ``generate.py`` integrate a standalone transcription of
each device ODE — they rule out a shared *integrator*, but a physics
*model-form* error copied into both the fitter and that transcription would
survive them. These traces raise the bar by sourcing ground truth from
jaxonomy, jaxterity's upstream engine, in one of two ways per device. jaxonomy
is only needed to regenerate; the test just loads the frozen ``.npz``.

**Independent physics** (SEA, BLDC) — the plant is assembled from jaxonomy's
*acausal* primitives, whose governing equations are authored and tested
independently of jaxterity. A sign/coupling error in jaxterity's own device ODE
fails to recover here even if copied identically into the Radau transcription.

* SEA: a two-inertia drivetrain wired from Inertia/Spring/Damper/TorqueSource.
* BLDC: an `IdealMotor` (its own `V = R*i + L*di/dt + K*w`, `tau = K*i`
  equations) driven open-loop. The current-loop P controller
  ``V_app = Kp_i*(i_ref - i)`` is folded *exactly* into the plant by algebra:
  the ``-Kp_i*i`` term is an extra resistive drop, so the closed loop equals an
  open-loop voltage ``Kp_i*i_ref`` into a motor of resistance ``R + Kp_i``. No
  causal feedback is needed, so the acausal boundary stays clean.

**Independent implementation** (Servo) — the servo's smooth-saturation feedback
is nonlinear and cannot be folded into a passive acausal network, and jaxonomy's
acausal *sensor* outputs cannot sit inside a causal feedback loop (the
context-build recurses). So the servo is built as an independent *causal*
block diagram — jaxonomy `Integrator`/`Gain`/`Adder`/`Trigonometric("tanh")`
primitives, solved by jaxonomy's BDF — which is a distinct assembly and engine
from both jaxterity's RK4 and the scipy transcription, but (unlike SEA/BLDC) is
still a transcription of the same equations, not independent physics.

Neither level reaches ground truth *external to the jax-y stack* — a physics
error shared across the stack would survive. That remains tracked in
``KNOWN_GAPS.md``.

Regenerate with::

    python test/sysid/data/independent/generate_jaxonomy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jaxonomy
import numpy as np
from jaxonomy import library as lib
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.acausal import rotational as rot
from jaxonomy.library.dynamics import Integrator
from jaxonomy.library.math_ops import Adder, Gain, Trigonometric
from jaxonomy.library.sources import Step

_HERE = Path(__file__).resolve().parent
_RTOL, _ATOL = 1e-10, 1e-12


def _resample(res, grid, names):
    """Linear-interpolate recorded (adaptive) signals onto the uniform grid."""
    tt = np.asarray(res.time, dtype=np.float64)
    return {
        out: np.interp(grid, tt, np.asarray(res.outputs[rec], dtype=np.float64))
        for out, rec in names.items()
    }


def _write(name: str, arrays: dict, meta: dict) -> None:
    out = _HERE / f"{name}.npz"
    np.savez(
        out,
        meta_json=np.asarray(json.dumps(meta)),
        **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()},
    )
    print(f"wrote {out.name}: {', '.join(arrays)}  (n={arrays['time'].size})")


# --- SEA: independent physics, passive acausal two-inertia drivetrain -------

_SEA_TRUTH = {
    "k": 120.0,
    "c": 0.15,
    "J_m": 1.2e-3,
    "J_l": 1.1e-2,
    "b_m": 1e-3,
    "b_l": 1e-3,
}
_SEA_WARM = {"k": 100.0, "c": 0.1, "J_m": 1e-3, "J_l": 1e-2, "b_m": 0.0, "b_l": 0.0}


def _gen_sea():
    fs, T = 2000.0, 1.0
    t = _SEA_TRUTH
    ev = EqnEnv()
    ad = AcausalDiagram()
    tin = rot.TorqueSource(
        ev, name="tin", enable_torque_port=True, enable_flange_b=False
    )
    Jm = rot.Inertia(
        ev,
        name="Jm",
        I=t["J_m"],
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    Jl = rot.Inertia(
        ev,
        name="Jl",
        I=t["J_l"],
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    spring = rot.Spring(ev, name="spring", K=t["k"])
    damper = rot.Damper(ev, name="damper", D=t["c"])
    bm = rot.Damper(ev, name="bm", D=t["b_m"])
    bl = rot.Damper(ev, name="bl", D=t["b_l"])
    gnd = rot.FixedAngle(ev, name="gnd")
    sm = rot.MotionSensor(ev, name="sm", enable_flange_b=False, enable_angle_port=True)
    sl = rot.MotionSensor(ev, name="sl", enable_flange_b=False, enable_angle_port=True)
    # motor node: torque source + motor inertia + spring/damper A-side
    # + ground damper + sensor all meet here
    ad.connect(tin, "flange_a", Jm, "flange")
    ad.connect(Jm, "flange", spring, "flange_a")
    ad.connect(Jm, "flange", damper, "flange_a")
    ad.connect(Jm, "flange", bm, "flange_a")
    ad.connect(Jm, "flange", sm, "flange_a")
    # load node: spring/damper B-side + load inertia + ground damper + sensor
    ad.connect(spring, "flange_b", Jl, "flange")
    ad.connect(damper, "flange_b", Jl, "flange")
    ad.connect(Jl, "flange", bl, "flange_a")
    ad.connect(Jl, "flange", sl, "flange_a")
    ad.connect(bm, "flange_b", gnd, "flange")
    ad.connect(bl, "flange_b", gnd, "flange")
    acsys = AcausalCompiler(ev, ad)()

    builder = jaxonomy.DiagramBuilder()
    acsys = builder.add(acsys)
    chirp = builder.add(lib.Chirp(f0=0.5, f1=30.0, stop_time=T, phi=0.0))
    gain = builder.add(Gain(0.5))
    builder.connect(chirp.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], acsys.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)
    qm = acsys.outsym_to_portid[sm.get_sym_by_port_name("ang_rel")]
    ql = acsys.outsym_to_portid[sl.get_sym_by_port_name("ang_rel")]
    rec = {
        "tau": gain.output_ports[0],
        "q_m": acsys.output_ports[qm],
        "q_l": acsys.output_ports[ql],
    }
    opts = jaxonomy.SimulatorOptions(
        max_major_step_length=1.0 / fs, rtol=_RTOL, atol=_ATOL, buffer_length=25000
    )
    res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=opts, recorded_signals=rec)
    grid = np.arange(int(fs * T) + 1, dtype=np.float64) / fs
    a = _resample(res, grid, {"tau_input": "tau", "q_motor": "q_m", "q_load": "q_l"})
    a["time"] = grid
    meta = {
        "device": "SeriesElasticActuator",
        "telemetry": "SeriesElasticTelemetry",
        "sample_rate_hz": fs,
        "integrator": {"method": "jaxonomy-acausal-BDF", "rtol": _RTOL, "atol": _ATOL},
        "independence": "physics",
        "truth": t,
        "warm": _SEA_WARM,
        "provenance": (
            "jaxonomy acausal two-inertia drivetrain; external to _sea_rhs."
        ),
    }
    return "sea_jaxonomy", a, meta


# --- BLDC: independent physics, acausal IdealMotor, current loop folded ------

_BLDC_TRUTH = {
    "R": 1.2,
    "L": 1.5e-3,
    "Kt": 0.05,
    "Ke": 0.05,
    "Kp_i": 12.0,
    "J": 1.2e-4,
    "b": 1e-4,
    "Fc": 0.0,
}
_BLDC_WARM = {
    "R": 1.0,
    "L": 1e-3,
    "Kt": 0.04,
    "Ke": 0.04,
    "Kp_i": 10.0,
    "J": 1e-4,
    "b": 0.0,
    "Fc": 0.0,
}


def _gen_bldc():
    fs, T = 2000.0, 1.0
    t = _BLDC_TRUTH
    # Fold the current-loop P gain into the winding resistance (exact algebra):
    # V_app = Kp_i*(i_ref - i)  ==>  L di/dt = Kp_i*i_ref - (R + Kp_i)*i - Ke*v.
    r_eff = t["R"] + t["Kp_i"]
    ev = EqnEnv()
    ad = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", enable_voltage_port=True)
    si = elec.CurrentSensor(ev, name="si")
    # IdealMotor forces Kt = Ke = K; our BLDC truth already has Kt == Ke.
    mot = elec.IdealMotor(
        ev,
        name="mot",
        R=r_eff,
        K=t["Kt"],
        L=t["L"],
        J=t["J"],
        initial_current=0.0,
        initial_current_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
        initial_angle=0.0,
        initial_angle_fixed=True,
    )
    gnd = elec.Ground(ev, name="gnd")
    bm = rot.Damper(ev, name="bm", D=t["b"])
    fa = rot.FixedAngle(ev, name="fa")
    sv = rot.MotionSensor(
        ev, name="sv", enable_flange_b=False, enable_velocity_port=True
    )
    ad.connect(vs, "p", si, "p")
    ad.connect(si, "n", mot, "pos")
    ad.connect(mot, "neg", vs, "n")
    ad.connect(gnd, "p", vs, "n")
    ad.connect(mot, "shaft", bm, "flange_a")
    ad.connect(bm, "flange_b", fa, "flange")
    ad.connect(mot, "shaft", sv, "flange_a")
    acsys = AcausalCompiler(
        ev, ad, scale=True
    )()  # scale=True: condition the DAE Jacobian

    builder = jaxonomy.DiagramBuilder()
    acsys = builder.add(acsys)
    chirp = builder.add(lib.Chirp(f0=0.5, f1=50.0, stop_time=T, phi=0.0))
    vsrc = builder.add(Gain(t["Kp_i"]))  # open-loop drive V = Kp_i * i_ref
    builder.connect(chirp.output_ports[0], vsrc.input_ports[0])
    builder.connect(vsrc.output_ports[0], acsys.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)
    i_idx = acsys.outsym_to_portid[si.get_sym_by_port_name("i")]
    w_idx = acsys.outsym_to_portid[sv.get_sym_by_port_name("w_rel")]
    rec = {
        "iref": chirp.output_ports[0],
        "i": acsys.output_ports[i_idx],
        "v": acsys.output_ports[w_idx],
    }
    opts = jaxonomy.SimulatorOptions(
        max_major_step_length=1.0 / fs, rtol=_RTOL, atol=_ATOL, buffer_length=40000
    )
    res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=opts, recorded_signals=rec)
    grid = np.arange(int(fs * T) + 1, dtype=np.float64) / fs
    a = _resample(res, grid, {"setpoint": "iref", "velocity": "v", "current": "i"})
    a["time"] = grid
    meta = {
        "device": "BLDC",
        "telemetry": "ActuatorChirpTelemetry",
        "sample_rate_hz": fs,
        "integrator": {"method": "jaxonomy-acausal-BDF", "rtol": _RTOL, "atol": _ATOL},
        "independence": "physics",
        "truth": t,
        "warm": _BLDC_WARM,
        "provenance": (
            "jaxonomy acausal IdealMotor (own R/L/backEMF/torque physics); "
            "current-loop P folded into R+Kp_i. Ground truth external to _bldc_rhs."
        ),
    }
    return "bldc_jaxonomy", a, meta


# --- Servo: independent implementation, causal block diagram -----------------

_SERVO_TRUTH = {
    "Kp": 20.0,
    "Kd": 0.3,
    "T_max": 5.0,
    "deadband": 1e-4,
    "J": 1e-3,
    "b": 0.0,
    "Fc": 0.0,
}
_SERVO_WARM = {
    "Kp": 15.0,
    "Kd": 0.4,
    "T_max": 5.0,
    "deadband": 1e-4,
    "J": 1.2e-3,
    "b": 0.0,
    "Fc": 0.0,
}


def _gen_servo():
    fs, T = 200.0, 5.0
    t = _SERVO_TRUTH
    b = jaxonomy.DiagramBuilder()
    q_int = b.add(Integrator(0.0, name="q"))
    v_int = b.add(Integrator(0.0, name="v"))
    # setpoint schedule [0, 0.5, 0.5, -0.2, 0] as summed steps (each self-events)
    s1 = b.add(Step(0.0, 0.5, 1.0))
    s2 = b.add(Step(0.0, -0.7, 3.0))
    s3 = b.add(Step(0.0, 0.2, 4.0))
    qset = b.add(Adder(3, operators="+++"))
    b.connect(s1.output_ports[0], qset.input_ports[0])
    b.connect(s2.output_ports[0], qset.input_ports[1])
    b.connect(s3.output_ports[0], qset.input_ports[2])
    # err = q_set - q
    err = b.add(Adder(2, operators="+-"))
    b.connect(qset.output_ports[0], err.input_ports[0])
    b.connect(q_int.output_ports[0], err.input_ports[1])
    # err_eff = err - deadband*tanh(err/deadband)
    g_din = b.add(Gain(1.0 / t["deadband"]))
    th_d = b.add(Trigonometric("tanh"))
    g_dout = b.add(Gain(t["deadband"]))
    err_eff = b.add(Adder(2, operators="+-"))
    b.connect(err.output_ports[0], g_din.input_ports[0])
    b.connect(g_din.output_ports[0], th_d.input_ports[0])
    b.connect(th_d.output_ports[0], g_dout.input_ports[0])
    b.connect(err.output_ports[0], err_eff.input_ports[0])
    b.connect(g_dout.output_ports[0], err_eff.input_ports[1])
    # tau_cmd = Kp*err_eff - Kd*v
    g_kp = b.add(Gain(t["Kp"]))
    g_kd = b.add(Gain(t["Kd"]))
    tau_cmd = b.add(Adder(2, operators="+-"))
    b.connect(err_eff.output_ports[0], g_kp.input_ports[0])
    b.connect(v_int.output_ports[0], g_kd.input_ports[0])
    b.connect(g_kp.output_ports[0], tau_cmd.input_ports[0])
    b.connect(g_kd.output_ports[0], tau_cmd.input_ports[1])
    # tau_act = T_max*tanh(tau_cmd/T_max); dv/dt = tau_act / J  (b = Fc = 0)
    g_sin = b.add(Gain(1.0 / t["T_max"]))
    th_s = b.add(Trigonometric("tanh"))
    g_sout = b.add(Gain(t["T_max"]))
    inv_j = b.add(Gain(1.0 / t["J"]))
    b.connect(tau_cmd.output_ports[0], g_sin.input_ports[0])
    b.connect(g_sin.output_ports[0], th_s.input_ports[0])
    b.connect(th_s.output_ports[0], g_sout.input_ports[0])
    b.connect(g_sout.output_ports[0], inv_j.input_ports[0])
    b.connect(inv_j.output_ports[0], v_int.input_ports[0])
    b.connect(v_int.output_ports[0], q_int.input_ports[0])
    diagram = b.build()
    ctx = diagram.create_context(check_types=True)
    rec = {"qset": qset.output_ports[0], "q": q_int.output_ports[0]}
    opts = jaxonomy.SimulatorOptions(
        max_major_step_length=1.0 / fs, rtol=_RTOL, atol=_ATOL, buffer_length=20000
    )
    res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=opts, recorded_signals=rec)
    grid = np.arange(int(fs * T), dtype=np.float64) / fs
    a = _resample(res, grid, {"setpoint": "qset", "position": "q"})
    a["time"] = grid
    meta = {
        "device": "Servo",
        "telemetry": "ActuatorStepTelemetry",
        "sample_rate_hz": fs,
        "integrator": {"method": "jaxonomy-causal-BDF", "rtol": _RTOL, "atol": _ATOL},
        "independence": "implementation",
        "truth": t,
        "warm": _SERVO_WARM,
        "provenance": (
            "jaxonomy causal block diagram (Integrator/Gain/Adder/tanh) on BDF — "
            "independent implementation, not independent physics."
        ),
    }
    return "servo_jaxonomy", a, meta


def main() -> None:
    for gen in (_gen_sea, _gen_bldc, _gen_servo):
        name, arrays, meta = gen()
        _write(name, arrays, meta)


if __name__ == "__main__":
    main()
