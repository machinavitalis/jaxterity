# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Regenerate the *cross-model* sysid oracle: a jaxonomy-built SEA trace.

The `Radau` oracle in ``generate.py`` integrates a standalone transcription of
the device ODE — so it rules out a shared *integrator*, but a physics
*model-form* error copied into both the fitter and that transcription would
survive it (the transcription and the fitter were both written from the same
device docstring).

This generator closes that residual for the series-elastic actuator. It does
**not** transcribe ``_sea_rhs`` at all. It assembles the two-inertia drivetrain
from jaxonomy's acausal *rotational primitives* — `Inertia`, `Spring`, `Damper`,
`TorqueSource` — and lets jaxonomy's acausal compiler build and simulate the
DAE. Those primitives are authored and tested independently of jaxterity (their
own conservation-law and analytic-response tests live in jaxonomy), and the SEA
structure emerges only from how they are *wired*. So if jaxterity's ``_sea_rhs``
had a sign or coupling error — wrong spring-torque sign, tau applied to the
wrong inertia, a swapped reaction — the fit would fail to recover the truth from
this trace. The physics ground truth is external to jaxterity's own equations.

Node balance of the wiring below reproduces exactly::

    tau_spring = k*(q_m - q_l) + c*(v_m - v_l)
    dv_m = (tau_in - tau_spring - b_m*v_m) / J_m
    dv_l = (tau_spring        - b_l*v_l) / J_l

This is still within the jax*y stack (jaxonomy is jaxterity's upstream engine),
so it is a stronger *model-form* cross-check, not a substitute for a truly
external trace (a third-party log or real hardware) — that remains the tracked
follow-up in ``KNOWN_GAPS.md``.

jaxonomy is required to regenerate; it is NOT required to run the test, which
only loads the frozen ``.npz``. Regenerate with::

    python test/sysid/data/independent/generate_jaxonomy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jaxonomy
import numpy as np
from jaxonomy import library as lib
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import rotational as rot

_HERE = Path(__file__).resolve().parent

# Same truth as the Radau SEA trace, so the two oracles are directly comparable.
_TRUTH = {"k": 120.0, "c": 0.15, "J_m": 1.2e-3, "J_l": 1.1e-2, "b_m": 1e-3, "b_l": 1e-3}
_WARM = {"k": 100.0, "c": 0.1, "J_m": 1e-3, "J_l": 1e-2, "b_m": 0.0, "b_l": 0.0}
_FS, _T = 2000.0, 1.0
_RTOL, _ATOL = 1e-10, 1e-12


def _build_sea_diagram():
    """Two-inertia SEA assembled from jaxonomy acausal rotational primitives.

    ``flange_a`` is the motor side of both the coupling Spring and Damper, which
    fixes the sign of the elastic/damping torque (`+k*(q_m-q_l)`,
    `+c*(v_m-v_l)`). Ground-referenced dampers give the per-inertia viscous
    terms. A torque-input source drives the motor node.
    """
    ev = EqnEnv()
    ad = AcausalDiagram()
    tin = rot.TorqueSource(
        ev, name="tin", enable_torque_port=True, enable_flange_b=False
    )
    Jm = rot.Inertia(
        ev,
        name="Jm",
        I=_TRUTH["J_m"],
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    Jl = rot.Inertia(
        ev,
        name="Jl",
        I=_TRUTH["J_l"],
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    spring = rot.Spring(ev, name="spring", K=_TRUTH["k"])
    damper = rot.Damper(ev, name="damper", D=_TRUTH["c"])
    bm = rot.Damper(ev, name="bm", D=_TRUTH["b_m"])
    bl = rot.Damper(ev, name="bl", D=_TRUTH["b_l"])
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
    # ground for the two viscous ground-dampers
    ad.connect(bm, "flange_b", gnd, "flange")
    ad.connect(bl, "flange_b", gnd, "flange")

    acsys = AcausalCompiler(ev, ad)()
    return acsys, sm, sl


def _simulate():
    acsys, sm, sl = _build_sea_diagram()
    builder = jaxonomy.DiagramBuilder()
    acsys = builder.add(acsys)
    chirp = builder.add(lib.Chirp(f0=0.5, f1=30.0, stop_time=_T, phi=0.0))
    gain = builder.add(lib.Gain(0.5))
    builder.connect(chirp.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], acsys.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)

    qm_idx = acsys.outsym_to_portid[sm.get_sym_by_port_name("ang_rel")]
    ql_idx = acsys.outsym_to_portid[sl.get_sym_by_port_name("ang_rel")]
    rec = {
        "tau": gain.output_ports[0],
        "q_m": acsys.output_ports[qm_idx],
        "q_l": acsys.output_ports[ql_idx],
    }
    opts = jaxonomy.SimulatorOptions(
        max_major_step_length=1.0 / _FS,
        rtol=_RTOL,
        atol=_ATOL,
        buffer_length=20000,
    )
    res = jaxonomy.simulate(diagram, ctx, (0.0, _T), options=opts, recorded_signals=rec)

    # Resample the (adaptive) recording onto the uniform telemetry grid. The
    # fitter reconstructs tau with a ZOH between these samples, so tau/q are kept
    # mutually consistent by interpolating all three to the same grid.
    tt = np.asarray(res.time, dtype=np.float64)
    grid = np.arange(int(_FS * _T) + 1, dtype=np.float64) / _FS
    return {
        "time": grid,
        "tau_input": np.interp(
            grid, tt, np.asarray(res.outputs["tau"], dtype=np.float64)
        ),
        "q_motor": np.interp(
            grid, tt, np.asarray(res.outputs["q_m"], dtype=np.float64)
        ),
        "q_load": np.interp(grid, tt, np.asarray(res.outputs["q_l"], dtype=np.float64)),
    }


def main() -> None:
    arrays = _simulate()
    meta = {
        "device": "SeriesElasticActuator",
        "telemetry": "SeriesElasticTelemetry",
        "sample_rate_hz": _FS,
        # Not the fitter's RK4, and not even a transcription of _sea_rhs: the
        # trace is jaxonomy's acausal DAE built from rotational primitives.
        "integrator": {"method": "jaxonomy-acausal-BDF", "rtol": _RTOL, "atol": _ATOL},
        "truth": _TRUTH,
        "warm": _WARM,
        "provenance": (
            "jaxonomy acausal two-inertia drivetrain (Inertia/Spring/Damper/"
            "TorqueSource) — physics ground truth external to jaxterity's own "
            "_sea_rhs. See generate_jaxonomy.py."
        ),
    }
    out = _HERE / "sea_jaxonomy.npz"
    np.savez(
        out,
        meta_json=np.asarray(json.dumps(meta)),
        **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()},
    )
    print(f"wrote {out.name}: {', '.join(arrays)}  (n={arrays['time'].size})")


if __name__ == "__main__":
    main()
