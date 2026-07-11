# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Render the real-LeKiwi self-check chain to a ~3-minute overlay video.

A dashboard animation over the self-check timeline, driven by **real** LeKiwi telemetry:
the captured arm joint motion (top), the twin's predicted servo temperature rising
to the thermal ceiling — where it **vetoes** while a body-only model proceeds
(middle) — and the real aggregate-load bus-voltage trace replayed through the
aged-battery model, sagging to the brownout floor (bottom). The banner carries the
chain stages and the reproducible ``run_hash``.

Writes an MP4 via ``imageio_ffmpeg`` (no system ffmpeg needed) and, with
``--rrd``, a Rerun recording of the same overlay via ``RobotRerunLogger``.

Run: ``python -m examples.selfcheck_demo_lekiwi_video --out <path.mp4>``
(add ``--rrd <path.rrd>`` to also write a Rerun recording).
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from examples.selfcheck_demo_lekiwi import run_demo
from jaxterity.calibrate import fit_battery_sag, load_bus_load_trace

_DATA = pathlib.Path(__file__).parent.parent / "test" / "calibrate" / "data" / "lekiwi"

# SO-101 thermal defaults + LeKiwi battery (the values the twin uses).
_C_TH, _R_TH, _K_PWR, _T_AMB = 15.0, 4.0, 6.0, 298.15
_AGED_FACTOR, _K_I = 5.0, 1.5
_CEILING_C = (_T_AMB + 35.0) - 273.15
_FLOOR_V = 10.0


def _arm_motion():
    """Real captured joint angles (deg) for shoulder_pan / lift / elbow."""
    import json

    out = {}
    for name in (
        "commanded_shoulder_pan",
        "commanded_shoulder_lift",
        "commanded_elbow_flex",
    ):
        d = json.loads((_DATA / f"{name}.json").read_text())
        c = d["calibration"]
        q = (
            c["sign"]
            * (np.asarray(d["pos_ticks"], float) - c["offset_ticks"])
            / c["ticks_per_rad"]
        )
        out[d["joint"]] = (np.asarray(d["t"], float), np.degrees(q))
    return out


def render(
    out_path: str, *, seconds: float = 180.0, fps: int = 12, rrd: str | None = None
):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run_demo(active=False)
    motion = _arm_motion()
    batt = fit_battery_sag(load_bus_load_trace(_DATA / "aggregate_load_trace.json"))
    v_oc, r_int = float(batt.fitted["V_oc"]), float(batt.fitted["R_int"])

    # thermal rise to the ceiling under a sustained tau=2 hold (lumped ODE).
    tau = 2.0
    t_ss = _T_AMB + _K_PWR * _R_TH * tau**2
    tau_c = _R_TH * _C_TH

    # real aggregate-load trace → aged-battery bus voltage (replayed across the window).
    trace = load_bus_load_trace(_DATA / "aggregate_load_trace.json")
    agg = trace.aggregate_duty
    v_bus_full = v_oc - (_AGED_FACTOR * r_int) * (2.7 * agg)  # aged pack browns out

    n_frames = int(seconds * fps)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    rr_logger = None
    rr_dof = ()
    if rrd:
        try:
            from jaxterity import zoo
            from jaxterity.calibrate import dof_joint_names
            from jaxterity.diagnostics.rerun_robot import RobotRerunLogger

            rr_robot = zoo.calibrated("so100")
            rr_dof = dof_joint_names(rr_robot)
            rr_logger = RobotRerunLogger(rr_robot, save_path=rrd)
        except Exception as e:  # the MP4 is the artifact; .rrd is a bonus
            print(f"(skipping .rrd: {e})")
            rr_logger = None

    fig = plt.figure(figsize=(9.6, 6.4), dpi=100)  # 960x640 (divisible by 16)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 1, 1, 1], hspace=0.6)
    ax_banner = fig.add_subplot(gs[0])
    ax_banner.axis("off")
    ax_arm, ax_temp, ax_volt = (fig.add_subplot(gs[i]) for i in (1, 2, 3))
    breach_s = result.thermal_time_to_breach_s or 30.0

    # --- static scene set up ONCE; only artist data is updated per frame (fast). ---
    ax_banner.text(
        0.5,
        0.7,
        "Real-LeKiwi self-check chain",
        ha="center",
        fontsize=14,
        weight="bold",
    )
    time_txt = ax_banner.text(0.5, 0.12, "", ha="center", fontsize=9)

    dots = {}
    for joint, (tt, deg) in motion.items():
        ax_arm.plot(np.linspace(0, seconds, len(deg)), deg, lw=0.6, alpha=0.5)
        (dots[joint],) = ax_arm.plot([], [], "o", ms=6, label=joint)
    ax_arm.set(
        xlim=(0, seconds),
        ylabel="joint (deg)",
        title="1. real commanded-excitation capture (active self-ID)",
    )
    ax_arm.legend(loc="upper right", fontsize=7, ncol=3)

    tg_full = np.linspace(0, seconds, 600)
    temp_full = (t_ss + (_T_AMB + 5.0 - t_ss) * np.exp(-tg_full / tau_c)) - 273.15
    ax_temp.axhline(_CEILING_C, color="r", ls="--", lw=1, label="thermal ceiling")
    (temp_line,) = ax_temp.plot([], [], color="C3", label="twin: predicted servo temp")
    veto_v = ax_temp.axvline(
        breach_s,
        color="k",
        ls=":",
        lw=1,
        visible=False,
        label=f"VETO @ {breach_s:.0f}s",
    )
    veto_txt = ax_temp.text(breach_s, _CEILING_C + 2, "", color="k", fontsize=8)
    ax_temp.set(
        xlim=(0, seconds),
        ylim=(20, temp_full.max() + 8),
        ylabel="°C",
        title="2. thermal check — twin vetoes; body-only is blind",
    )
    ax_temp.legend(loc="lower right", fontsize=7)

    vt_full = np.linspace(0, seconds, len(v_bus_full))
    ax_volt.axhline(_FLOOR_V, color="r", ls="--", lw=1, label="brownout floor")
    (volt_line,) = ax_volt.plot(
        [], [], color="C0", lw=0.7, label="twin: bus V (aged pack, real load)"
    )
    brown_txt = ax_volt.text(seconds * 0.02, _FLOOR_V - 1.4, "", color="k", fontsize=8)
    ax_volt.set(
        xlim=(0, seconds),
        ylim=(5, 13),
        xlabel="s",
        ylabel="V",
        title="3. brownout check — twin vetoes; body-only has no battery",
    )
    ax_volt.legend(loc="lower right", fontsize=7)

    for k in range(n_frames):
        ts = seconds * k / max(1, n_frames - 1)
        time_txt.set_text(
            f"twin {result.twin_handle[:16]}…   "
            f"run_hash {result.run_hash[:16]}…   t = {ts:5.1f}s"
        )
        for joint, (tt, deg) in motion.items():
            idx = int((ts / seconds) * len(deg)) % len(deg)
            dots[joint].set_data([ts], [deg[idx]])
        m = tg_full <= ts
        temp_line.set_data(tg_full[m], temp_full[m])
        if ts >= breach_s:
            veto_v.set_visible(True)
            veto_txt.set_text(" twin VETO  (body-only: proceed)")
        nv = max(2, int((ts / seconds) * len(v_bus_full)))
        volt_line.set_data(vt_full[:nv], v_bus_full[:nv])
        if (v_bus_full[:nv] < _FLOOR_V).any():
            brown_txt.set_text("twin VETO [bus_voltage]  (body-only: proceed)")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        if rr_logger is not None and k % fps == 0:
            q_full = np.zeros(len(rr_dof))
            for joint, (tt, deg) in motion.items():
                if joint in rr_dof:
                    idx = int((ts / seconds) * len(deg)) % len(deg)
                    q_full[rr_dof.index(joint)] = np.radians(deg[idx])
            rr_logger.log_step(q_full, time=ts)

    writer.close()
    plt.close(fig)
    # The .rrd is streamed to disk by rr.save() at init; no explicit close needed.
    print(
        f"wrote {out_path} ({n_frames} frames @ {fps} fps ≈ {seconds:.0f}s)"
        + (f" + {rrd}" if rrd else "")
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/media/selfcheck_lekiwi.mp4")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--rrd", default=None, help="also write a Rerun .rrd recording")
    args = ap.parse_args()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render(args.out, seconds=args.seconds, fps=args.fps, rrd=args.rrd)


if __name__ == "__main__":
    main()
