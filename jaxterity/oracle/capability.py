# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Generate a robot's capability sheet from its (measured) model.

WHY THIS EXISTS. Agentic-safety evaluations supply a robot's physical limits
as hand-written prompt text ("my maximum payload is 500 grams"), and
feasibility-judgment accuracy is strongly driven by the quality of that
supplied context (Gemini Robotics 2 safety report: 62.0% → 95.8% as
hand-authored capability context is enriched). This module generates that
context **from the model of the specific unit** instead of from an author's
memory of a datasheet: each field carries a value, units, and — the part a
hand-written prompt never carries — a **provenance tag** saying whether the
number is *fitted* (identified from this unit's telemetry), *derived*
(computed from the model's kinematics/dynamics), *nominal* (supplied
spec-sheet value), or *representative* (a default that has NOT been measured
— see ``KNOWN_GAPS.md``). :meth:`CapabilitySheet.to_text` renders the block
an agent consumes; :meth:`CapabilitySheet.to_dict` is the structured form.

The computations are deliberately plain numpy + MuJoCo (deterministic, no
JAX compile): forward-kinematic reach sweep, and a worst-case-pose static
payload bound — for sampled poses, the per-joint gravity torque per kilogram
at the end body (translational Jacobian row against gravity) is compared
against per-joint torque budgets net of the arm's own static load. The
payload bound is **static and conservative**: it ignores dynamic/inertial
loads and assumes rigid links (dynamic manipulation needs a larger margin,
which the sheet says out loud).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..errors import JaxterityError
from ..robot import Robot

#: Gravity magnitude used by the static analyses, m/s^2.
_GRAVITY = 9.81
#: Default bus-voltage floor for the battery-margin field, V.
_DEFAULT_BUS_FLOOR_V = 11.0


@dataclasses.dataclass(frozen=True)
class CapabilityField:
    """One capability entry: value, units, and where the number came from.

    ``provenance`` is one of ``"fitted"`` (identified from this unit's own
    telemetry), ``"derived"`` (computed from the model), ``"nominal"``
    (supplied spec value), ``"representative"`` (unmeasured default).
    """

    name: str
    value: Any
    units: str
    provenance: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CapabilitySheet:
    """A unit's capability context, structured and renderable.

    Built by :func:`capability_sheet`. ``fields`` preserves insertion order;
    ``attestation_handle`` binds the sheet to the exact model state that
    produced it (a re-fit changes the handle, so stale sheets are
    detectable).
    """

    robot_name: str
    attestation_handle: str
    fields: tuple[CapabilityField, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "attestation_handle": self.attestation_handle,
            "fields": [f.to_dict() for f in self.fields],
        }

    def to_text(self) -> str:
        """Render the agent-consumable capability block.

        The format is deliberately close to the hand-written constraint
        blocks agentic-safety benchmarks use, with provenance tags appended
        so the reader (human or model) can tell measured facts from
        assumptions.
        """
        lines = [
            f"ROBOT CAPABILITY SHEET — {self.robot_name} "
            f"(model attestation {self.attestation_handle[:12]}…)",
            "Each line is tagged with its provenance: [fitted]=identified "
            "from this unit's telemetry, [derived]=computed from the model, "
            "[nominal]=spec value, [representative]=unmeasured default.",
        ]
        for f in self.fields:
            val = f.value
            if isinstance(val, float):
                val = f"{val:.3g}"
            line = f"- {f.name}: {val} {f.units} [{f.provenance}]"
            if f.note:
                line += f" — {f.note}"
            lines.append(line)
        return "\n".join(lines)


def _pose_grid(lo: np.ndarray, hi: np.ndarray, n_random: int, seed: int) -> np.ndarray:
    """Sampled joint configurations: range corners capped + uniform draws."""
    rng = np.random.default_rng(seed)
    mid = (lo + hi) / 2.0
    poses = [mid, lo.copy(), hi.copy()]
    poses.extend(rng.uniform(lo, hi, size=(n_random, lo.shape[0])))
    return np.asarray(poses)


def capability_sheet(
    robot: Robot,
    *,
    robot_name: str = "robot",
    tau_max_nm: float | None = None,
    battery: dict[str, float] | None = None,
    bus_floor_v: float = _DEFAULT_BUS_FLOOR_V,
    hold_current_a: float | None = None,
    n_random_poses: int = 200,
    seed: int = 0,
) -> CapabilitySheet:
    """Generate the capability sheet for a robot model.

    Args
    ----
    robot : Robot
        The (ideally calibrated) robot model. Fitted friction and provenance
        travel with it; the attestation handle binds the sheet to it.
    robot_name : str
        Display name for the sheet header.
    tau_max_nm : float | None
        Per-joint torque budget for the static payload bound. If the MJCF
        carries usable actuator force ranges those are used (provenance
        "derived"); otherwise this supplied value is used (provenance
        "nominal"); if neither exists the payload field is omitted with a
        named note — never silently guessed.
    battery : dict | None
        Optional ``{"V_oc": V, "R_int": Ω}`` (e.g. a unit's fitted Thévenin
        pair) for the bus-margin field, tagged "fitted".
    bus_floor_v : float
        Operator-declared brownout floor for the margin computation.
    hold_current_a : float | None
        Aggregate hold-current estimate for the bus-margin field; required
        for that field when ``battery`` is given.
    n_random_poses : int
        Random poses added to the corner poses for the reach/payload sweeps.
    seed : int
        RNG seed — the sheet is deterministic given identical inputs.

    Returns
    -------
    CapabilitySheet

    Raises
    ------
    JaxterityError
        If the model has no hinge joints to analyze.
    """
    import mujoco

    mj = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    data = mujoco.MjData(mj)

    hinge_ids = [
        j for j in range(mj.njnt) if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if not hinge_ids:
        raise JaxterityError("capability_sheet: model has no hinge joints to analyze.")
    names = [mj.joint(j).name for j in hinge_ids]
    qadr = np.array([mj.jnt_qposadr[j] for j in hinge_ids])
    dadr = np.array([mj.jnt_dofadr[j] for j in hinge_ids])
    lo = mj.jnt_range[hinge_ids, 0].copy()
    hi = mj.jnt_range[hinge_ids, 1].copy()
    unlimited = ~mj.jnt_limited[hinge_ids].astype(bool)
    lo[unlimited], hi[unlimited] = -np.pi, np.pi

    fields: list[CapabilityField] = [
        CapabilityField(
            name="joints",
            value=", ".join(
                f"{n} [{lo_i:.2f}, {hi_i:.2f}]" for n, lo_i, hi_i in zip(names, lo, hi)
            ),
            units="rad",
            provenance="derived",
            note="hinge ranges from the model",
        )
    ]

    # End body: the last body in the kinematic chain (deepest child).
    end_body = mj.nbody - 1
    base_pos = data.xpos[0].copy()

    poses = _pose_grid(lo, hi, n_random_poses, seed)
    reach = 0.0
    # Torque budget per joint.
    force_range = np.abs(mj.actuator_forcerange) if mj.nu else np.zeros((0, 2))
    if mj.nu and np.any(force_range[:, 1] > 0):
        tau_budget = np.full(len(hinge_ids), float(np.min(force_range[:, 1])))
        tau_provenance = "derived"
        tau_note = "min actuator force range in the model"
    elif tau_max_nm is not None:
        tau_budget = np.full(len(hinge_ids), float(tau_max_nm))
        tau_provenance = "nominal"
        tau_note = f"supplied per-joint budget {tau_max_nm} N·m"
    else:
        tau_budget = None
        tau_provenance = ""
        tau_note = ""

    payload_min = np.inf
    jacp = np.zeros((3, mj.nv))
    for q in poses:
        data.qpos[:] = 0.0
        data.qpos[qadr] = q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(mj, data)
        reach = max(reach, float(np.linalg.norm(data.xpos[end_body] - base_pos)))
        if tau_budget is None:
            continue
        # Static arm-gravity torque at this pose.
        mujoco.mj_inverse(mj, data)
        tau_static = np.abs(data.qfrc_inverse[dadr])
        # Per-kilogram gravity torque at the end body.
        mujoco.mj_jacBody(mj, data, jacp, None, end_body)
        tau_per_kg = np.abs(jacp[2, dadr]) * _GRAVITY
        headroom = tau_budget - tau_static
        with np.errstate(divide="ignore", invalid="ignore"):
            m_max = np.where(tau_per_kg > 1e-9, headroom / tau_per_kg, np.inf)
        m_max = np.where(headroom <= 0.0, 0.0, m_max)
        payload_min = min(payload_min, float(np.min(m_max)))

    fields.append(
        CapabilityField(
            name="max_reach",
            value=round(reach, 4),
            units="m",
            provenance="derived",
            note=f"kinematic sweep over {len(poses)} poses",
        )
    )
    if tau_budget is not None and np.isfinite(payload_min):
        fields.append(
            CapabilityField(
                name="max_payload_static",
                value=round(max(payload_min, 0.0), 4),
                units="kg",
                provenance=tau_provenance,
                note=(
                    "worst-pose static bound (ignores dynamics — use margin "
                    f"for motion); torque budget: {tau_note}"
                ),
            )
        )
    else:
        fields.append(
            CapabilityField(
                name="max_payload_static",
                value="unavailable",
                units="",
                provenance="representative",
                note="no actuator torque limits in model and no tau_max_nm "
                "supplied — payload NOT computed (never guessed)",
            )
        )

    if battery is not None:
        v_oc = float(battery["V_oc"])
        r_int = float(battery["R_int"])
        if hold_current_a is not None:
            v_min = v_oc - r_int * float(hold_current_a)
            fields.append(
                CapabilityField(
                    name="bus_margin_at_hold",
                    value=round(v_min - bus_floor_v, 3),
                    units="V",
                    provenance="fitted",
                    note=(
                        f"V_oc {v_oc:.3f} V, R_int {r_int * 1e3:.1f} mΩ "
                        f"(this unit), {hold_current_a:.1f} A hold vs "
                        f"{bus_floor_v:.2f} V floor"
                    ),
                )
            )
        else:
            fields.append(
                CapabilityField(
                    name="battery",
                    value=f"V_oc {v_oc:.3f}, R_int {r_int * 1e3:.1f}",
                    units="V, mΩ",
                    provenance="fitted",
                    note="pass hold_current_a for a bus-margin figure",
                )
            )

    return CapabilitySheet(
        robot_name=robot_name,
        attestation_handle=robot.attestation_handle,
        fields=tuple(fields),
    )


__all__ = ["CapabilityField", "CapabilitySheet", "capability_sheet"]
