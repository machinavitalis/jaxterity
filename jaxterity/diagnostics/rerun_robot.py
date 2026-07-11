# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Robot-agnostic Rerun.io state streaming + signal inspection.

Where :mod:`~jaxterity.diagnostics.rerun_adapter` logs a *post-hoc* sim-vs-real
scalar comparison, this module streams a **live, articulated** view of any
:class:`~jaxterity.robot.Robot`: it runs the robot's forward kinematics (via the
same MJX model the dynamics use) and logs every link's 3D pose plus the
kinematic skeleton, alongside structured per-joint scalar signals (position,
velocity, optional applied torque). Because the geometry comes from FK, it works
for every zoo robot — the fixed-base arms, the floating-base Crazyflie, and the
29-DoF G1 — with no per-robot drawing code.

Two entry points:

- :class:`RobotRerunLogger` — the **live** primitive. Construct once, then call
  :meth:`~RobotRerunLogger.log_step` per control-loop / simulation step to stream
  state as it is produced.
- :func:`log_robot_trajectory` — a convenience that logs a whole recorded
  trajectory (a ``(T, nq)`` array of configurations) in one call.

Entity layout under the recording (``ns`` defaults to the robot name)::

    {ns}/bodies/{link}      Transform3D + a small Boxes3D  (one per link, oriented)
    {ns}/skeleton           LineStrips3D  (bones connecting each link to its parent)
    {ns}/signals/{joint}/position   Scalars  (per-joint q; one series per DOF)
    {ns}/signals/{joint}/velocity   Scalars  (per-joint q̇, when qd is given)
    {ns}/signals/{joint}/torque     Scalars  (per-joint τ, when tau is given)

``rerun-sdk`` is an optional dependency (the ``[rerun]`` extra). If it is not
installed, constructing a logger raises a typed
:class:`~jaxterity.errors.JaxterityError` with install instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..errors import JaxterityError

if TYPE_CHECKING:  # avoid a runtime import cycle (robot -> diagnostics -> here)
    from ..robot import Robot

# MuJoCo joint-type codes (mjtJoint): free=0, ball=1, slide=2, hinge=3.
# (nq, nv) width per joint type, for slicing qpos / qvel / qfrc per joint.
_JOINT_WIDTHS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}


def _require_rerun():
    try:
        import rerun as rr

        return rr
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise JaxterityError(
            'rerun-sdk is not installed. Install with `pip install "jaxterity[rerun]"` '
            "(or `pip install rerun-sdk>=0.20`) to use the robot Rerun logger."
        ) from e


def _build_kinematics(robot: Robot):
    """Compile the robot's MJX model and a jitted forward-kinematics closure.

    Returns ``(fk, body_names, parent_ids, joints)`` where ``fk(q) -> (xpos,
    xquat)`` gives every body's world position / orientation (MuJoCo ``wxyz``
    quaternion) and ``joints`` is a list of ``(name, qpos_adr, nq, dof_adr, nv)``.
    """
    import jax
    import mujoco
    from mujoco import mjx

    mj_model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    mjx_model = mjx.put_model(mj_model)

    body_names = [mj_model.body(i).name or f"body{i}" for i in range(mj_model.nbody)]
    parent_ids = [int(mj_model.body_parentid[i]) for i in range(mj_model.nbody)]

    joints = []
    for j in range(mj_model.njnt):
        nq, nv = _JOINT_WIDTHS[int(mj_model.jnt_type[j])]
        joints.append(
            (
                mj_model.joint(j).name or f"joint{j}",
                int(mj_model.jnt_qposadr[j]),
                nq,
                int(mj_model.jnt_dofadr[j]),
                nv,
            )
        )

    def _fk(q):
        data = mjx.make_data(mjx_model).replace(qpos=q)
        data = mjx.kinematics(mjx_model, data)
        return data.xpos, data.xquat

    return (
        jax.jit(_fk),
        body_names,
        parent_ids,
        joints,
        int(mj_model.nq),
        int(mj_model.nv),
    )


class RobotRerunLogger:
    """Stream a :class:`~jaxterity.robot.Robot`'s state to Rerun, step by step.

    Construct once for a robot, then call :meth:`log_step` for each state you
    want to visualize — inside a simulation loop, a control loop, or over a
    recorded trajectory. The logger holds a compiled forward-kinematics function,
    so per-step logging is cheap.

    Args
    ----
    robot : Robot
        The robot to visualize. Its current parameters drive the FK, so a
        calibrated robot renders with its calibrated geometry.
    namespace : str, optional
        Root entity path. Defaults to ``robot.name``.
    spawn : bool
        If ``True``, spawn the Rerun viewer (live streaming). Default ``False``.
    save_path : str or pathlib.Path, optional
        If given, write the recording to this ``.rrd`` file instead of spawning.
    recording_id : str, optional
        Passed to ``rerun.init`` to append to an existing recording.
    link_size : float
        Half-size (m) of the small box drawn at each link, so link orientation
        is visible. Default ``0.02``.
    timeline : str
        Name of the Rerun timeline used by :meth:`log_step`. Default ``"time"``.

    Raises
    ------
    JaxterityError
        If ``rerun-sdk`` is not installed.
    """

    def __init__(
        self,
        robot: Robot,
        *,
        namespace: str | None = None,
        spawn: bool = False,
        save_path=None,
        recording_id: str | None = None,
        link_size: float = 0.02,
        timeline: str = "time",
    ) -> None:
        self._rr = _require_rerun()
        self.namespace = namespace or robot.name or "robot"
        self.link_size = float(link_size)
        self.timeline = timeline
        (
            self._fk,
            self._body_names,
            self._parent_ids,
            self._joints,
            self._nq,
            self._nv,
        ) = _build_kinematics(robot)

        self._rr.init(
            f"jaxterity-robot-{self.namespace}",
            recording_id=recording_id,
            spawn=spawn,
        )
        if save_path is not None:
            self._rr.save(str(save_path))

    def log_step(self, q, *, qd=None, tau=None, time: float | None = None) -> None:
        """Log one state: link transforms, the skeleton, and joint signals.

        Args
        ----
        q : array-like, shape ``(nq,)``
            Configuration (joint positions; for a floating base the first 7 are
            the free joint's position + ``wxyz`` quaternion).
        qd : array-like, shape ``(nv,)``, optional
            Generalized velocity. Logged as the per-joint ``velocity`` signal.
        tau : array-like, shape ``(nv,)``, optional
            Applied generalized force/torque. Logged as the ``torque`` signal.
        time : float, optional
            Timeline value for this step. If omitted, an internal step counter
            (incremented per call) is used as a sequence index.
        """
        rr = self._rr
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self._nq:
            raise JaxterityError(
                f"RobotRerunLogger.log_step: q has length {q.shape[0]}, expected "
                f"nq={self._nq} for robot {self.namespace!r}."
            )
        if time is None:
            rr.set_time(self.timeline, sequence=getattr(self, "_seq", 0))
            self._seq = getattr(self, "_seq", 0) + 1
        else:
            rr.set_time(self.timeline, duration=float(time))

        xpos, xquat = self._fk(q)
        xpos = np.asarray(xpos, dtype=np.float64)
        xquat = np.asarray(xquat, dtype=np.float64)

        # Per-link transform + a small oriented box (skip the world body, 0).
        hs = [self.link_size] * 3
        for b in range(1, len(self._body_names)):
            w, x, y, z = xquat[b]
            path = f"{self.namespace}/bodies/{self._body_names[b]}"
            rr.log(
                path,
                rr.Transform3D(
                    translation=xpos[b], rotation=rr.Quaternion(xyzw=[x, y, z, w])
                ),
            )
            rr.log(path, rr.Boxes3D(half_sizes=[hs]))

        # Kinematic skeleton: a bone from each non-root body to its parent.
        bones = [
            [xpos[self._parent_ids[b]].tolist(), xpos[b].tolist()]
            for b in range(1, len(self._body_names))
            if self._parent_ids[b] != 0
        ]
        if bones:
            rr.log(f"{self.namespace}/skeleton", rr.LineStrips3D(bones))

        # Per-joint scalar signals.
        qd_arr = None if qd is None else np.asarray(qd, dtype=np.float64).reshape(-1)
        tau_arr = None if tau is None else np.asarray(tau, dtype=np.float64).reshape(-1)
        for name, qadr, nq_j, dadr, nv_j in self._joints:
            self._log_signal(f"signals/{name}/position", q[qadr : qadr + nq_j])
            if qd_arr is not None:
                self._log_signal(f"signals/{name}/velocity", qd_arr[dadr : dadr + nv_j])
            if tau_arr is not None:
                self._log_signal(f"signals/{name}/torque", tau_arr[dadr : dadr + nv_j])

    def _log_signal(self, subpath: str, values: np.ndarray) -> None:
        rr = self._rr
        multi = values.shape[0] > 1
        for i, v in enumerate(values):
            suffix = f".{i}" if multi else ""
            rr.log(f"{self.namespace}/{subpath}{suffix}", rr.Scalars(float(v)))


def log_robot_trajectory(
    robot: Robot,
    qs,
    *,
    qds=None,
    taus=None,
    times=None,
    dt: float | None = None,
    namespace: str | None = None,
    spawn: bool = False,
    save_path=None,
    recording_id: str | None = None,
    link_size: float = 0.02,
) -> None:
    """Stream a whole recorded trajectory of a robot to Rerun.

    A convenience wrapper over :class:`RobotRerunLogger`: builds the logger and
    calls :meth:`~RobotRerunLogger.log_step` for each row of ``qs``.

    Args
    ----
    robot : Robot
        The robot to visualize.
    qs : array-like, shape ``(T, nq)``
        Configuration at each timestep.
    qds, taus : array-like, shape ``(T, nv)``, optional
        Per-step velocity / applied torque, logged as signals.
    times : array-like, shape ``(T,)``, optional
        Explicit time values. If omitted and ``dt`` is given, uses
        ``arange(T) * dt``; otherwise a per-step sequence index is used.
    dt : float, optional
        Timestep, used to synthesize ``times`` when ``times`` is not given.
    namespace, spawn, save_path, recording_id, link_size
        Forwarded to :class:`RobotRerunLogger`.

    Raises
    ------
    JaxterityError
        If ``rerun-sdk`` is not installed, or array lengths disagree.
    """
    qs = np.asarray(qs, dtype=np.float64)
    if qs.ndim != 2:
        raise JaxterityError(
            f"log_robot_trajectory: qs must be (T, nq); got shape {qs.shape}."
        )
    t = qs.shape[0]
    if times is None and dt is not None:
        times = np.arange(t, dtype=np.float64) * float(dt)
    for arr, label in ((qds, "qds"), (taus, "taus")):
        if arr is not None and np.asarray(arr).shape[0] != t:
            raise JaxterityError(
                f"log_robot_trajectory: {label} has {np.asarray(arr).shape[0]} rows, "
                f"expected {t} to match qs."
            )

    logger = RobotRerunLogger(
        robot,
        namespace=namespace,
        spawn=spawn,
        save_path=save_path,
        recording_id=recording_id,
        link_size=link_size,
    )
    for i in range(t):
        logger.log_step(
            qs[i],
            qd=None if qds is None else qds[i],
            tau=None if taus is None else taus[i],
            time=None if times is None else float(times[i]),
        )


__all__ = ["RobotRerunLogger", "log_robot_trajectory"]
