# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Generate MuJoCo MJCF from a Jaxterity ``Joint``/``Link`` tree.

MJX (MuJoCo's JAX physics) is Jaxterity's dynamics backend (ADR-016). MuJoCo
parses MJCF strings directly — no Gazebo/``sdformat`` dependency — so
``Robot.to_diagram`` lowers the kinematic tree to MJCF here, then ``mjx.put_model``
turns it into a differentiable JAX model.

Scope: kinematic trees with revolute / continuous / prismatic / fixed
joints, optionally on a single floating base (one ``FLOATING`` joint → a MuJoCo
``<freejoint>``). Planar joints and multi-root models raise a clear error.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ..errors import JaxterityError
from .frame import Frame, FramedQuantity, Position, Transform
from .structure import CollisionGeom, Joint, JointLimits, JointType, Link

if TYPE_CHECKING:
    from .robot import Robot

_MJ_JOINT = {
    JointType.REVOLUTE: "hinge",
    JointType.CONTINUOUS: "hinge",
    JointType.PRISMATIC: "slide",
}


def _fmt(vec) -> str:
    return " ".join(repr(float(x)) for x in np.asarray(vec).ravel())


def _matrix_to_quat(rot) -> np.ndarray:
    """A 3x3 rotation matrix to a MuJoCo ``(w, x, y, z)`` quaternion.

    Inverse of :func:`_quat_to_matrix` (Shepperd's method — numerically stable
    across all rotations). Used to carry a joint's origin rotation onto the MJCF
    ``<body quat=...>`` so the joint frame (and thus its axis) is oriented
    correctly; dropping it silently flattens every joint onto the parent frame.
    """
    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    return np.array([w, x, y, z])


def _inertial_xml(link: Link) -> str:
    com = np.asarray(link.com).ravel()
    inertia = np.asarray(link.inertia)
    mass = float(link.mass)
    # MuJoCo requires a positive mass and positive inertia eigenvalues. Frame-only
    # links (common in real URDFs, e.g. a gripper TCP frame) carry zero inertia;
    # floor them to a tiny valid value. Real links (already positive-definite) are
    # untouched. A welded frame link's tiny inertia does not affect the dynamics.
    if mass <= 1e-9 or float(np.min(np.linalg.eigvalsh(inertia))) <= 1e-12:
        mass = max(mass, 1e-9)
        inertia = np.eye(3) * 1e-9
    full = [
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    ]
    return f'<inertial pos="{_fmt(com)}" mass="{mass!r}" fullinertia="{_fmt(full)}"/>'


def _collision_geoms(link: Link) -> tuple[CollisionGeom, ...]:
    """The link's lowerable collision primitives (empty for opaque handles).

    URDF import stores mesh filename strings in ``Link.collision``; those stay
    opaque (inertials drive the dynamics). Only a tuple/list of
    :class:`~jaxterity.robot.CollisionGeom` is lowered to MJCF ``<geom>``
    elements.
    """
    collision = link.collision
    if isinstance(collision, (tuple, list)) and all(
        isinstance(g, CollisionGeom) for g in collision
    ):
        return tuple(collision)
    return ()


def _geom_xml(geom: CollisionGeom, pad: str) -> str:
    """Emit one MuJoCo ``<geom>`` for a :class:`CollisionGeom`."""
    attrs = [f'type="{geom.kind}"', f'size="{_fmt(geom.size)}"']
    if geom.name is not None:
        attrs.insert(0, f'name="{geom.name}"')
    if any(abs(v) > 0.0 for v in geom.pos):
        attrs.append(f'pos="{_fmt(geom.pos)}"')
    if geom.quat is not None:
        attrs.append(f'quat="{_fmt(geom.quat)}"')
    if geom.friction is not None:
        attrs.append(f'friction="{_fmt(geom.friction)}"')
    # The geom is collision-only: density 0 would zero its mass anyway, but
    # explicit <inertial> elements already own the mass properties, so MuJoCo
    # ignores geom densities here. condim 3 (the default) gives regularized
    # sliding friction, which is what foot–ground contact needs.
    return f"{pad}<geom {' '.join(attrs)}/>"


def _body_xml(
    link: Link,
    joint: Joint | None,
    links: dict[str, Link],
    children: dict[str, list[Joint]],
    indent: int,
    freejoint: Joint | None = None,
) -> str:
    """Recursively emit the MJCF ``<body>`` for ``link`` (entered via ``joint``).

    ``freejoint`` (set only on a floating base) makes the body a free 6-DOF body
    via ``<freejoint>`` instead of a fixed attachment to the world.
    """
    pad = "  " * indent
    pos = _fmt(np.asarray(joint.origin.translation)) if joint is not None else "0 0 0"
    # Carry the joint origin's rotation onto the body frame so the joint axis is
    # oriented correctly. Without this every joint frame collapses onto its
    # parent's, so a real multi-axis arm (whose URDF expresses each axis in a
    # rotated joint frame) ends up with all axes parallel — and an arm with all
    # axes along gravity cannot move (a silent, golden-hiding failure).
    quat_attr = ""
    if joint is not None:
        quat = _matrix_to_quat(np.asarray(joint.origin.rotation))
        if not np.allclose(quat, [1.0, 0.0, 0.0, 0.0]):
            quat_attr = f' quat="{_fmt(quat)}"'
    lines = [f'{pad}<body name="{link.name}" pos="{pos}"{quat_attr}>']

    if freejoint is not None:
        # A floating base: 6 DOF (3 translation + 3 rotation) to the world.
        lines.append(f'{pad}  <freejoint name="{freejoint.name}"/>')

    if joint is not None and joint.joint_type is not JointType.FIXED:
        mj_type = _MJ_JOINT.get(joint.joint_type)
        if mj_type is None:
            raise JaxterityError(
                f"Joint {joint.name!r} has type {joint.joint_type.value!r}, which "
                "is not supported by the MJCF backend yet (planar joints are future "
                "work). Use revolute/continuous/prismatic/fixed, or FLOATING for a "
                "free base."
            )
        axis = _fmt(np.asarray(joint.axis.value.xyz))
        limited = "true" if joint.joint_type is not JointType.CONTINUOUS else "false"
        rng = (
            f' range="{joint.limits.lower!r} {joint.limits.upper!r}"'
            if joint.joint_type is not JointType.CONTINUOUS
            else ""
        )
        lines.append(
            f'{pad}  <joint name="{joint.name}" type="{mj_type}" axis="{axis}" '
            f'pos="0 0 0" limited="{limited}"{rng} '
            f'damping="{float(joint.damping)!r}" '
            f'frictionloss="{float(joint.friction)!r}" '
            f'armature="{float(joint.armature)!r}"/>'
        )

    lines.append(f"{pad}  {_inertial_xml(link)}")
    for geom in _collision_geoms(link):
        lines.append(_geom_xml(geom, pad + "  "))

    for child_joint in children.get(link.name, []):
        lines.append(
            _body_xml(
                links[child_joint.child_link],
                child_joint,
                links,
                children,
                indent + 1,
            )
        )
    lines.append(f"{pad}</body>")
    return "\n".join(lines)


def to_mjcf(
    links: dict[str, Link],
    joints: dict[str, Joint],
    *,
    name: str = "robot",
    gravity: float = 9.81,
    ground_plane: bool = False,
) -> str:
    """Lower a fixed-base ``Joint``/``Link`` tree to an MJCF string.

    Args
    ----
    links : dict[str, Link]
        Links by name.
    joints : dict[str, Joint]
        Joints by name (each names its parent and child link).
    name : str
        Model name.
    gravity : float
        Gravitational acceleration magnitude (m/s²); applied along -z.
    ground_plane : bool
        Emit an infinite ``<geom name="floor" type="plane">`` at ``z = 0``
        (world frame). Links carrying :class:`CollisionGeom` entries then make
        contact with it — the foot–ground case. Default False (the
        historical geometry-free model; the handle-stable choice).

    Returns
    -------
    str
        An MJCF document parseable by ``mujoco.MjModel.from_xml_string``.

    Raises
    ------
    JaxterityError
        If the tree has no unique root link, or uses an unsupported joint type.

    Notes
    -----
    A single :class:`~jaxterity.robot.JointType.FLOATING` joint (parent
    ``"world"``, child = the root link) makes the model floating-base: the root
    body gets a ``<freejoint>`` (6 DOF, ``nq = nv + 1`` per free joint — a unit
    quaternion). The articulation tree below it is unchanged.
    """
    # A FLOATING joint attaches the root link to the world (parent "world"); it
    # is not part of the articulation adjacency.
    floating = {
        j.child_link: j for j in joints.values() if j.joint_type is JointType.FLOATING
    }
    if len(floating) > 1:
        raise JaxterityError(
            f"At most one floating base is supported, found {sorted(floating)}."
        )

    children: dict[str, list[Joint]] = {}
    child_links = set()
    for joint in joints.values():
        if joint.joint_type is JointType.FLOATING:
            continue
        children.setdefault(joint.parent_link, []).append(joint)
        child_links.add(joint.child_link)

    roots = [name_ for name_ in links if name_ not in child_links]
    if len(roots) != 1:
        raise JaxterityError(
            f"Expected exactly one root link (a link that is no joint's child), "
            f"found {roots}. Trees must have a single root (the floating base, if "
            "any). Multi-root models are future work."
        )
    if floating and roots[0] not in floating:
        raise JaxterityError(
            f"The floating joint's child {sorted(floating)} must be the root link "
            f"{roots[0]!r}; floating a non-root link is not supported."
        )

    body = _body_xml(
        links[roots[0]],
        None,
        links,
        children,
        indent=3,
        freejoint=floating.get(roots[0]),
    )
    # MuJoCo plane size: two zeros = infinite extent; the third is a render
    # half-thickness only (no dynamics effect).
    floor = '      <geom name="floor" type="plane" size="0 0 0.01"/>\n'
    worldbody = f"{floor if ground_plane else ''}{body}"
    return (
        f'<mujoco model="{name}">\n'
        # MuJoCo defaults to angle="degree"; Jaxterity is SI/radian everywhere
        # (ADR-006), so joint ranges and any rpy must be read as radians.
        f'  <compiler angle="radian"/>\n'
        f'  <option gravity="0 0 {-abs(gravity)!r}"/>\n'
        f"  <worldbody>\n{worldbody}\n  </worldbody>\n"
        f"</mujoco>\n"
    )


class MjcfWarning(UserWarning):
    """An unsupported or lossy MJCF feature was encountered during import."""


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def from_mjcf(path) -> "Robot":
    """Parse an MJCF file into a :class:`~jaxterity.robot.Robot`.

    Uses MuJoCo to parse the MJCF (no Gazebo), then reads its kinematic tree:
    each body becomes a :class:`~jaxterity.robot.Link` (mass, COM, inertia tensor
    reconstructed from MuJoCo's principal-frame representation) and each joint a
    :class:`~jaxterity.robot.Joint`. Bodies welded to a parent (no joint) become
    ``FIXED`` joints. ``free``/``ball`` joints raise :class:`MjcfWarning` and are
    skipped.

    Args
    ----
    path : str | pathlib.Path
        Path to the ``.xml`` / ``.mjcf`` model.

    Returns
    -------
    Robot
        An uncalibrated robot.

    Units / Frame
    -------------
    MuJoCo is SI; angles are read as radians (the model's own ``compiler``
    setting governs parsing). Inertia is returned as a full 3x3 tensor in the
    link body frame.
    """
    import mujoco

    from .robot import Robot

    m = mujoco.MjModel.from_xml_path(str(path))
    mj_type = {
        int(mujoco.mjtJoint.mjJNT_HINGE): None,  # REVOLUTE/CONTINUOUS by jnt_limited
        int(mujoco.mjtJoint.mjJNT_SLIDE): JointType.PRISMATIC,
    }

    links: list[Link] = []
    joints: list[Joint] = []
    for b in range(1, m.nbody):  # skip body 0 (world)
        name = m.body(b).name or f"body{b}"
        R = _quat_to_matrix(m.body_iquat[b])
        inertia = R @ np.diag(np.asarray(m.body_inertia[b])) @ R.T
        links.append(
            Link(
                name=name,
                body_frame=Frame(name),
                mass=jnp.asarray(float(m.body_mass[b])),
                com=jnp.asarray(np.asarray(m.body_ipos[b])),
                inertia=jnp.asarray(inertia),
            )
        )
        parent = m.body(int(m.body_parentid[b])).name or "world"
        if parent == "world":
            continue  # root link: welded to the world (fixed base), no joint

        origin = Transform(
            rotation=jnp.asarray(_quat_to_matrix(m.body_quat[b])),
            translation=jnp.asarray(np.asarray(m.body_pos[b])),
            source=Frame(name),
            target=Frame(parent),
        )
        if int(m.body_jntnum[b]) == 0:
            joints.append(_fixed_joint(f"{parent}_to_{name}", parent, name, origin))
            continue
        for k in range(int(m.body_jntnum[b])):
            j = int(m.body_jntadr[b]) + k
            jt = int(m.jnt_type[j])
            mapped = mj_type.get(jt)
            if jt == int(mujoco.mjtJoint.mjJNT_HINGE):
                mapped = (
                    JointType.REVOLUTE
                    if bool(m.jnt_limited[j])
                    else JointType.CONTINUOUS
                )
            if mapped is None:
                warnings.warn(
                    f"MJCF joint {m.joint(j).name!r} has unsupported type "
                    f"(free/ball); skipping it.",
                    MjcfWarning,
                    stacklevel=2,
                )
                continue
            dof = int(m.jnt_dofadr[j])
            lo, hi = (float(v) for v in m.jnt_range[j])
            joints.append(
                Joint(
                    name=m.joint(j).name or f"joint{j}",
                    joint_type=mapped,
                    parent_link=parent,
                    child_link=name,
                    axis=FramedQuantity(
                        Position(jnp.asarray(np.asarray(m.jnt_axis[j]))), Frame(parent)
                    ),
                    limits=JointLimits(lower=lo, upper=hi, effort=0.0, velocity=0.0),
                    origin=origin,
                    damping=jnp.asarray(float(m.dof_damping[dof])),
                    friction=jnp.asarray(float(m.dof_frictionloss[dof])),
                    armature=jnp.asarray(float(m.dof_armature[dof])),
                )
            )
    return Robot(
        links=tuple(links),
        joints=tuple(joints),
        name=m.names.split(b"\x00")[0].decode() or "robot",
        source=str(path),
    )


def _fixed_joint(name: str, parent: str, child: str, origin: Transform) -> Joint:
    return Joint(
        name=name,
        joint_type=JointType.FIXED,
        parent_link=parent,
        child_link=child,
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame(parent)),
        limits=JointLimits(lower=0.0, upper=0.0, effort=0.0, velocity=0.0),
        origin=origin,
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
        armature=jnp.asarray(0.0),
    )


__all__ = ["to_mjcf", "from_mjcf", "MjcfWarning"]
