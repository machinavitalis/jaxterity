# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""URDF import/export.

``from_urdf`` parses the standard URDF subset Jaxterity needs — links with
inertials and revolute / continuous / prismatic / fixed joints — into a
:class:`~jaxterity.robot.Robot`. URDF is already SI (metres, radians, kg), so no
unit conversion is needed; only the joint-origin ``rpy`` is converted to a
rotation matrix. Unsupported features (floating/planar joints, transmissions,
gazebo tags) raise a structured :class:`UrdfWarning` rather than being dropped
silently. ``to_urdf`` writes a robot back out (semantic round-trip).

A small stdlib (``xml.etree``) parser is used deliberately — the subset is
narrow and this keeps URDF import dependency-free (no trimesh/networkx pull-in).
Mesh geometry is kept as an opaque handle (the source filename), not loaded.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np

from ..errors import JaxterityError
from .frame import Frame, FramedQuantity, Position, Transform
from .robot import Robot
from .structure import Joint, JointLimits, JointType, Link

_URDF_JOINT_TYPE = {
    "revolute": JointType.REVOLUTE,
    "continuous": JointType.CONTINUOUS,
    "prismatic": JointType.PRISMATIC,
    "fixed": JointType.FIXED,
}


class UrdfWarning(UserWarning):
    """An unsupported or lossy URDF feature was encountered during import."""


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF roll-pitch-yaw (extrinsic XYZ) to a 3x3 rotation matrix."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`_rpy_to_matrix` (one valid rpy triple)."""
    rot = np.asarray(rot)
    p = math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0]))
    r = math.atan2(rot[2, 1], rot[2, 2])
    y = math.atan2(rot[1, 0], rot[0, 0])
    return r, p, y


def _floats(text: str | None, default) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def _parse_link(elem: ET.Element) -> Link:
    name = elem.get("name")
    if name is None:
        raise JaxterityError("URDF <link> is missing a 'name' attribute.")
    inertial = elem.find("inertial")
    if inertial is None:
        warnings.warn(
            f"URDF link {name!r} has no <inertial>; defaulting to a tiny mass. "
            "Fixed/massless links are fine; a moving link without inertia is a "
            "modelling error.",
            UrdfWarning,
            stacklevel=2,
        )
        mass, com, inertia = 1e-9, np.zeros(3), np.eye(3) * 1e-12
    else:
        origin = inertial.find("origin")
        com = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
        mass_elem = inertial.find("mass")
        assert mass_elem is not None, f"URDF link {name!r} <inertial> missing <mass>"
        mass = float(mass_elem.get("value") or 0.0)
        i = inertial.find("inertia")
        assert i is not None, f"URDF link {name!r} <inertial> missing <inertia>"
        ixx = float(i.get("ixx") or 0.0)
        iyy = float(i.get("iyy") or 0.0)
        izz = float(i.get("izz") or 0.0)
        ixy, ixz, iyz = (
            float(i.get("ixy", 0.0)),
            float(i.get("ixz", 0.0)),
            float(i.get("iyz", 0.0)),
        )
        inertia = np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float
        )

    def _geom_handle(tag: str):
        node = elem.find(tag)
        if node is None:
            return None
        geom = node.find("geometry")
        mesh = geom.find("mesh") if geom is not None else None
        return mesh.get("filename") if mesh is not None else "primitive"

    return Link(
        name=name,
        body_frame=Frame(name),
        mass=jnp.asarray(mass),
        com=jnp.asarray(com),
        inertia=jnp.asarray(inertia),
        visual=_geom_handle("visual"),
        collision=_geom_handle("collision"),
    )


def _parse_joint(elem: ET.Element) -> Joint | None:
    name = elem.get("name")
    if name is None:
        raise JaxterityError("URDF <joint> is missing a 'name' attribute.")
    urdf_type = elem.get("type")
    jtype = _URDF_JOINT_TYPE.get(urdf_type) if urdf_type is not None else None
    if jtype is None:
        warnings.warn(
            f"URDF joint {name!r} has unsupported type {urdf_type!r} "
            "(floating/planar and others are not yet supported); skipping it.",
            UrdfWarning,
            stacklevel=2,
        )
        return None
    parent_elem = elem.find("parent")
    child_elem = elem.find("child")
    assert parent_elem is not None and child_elem is not None, (
        f"URDF joint {name!r} missing <parent> or <child>"
    )
    parent = parent_elem.get("link")
    child = child_elem.get("link")
    assert parent is not None and child is not None, (
        f"URDF joint {name!r} <parent>/<child> missing 'link' attribute"
    )
    origin = elem.find("origin")
    xyz = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
    rpy = _floats(origin.get("rpy") if origin is not None else None, [0, 0, 0])
    axis_elem = elem.find("axis")
    axis = _floats(axis_elem.get("xyz") if axis_elem is not None else None, [1, 0, 0])
    limit = elem.find("limit")
    limits = JointLimits(
        lower=float(limit.get("lower", -3.141592653589793))
        if limit is not None
        else -3.141592653589793,
        upper=float(limit.get("upper", 3.141592653589793))
        if limit is not None
        else 3.141592653589793,
        effort=float(limit.get("effort", 0.0)) if limit is not None else 0.0,
        velocity=float(limit.get("velocity", 0.0)) if limit is not None else 0.0,
    )
    dynamics = elem.find("dynamics")
    damping = float(dynamics.get("damping", 0.0)) if dynamics is not None else 0.0
    friction = float(dynamics.get("friction", 0.0)) if dynamics is not None else 0.0

    return Joint(
        name=name,
        joint_type=jtype,
        parent_link=parent,
        child_link=child,
        axis=FramedQuantity(Position(jnp.asarray(axis)), Frame(parent)),
        limits=limits,
        origin=Transform(
            rotation=jnp.asarray(_rpy_to_matrix(rpy)),
            translation=jnp.asarray(xyz),
            source=Frame(child),
            target=Frame(parent),
        ),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(friction),
        # URDF has no armature/rotor-inertia element; identifiable, default 0.
        armature=jnp.asarray(0.0),
    )


def from_urdf(path) -> Robot:
    """Parse a URDF file into a :class:`~jaxterity.robot.Robot`.

    Args
    ----
    path : str | pathlib.Path
        Path to the ``.urdf`` file.

    Returns
    -------
    Robot
        An uncalibrated robot. Joints of unsupported type and links without
        inertials raise a :class:`UrdfWarning` (never a silent drop).

    Raises
    ------
    JaxterityError
        If the file is not a ``<robot>`` document or a link lacks a name.

    Units / Frame
    -------------
    URDF is SI (m, rad, kg); values are taken as-is. Joint-origin ``rpy`` is
    converted to a rotation matrix.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise JaxterityError(
            f"Expected a URDF <robot> root element, got <{root.tag}> in {path}."
        )
    links = tuple(_parse_link(e) for e in root.findall("link"))
    joints = tuple(
        j for j in (_parse_joint(e) for e in root.findall("joint")) if j is not None
    )
    return Robot(
        links=links, joints=joints, name=root.get("name", "robot"), source=str(path)
    )


def to_urdf(robot: Robot, path) -> None:
    """Write a :class:`~jaxterity.robot.Robot` back out as a URDF file.

    Round-trips semantically: ``from_urdf`` of the output reproduces the robot's
    links, joints, and identifiable parameters.
    """
    root = ET.Element("robot", name=robot.name)
    for link in robot.links:
        le = ET.SubElement(root, "link", name=link.name)
        inertial = ET.SubElement(le, "inertial")
        ET.SubElement(inertial, "origin", xyz=_fmt(link.com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=repr(float(link.mass)))
        ix = np.asarray(link.inertia)
        ET.SubElement(
            inertial,
            "inertia",
            ixx=repr(float(ix[0, 0])),
            iyy=repr(float(ix[1, 1])),
            izz=repr(float(ix[2, 2])),
            ixy=repr(float(ix[0, 1])),
            ixz=repr(float(ix[0, 2])),
            iyz=repr(float(ix[1, 2])),
        )
    for joint in robot.joints:
        je = ET.SubElement(
            root, "joint", name=joint.name, type=_urdf_type_name(joint.joint_type)
        )
        ET.SubElement(je, "parent", link=joint.parent_link)
        ET.SubElement(je, "child", link=joint.child_link)
        r, p, y = _matrix_to_rpy(np.asarray(joint.origin.rotation))
        ET.SubElement(
            je,
            "origin",
            xyz=_fmt(joint.origin.translation),
            rpy=f"{r!r} {p!r} {y!r}",
        )
        ET.SubElement(je, "axis", xyz=_fmt(joint.axis.value.xyz))
        ET.SubElement(
            je,
            "limit",
            lower=repr(joint.limits.lower),
            upper=repr(joint.limits.upper),
            effort=repr(joint.limits.effort),
            velocity=repr(joint.limits.velocity),
        )
        ET.SubElement(
            je,
            "dynamics",
            damping=repr(float(joint.damping)),
            friction=repr(float(joint.friction)),
        )
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


def _fmt(vec) -> str:
    return " ".join(repr(float(x)) for x in np.asarray(vec).ravel())


def _urdf_type_name(jtype: JointType) -> str:
    for name, value in _URDF_JOINT_TYPE.items():
        if value is jtype:
            return name
    raise JaxterityError(f"No URDF name for joint type {jtype!r}.")


__all__ = ["from_urdf", "to_urdf", "UrdfWarning"]
