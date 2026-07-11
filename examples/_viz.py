# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Rendering helpers for the Jaxterity tutorial GIFs.

Jaxterity's ``Robot.to_mjcf`` emits ``<inertial>`` elements but no visual
``<geom>``, so a raw render shows an empty scene. These helpers inject *cosmetic*
proxy geometry (capsules linking each body to its children, a sphere at each
leaf) without touching masses or inertias, then add a ground plane, lights, a
camera, and materials. The physics is unchanged: rendering reads ``data.xpos``,
which is driven entirely by the inertials.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

_LINK_RGBA = "0.20 0.45 0.78 1"  # blue links
_JOINT_RGBA = "0.93 0.55 0.10 1"  # orange joint markers
_TIP_RGBA = "0.85 0.20 0.25 1"  # red end-effector tip


def _children(body: ET.Element) -> list[ET.Element]:
    return [c for c in body if c.tag == "body"]


def add_proxy_geoms(
    mjcf: str,
    *,
    link_radius: float = 0.010,
    joint_radius: float = 0.014,
    tip_radius: float = 0.018,
) -> str:
    """Return ``mjcf`` with cosmetic proxy geoms added to every body.

    A capsule is drawn from each body origin toward each child body origin (so
    the kinematic chain reads as a connected arm); a small sphere marks each
    joint location, and a larger sphere marks each leaf (end-effector tip). All
    proxy geoms are ``contype="0" conaffinity="0"`` (visual only, no contact)
    and carry no mass, so dynamics are untouched.
    """
    root = ET.fromstring(mjcf)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")

    def decorate(body: ET.Element) -> None:
        kids = _children(body)
        # Joint marker sphere at this body's origin (where its joint sits).
        if body.find("joint") is not None:
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "sphere",
                    "size": f"{joint_radius}",
                    "rgba": _JOINT_RGBA,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
        if not kids:
            # Leaf body: mark the tip at the inertial COM if present, else origin.
            # Skip negligible-mass frame stubs (e.g. a 1e-9 kg gripper frame) so
            # they don't clutter the end-effector.
            inertial = body.find("inertial")
            pos = inertial.get("pos", "0 0 0") if inertial is not None else "0 0 0"
            mass = float(inertial.get("mass", "1")) if inertial is not None else 1.0
            if mass > 1e-6:
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "type": "sphere",
                        "pos": pos,
                        "size": f"{tip_radius}",
                        "rgba": _TIP_RGBA,
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "1",
                    },
                )
        for child in kids:
            cpos = np.array([float(v) for v in child.get("pos", "0 0 0").split()])
            if np.linalg.norm(cpos) < 1e-6:
                continue  # coincident: nothing to draw a segment for
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "capsule",
                    "fromto": f"0 0 0 {cpos[0]} {cpos[1]} {cpos[2]}",
                    "size": f"{link_radius}",
                    "rgba": _LINK_RGBA,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
            decorate(child)

    for body in _children(worldbody):
        decorate(body)

    return ET.tostring(root, encoding="unicode")


def add_scene(
    mjcf: str,
    *,
    floor: bool = True,
    camera: tuple[float, float, float, str] | None = None,
    offwidth: int = 960,
    offheight: int = 960,
) -> str:
    """Add ``<visual>``, ``<asset>`` materials, lights, an optional floor and a
    named camera ``"showcase"`` to a (proxy-decorated) MJCF string."""
    root = ET.fromstring(mjcf)

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "global", {"offwidth": str(offwidth), "offheight": str(offheight)}
    )
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "8"})
    ET.SubElement(
        visual, "headlight", {"ambient": "0.4 0.4 0.4", "diffuse": "0.5 0.5 0.5"}
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.18 0.20 0.24",
            "rgb2": "0.24 0.27 0.31",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "grid", "texture": "grid", "texrepeat": "6 6", "reflectance": "0.15"},
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sky",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.55 0.62 0.72",
            "rgb2": "0.10 0.12 0.16",
            "width": "256",
            "height": "256",
        },
    )

    worldbody = root.find("worldbody")
    # Raked key light + soft fill.
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key",
            "pos": "1.0 -1.0 1.5",
            "dir": "-0.6 0.6 -1",
            "directional": "true",
            "diffuse": "0.7 0.7 0.7",
            "specular": "0.2 0.2 0.2",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill",
            "pos": "-1.0 -0.5 0.8",
            "dir": "0.6 0.3 -0.6",
            "directional": "true",
            "diffuse": "0.3 0.3 0.35",
        },
    )
    if floor:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "2 2 0.05",
                "material": "grid",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    if camera is not None:
        cx, cy, cz, xyaxes = camera
        ET.SubElement(
            worldbody,
            "camera",
            {"name": "showcase", "pos": f"{cx} {cy} {cz}", "xyaxes": xyaxes},
        )

    return ET.tostring(root, encoding="unicode")
