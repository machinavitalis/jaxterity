# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Vendor the G1 locomotion scene + pretrained joystick policy.

Downloads, at a pinned upstream commit, the MuJoCo Playground G1 "feet-only"
locomotion model, the flat-terrain scene, and the pretrained joystick policy
(``g1_policy.onnx``, Apache-2.0), then:

1. Reduces the model to a **mesh-free** MJCF (the zoo policy: inertials drive
   dynamics) keeping everything the policy was trained against — options
   (Euler @ 2 ms, solver iterations), joint armature/damping/frictionloss
   classes, position actuators (kp per class), the two foot collision boxes,
   the floor plane, the explicit contact pairs, the pelvis IMU site +
   velocimeter/gyro sensors, and both keyframes — and writes it to
   ``jaxterity/zoo/unitree_g1/g1_walk.xml``.
2. Converts the ONNX policy to a plain ``.npz`` of MLP weights
   (``jaxterity/zoo/unitree_g1/g1_policy.npz``) so the runtime needs no ONNX
   dependency: observation normalization (mean, reciprocal std), three
   swish-activated hidden layers, and the action head (the first 29 outputs,
   tanh-squashed; the remaining 29 are the discarded log-std head).
3. Records a golden I/O set (``g1_policy_golden.npz``): random observations
   and the onnxruntime outputs, so the pure-JAX reimplementation is parity-
   tested without ONNX at test time.

Requires network access plus the ``onnx`` and ``onnxruntime`` packages (dev
tools only — the vendored artifacts have no such dependency). Re-run only to
bump the upstream pin; the outputs are committed.

Usage::

    python scripts/vendor_g1_walk_scene.py
"""

from __future__ import annotations

import io
import pathlib
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

#: Pinned upstream commit (google-deepmind/mujoco_playground, Apache-2.0).
UPSTREAM_SHA = "33f1b2843a7ec5537c4882177aa2a9f236e9b692"
_RAW = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_playground/"
    f"{UPSTREAM_SHA}/mujoco_playground"
)
MODEL_URL = f"{_RAW}/_src/locomotion/g1/xmls/g1_mjx_feetonly.xml"
SCENE_URL = f"{_RAW}/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"
POLICY_URL = f"{_RAW}/experimental/sim2sim/onnx/g1_policy.onnx"

ZOO_DIR = pathlib.Path(__file__).resolve().parents[1] / "jaxterity/zoo/unitree_g1"

#: Geoms that participate in the trained contact dynamics (everything else is
#: visual or an unused collision proxy — the model collides via explicit
#: <pair> elements only, since the g1 default class zeroes contype/conaffinity).
_CONTACT_GEOMS = {"left_foot", "right_foot", "floor"}

#: The sensors the deployed controller reads (the rest are training-only).
_KEPT_SENSORS = {"local_linvel_pelvis", "gyro_pelvis", "upvector_pelvis"}


def _fetch(url: str) -> bytes:
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _strip_visuals(parent: ET.Element) -> None:
    """Recursively drop meshes, visual/collision-proxy geoms, lights, cameras."""
    for child in list(parent):
        if child.tag in ("light", "camera"):
            parent.remove(child)
            continue
        if child.tag == "geom":
            cls = child.get("class")
            name = child.get("name")
            mesh_bound = child.get("mesh") is not None or cls == "visual"
            collision_proxy = cls == "collision" and name not in _CONTACT_GEOMS
            if mesh_bound or collision_proxy:
                parent.remove(child)
                continue
        _strip_visuals(child)


def reduce_model(model_xml: bytes, scene_xml: bytes) -> str:
    """Inline the scene into a mesh-free, self-contained locomotion MJCF."""
    root = ET.fromstring(model_xml)
    scene = ET.fromstring(scene_xml)
    root.set("model", "g1_walk")

    for asset in root.findall("asset"):
        root.remove(asset)

    # Drop the visual default class (its material no longer exists); keep the
    # collision/foot classes — the foot geoms reference them for group/size.
    for default in root.iter("default"):
        for child in list(default):
            if child.tag == "default" and child.get("class") == "visual":
                default.remove(child)

    worldbody = root.find("worldbody")
    assert worldbody is not None, "upstream model has no <worldbody>"
    _strip_visuals(worldbody)
    floor = ET.Element("geom", {"name": "floor", "type": "plane", "size": "0 0 0.01"})
    worldbody.insert(0, floor)

    # Contact pairs: keep only those whose geoms survive the reduction.
    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact):
            geoms = {pair.get("geom1"), pair.get("geom2")}
            if not geoms <= _CONTACT_GEOMS:
                contact.remove(pair)

    sensors = root.find("sensor")
    if sensors is not None:
        for sensor in list(sensors):
            if sensor.get("name") not in _KEPT_SENSORS:
                sensors.remove(sensor)

    # Keyframes (home / knees_bent qpos + ctrl) live in the scene file.
    keyframe = scene.find("keyframe")
    assert keyframe is not None, "upstream scene has no <keyframe>"
    root.append(keyframe)

    header = (
        "<!--\n"
        "  Unitree G1 locomotion scene (mesh-free), for the vendored joystick\n"
        "  walking policy. Derived at pinned commit\n"
        f"  {UPSTREAM_SHA}\n"
        "  from google-deepmind/mujoco_playground\n"
        "  _src/locomotion/g1/xmls/g1_mjx_feetonly.xml (+ the flat-terrain\n"
        "  scene keyframes), Apache-2.0 — see LICENSE-playground in this dir.\n"
        "  Reduction (scripts/vendor_g1_walk_scene.py): removed meshes, visual\n"
        "  geoms, unused collision proxies, lights, cameras, and training-only\n"
        "  sensors; kept the trained contact set (foot boxes + floor + pairs),\n"
        "  actuator/joint classes, options, the pelvis IMU sensors, and the\n"
        "  keyframes. Regenerate by re-running the script.\n"
        "-->\n"
    )
    ET.indent(root, space="  ")
    return header + ET.tostring(root, encoding="unicode") + "\n"


def convert_policy(onnx_bytes: bytes) -> dict[str, np.ndarray]:
    """Extract the MLP weights from the ONNX graph (see module docstring)."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load_from_string(onnx_bytes)
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    def by_suffix(suffix: str) -> np.ndarray:
        matches = [v for k, v in init.items() if k.endswith(suffix)]
        assert len(matches) == 1, f"expected one initializer ending {suffix!r}"
        return np.asarray(matches[0])

    w_head = by_suffix("hidden_3_1/Cast/ReadVariableOp:0")
    b_head = by_suffix("hidden_3_1/BiasAdd/ReadVariableOp:0")
    n_act = w_head.shape[1] // 2  # mean head; the log-std half is discarded
    return {
        "obs_mean": by_suffix("sub/ReadVariableOp:0"),
        "obs_recip_std": by_suffix("truediv_recip:0"),
        "w0": by_suffix("hidden_0_1/Cast/ReadVariableOp:0"),
        "b0": by_suffix("hidden_0_1/BiasAdd/ReadVariableOp:0"),
        "w1": by_suffix("hidden_1_1/Cast/ReadVariableOp:0"),
        "b1": by_suffix("hidden_1_1/BiasAdd/ReadVariableOp:0"),
        "w2": by_suffix("hidden_2_1/Cast/ReadVariableOp:0"),
        "b2": by_suffix("hidden_2_1/BiasAdd/ReadVariableOp:0"),
        "w_out": w_head[:, :n_act],
        "b_out": b_head[:n_act],
    }


def golden_io(onnx_bytes: bytes, n_obs: int, obs_dim: int) -> dict[str, np.ndarray]:
    """Run onnxruntime on seeded random observations for the parity golden."""
    import onnxruntime

    session = onnxruntime.InferenceSession(
        onnx_bytes, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(20260610)
    obs = rng.standard_normal((n_obs, obs_dim)).astype(np.float32)
    actions = np.stack(
        [session.run(["continuous_actions"], {"obs": o[None, :]})[0][0] for o in obs]
    )
    return {"obs": obs, "actions": actions}


def main() -> None:
    model_xml = _fetch(MODEL_URL)
    scene_xml = _fetch(SCENE_URL)
    onnx_bytes = _fetch(POLICY_URL)

    mjcf = reduce_model(model_xml, scene_xml)
    (ZOO_DIR / "g1_walk.xml").write_text(mjcf)
    print(f"wrote {ZOO_DIR / 'g1_walk.xml'} ({len(mjcf)} chars)")

    weights = convert_policy(onnx_bytes)
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **weights)
        (ZOO_DIR / "g1_policy.npz").write_bytes(buf.getvalue())
    shapes = {k: v.shape for k, v in weights.items()}
    print(f"wrote {ZOO_DIR / 'g1_policy.npz'}: {shapes}")

    golden = golden_io(onnx_bytes, n_obs=8, obs_dim=weights["obs_mean"].shape[0])
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **golden)
        (ZOO_DIR / "g1_policy_golden.npz").write_bytes(buf.getvalue())
    print(f"wrote {ZOO_DIR / 'g1_policy_golden.npz'}")

    # Sanity: the reduced scene must compile and keep the trained topology.
    import mujoco

    m = mujoco.MjModel.from_xml_string(mjcf)
    assert (m.nq, m.nv, m.nu) == (36, 35, 29), (m.nq, m.nv, m.nu)
    assert m.ngeom == 3, m.ngeom
    mujoco.mj_resetDataKeyframe(m, mujoco.MjData(m), m.keyframe("knees_bent").id)
    print(f"scene OK: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom}")


if __name__ == "__main__":
    main()
