# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Browser output surface — export a Jaxterity demo to a static web bundle.

Rerun and Mermaid are Jaxterity's read-only visual surfaces (CONTEXT.md); this
adds a third: a self-contained, dependency-free web page that runs a trained
Jaxterity policy on a calibrated robot **in the browser**. It is the producer
side of the browser-runnable demo on MuJoCo-WASM.

:func:`~jaxterity.web.bundle.write_bundle` emits a directory of static files
(`index.html`, the policy as JSON, an MJCF) that needs no server beyond a static
file host. The reference page (the nanojaxterity cartpole) runs two physics
backends: a tiny **closed-form** integrator that ships in the page (so the demo
runs anywhere, today) and an optional **MuJoCo-WASM** backend that simulates the
exported MJCF at full fidelity once the user drops a MuJoCo-WASM build in (see
the bundle's ``MUJOCO_WASM.md``). The trained policy is ported to a few lines of
JavaScript whose action choice is verified equal to the JAX policy
(``test/web/test_policy_parity.py``, run under Node).
"""

from __future__ import annotations

from . import bundle

__all__ = ["bundle"]
