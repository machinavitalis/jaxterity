# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Identifiable-parameter metadata.

Jaxterity stateful objects (``Link``, ``Joint``, actuators, sensors) are
``equinox.Module`` PyTrees whose *identifiable* parameters are JAX array leaves
and whose structural metadata is static (ADR-007). A :class:`ParameterSpec`
records which leaf fields are identifiable — the subset a system-identification
recipe is allowed to fit — separating them from fixed structural fields.

The leaves are plain JAX arrays so ``jax.grad`` / ``jax.vmap`` flow through them
directly. They are bound to ``jaxonomy.Parameter`` (so ``fit_parameters`` can
identify them) at ``Robot.to_diagram()`` time, which is the single
boundary where the robotics PyTree meets the engine's parameter system.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """The identifiable (fittable) parameter fields of an object.

    Args
    ----
    identifiable : tuple[str, ...]
        Names of the object's array-leaf fields a sysid recipe may fit. Fields
        not listed are treated as fixed.

    Examples
    --------
    >>> spec = ParameterSpec(("mass", "com", "inertia"))
    >>> "mass" in spec
    True
    >>> len(spec)
    3
    """

    identifiable: tuple[str, ...]

    def __contains__(self, name: str) -> bool:
        return name in self.identifiable

    def __len__(self) -> int:
        return len(self.identifiable)

    def __iter__(self):
        return iter(self.identifiable)


__all__ = ["ParameterSpec"]
