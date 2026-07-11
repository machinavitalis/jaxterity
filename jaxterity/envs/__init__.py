# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Training-environment building blocks.

:mod:`jaxterity.envs.internals` puts the off-body physics (motor thermals,
the battery bus) *inside* a training rollout's step loop — the per-motor
state is carried across steps, fed back into the delivered actuation, and
optionally terminates the episode (brownout). Where the twin runs
these models *beside* a controller (the self-check co-processor), this
package runs them *inside the plant a policy trains against*.
"""

from .internals import (
    ConditionDraw,
    InternalsParams,
    InternalsState,
    actuation_scale,
    apply_actuation_scale,
    init_state,
    sample_conditions,
    step_internals,
)

__all__ = [
    "InternalsParams",
    "InternalsState",
    "ConditionDraw",
    "init_state",
    "actuation_scale",
    "step_internals",
    "apply_actuation_scale",
    "sample_conditions",
]
