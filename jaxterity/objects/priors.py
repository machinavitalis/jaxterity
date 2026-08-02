# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Semantic priors: a guess about *what* an object is, turned into search bounds.

Inertial identification is a well-conditioned problem with badly-behaved local
minima: started from an arbitrary point, a Levenberg-Marquardt fit for a
payload's mass and centre of mass can settle far from the truth, and the fit
reports a small residual either way. The cheapest escape is information nobody
has to measure — a vision-language model looking at the grasped object already
"knows" it is a steel wrench rather than a foam replica of one, and that
sentence bounds the mass to within an order of magnitude before the arm moves.

This module turns that sentence into two numbers the fit can use: a **seed**
(where :func:`~jaxterity.objects.identify_payload` starts) and a **box**
(where it is allowed to go). It composes with — never replaces — the lane's
existing honesty guards: the Fisher identifiability report still runs before
the fit, physical non-negativity still projects during it, and the covariance
flags still fire after it. A prior narrows the search; it does not certify the
answer.

Honesty rule, enforced here rather than left to the caller: **a fit seeded by a
guess records that it was.** :func:`~jaxterity.objects.identify_payload` names
the prior in its result's ``recipe_name`` (so the seeded provenance reaches the
attestation handle) and in a declaration note carrying the label, the source,
and the bounds. If the estimate then lands *on* one of those bounds, the result
says so too: that value is the guess, not the measurement.

No model is called from here. A :class:`SemanticPrior` is a plain declared
value — whoever produced it (a VLM, a barcode lookup, an operator) names
themselves in ``source``.

Units
-----
kg, m, kg·m² throughout; density in kg/m³.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from ..errors import JaxterityError
from .payload import RigidPayload, _fit_layout

#: Rough density ranges (kg/m³) for the materials a vision model actually names,
#: deliberately wide: the point of the prior is an order-of-magnitude bracket
#: nobody has to measure, not a materials-science claim. ``"unknown"`` is the
#: honest fallback — it still excludes "a 40 kg wrench" and little else.
MATERIAL_DENSITY_KG_M3: dict[str, tuple[float, float]] = {
    "foam": (15.0, 200.0),
    "cardboard_box": (30.0, 200.0),
    "wood": (350.0, 900.0),
    "plastic": (850.0, 1450.0),
    "water": (990.0, 1010.0),
    "rubber": (1100.0, 1600.0),
    "glass": (2400.0, 2800.0),
    "aluminum": (2600.0, 2850.0),
    "steel": (7600.0, 8100.0),
    "brass": (8300.0, 8800.0),
    "unknown": (50.0, 3000.0),
}

#: Default bracket on how much of the named bounding box is actually material —
#: a bottle is mostly its contents, a wrench nearly fills its box. Wide on
#: purpose; a caller who knows better passes their own.
_DEFAULT_FILL = (0.4, 1.0)


@dataclasses.dataclass(frozen=True)
class SemanticPrior:
    """A declared semantic guess about a held object: a seed and a search box.

    Args
    ----
    label : str
        What the object is believed to be, verbatim as stated (e.g.
        ``"steel adjustable wrench, ~200 mm"``). Reported, never parsed.
    mass_kg : tuple[float, float]
        Plausible mass bracket ``(lo, hi)``, kg. Both > 0, ``lo <= hi``. The
        fit starts at the geometric mean and is confined to the bracket.
    source : str
        Who produced the guess — a model id, a lookup, an operator name. This
        is the provenance the seeded fit records.
    com_halfwidth_m : tuple[float, float, float] | None
        Per-axis bound on how far the centre of mass may sit from the payload
        frame's origin (the grasp point), m. ``None`` leaves the CoM
        unbounded — the fit is then seeded but not boxed in those components.
    inertia_max : tuple[float, float, float] | None
        Per-axis upper bound on the principal moments, kg·m². ``None`` leaves
        them bounded below by physical non-negativity only.
    confidence : float
        The producer's stated confidence in ``label``, in ``[0, 1]``. Carried
        into the declaration; it does not scale the bounds (a half-confident
        guess with a wide bracket is the honest encoding — widen the bracket,
        do not lower the number).

    Raises
    ------
    JaxterityError
        If the mass bracket is not positive and ordered, if a half-width or
        inertia bound is negative, or if ``confidence`` is outside ``[0, 1]``.

    Frame
    -----
    ``com_halfwidth_m`` is in the payload's own body frame — the same frame
    :class:`~jaxterity.objects.RigidPayload.com` lives in, whose origin is the
    grasp point.

    Units
    -----
    kg, m, kg·m².

    Examples
    --------
    >>> prior = SemanticPrior(
    ...     label="full 500 ml plastic water bottle",
    ...     mass_kg=(0.45, 0.60),
    ...     source="vlm/claude-opus-5",
    ...     com_halfwidth_m=(0.03, 0.03, 0.12),
    ... )
    >>> round(prior.seed_mass, 3)
    0.52
    """

    label: str
    mass_kg: tuple[float, float]
    source: str
    com_halfwidth_m: tuple[float, float, float] | None = None
    inertia_max: tuple[float, float, float] | None = None
    confidence: float = 0.5

    def __post_init__(self) -> None:
        lo, hi = (float(self.mass_kg[0]), float(self.mass_kg[1]))
        if not (lo > 0.0 and hi >= lo):
            raise JaxterityError(
                f"SemanticPrior({self.label!r}): mass_kg must be a positive, "
                f"ordered bracket (lo, hi); got ({lo!r}, {hi!r}). The lower "
                "bound seeds the fit's linearization point, so it cannot be "
                "zero — widen the bracket instead of dropping it."
            )
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise JaxterityError(
                f"SemanticPrior({self.label!r}): confidence must be in [0, 1]; "
                f"got {self.confidence!r}."
            )
        for field, values in (
            ("com_halfwidth_m", self.com_halfwidth_m),
            ("inertia_max", self.inertia_max),
        ):
            if values is None:
                continue
            if len(values) != 3 or any(float(v) < 0.0 for v in values):
                raise JaxterityError(
                    f"SemanticPrior({self.label!r}): {field} must be three "
                    f"non-negative per-axis bounds; got {values!r}."
                )

    @classmethod
    def from_material(
        cls,
        label: str,
        material: str,
        extent_m: tuple[float, float, float],
        *,
        source: str,
        fill: tuple[float, float] = _DEFAULT_FILL,
        confidence: float = 0.5,
    ) -> "SemanticPrior":
        """Build a prior from the two things a vision model can actually say.

        A material name and a rough bounding box are the realistic output of
        "look at the thing in the gripper". This converts them into a mass
        bracket (density range × box volume × fill bracket), a CoM box (the
        centre of mass of a rigid object lies inside its own bounding box), and
        an upper bound on the principal moments (a solid box at the heaviest
        plausible mass).

        Args
        ----
        label : str
            The guess as stated, for the record.
        material : str
            A key of :data:`MATERIAL_DENSITY_KG_M3`. Use ``"unknown"`` rather
            than inventing a density.
        extent_m : tuple[float, float, float]
            Bounding-box side lengths, m. All > 0.
        source : str
            Who produced the guess.
        fill : tuple[float, float]
            Bracket on the fraction of the box that is material,
            dimensionless. Default :data:`_DEFAULT_FILL`.
        confidence : float
            The producer's confidence in ``label``, ``[0, 1]``.

        Returns
        -------
        SemanticPrior
            The seed + box.

        Raises
        ------
        JaxterityError
            On an unknown material, a non-positive extent, or a fill bracket
            outside ``(0, 1]``.

        Units
        -----
        m in; kg, m, kg·m² out.

        Examples
        --------
        >>> p = SemanticPrior.from_material(
        ...     "aluminum block", "aluminum", (0.05, 0.04, 0.03),
        ...     source="vlm/claude-opus-5")
        >>> round(p.mass_kg[0], 3), round(p.mass_kg[1], 3)
        (0.062, 0.171)
        """
        if material not in MATERIAL_DENSITY_KG_M3:
            raise JaxterityError(
                f"SemanticPrior.from_material: unknown material {material!r}. "
                f"Known: {sorted(MATERIAL_DENSITY_KG_M3)}. Use 'unknown' (a "
                "deliberately wide bracket) rather than inventing a density."
            )
        extents = tuple(float(e) for e in extent_m)
        if len(extents) != 3 or any(e <= 0.0 for e in extents):
            raise JaxterityError(
                f"SemanticPrior.from_material: extent_m must be three positive "
                f"side lengths in m; got {extent_m!r}."
            )
        fill_lo, fill_hi = float(fill[0]), float(fill[1])
        if not (0.0 < fill_lo <= fill_hi <= 1.0):
            raise JaxterityError(
                f"SemanticPrior.from_material: fill must satisfy "
                f"0 < lo <= hi <= 1; got {fill!r}."
            )
        rho_lo, rho_hi = MATERIAL_DENSITY_KG_M3[material]
        volume = extents[0] * extents[1] * extents[2]
        mass_lo = rho_lo * volume * fill_lo
        mass_hi = rho_hi * volume * fill_hi
        # A solid box at the heaviest plausible mass bounds every principal
        # moment: I_x = m (b² + c²) / 12, etc.
        m = mass_hi
        a, b, c = extents
        inertia_max = (
            m * (b * b + c * c) / 12.0,
            m * (a * a + c * c) / 12.0,
            m * (a * a + b * b) / 12.0,
        )
        return cls(
            label=label,
            mass_kg=(mass_lo, mass_hi),
            source=source,
            com_halfwidth_m=(a / 2.0, b / 2.0, c / 2.0),
            inertia_max=inertia_max,
            confidence=confidence,
        )

    @property
    def seed_mass(self) -> float:
        """The bracket's geometric mean, kg — the fit's starting mass.

        Geometric rather than arithmetic because the bracket is an
        order-of-magnitude statement: the midpoint of ``(0.05, 5.0)`` is
        ``0.5``, not ``2.5``.
        """
        lo, hi = float(self.mass_kg[0]), float(self.mass_kg[1])
        return float(np.sqrt(lo * hi))

    def seed(self, payload: RigidPayload) -> RigidPayload:
        """Return the payload re-seeded at this prior (structure preserved).

        Replaces the payload's prior mass with :attr:`seed_mass` and its
        inertia with a solid-box inertia at that mass (when the prior carries
        moment bounds), keeping the attach link, name, grasp offset, and CoM
        prior. This is the point the fit linearizes at *and* starts from.

        Args
        ----
        payload : RigidPayload
            The object as declared by the caller.

        Returns
        -------
        RigidPayload
            A new payload; the input is unchanged.

        Units
        -----
        kg, m, kg·m².
        """
        mass = self.seed_mass
        if self.inertia_max is None:
            scale = mass / max(float(payload.mass), np.finfo(float).tiny)
            inertia = payload.inertia * scale
        else:
            # Mid-bracket moments: the max bound scaled to the seed mass.
            factor = mass / float(self.mass_kg[1])
            inertia = jnp.diag(
                jnp.asarray([float(i) * factor for i in self.inertia_max])
            )
        return RigidPayload(
            payload.attach_link,
            mass=mass,
            com=payload.com,
            inertia=inertia,
            name=payload.name,
            offset=payload.offset,
        )

    def bounds(
        self, payload: RigidPayload, fit: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-scalar search box for a fit vector, in ``_fit_layout`` order.

        Args
        ----
        payload : RigidPayload
            The payload being fitted (supplies the parameter layout and the
            CoM prior the half-widths are centred on).
        fit : tuple[str, ...]
            The fit groups, a subset of ``("mass", "com", "inertia_diag")``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(lower, upper)`` per fitted scalar. Unbounded entries are
            ``∓inf``; ``mass`` and ``inertia_diag`` keep their physical
            non-negativity floor even where the prior says nothing.

        Units
        -----
        kg, m, kg·m².
        """
        names, prior_vec, physical_lower = _fit_layout(payload, tuple(fit))
        lower = np.array(physical_lower, dtype=float)
        upper = np.full(len(names), np.inf)
        k = 0
        if "mass" in fit:
            lower[k] = max(lower[k], float(self.mass_kg[0]))
            upper[k] = float(self.mass_kg[1])
            k += 1
        if "com" in fit:
            if self.com_halfwidth_m is not None:
                for i in range(3):
                    half = float(self.com_halfwidth_m[i])
                    lower[k + i] = prior_vec[k + i] - half
                    upper[k + i] = prior_vec[k + i] + half
            k += 3
        if "inertia_diag" in fit:
            if self.inertia_max is not None:
                for i in range(3):
                    upper[k + i] = float(self.inertia_max[i])
            k += 3
        return lower, upper

    def declaration(self) -> str:
        """One line naming the prior, its source, and the bracket it imposed.

        The string :func:`~jaxterity.objects.identify_payload` records so a
        seeded fit is legible as seeded.
        """
        com = (
            "unbounded"
            if self.com_halfwidth_m is None
            else "±("
            + ", ".join(f"{float(h):.3g}" for h in self.com_halfwidth_m)
            + ") m"
        )
        return (
            f"semantic prior {self.label!r} (source {self.source!r}, confidence "
            f"{float(self.confidence):.2f}): mass ∈ [{float(self.mass_kg[0]):.3g}, "
            f"{float(self.mass_kg[1]):.3g}] kg, CoM {com}"
        )


__all__ = ["SemanticPrior", "MATERIAL_DENSITY_KG_M3"]
