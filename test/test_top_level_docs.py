# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Anti-drift tests for the top-level CLAIMS.md / KNOWN_GAPS.md contract docs.

CLAIMS.md says what Jaxterity guarantees today (mapped to evidence);
KNOWN_GAPS.md says what it explicitly does not yet do. They are symmetric —
a capability that lands moves text from gaps to claims. These tests catch the
mechanical drift modes: a file gets deleted, a cross-reference breaks, the
README stops pointing at the contract, or the gap-anchor phrases disappear.

Deliberately file-content-only (no ``import jaxterity``) so the contract stays
checkable without the optional MJX / MuJoCo stack installed.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAIMS = REPO / "CLAIMS.md"
KNOWN_GAPS = REPO / "KNOWN_GAPS.md"
README = REPO / "README.md"


def test_claims_file_exists():
    assert CLAIMS.exists(), "CLAIMS.md missing at repo root"


def test_known_gaps_file_exists():
    assert KNOWN_GAPS.exists(), "KNOWN_GAPS.md missing at repo root"


def test_claims_cross_references_known_gaps():
    assert "KNOWN_GAPS.md" in CLAIMS.read_text(), (
        "CLAIMS.md must cross-reference KNOWN_GAPS.md so readers see both "
        "halves of the contract."
    )


def test_known_gaps_cross_references_claims():
    assert "CLAIMS.md" in KNOWN_GAPS.read_text(), (
        "KNOWN_GAPS.md must cross-reference CLAIMS.md so readers see both "
        "halves of the contract."
    )


def test_readme_cross_references_both_contract_docs():
    text = README.read_text()
    assert "CLAIMS.md" in text, "README must point at CLAIMS.md"
    assert "KNOWN_GAPS.md" in text, "README must point at KNOWN_GAPS.md"


def test_both_docs_carry_doc_drift_banner():
    for path in (CLAIMS, KNOWN_GAPS):
        text = path.read_text().lower()
        assert "doc-drift rule" in text, f"{path.name} lost the doc-drift banner"
        assert "test_top_level_docs.py" in text, (
            f"{path.name} must name this test so the rule stays discoverable."
        )


def test_known_gaps_keeps_gap_anchor_phrases():
    """KNOWN_GAPS uses canonical status phrases as grep anchors for the next
    agent looking for a gap to close. Losing all of them means the file has
    eroded into untracked prose."""
    text = KNOWN_GAPS.read_text().lower()
    canonical = ("not yet", "partial", "known limitation", "experimental")
    assert any(phrase in text for phrase in canonical), (
        "KNOWN_GAPS.md lost every canonical gap-status phrase."
    )


def test_known_gaps_names_the_deployment_and_mjx_caveats():
    """The two highest-surprise gaps must be named: deployment is owned by
    Jaxility (not this repo), and some policy paths bypass the MJX solver."""
    text = KNOWN_GAPS.read_text()
    assert "Jaxility" in text, "KNOWN_GAPS must name Jaxility as the deployment owner"
    assert "MJX" in text, "KNOWN_GAPS must name the MJX-path caveat"


def test_known_gaps_marks_jaxonomy_primitives_out_of_scope():
    """ADR-002: general simulation primitives are imported from Jaxonomy, not
    re-implemented here. The 'out of scope' framing must survive so a reader
    does not mistake it for a missing feature."""
    text = KNOWN_GAPS.read_text()
    assert "ADR-002" in text
    assert "Jaxonomy" in text
