"""Unit tests for EvidenceValidator verifying threshold checks, fact verification, and rejection."""

from src.evidence_validator import EvidenceValidator
from src.retriever import RetrievedPassage


def test_distance_threshold_rejection():
    """Verify that passages with distance exceeding threshold are rejected."""
    validator = EvidenceValidator(max_distance=0.55)
    weak_passage = RetrievedPassage(
        chunk_id="c_weak",
        text="Gas turbine general operating principles.",
        page=1,
        printed_page="N/A",
        section="Title",
        source="GER-3620-P.pdf",
        distance=0.72,  # Exceeds 0.55
        similarity=0.28,
        score=0.28,
    )

    res = validator.validate("What is the melting point of titanium?", [weak_passage])
    assert not res.is_supported
    assert "exceeds threshold" in res.reason


def test_rejects_purchase_price_even_with_low_distance():
    """Verify that purchase price queries are rejected even if turbine passages match words."""
    validator = EvidenceValidator(max_distance=0.60)
    cost_passage = RetrievedPassage(
        chunk_id="c_cost",
        text="Maintenance costs and machine availability are two of the most important concerns to an owner.",
        page=5,
        printed_page="1",
        section="Maintenance Planning",
        source="GER-3620-P.pdf",
        distance=0.40,  # Good similarity to words
        similarity=0.60,
        score=0.65,
    )

    res = validator.validate("What is the purchase price of this gas turbine?", [cost_passage])
    assert not res.is_supported
    assert "purchase price" in res.reason.lower()


def test_rejects_melting_point_of_titanium():
    """Verify that physical constants like melting points are rejected."""
    validator = EvidenceValidator(max_distance=0.60)
    material_passage = RetrievedPassage(
        chunk_id="c_mat",
        text="The stage 1 bucket uses high temperature superalloys subject to oxidation and creep life limits.",
        page=29,
        printed_page="25",
        section="Hot Gas Path Parts",
        source="GER-3620-P.pdf",
        distance=0.45,
        similarity=0.55,
        score=0.55,
    )

    res = validator.validate("What is the melting point of titanium?", [material_passage])
    assert not res.is_supported
    assert "melting point" in res.reason.lower()


def test_validates_grounded_maintenance_factor():
    """Verify that genuine grounded passages for maintenance factor of two are accepted."""
    validator = EvidenceValidator(max_distance=0.55)
    mf_passage = RetrievedPassage(
        chunk_id="c_mf",
        text="For example, a maintenance factor of two would indicate a maintenance interval that is half of the baseline interval.",
        page=9,
        printed_page="5",
        section="Starts and Hours Criteria",
        source="GER-3620-P.pdf",
        distance=0.22,
        similarity=0.78,
        score=0.85,
    )

    res = validator.validate("What does a maintenance factor of two mean?", [mf_passage])
    assert res.is_supported
    assert res.best_passage.page == 9


def test_validates_grounded_types_of_inspection():
    """Verify that genuine grounded passages for inspection types are accepted."""
    validator = EvidenceValidator(max_distance=0.55)
    insp_passage = RetrievedPassage(
        chunk_id="c_insp",
        text="Maintenance inspection types may be broadly classified as standby, running, and disassembly inspections.",
        page=25,
        printed_page="21",
        section="Maintenance Inspections",
        source="GER-3620-P.pdf",
        distance=0.25,
        similarity=0.75,
        score=0.80,
    )

    res = validator.validate("What are the three main types of inspection?", [insp_passage])
    assert res.is_supported
    assert res.best_passage.page == 25
