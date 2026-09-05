"""
Comprehensive Test Suite for LeaseGuard Review Assistant
Track 5: Legal Operations
"""

import os
from pathlib import Path
from fastapi.testclient import TestClient

from app import app
from leaseguard.models import ReviewReport
from leaseguard.parser import LeaseParser
from leaseguard.retriever import StandardPositionRetriever
from leaseguard.analyzer import LeaseAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"


def test_standard_positions_definitions():
    retriever = StandardPositionRetriever()
    assert len(retriever.positions) == 8
    expected_ids = {"SP-1", "SP-2", "SP-3", "SP-4", "SP-5", "SP-6", "SP-7", "SP-8"}
    assert set(retriever.positions.keys()) == expected_ids

    for sp_id, pos in retriever.positions.items():
        assert pos.id.startswith("SP-")
        assert len(pos.category) > 0
        assert len(pos.title) > 0
        assert len(pos.summary) > 0
        assert len(pos.standard_terms) > 0
        assert len(pos.acceptable_range) > 0
        assert len(pos.silence_risk) > 0
        assert len(pos.standard_clause_text) > 0


def test_clause_parser():
    compliant_path = SAMPLES_DIR / "sample_compliant_lease.txt"
    with open(compliant_path, "r", encoding="utf-8") as f:
        text = f.read()

    title, meta = LeaseParser.extract_title_and_metadata(text)
    clauses = LeaseParser.parse_clauses(text)

    assert "COMMERCIAL LEASE AGREEMENT" in title
    assert len(clauses) == 8
    assert clauses[0].number == "1"
    assert "Rent" in clauses[0].title
    assert len(clauses[0].text) > 50


def test_compliant_lease_is_clean():
    """
    TASK.md rule: 'A clean agreement must come back clean; the system must not manufacture
    findings to look useful.'
    """
    compliant_path = SAMPLES_DIR / "sample_compliant_lease.txt"
    with open(compliant_path, "r", encoding="utf-8") as f:
        text = f.read()

    analyzer = LeaseAnalyzer()
    report = analyzer.review_agreement(text)

    assert report.overall_status == "COMPLIANT"
    assert report.departures_count == 0
    assert report.missing_count == 0
    assert report.matches_count == 8
    assert len(report.clause_findings) == 8

    for finding in report.clause_findings:
        assert finding.status == "MATCHES"
        assert finding.risk_level == "NONE"
        assert finding.departure_details is None
        assert finding.recommended_amendment is None
        assert finding.standard_position_id.startswith("SP-")


def test_departures_lease_flags_all_violations():
    departures_path = SAMPLES_DIR / "sample_departures_lease.txt"
    with open(departures_path, "r", encoding="utf-8") as f:
        text = f.read()

    analyzer = LeaseAnalyzer()
    report = analyzer.review_agreement(text)

    assert report.overall_status == "NON_COMPLIANT"
    assert report.departures_count == 8
    assert report.matches_count == 0

    flagged_sp_ids = set()
    for finding in report.clause_findings:
        assert finding.status == "DEPARTS"
        assert finding.risk_level in ("MEDIUM", "HIGH")
        assert finding.departure_details is not None
        assert finding.recommended_amendment is not None
        flagged_sp_ids.add(finding.standard_position_id)

    # All SP-1 through SP-8 are cited in departures
    assert flagged_sp_ids == {"SP-1", "SP-2", "SP-3", "SP-4", "SP-5", "SP-6", "SP-7", "SP-8"}


def test_silent_lease_detects_omitted_positions():
    """
    TASK.md rule: 'one that is silent on two important terms... silence in the agreement is a finding too.'
    """
    silent_path = SAMPLES_DIR / "sample_silent_lease.txt"
    with open(silent_path, "r", encoding="utf-8") as f:
        text = f.read()

    analyzer = LeaseAnalyzer()
    report = analyzer.review_agreement(text)

    assert report.overall_status == "CAUTION"
    assert report.matches_count == 6
    assert report.departures_count == 0
    assert report.missing_count == 2

    missing_ids = [m.standard_position_id for m in report.missing_findings]
    assert "SP-5" in missing_ids  # Subletting & Assignment omitted
    assert "SP-8" in missing_ids  # Alterations & Improvements omitted

    for m in report.missing_findings:
        assert m.status == "MISSING"
        assert len(m.silence_risk) > 0
        assert len(m.recommended_clause_to_insert) > 0


def test_api_integration():
    client = TestClient(app)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["positions_loaded"] == 8

    # 2. GET / (UI)
    res = client.get("/")
    assert res.status_code == 200
    assert "LeaseGuard" in res.text
    assert "Standard Positions" in res.text

    # 3. GET /api/standard-positions
    res = client.get("/api/standard-positions")
    assert res.status_code == 200
    assert len(res.json()) == 8

    # 4. GET /api/samples
    res = client.get("/api/samples")
    assert res.status_code == 200
    assert len(res.json()) == 3

    # 5. POST /api/review with valid text
    sample_text = res.json()[0]["content"]
    res = client.post("/api/review", json={"agreement_text": sample_text, "agreement_title": "Test Clean"})
    assert res.status_code == 200
    report = res.json()
    assert report["overall_status"] == "COMPLIANT"
    assert report["matches_count"] == 8

    # 6. POST /api/review with empty text
    res = client.post("/api/review", json={"agreement_text": "", "agreement_title": "Empty"})
    assert res.status_code == 400

    # 7. POST /api/report/markdown
    res = client.post("/api/report/markdown", json={"agreement_text": sample_text, "agreement_title": "Test Clean"})
    assert res.status_code == 200
    assert "# Legal Review Memorandum" in res.text
