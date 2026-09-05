"""
LeaseGuard - Lease Agreement Review Assistant
Track 5: Legal Operations
"""

import os
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from leaseguard.models import ReviewRequest, ReviewReport
from leaseguard.analyzer import LeaseAnalyzer
from leaseguard.retriever import StandardPositionRetriever
from leaseguard.parser import LeaseParser

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="LeaseGuard",
    description="Clause-by-clause lease agreement review assistant against company standard positions",
    version="1.0.0",
)

# Initialize retriever and analyzer
retriever = StandardPositionRetriever(str(DATA_DIR / "standard_positions.json"))
analyzer = LeaseAnalyzer(retriever=retriever)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "LeaseGuard",
        "track": "Track 5 - Legal Operations",
        "positions_loaded": len(retriever.positions),
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY"))
    }


@app.get("/api/standard-positions")
def get_standard_positions():
    """Returns all company standard positions SP-1 through SP-8."""
    return list(retriever.positions.values())


@app.get("/api/samples")
def get_samples():
    """Returns the 3 built-in sample lease agreements."""
    samples = []
    sample_files = [
        ("sample_compliant_lease.txt", "Highland Tower Commercial Lease", "Broadly Compliant (Clean Agreement)", "compliant"),
        ("sample_departures_lease.txt", "Metro Plaza Triple-Net Lease", "Multiple Severe Departures (Landlord-Favorable)", "departures"),
        ("sample_silent_lease.txt", "Bayview Business Center Lease", "Silent on Subletting & Alterations (Missing Terms)", "silent")
    ]
    for filename, display_name, description, sample_type in sample_files:
        filepath = SAMPLES_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            samples.append({
                "id": filename,
                "title": display_name,
                "description": description,
                "type": sample_type,
                "content": content
            })
    return samples


@app.post("/api/review", response_model=ReviewReport)
def review_lease(request: ReviewRequest):
    """
    Executes a clause-by-clause review of the provided lease text against company standard positions.
    """
    if not request.agreement_text or not request.agreement_text.strip():
        raise HTTPException(status_code=400, detail="Agreement text cannot be empty.")

    try:
        report = analyzer.review_agreement(
            agreement_text=request.agreement_text,
            agreement_title=request.agreement_title
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


@app.post("/api/report/markdown", response_class=PlainTextResponse)
def export_markdown_report(request: ReviewRequest):
    """Generates an exportable legal desk Markdown review memorandum."""
    report = analyzer.review_agreement(request.agreement_text, request.agreement_title)

    lines = [
        f"# Legal Review Memorandum: {report.agreement_title}",
        f"**Track**: 5 - Legal Operations | **System**: LeaseGuard Legal Assistant",
        f"**Verdict**: **{report.overall_status}**",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        f"- **Total Clauses Analyzed**: {report.total_clauses_analyzed}",
        f"- **Compliant Matches**: {report.matches_count}",
        f"- **Contractual Departures**: {report.departures_count}",
        f"- **Missing Company Standard Terms (Silence Findings)**: {report.missing_count}",
        "",
        "---",
        "",
        "## Clause-by-Clause Audit Findings",
        ""
    ]

    for f in report.clause_findings:
        status_icon = "PASS" if f.status == "MATCHES" else "DEPARTURE"
        lines.append(f"### [{status_icon}] Clause {f.clause_number}: {f.clause_title}")
        lines.append(f"**Checked Against Standard Position**: [{f.standard_position_id}] {f.standard_position_title}")
        lines.append(f"**Compliance Status**: `{f.status}` | **Risk Level**: `{f.risk_level}`")
        lines.append(f"**Finding**: {f.finding_summary}")
        if f.departure_details:
            lines.append(f"\n> **Departure Analysis**: {f.departure_details}")
        if f.recommended_amendment:
            lines.append(f"\n**Recommended Amendment / Redline**:\n```text\n{f.recommended_amendment}\n```")
        lines.append("")

    if report.missing_findings:
        lines.append("---")
        lines.append("## Silence Findings (Omitted Standard Positions)")
        lines.append("The agreement completely omits the following standard company positions:\n")
        for m in report.missing_findings:
            lines.append(f"### [MISSING] {m.standard_position_id}: {m.standard_position_title}")
            lines.append(f"**Risk**: `{m.risk_level}` | **Silence Impact**: {m.silence_risk}")
            lines.append(f"**Mandatory Standard Clause to Insert**:\n```text\n{m.recommended_clause_to_insert}\n```\n")

    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the primary single-page web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)

    # Fallback minimal HTML if index.html is being prepared
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html>
<head><title>LeaseGuard - Lease Review Assistant</title></head>
<body>
  <h1>LeaseGuard - Track 5 Legal Operations</h1>
  <p>System is online. UI assets loading...</p>
</body>
</html>""",
        status_code=200
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
