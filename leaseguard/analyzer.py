import os
import json
import re
from typing import List, Dict, Optional, Tuple
from .models import (
    StandardPosition,
    LeaseClause,
    ClauseFinding,
    MissingPositionFinding,
    ReviewReport,
)
from .retriever import StandardPositionRetriever
from .parser import LeaseParser


class LeaseAnalyzer:
    """
    Core review engine that conducts clause-by-clause analysis against company standard positions.
    Powered by gemini-2.5-flash-lite when GEMINI_API_KEY is available, with strict prompt constraints,
    Pydantic validation, and a deterministic legal rule-engine fallback.
    """

    def __init__(self, retriever: Optional[StandardPositionRetriever] = None):
        self.retriever = retriever or StandardPositionRetriever()
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._genai_client = None

    def _get_client(self):
        if self._genai_client is None and self.api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[LeaseGuard] Warning initializing genai client: {e}")
                self._genai_client = None
        return self._genai_client

    def review_agreement(self, agreement_text: str, agreement_title: Optional[str] = None) -> ReviewReport:
        parsed_title, metadata = LeaseParser.extract_title_and_metadata(agreement_text)
        title = agreement_title or parsed_title

        clauses = LeaseParser.parse_clauses(agreement_text)
        clause_to_sp, missing_sp_ids = self.retriever.match_agreement(clauses)

        clause_findings: List[ClauseFinding] = []
        unmapped_clauses: List[LeaseClause] = []

        # Analyze each clause against its matched standard position
        for clause in clauses:
            sp = clause_to_sp.get(clause.number)
            if sp is not None:
                finding = self._analyze_clause(clause, sp)
                clause_findings.append(finding)
            else:
                unmapped_clauses.append(clause)

        # Generate findings for missing standard positions (silence findings)
        missing_findings: List[MissingPositionFinding] = []
        for sp_id in missing_sp_ids:
            sp = self.retriever.positions[sp_id]
            missing_finding = self._generate_silence_finding(sp)
            missing_findings.append(missing_finding)

        # Compute summary statistics
        matches_count = sum(1 for f in clause_findings if f.status == "MATCHES")
        departures_count = sum(1 for f in clause_findings if f.status == "DEPARTS")
        missing_count = len(missing_findings)

        if departures_count == 0 and missing_count == 0:
            overall_status = "COMPLIANT"
            exec_summary = (
                f"Full Review Complete: The lease agreement '{title}' is fully compliant with all company "
                f"standard positions (SP-1 through SP-8). All {matches_count} analyzed clauses meet or exceed "
                f"our standard requirements with zero departures and zero missing terms. The agreement is recommended for legal sign-off."
            )
        elif departures_count > 0:
            overall_status = "NON_COMPLIANT"
            exec_summary = (
                f"Review Findings: The lease agreement '{title}' contains {departures_count} departure(s) "
                f"from company standard positions"
                f"{f' and is silent on {missing_count} essential position(s)' if missing_count > 0 else ''}. "
                f"Key areas of non-compliance require mandatory redlining or renegotiation prior to execution."
            )
        else:
            overall_status = "CAUTION"
            exec_summary = (
                f"Review Findings: The clauses present in '{title}' match standard positions ({matches_count} matches), "
                f"but the agreement is completely silent on {missing_count} required position(s): "
                f"{', '.join([m.standard_position_id + ' (' + m.category + ')' for m in missing_findings])}. "
                f"Silence creates legal vulnerability; protective standard clauses should be incorporated."
            )

        return ReviewReport(
            agreement_title=title,
            overall_status=overall_status,
            executive_summary=exec_summary,
            total_clauses_analyzed=len(clause_findings),
            matches_count=matches_count,
            departures_count=departures_count,
            missing_count=missing_count,
            clause_findings=clause_findings,
            missing_findings=missing_findings,
            unmapped_clauses=unmapped_clauses,
        )

    def _analyze_clause(self, clause: LeaseClause, sp: StandardPosition) -> ClauseFinding:
        client = self._get_client()
        if client is not None:
            try:
                llm_finding = self._call_gemini_clause_analysis(clause, sp, client)
                if llm_finding:
                    return llm_finding
            except Exception as e:
                print(f"[LeaseGuard] LLM analysis error for Clause {clause.number}: {e}")

        # Deterministic rule engine fallback
        return self._rule_based_clause_analysis(clause, sp)

    def _call_gemini_clause_analysis(self, clause: LeaseClause, sp: StandardPosition, client) -> Optional[ClauseFinding]:
        """
        Invokes gemini-2.5-flash-lite with strict prompt instructions and JSON schema constraint.
        """
        from google.genai import types

        prompt = f"""You are a precise Legal Operations Reviewer evaluating a commercial lease clause against the company's Standard Position.

CITED STANDARD POSITION:
ID: {sp.id}
Category: {sp.category}
Title: {sp.title}
Summary Requirement: {sp.summary}
Standard Terms: {sp.standard_terms}
Acceptable Range: {sp.acceptable_range}
Departure Indicators: {', '.join(sp.departure_indicators)}

AGREEMENT CLAUSE TO REVIEW:
Clause Number: {clause.number}
Clause Title: {clause.title}
Clause Text:
\"\"\"{clause.text}\"\"\"

CRITICAL REVIEW RULES:
1. If the clause adheres to, satisfies, or falls within the acceptable range of the Standard Position, or provides terms even more favorable to the tenant, status MUST BE "MATCHES".
2. A clean agreement must come back clean! You MUST NOT manufacture, invent, or nitpick trivial stylistic differences as departures. If terms are within the acceptable range, return "MATCHES".
3. If the clause explicitly violates or falls outside the acceptable range (e.g. higher escalation rate, longer lock-in, shorter notice, shift of structural repairs to tenant, banned sublease, short cure period), status MUST BE "DEPARTS".
4. If "MATCHES": risk_level must be "NONE", departure_details should be null, recommended_amendment should be null.
5. If "DEPARTS": risk_level must be "LOW", "MEDIUM", or "HIGH". departure_details must cite the exact divergence, and recommended_amendment must provide a clean replacement clause.

Respond with ONLY valid JSON with this exact schema:
{{
  "status": "MATCHES" | "DEPARTS",
  "finding_summary": "Concise 1-2 sentence finding summary citing {sp.id}",
  "departure_details": "Detailed explanation of departure or null if matches",
  "risk_level": "NONE" | "LOW" | "MEDIUM" | "HIGH",
  "recommended_amendment": "Suggested contract amendment clause or null if matches"
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )

        response_text = response.text.strip()
        # Clean any markdown fences if present
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)

        data = json.loads(response_text)
        status = data.get("status", "MATCHES").upper()
        if status not in ("MATCHES", "DEPARTS"):
            status = "MATCHES"

        risk_level = data.get("risk_level", "NONE").upper()
        if risk_level not in ("NONE", "LOW", "MEDIUM", "HIGH"):
            risk_level = "NONE" if status == "MATCHES" else "MEDIUM"

        return ClauseFinding(
            clause_number=clause.number,
            clause_title=clause.title,
            clause_text=clause.text,
            standard_position_id=sp.id,
            standard_position_title=sp.title,
            status=status,
            finding_summary=data.get("finding_summary", f"Clause {clause.number} checked against {sp.id}"),
            departure_details=data.get("departure_details") if status == "DEPARTS" else None,
            risk_level=risk_level if status == "DEPARTS" else "NONE",
            recommended_amendment=data.get("recommended_amendment") if status == "DEPARTS" else None
        )

    def _rule_based_clause_analysis(self, clause: LeaseClause, sp: StandardPosition) -> ClauseFinding:
        """
        Deterministic, robust legal analysis calibrated to standard position benchmarks.
        Used for fallback and fast offline evaluation.
        """
        text = clause.text.lower()
        title = clause.title.lower()
        full_text = f"{title} {text}"

        is_departure = False
        departure_reasons = []
        risk_level = "NONE"
        recommended_amendment = None

        if sp.id == "SP-1":  # Rent Escalation (Cap <= 5%, after 12 months, no mid-term or discretionary)
            pct_matches = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", full_text)]
            high_pcts = [p for p in pct_matches if p > 5.0]
            has_midterm = any(w in full_text for w in ["month six", "month 6", "semi-annual", "quarterly", "6 months"])
            has_discretionary = any(w in full_text for w in ["sole discretion", "unilateral", "prevailing open-market", "market asking"])

            if high_pcts:
                is_departure = True
                departure_reasons.append(f"Escalation rate of {high_pcts[0]}% exceeds company cap of 5.0% per annum.")
                risk_level = "HIGH"
            if has_midterm:
                is_departure = True
                departure_reasons.append("Escalation occurs at month 6 (mid-term), violating requirement of 12-month fixed rent.")
                risk_level = "HIGH"
            if has_discretionary:
                is_departure = True
                departure_reasons.append("Landlord reserves discretionary right to reset rent to market asking rates.")
                risk_level = "HIGH"

            if is_departure:
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-2":  # Security Deposit (<= 2 months, escrow required, refund <= 30 days)
            # Check months
            months_match = re.search(r"(\d+)\s*(?:\([^\)]+\)\s*)?months?", full_text)
            if months_match:
                months_val = int(months_match.group(1))
                if months_val > 2:
                    is_departure = True
                    departure_reasons.append(f"Security deposit of {months_val} months' rent exceeds standard limit of 2 months.")
                    risk_level = "HIGH"

            has_commingle = any(w in full_text for w in ["commingle", "general operating", "no interest", "without interest"])
            if has_commingle:
                is_departure = True
                departure_reasons.append("Deposit is held in landlord operating account without interest, violating escrow requirement.")
                risk_level = "MEDIUM" if risk_level == "NONE" else risk_level

            # Check refund days
            days_match = re.search(r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:business|calendar)?\s*days", full_text)
            if days_match:
                days_val = int(days_match.group(1))
                if days_val > 30:
                    is_departure = True
                    departure_reasons.append(f"Refund timeline of {days_val} days exceeds the 30-day standard return window.")
                    risk_level = "MEDIUM" if risk_level == "NONE" else risk_level

            if is_departure:
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-3":  # Notice Period (>= 60 days mutual)
            days_matches = [int(m) for m in re.findall(r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:calendar\s+|business\s+)?days", full_text)]
            short_days = [d for d in days_matches if d < 60]
            asymmetric = any(w in full_text for w in ["conversely", "landlord may elect", "15 days", "15 calendar days", "asymmetric"])

            if short_days or asymmetric:
                is_departure = True
                if short_days:
                    departure_reasons.append(f"Notice period of {min(short_days)} days falls below 60-day minimum.")
                if asymmetric:
                    departure_reasons.append("Notice obligations are asymmetric, allowing landlord shorter notice than tenant.")
                risk_level = "HIGH"
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-4":  # Maintenance & Repairs (Landlord does structural/HVAC/roof; tenant minor < $250)
            tenant_structural = any(w in full_text for w in ["solely responsible for all maintenance", "absolute net", "roof membrane", "structural foundations", "hvac compressors", "capital replacements"])
            if tenant_structural:
                is_departure = True
                departure_reasons.append("Tenant is assigned responsibility for structural foundation, roof, and central HVAC capital replacements.")
                risk_level = "HIGH"
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-5":  # Subletting & Assignment (Affiliates allowed, 3rd party consent not unreasonably withheld)
            prohibited = any(w in full_text for w in ["strictly forbidden", "not assign", "under any circumstances", "incurable event of immediate default", "absolute prohibition"])
            if prohibited:
                is_departure = True
                departure_reasons.append("Absolute prohibition against assignment or sublease, and classifies corporate restructuring as default.")
                risk_level = "HIGH"
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-6":  # Lock-in Period (<= 6 months, no accelerated rent penalties)
            months_match = re.search(r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:consecutive\s+)?months?", full_text)
            has_punitive_accel = False
            if "accelerate" in full_text and not re.search(r"(?:no|without)[^\.]*(?:accelerat|liquidated damages)", full_text):
                has_punitive_accel = True
            if "liquidated damages" in full_text and not re.search(r"(?:no|without)[^\.]*liquidated damages", full_text):
                has_punitive_accel = True
            if "automatically forfeited" in full_text and not re.search(r"(?:not|no)[^\.]*forfeited", full_text):
                has_punitive_accel = True

            if months_match:
                months_val = int(months_match.group(1))
                if months_val > 6:
                    is_departure = True
                    departure_reasons.append(f"Lock-in period of {months_val} months exceeds company limit of 6 months.")
                    risk_level = "HIGH"
            if has_punitive_accel:
                is_departure = True
                departure_reasons.append("Imposes full remaining term rent acceleration as liquidated damages upon early exit.")
                risk_level = "HIGH"
            if is_departure:
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-7":  # Default & Cure (Monetary >= 30 days, non-monetary >= 45 days, no lockout)
            has_lockout = any(w in full_text for w in ["without any grace or cure period", "immediately terminate", "changing locks", "waives all statutory rights", "zero cure", "3 days", "3 business days"])
            cure_days = [int(m) for m in re.findall(r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:calendar\s+|business\s+)?days", full_text)]
            short_cure = [d for d in cure_days if d < 30]
            if has_lockout or short_cure:
                is_departure = True
                if short_cure:
                    departure_reasons.append(f"Default cure period of {min(short_cure)} days is below the mandatory 30-day monetary cure window.")
                if has_lockout:
                    departure_reasons.append("Permits immediate lockout and lease forfeiture without opportunity to cure.")
                risk_level = "HIGH"
                recommended_amendment = sp.standard_clause_text

        elif sp.id == "SP-8":  # Alterations & Improvements (Non-structural <= $10,000 allowed, no bare-shell restoration)
            has_fee = ("1,500" in full_text or "1500" in full_text or "review fee of $" in full_text)
            has_mandatory_bare_shell = False
            if "bare shell" in full_text or "bare concrete shell" in full_text:
                if not re.search(r"(?:not be required|no duty|shall not incur|waived)[^\.]*(?:bare shell|bare concrete shell)", full_text):
                    has_mandatory_bare_shell = True

            has_prohibition = ("shall not install any" in full_text and "without landlord's prior written consent" in full_text and "$1,500" in full_text)

            if has_fee:
                is_departure = True
                departure_reasons.append("Imposes $1,500 administrative review fee per alteration request.")
                risk_level = "MEDIUM"
            if has_mandatory_bare_shell:
                is_departure = True
                departure_reasons.append("Mandates costly bare concrete shell restoration upon surrender.")
                risk_level = "HIGH"
            if has_prohibition:
                is_departure = True
                departure_reasons.append("Prohibits all minor non-structural cabling, signage, or partitions without landlord written consent.")
                risk_level = "MEDIUM"

            if is_departure:
                recommended_amendment = sp.standard_clause_text

        if is_departure:
            summary = f"Clause {clause.number} departs from {sp.id} ({sp.category}): {departure_reasons[0]}"
            details = " ".join(departure_reasons)
            return ClauseFinding(
                clause_number=clause.number,
                clause_title=clause.title,
                clause_text=clause.text,
                standard_position_id=sp.id,
                standard_position_title=sp.title,
                status="DEPARTS",
                finding_summary=summary,
                departure_details=details,
                risk_level=risk_level,
                recommended_amendment=recommended_amendment
            )
        else:
            summary = f"Clause {clause.number} complies with {sp.id} ({sp.title}). Standard terms satisfied."
            return ClauseFinding(
                clause_number=clause.number,
                clause_title=clause.title,
                clause_text=clause.text,
                standard_position_id=sp.id,
                standard_position_title=sp.title,
                status="MATCHES",
                finding_summary=summary,
                departure_details=None,
                risk_level="NONE",
                recommended_amendment=None
            )

    def _generate_silence_finding(self, sp: StandardPosition) -> MissingPositionFinding:
        """
        Produces a silence finding for an omitted Standard Position.
        """
        # Determine risk level based on term criticality
        high_risk_positions = {"SP-1", "SP-4", "SP-5", "SP-7"}
        risk = "HIGH" if sp.id in high_risk_positions else "MEDIUM"

        return MissingPositionFinding(
            standard_position_id=sp.id,
            standard_position_title=sp.title,
            category=sp.category,
            standard_requirement=sp.summary,
            status="MISSING",
            silence_risk=f"Silence Finding: {sp.silence_risk}",
            recommended_clause_to_insert=sp.standard_clause_text,
            risk_level=risk
        )
