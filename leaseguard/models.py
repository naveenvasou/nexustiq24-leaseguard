from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class StandardPosition(BaseModel):
    id: str = Field(description="Citable ID, e.g., SP-1")
    category: str = Field(description="Short category name")
    title: str = Field(description="Full title of the position")
    summary: str = Field(description="Executive summary of position")
    standard_terms: str = Field(description="Detailed standard requirements")
    acceptable_range: str = Field(description="Acceptable parameter thresholds")
    departure_indicators: List[str] = Field(default_factory=list, description="Signs of departure")
    silence_risk: str = Field(description="Legal risk if agreement is silent on this term")
    standard_clause_text: str = Field(description="Model clause text to insert")


class LeaseClause(BaseModel):
    number: str
    title: str
    text: str
    raw_header: Optional[str] = None


class MatchScore(BaseModel):
    standard_position_id: str
    score: float
    category: str
    title: str


class ClauseFinding(BaseModel):
    clause_number: str
    clause_title: str
    clause_text: str
    standard_position_id: str
    standard_position_title: str
    status: Literal["MATCHES", "DEPARTS"]
    finding_summary: str
    departure_details: Optional[str] = None
    risk_level: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    recommended_amendment: Optional[str] = None


class MissingPositionFinding(BaseModel):
    standard_position_id: str
    standard_position_title: str
    category: str
    standard_requirement: str
    status: Literal["MISSING"] = "MISSING"
    silence_risk: str
    recommended_clause_to_insert: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]


class ReviewReport(BaseModel):
    agreement_title: str
    overall_status: Literal["COMPLIANT", "CAUTION", "NON_COMPLIANT"]
    executive_summary: str
    total_clauses_analyzed: int
    matches_count: int
    departures_count: int
    missing_count: int
    clause_findings: List[ClauseFinding]
    missing_findings: List[MissingPositionFinding]
    unmapped_clauses: List[LeaseClause] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    agreement_text: str
    agreement_title: Optional[str] = "Lease Agreement"
