import re
from typing import List, Tuple, Optional
from .models import LeaseClause


class LeaseParser:
    """
    Parses lease agreements into structured numbered clauses.
    Detects patterns such as 'Clause 1: ...', 'Section 2. ...', 'Article 3 - ...', or '1. ...'
    """

    HEADER_PATTERN = re.compile(
        r"(?:^|\n)"
        r"(?:(?:Clause|Section|Article)\s+)?(\d+(?:\.\d+)?)\s*[:.\-]\s*"
        r"([^\n]+)\n",
        re.IGNORECASE
    )

    ALT_HEADER_PATTERN = re.compile(
        r"(?:^|\n)"
        r"(?:Clause|Section|Article)\s+(\d+(?:\.\d+)?)\s*\n",
        re.IGNORECASE
    )

    @classmethod
    def extract_title_and_metadata(cls, text: str) -> Tuple[str, dict]:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        title = "Commercial Lease Agreement"
        metadata = {}

        if lines:
            first_line = lines[0]
            if len(first_line) < 120 and ("agreement" in first_line.lower() or "lease" in first_line.lower()):
                title = first_line

        for line in lines[:10]:
            if ":" in line:
                key, val = line.split(":", 1)
                k_clean = key.strip().lower()
                if k_clean in ("premises", "landlord", "tenant", "commencement date", "term"):
                    metadata[k_clean] = val.strip()

        return title, metadata

    @classmethod
    def parse_clauses(cls, text: str) -> List[LeaseClause]:
        matches = list(cls.HEADER_PATTERN.finditer(text))
        clauses: List[LeaseClause] = []

        if matches:
            for i, match in enumerate(matches):
                num = match.group(1).strip()
                title = match.group(2).strip()
                start_pos = match.end()

                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                clause_body = text[start_pos:end_pos].strip()

                # Strip closing signature blocks if in last clause
                if i == len(matches) - 1:
                    witness_match = re.search(r"\n\s*(?:IN WITNESS WHEREOF|EXECUTED as of|SIGNATURES|Signed by)", clause_body, re.IGNORECASE)
                    if witness_match:
                        clause_body = clause_body[:witness_match.start()].strip()

                clauses.append(LeaseClause(
                    number=num,
                    title=title,
                    text=clause_body,
                    raw_header=match.group(0).strip()
                ))
            return clauses

        # Fallback to alternative pattern or paragraphs
        alt_matches = list(cls.ALT_HEADER_PATTERN.finditer(text))
        if alt_matches:
            for i, match in enumerate(alt_matches):
                num = match.group(1).strip()
                start_pos = match.end()
                end_pos = alt_matches[i + 1].start() if i + 1 < len(alt_matches) else len(text)
                content = text[start_pos:end_pos].strip()
                lines = content.split("\n", 1)
                title = lines[0].strip() if lines else f"Clause {num}"
                body = lines[1].strip() if len(lines) > 1 else ""
                clauses.append(LeaseClause(
                    number=num,
                    title=title,
                    text=body,
                    raw_header=match.group(0).strip()
                ))
            return clauses

        # Fallback: Paragraph-based splitting
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs, start=1):
            if len(p) < 40 and idx == 1:
                continue
            lines = p.split("\n", 1)
            title = lines[0][:60] if len(lines[0]) < 60 else f"Section {idx}"
            body = lines[1] if len(lines) > 1 else p
            clauses.append(LeaseClause(
                number=str(idx),
                title=title,
                text=body,
                raw_header=f"Section {idx}"
            ))

        return clauses
