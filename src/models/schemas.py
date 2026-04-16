"""Pydantic v2 schemas for the DDR pipeline.

Data flows: Ingest → Structure → Merge → Generate → Render
Each stage consumes the output of the previous one.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1 — Ingest  (PyMuPDF extraction, no LLM)
# ---------------------------------------------------------------------------

class ExtractedImage(BaseModel):
    """A single image pulled from the PDF."""
    image_id: str = Field(..., description="Stable identifier, e.g. 'img_p3_0'")
    source_page: int
    bbox: tuple[float, float, float, float] = Field(
        ..., description="(x0, y0, x1, y1) in PDF coordinates"
    )
    file_path: str = Field(..., description="Path where the image was saved on disk")
    caption_text: str = Field(
        default="", description="Nearby text within ~100px, if any"
    )


class ExtractedPage(BaseModel):
    """One page of extracted content."""
    page_number: int
    text: str
    ocr_used: bool = Field(
        default=False, description="True if pytesseract was needed for this page"
    )


class ExtractedDocument(BaseModel):
    """Full extraction result for one PDF — output of the Ingest stage."""
    doc_type: Literal["inspection", "thermal"]
    file_path: str
    pages: list[ExtractedPage]
    images: list[ExtractedImage]
    page_count: int
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2 — Structure  (LLM converts raw text → typed observations)
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class Observation(BaseModel):
    """A single finding extracted by the Structure stage."""
    area: str = Field(..., description="Physical location, e.g. 'Hall ceiling'")
    finding: str = Field(..., description="Plain-English description of what was found")
    source_doc: Literal["inspection", "thermal"]
    source_page: int
    image_ids: list[str] = Field(default_factory=list)
    severity_hint: SeverityLevel | None = Field(
        default=None, description="Optional severity suggested by the source text"
    )
    measurement: str | None = Field(
        default=None,
        description="Quantitative reading if present, e.g. '24.0°C', '12% moisture'",
    )


class StructuredDocument(BaseModel):
    """Output of the Structure stage for one document."""
    doc_type: Literal["inspection", "thermal"]
    observations: list[Observation]
    extraction_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 — Merge  (fuse inspection + thermal by area)
# ---------------------------------------------------------------------------

class SourceRef(BaseModel):
    """Provenance pointer back to a source document."""
    doc_type: Literal["inspection", "thermal"]
    page_number: int
    image_ids: list[str] = Field(default_factory=list)


class MergedFinding(BaseModel):
    """One finding within an area, potentially backed by both documents."""
    description: str
    source_refs: list[SourceRef]
    measurement: str | None = None
    conflict_notes: str | None = Field(
        default=None,
        description="Verbatim quotes from both sources when they disagree",
    )
    needs_review: bool = False


class AreaFindings(BaseModel):
    """All merged findings for a single physical area."""
    area_name: str
    findings: list[MergedFinding]
    image_ids: list[str] = Field(
        default_factory=list, description="All image IDs relevant to this area"
    )


class CoverageGap(BaseModel):
    """An area present in one document but absent in the other."""
    area_name: str
    present_in: Literal["inspection", "thermal"]
    missing_from: Literal["inspection", "thermal"]


class MergedFindings(BaseModel):
    """Output of the Merge stage — input to the Generate stage."""
    areas: list[AreaFindings]
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    merge_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 4 — Generate  (LLM produces 7-section DDR as structured JSON)
# ---------------------------------------------------------------------------

class ImageRef(BaseModel):
    """Reference to an image to be placed inline by the renderer."""
    image_id: str
    file_path: str
    caption: str = ""


class AreaObservation(BaseModel):
    """One area's entry in Section 2 of the DDR."""
    area_name: str
    findings: list[str] = Field(..., description="Plain-English finding descriptions")
    image_refs: list[ImageRef] = Field(default_factory=list)
    image_status: str | None = Field(
        default=None,
        description="Set to 'Image Not Available' when no images exist for this area",
    )
    severity: SeverityLevel


class RecommendedAction(BaseModel):
    """A single recommended action with priority."""
    action: str
    priority: Literal["immediate", "short-term", "long-term"]


class SeverityAssessment(BaseModel):
    """Overall severity rating with justification."""
    overall: SeverityLevel
    reasoning: str = Field(
        ..., description="Must cite specific findings to justify the rating"
    )


class DDRReport(BaseModel):
    """The final 7-section DDR — output of the Generate stage, input to Render."""

    # Section 1
    property_issue_summary: str = Field(
        ..., description="3-5 sentence executive summary in plain English"
    )
    # Section 2
    area_observations: list[AreaObservation]
    # Section 3
    probable_root_cause: str = Field(
        ..., description="Grounded analysis referencing specific findings"
    )
    # Section 4
    severity_assessment: SeverityAssessment
    # Section 5
    recommended_actions: list[RecommendedAction]
    # Section 6
    additional_notes: list[str] = Field(
        default_factory=list,
        description="Conflicts, caveats, assumptions",
    )
    # Section 7
    missing_or_unclear: list[str] = Field(
        default_factory=list,
        description="Gaps — each entry uses 'Not Available' where info is missing",
    )


# ---------------------------------------------------------------------------
# LLM call logging
# ---------------------------------------------------------------------------

class LLMCallLog(BaseModel):
    """One row in outputs/run_log.jsonl."""
    run_id: str
    stage: str
    provider: Literal["gemini", "groq"]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: str | None = None
