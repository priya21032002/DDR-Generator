"""Stage 2 — Structure: LLM converts raw extracted text into typed Observations.

For thermal documents, uses Gemini vision (page-as-image) because thermal PDFs
contain visual reference photos needed to identify room/area names.
Pages are batched to stay within free-tier rate limits.
For inspection documents, text-only is sufficient.
"""

from __future__ import annotations

import asyncio
import logging

import fitz  # PyMuPDF
from google.genai import types as genai_types

from src.llm.client import call_gemini_vision, call_llm
from src.llm.prompts import (
    STRUCTURE_SYSTEM,
    STRUCTURE_THERMAL_VISION_SYSTEM,
    STRUCTURE_USER_TEMPLATE,
    STRUCTURE_THERMAL_VISION_USER,
)
from src.models.schemas import ExtractedDocument, StructuredDocument

logger = logging.getLogger(__name__)

THERMAL_BATCH_SIZE = 10  # pages per vision call


def _render_pages_as_images(
    pdf_path: str, start: int = 0, end: int | None = None, dpi: int = 150,
) -> list[genai_types.Part]:
    """Render PDF pages [start, end) as PNG Gemini Part objects."""
    doc = fitz.open(pdf_path)
    if end is None:
        end = len(doc)
    parts: list[genai_types.Part] = []
    for page_idx in range(start, min(end, len(doc))):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        parts.append(genai_types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
    doc.close()
    return parts


async def _structure_thermal_batch(
    doc: ExtractedDocument,
    batch_start: int,
    batch_end: int,
    run_id: str,
) -> StructuredDocument:
    """Structure one batch of thermal pages using Gemini vision."""
    image_parts = _render_pages_as_images(doc.file_path, batch_start, batch_end)

    # Send text for the batch pages only
    batch_text_parts = []
    for p in doc.pages:
        if batch_start < p.page_number <= batch_end and p.text.strip():
            batch_text_parts.append(f"=== Page {p.page_number} ===\n{p.text}")
    all_text = "\n\n".join(batch_text_parts) or "(No readable text for these pages)"

    user_prompt = STRUCTURE_THERMAL_VISION_USER.format(
        page_count=batch_end - batch_start,
        all_text=all_text,
    )

    result = await call_gemini_vision(
        prompt=user_prompt,
        system=STRUCTURE_THERMAL_VISION_SYSTEM,
        image_parts=image_parts,
        response_schema=StructuredDocument,
        run_id=run_id,
        stage=f"structure_thermal_p{batch_start + 1}_{batch_end}",
    )

    # Fix doc_type and source_doc on all observations
    result.doc_type = "thermal"
    for obs in result.observations:
        obs.source_doc = "thermal"
        # Offset page numbers to be absolute (batch pages are 0-indexed in the prompt
        # but the LLM should already output the correct page number from the text metadata)

    return result


async def _structure_thermal_vision(
    doc: ExtractedDocument,
    run_id: str,
) -> StructuredDocument:
    """Structure thermal document using Gemini vision, batched by page groups.

    Sends pages in batches of THERMAL_BATCH_SIZE to avoid overwhelming
    Gemini's free-tier rate limits with too many images at once.
    """
    total_pages = doc.page_count
    batches: list[tuple[int, int]] = []
    for start in range(0, total_pages, THERMAL_BATCH_SIZE):
        end = min(start + THERMAL_BATCH_SIZE, total_pages)
        batches.append((start, end))

    logger.info(
        "Structuring thermal doc (%d pages) in %d batches of up to %d",
        total_pages, len(batches), THERMAL_BATCH_SIZE,
    )

    # Run batches sequentially to respect rate limits
    all_observations = []
    all_notes = []
    for batch_start, batch_end in batches:
        logger.info("  Processing thermal pages %d-%d...", batch_start + 1, batch_end)
        batch_result = await _structure_thermal_batch(doc, batch_start, batch_end, run_id)
        all_observations.extend(batch_result.observations)
        all_notes.extend(batch_result.extraction_notes)
        # Delay between batches to respect Gemini free-tier rate limits
        if batch_end < total_pages:
            logger.info("  Waiting 15s between batches for rate limit...")
            await asyncio.sleep(15)

    merged = StructuredDocument(
        doc_type="thermal",
        observations=all_observations,
        extraction_notes=all_notes,
    )

    logger.info(
        "Structured thermal (vision): %d observations, areas: %s",
        len(merged.observations),
        sorted({o.area for o in merged.observations}),
    )
    return merged


async def _structure_text(
    doc: ExtractedDocument,
    run_id: str,
) -> StructuredDocument:
    """Structure inspection document using text-only LLM."""
    all_text = "\n\n".join(
        f"=== Page {p.page_number} ===\n{p.text}"
        for p in doc.pages
        if p.text.strip()
    )

    if not all_text.strip():
        logger.warning("No text extracted from %s — returning empty observations", doc.file_path)
        return StructuredDocument(
            doc_type=doc.doc_type,
            observations=[],
            extraction_notes=["No text could be extracted from the document."],
        )

    user_prompt = STRUCTURE_USER_TEMPLATE.format(
        doc_type=doc.doc_type,
        page_count=doc.page_count,
        all_text=all_text,
    )

    result = await call_llm(
        prompt=user_prompt,
        system=STRUCTURE_SYSTEM,
        response_schema=StructuredDocument,
        run_id=run_id,
        stage=f"structure_{doc.doc_type}",
    )

    result.doc_type = doc.doc_type
    logger.info(
        "Structured %s: %d observations extracted",
        doc.doc_type, len(result.observations),
    )
    return result


async def structure_document(
    doc: ExtractedDocument,
    run_id: str,
) -> StructuredDocument:
    """Convert an ExtractedDocument into a StructuredDocument with typed Observations.

    Uses Gemini vision for thermal documents (need to see reference photos),
    text-only LLM for inspection documents.
    """
    if doc.doc_type == "thermal":
        return await _structure_thermal_vision(doc, run_id)
    return await _structure_text(doc, run_id)
