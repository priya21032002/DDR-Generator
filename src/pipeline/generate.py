"""Stage 4 — Generate: LLM produces the final 7-section DDR as structured JSON.

Takes MergedFindings + available images and produces a DDRReport.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from src.llm.client import call_llm
from src.llm.prompts import GENERATE_SYSTEM, GENERATE_USER_TEMPLATE
from src.models.schemas import (
    DDRReport,
    ExtractedImage,
    ImageRef,
    MergedFindings,
)

logger = logging.getLogger(__name__)


def _build_images_json(images: list[ExtractedImage]) -> str:
    """Build a JSON summary of available images for the LLM."""
    img_data = [
        {
            "image_id": img.image_id,
            "file_path": img.file_path,
            "source_page": img.source_page,
            "caption": img.caption_text,
        }
        for img in images
    ]
    return json.dumps(img_data, indent=2)


def _build_image_lookup(images: list[ExtractedImage]) -> dict[str, ExtractedImage]:
    """Create a lookup from image_id to ExtractedImage."""
    return {img.image_id: img for img in images}


def _assign_images_to_areas(
    report: DDRReport,
    merged: MergedFindings,
    all_images: list[ExtractedImage],
) -> None:
    """Post-process: assign extracted images to area observations.

    Strategy:
    1. If merge stage tagged image_ids on area findings, use those.
    2. Match images to areas by page number overlap with source_refs.
    3. Set "Image Not Available" for areas with no images.
    """
    image_lookup = _build_image_lookup(all_images)

    # Build a map: area_name → set of source pages
    area_pages: dict[str, set[int]] = defaultdict(set)
    for area in merged.areas:
        for finding in area.findings:
            for ref in finding.source_refs:
                area_pages[area.area_name].add(ref.page_number)
        # Also include image_ids directly tagged on the area
        for img_id in area.image_ids:
            if img_id in image_lookup:
                area_pages[area.area_name].add(image_lookup[img_id].source_page)

    # Build a map: page_number → list of images
    page_images: dict[int, list[ExtractedImage]] = defaultdict(list)
    for img in all_images:
        page_images[img.source_page].append(img)

    used_image_ids: set[str] = set()

    for area_obs in report.area_observations:
        area_name = area_obs.area_name

        # Collect images from matching pages
        matching_pages = area_pages.get(area_name, set())
        area_images: list[ExtractedImage] = []
        for page_num in sorted(matching_pages):
            for img in page_images.get(page_num, []):
                if img.image_id not in used_image_ids:
                    area_images.append(img)

        # Limit to 4 images per area to keep report readable
        area_images = area_images[:4]

        if area_images:
            area_obs.image_refs = [
                ImageRef(
                    image_id=img.image_id,
                    file_path=img.file_path,
                    caption=img.caption_text or f"Image from page {img.source_page}",
                )
                for img in area_images
            ]
            used_image_ids.update(img.image_id for img in area_images)
            area_obs.image_status = None
        else:
            area_obs.image_refs = []
            area_obs.image_status = "Image Not Available"


async def generate_ddr(
    merged: MergedFindings,
    all_images: list[ExtractedImage],
    run_id: str,
) -> DDRReport:
    """Generate the final 7-section DDR report from merged findings."""
    merged_json = json.dumps(merged.model_dump(mode="json"), indent=2)
    images_json = _build_images_json(all_images)

    user_prompt = GENERATE_USER_TEMPLATE.format(
        merged_json=merged_json,
        images_json=images_json,
    )

    result = await call_llm(
        prompt=user_prompt,
        system=GENERATE_SYSTEM,
        response_schema=DDRReport,
        run_id=run_id,
        stage="generate",
    )

    # Post-process: assign images to areas using page-matching
    _assign_images_to_areas(result, merged, all_images)

    logger.info(
        "Generated DDR: %d areas, %d actions, %d areas with images",
        len(result.area_observations),
        len(result.recommended_actions),
        sum(1 for a in result.area_observations if a.image_refs),
    )
    return result
