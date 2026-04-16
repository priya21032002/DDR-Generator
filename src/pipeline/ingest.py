"""Stage 1 — Ingest: Extract text and images from PDFs using PyMuPDF.

No LLM calls. Tesseract OCR fallback for scanned/image-only pages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from PIL import Image

from src.models.schemas import ExtractedDocument, ExtractedImage, ExtractedPage

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 30
MIN_IMAGE_DIM = 50


def _try_ocr(page: fitz.Page) -> str:
    """Attempt OCR on a page rendered as an image."""
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed — skipping OCR")
        return ""
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img).strip()
    except Exception as exc:
        logger.warning("OCR failed on page %d: %s", page.number + 1, exc)
        return ""


def _extract_images(
    doc: fitz.Document,
    output_dir: Path,
    doc_label: str,
) -> list[ExtractedImage]:
    """Extract all meaningful images from the PDF and save to disk."""
    images: list[ExtractedImage] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        img_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                width = base_image["width"]
                height = base_image["height"]
                if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
                    continue

                ext = base_image["ext"]
                image_id = f"img_p{page_idx + 1}_{img_idx}"
                filename = f"{doc_label}_{image_id}.{ext}"
                file_path = output_dir / filename

                with open(file_path, "wb") as f:
                    f.write(base_image["image"])

                caption = _find_caption(page, xref)

                images.append(ExtractedImage(
                    image_id=image_id,
                    source_page=page_idx + 1,
                    bbox=(0, 0, float(width), float(height)),
                    file_path=str(file_path),
                    caption_text=caption,
                ))
            except Exception as exc:
                logger.warning(
                    "Failed to extract image xref=%d from page %d: %s",
                    xref, page_idx + 1, exc,
                )
    return images


def _find_caption(page: fitz.Page, xref: int) -> str:
    """Try to find text near an image that might serve as a caption."""
    try:
        for img_rect in page.get_image_rects(xref):
            search_rect = fitz.Rect(
                img_rect.x0, img_rect.y1,
                img_rect.x1, img_rect.y1 + 50,
            )
            text = page.get_text("text", clip=search_rect).strip()
            if text:
                return text[:200]
    except Exception:
        pass
    return ""


def ingest_pdf(
    pdf_path: str | Path,
    doc_type: Literal["inspection", "thermal"],
    output_dir: str | Path = "outputs",
) -> ExtractedDocument:
    """Extract text and images from a single PDF."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir) / "images" / doc_type

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: list[ExtractedPage] = []
    warnings: list[str] = []

    logger.info("Ingesting %s (%d pages): %s", doc_type, len(doc), pdf_path.name)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        ocr_used = False

        if len(text) < MIN_TEXT_LENGTH:
            ocr_text = _try_ocr(page)
            if ocr_text:
                text = ocr_text
                ocr_used = True
            else:
                warnings.append(
                    f"Page {page_idx + 1}: minimal text and OCR failed"
                )

        pages.append(ExtractedPage(
            page_number=page_idx + 1,
            text=text,
            ocr_used=ocr_used,
        ))

    images = _extract_images(doc, output_dir, doc_type)
    logger.info(
        "Extracted %d pages, %d images from %s",
        len(pages), len(images), pdf_path.name,
    )
    doc.close()

    return ExtractedDocument(
        doc_type=doc_type,
        file_path=str(pdf_path),
        pages=pages,
        images=images,
        page_count=len(pages),
        warnings=warnings,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.ingest <pdf_path> [inspection|thermal]")
        sys.exit(1)

    path = sys.argv[1]
    dtype = sys.argv[2] if len(sys.argv) > 2 else "inspection"
    result = ingest_pdf(path, dtype)  # type: ignore[arg-type]
    print(f"Pages: {result.page_count}, Images: {len(result.images)}")
    for p in result.pages:
        print(f"  Page {p.page_number}: {len(p.text)} chars, OCR={p.ocr_used}")
    for img in result.images:
        print(f"  {img.image_id}: page {img.source_page} -> {img.file_path}")
