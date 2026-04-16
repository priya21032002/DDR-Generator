"""Stage 5 — Render: Convert DDRReport JSON into HTML and PDF.

Uses Jinja2 for templating and WeasyPrint for PDF generation. No LLM calls.
Image paths are resolved to absolute file:// URIs for PDF rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.models.schemas import DDRReport

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def _resolve_image_paths(report: DDRReport, output_dir: Path) -> DDRReport:
    """Convert relative image paths to absolute paths for PDF rendering.

    WeasyPrint/xhtml2pdf need file:// URIs or absolute paths to find images.
    """
    for area in report.area_observations:
        for img_ref in area.image_refs:
            p = Path(img_ref.file_path)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                img_ref.file_path = str(p)
            else:
                logger.warning("Image not found: %s", img_ref.file_path)
                # Keep original path; will show as broken image
    return report


def render_html(report: DDRReport, run_id: str, output_dir: str | Path = "outputs") -> Path:
    """Render the DDRReport as an HTML file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve image paths before rendering
    report = _resolve_image_paths(report, output_dir)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("ddr_template.html")

    html_content = template.render(
        report=report,
        run_id=run_id,
    )

    html_path = output_dir / f"{run_id}_ddr_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("HTML report written to %s", html_path)
    return html_path


def render_pdf(html_path: Path, output_dir: str | Path = "outputs") -> Path:
    """Convert the HTML report to PDF. Tries WeasyPrint first, then xhtml2pdf fallback."""
    output_dir = Path(output_dir)
    pdf_path = (output_dir / html_path.stem).with_suffix(".pdf")

    # Try WeasyPrint first (best quality, needs GTK on Windows)
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path), base_url=str(Path.cwd())).write_pdf(str(pdf_path))
        logger.info("PDF report written (WeasyPrint) to %s", pdf_path)
        return pdf_path
    except Exception as wp_err:
        logger.warning("WeasyPrint failed (%s), trying xhtml2pdf fallback", wp_err)

    # Fallback: xhtml2pdf (pure Python, no system deps)
    try:
        from xhtml2pdf import pisa
        html_content = html_path.read_text(encoding="utf-8")
        with open(pdf_path, "wb") as f:
            status = pisa.CreatePDF(html_content, dest=f)
        if not status.err:
            logger.info("PDF report written (xhtml2pdf) to %s", pdf_path)
            return pdf_path
        else:
            raise RuntimeError(f"xhtml2pdf returned errors: {status.err}")
    except ImportError:
        logger.error("Neither WeasyPrint nor xhtml2pdf available for PDF generation")
        raise
    except Exception as exc:
        logger.error("PDF generation failed: %s. HTML is at %s", exc, html_path)
        raise


def render_report(
    report: DDRReport,
    run_id: str,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Render both HTML and PDF versions of the DDR report."""
    html_path = render_html(report, run_id, output_dir)
    try:
        pdf_path = render_pdf(html_path, output_dir)
    except Exception:
        logger.warning("PDF generation failed; returning HTML only")
        pdf_path = html_path  # Fallback
    return html_path, pdf_path
