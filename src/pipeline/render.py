"""Stage 5 — Render: Convert DDRReport JSON into HTML (and PDF if available).

Uses Jinja2 for templating. PDF generation is optional — tries xhtml2pdf
if installed, otherwise returns HTML only (users can print to PDF from browser).
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.models.schemas import DDRReport

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def _resolve_image_paths(report: DDRReport) -> DDRReport:
    """Convert relative image paths to absolute paths for rendering."""
    for area in report.area_observations:
        for img_ref in area.image_refs:
            p = Path(img_ref.file_path)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                img_ref.file_path = str(p)
            else:
                logger.warning("Image not found: %s", img_ref.file_path)
    return report


def render_html(report: DDRReport, run_id: str, output_dir: str | Path = "outputs") -> Path:
    """Render the DDRReport as an HTML file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _resolve_image_paths(report)

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


def render_pdf(html_path: Path, output_dir: str | Path = "outputs") -> Path | None:
    """Try to convert HTML to PDF. Returns None if no PDF library is available."""
    output_dir = Path(output_dir)
    pdf_path = (output_dir / html_path.stem).with_suffix(".pdf")

    # Try xhtml2pdf if installed (optional dependency)
    try:
        from xhtml2pdf import pisa
        html_content = html_path.read_text(encoding="utf-8")
        with open(pdf_path, "wb") as f:
            status = pisa.CreatePDF(html_content, dest=f)
        if not status.err:
            logger.info("PDF report written to %s", pdf_path)
            return pdf_path
    except ImportError:
        logger.info("xhtml2pdf not installed — PDF generation skipped, HTML is primary output")
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)

    return None


def render_report(
    report: DDRReport,
    run_id: str,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Render HTML and optionally PDF. Always returns (html_path, best_path)."""
    html_path = render_html(report, run_id, output_dir)
    pdf_path = render_pdf(html_path, output_dir)
    return html_path, pdf_path or html_path
