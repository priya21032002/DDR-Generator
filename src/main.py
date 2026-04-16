"""FastAPI application — DDR Generator.

Provides:
- POST /generate  — Upload two PDFs, get back a DDR report
- GET  /           — Simple upload form
- GET  /outputs/{filename} — Download generated reports
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline.ingest import ingest_pdf
from src.pipeline.structure import structure_document
from src.pipeline.merge import merge_documents
from src.pipeline.generate import generate_ddr
from src.pipeline.render import render_report

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DDR Generator",
    description="Converts inspection + thermal PDFs into a Detailed Diagnostic Report",
    version="1.0.0",
)

OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR = Path("uploads")

# Serve generated outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR), check_dir=False), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the upload form."""
    upload_html = Path("static/upload.html")
    if upload_html.exists():
        return HTMLResponse(upload_html.read_text(encoding="utf-8"))
    return HTMLResponse("""
    <html><body>
    <h1>DDR Generator</h1>
    <form action="/generate" method="post" enctype="multipart/form-data">
        <p><label>Inspection Report PDF:<br><input type="file" name="inspection_pdf" accept=".pdf" required></label></p>
        <p><label>Thermal Report PDF:<br><input type="file" name="thermal_pdf" accept=".pdf" required></label></p>
        <p><button type="submit">Generate DDR</button></p>
    </form>
    </body></html>
    """)


@app.post("/generate")
async def generate(
    inspection_pdf: UploadFile = File(...),
    thermal_pdf: UploadFile = File(...),
):
    """Run the full 5-stage pipeline and return the generated DDR."""
    run_id = uuid.uuid4().hex[:12]
    logger.info("Starting DDR generation run_id=%s", run_id)

    # Save uploaded files
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    inspection_path = UPLOADS_DIR / f"{run_id}_inspection.pdf"
    thermal_path = UPLOADS_DIR / f"{run_id}_thermal.pdf"

    inspection_path.write_bytes(await inspection_pdf.read())
    thermal_path.write_bytes(await thermal_pdf.read())

    try:
        # Stage 1: Ingest
        logger.info("[%s] Stage 1: Ingesting PDFs...", run_id)
        inspection_doc = ingest_pdf(inspection_path, "inspection", str(OUTPUTS_DIR))
        thermal_doc = ingest_pdf(thermal_path, "thermal", str(OUTPUTS_DIR))

        # Stage 2: Structure (sequential to respect Gemini rate limits)
        logger.info("[%s] Stage 2a: Structuring inspection (Groq)...", run_id)
        inspection_structured = await structure_document(inspection_doc, run_id)
        logger.info("[%s] Stage 2b: Structuring thermal (Gemini vision)...", run_id)
        thermal_structured = await structure_document(thermal_doc, run_id)

        # Stage 3: Merge
        logger.info("[%s] Stage 3: Merging findings...", run_id)
        merged = await merge_documents(inspection_structured, thermal_structured, run_id)

        # Stage 4: Generate
        logger.info("[%s] Stage 4: Generating DDR...", run_id)
        all_images = inspection_doc.images + thermal_doc.images
        ddr_report = await generate_ddr(merged, all_images, run_id)

        # Stage 5: Render
        logger.info("[%s] Stage 5: Rendering report...", run_id)
        html_path, pdf_path = render_report(ddr_report, run_id, str(OUTPUTS_DIR))

        # Save intermediate artifacts for debugging
        _save_artifact(run_id, "ingest_inspection", inspection_doc)
        _save_artifact(run_id, "ingest_thermal", thermal_doc)
        _save_artifact(run_id, "structure_inspection", inspection_structured)
        _save_artifact(run_id, "structure_thermal", thermal_structured)
        _save_artifact(run_id, "merged", merged)
        _save_artifact(run_id, "ddr_report", ddr_report)

        logger.info("[%s] DDR generation complete!", run_id)

        return {
            "run_id": run_id,
            "html_report": f"/outputs/{html_path.name}",
            "pdf_report": f"/outputs/{pdf_path.name}",
            "stages_completed": 5,
        }

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")


def _save_artifact(run_id: str, stage: str, data) -> None:
    """Save a pipeline artifact as JSON for debugging."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{run_id}_{stage}.json"
    path.write_text(data.model_dump_json(indent=2), encoding="utf-8")


@app.get("/report/{run_id}")
async def get_report(run_id: str):
    """Download a generated report by run_id."""
    pdf_path = OUTPUTS_DIR / f"{run_id}_ddr_report.pdf"
    html_path = OUTPUTS_DIR / f"{run_id}_ddr_report.html"

    if pdf_path.exists():
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
    elif html_path.exists():
        return FileResponse(html_path, media_type="text/html", filename=html_path.name)
    else:
        raise HTTPException(status_code=404, detail="Report not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
