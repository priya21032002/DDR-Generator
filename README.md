# DDR Generator - AI-Powered Detailed Diagnostic Report System

An AI system that reads property inspection and thermal imaging PDFs, then generates a professional Detailed Diagnostic Report (DDR) combining visual observations with thermal analysis.

## What It Does

**Input:** 2 PDFs (Inspection Report + Thermal Imaging Report)  
**Output:** A structured, client-friendly DDR with 7 sections, inline images, and actionable recommendations

The system extracts text and images from both documents, uses AI to identify areas and correlate findings, merges inspection + thermal data, and renders a professional PDF report.

## Architecture - 5-Stage Pipeline

```
PDF Upload --> [1. Ingest] --> [2. Structure] --> [3. Merge] --> [4. Generate] --> [5. Render] --> DDR Report
```

| Stage | What It Does | Technology |
|-------|-------------|------------|
| **Ingest** | Extract text + images from PDFs | PyMuPDF, Tesseract OCR |
| **Structure** | Convert raw text to typed observations | Groq (text), Gemini Vision (thermal) |
| **Merge** | Group by area, fuse inspection + thermal data | Groq LLM |
| **Generate** | Produce 7-section DDR as structured JSON | Groq LLM |
| **Render** | Convert JSON to HTML and PDF | Jinja2, xhtml2pdf |

## Key Features

- **Thermal Vision Analysis**: Uses Gemini's vision capability to identify rooms from thermal reference photos (thermal PDFs have no text labels)
- **Intelligent Merging**: Correlates inspection findings with thermal readings for the same physical location
- **Image Placement**: Automatically links extracted images to the correct report sections
- **Conflict Detection**: Flags contradictions between inspection and thermal data
- **Gap Identification**: Notes areas covered by one report but missing from the other
- **Source Traceability**: Every claim in the DDR traces back to a specific page in the source documents

## DDR Output Sections

1. **Property Issue Summary** - Executive summary
2. **Area-wise Observations** - Findings per area with inline images
3. **Probable Root Cause** - Analysis of why issues exist
4. **Severity Assessment** - Overall rating with reasoning
5. **Recommended Actions** - Prioritized (immediate / short-term / long-term)
6. **Additional Notes** - Conflicts, caveats, disclaimers
7. **Missing or Unclear Information** - Explicit gaps marked "Not Available"

## Setup & Installation

### Prerequisites
- Python 3.11+
- Free API keys: [Groq](https://console.groq.com/) and [Google AI Studio](https://aistudio.google.com/apikey)

### Install

```bash
git clone https://github.com/priya21032002/DDR-Generator.git
cd DDR-Generator

pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Run

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, upload the two PDFs, and wait ~2-3 minutes for the report.

## Project Structure

```
src/
  main.py                  # FastAPI app with /generate endpoint
  models/
    schemas.py             # Pydantic v2 schemas for all pipeline stages
  llm/
    client.py              # Unified LLM client (Groq primary, Gemini fallback)
    prompts.py             # System prompts for each pipeline stage
  pipeline/
    ingest.py              # Stage 1: PyMuPDF text + image extraction
    structure.py           # Stage 2: LLM structuring (vision for thermal)
    merge.py               # Stage 3: Area-based merging
    generate.py            # Stage 4: 7-section DDR generation
    render.py              # Stage 5: HTML/PDF rendering
  templates/
    ddr_template.html      # Jinja2 template for the report
static/
  upload.html              # Upload form UI
sample_inputs/             # Sample inspection + thermal PDFs for testing
```

## LLM Configuration

| Provider | Model | Role | Cost |
|----------|-------|------|------|
| Groq | llama-3.3-70b-versatile | Primary for text tasks (structure, merge, generate) | Free |
| Gemini | gemini-2.5-flash | Thermal vision analysis + text fallback | Free |
| Gemini | gemini-2.0-flash | Secondary fallback when 2.5-flash quota exhausted | Free |

All calls use temperature=0 for deterministic output. Every LLM call is logged with token counts and latency to `outputs/run_log.jsonl`.

## Design Decisions

- **Why Gemini Vision for thermal?** Thermal PDFs contain only temperature numbers as text. Room identification requires looking at the visual reference photo on each page.
- **Why batched thermal processing?** 30 thermal pages sent at once exceeds free-tier rate limits. Batches of 10 pages with delays between calls stay within limits.
- **Why Groq primary?** Faster response times for text-only calls. Gemini is reserved for vision tasks and fallback.
- **Why xhtml2pdf?** WeasyPrint requires GTK system libraries. xhtml2pdf is pure Python and works everywhere.

## Limitations

- Gemini free tier has rate limits (~20 requests/minute). The pipeline includes retry logic with exponential backoff.
- xhtml2pdf has limited CSS support compared to WeasyPrint. Some advanced styling is simplified for compatibility.
- Thermal area identification depends on Gemini's ability to recognize rooms from photos. Ambiguous photos may result in generic area names.
- The system works on similar inspection reports but is optimized for the UrbanRoof-style format.

## How to Improve

- Add a paid Gemini API tier for higher rate limits and faster processing
- Integrate WeasyPrint with GTK for higher-quality PDF rendering
- Add a progress bar / WebSocket updates for real-time pipeline status
- Support more inspection report formats with configurable extraction templates
- Add caching to avoid re-processing the same PDFs
