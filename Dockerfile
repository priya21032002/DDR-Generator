FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMuPDF, Tesseract, and building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    build-essential \
    libffi-dev \
    libssl-dev \
    cargo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to slim down the final image
RUN apt-get purge -y --auto-remove build-essential cargo

COPY src/ src/
COPY static/ static/
COPY sample_inputs/ sample_inputs/

# Create output directories
RUN mkdir -p outputs uploads

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
