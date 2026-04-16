"""Unified LLM client — Groq primary, Gemini fallback.

Both providers are free-tier. Groq is faster for text-only calls;
Gemini supports vision (required for Structure stage with page images).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TypeVar

from google import genai
from google.genai import types as genai_types
from groq import AsyncGroq
from pydantic import BaseModel

from src.models.schemas import LLMCallLog

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
RUN_LOG_PATH = Path("outputs/run_log.jsonl")


def _ensure_log_dir() -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log_call(entry: LLMCallLog) -> None:
    """Append one JSONL row to the run log."""
    _ensure_log_dir()
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def _schema_instruction(schema_cls: type[BaseModel]) -> str:
    """Return a prompt fragment instructing the LLM to output this schema."""
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    return (
        f"Respond with valid JSON matching this schema:\n"
        f"```json\n{schema_json}\n```\n"
        f"Output ONLY the JSON object, no markdown fences."
    )


# ---------------------------------------------------------------------------
# Groq (primary — text only)
# ---------------------------------------------------------------------------

async def _call_groq(
    prompt: str,
    system: str,
    response_schema: type[T],
    run_id: str,
    stage: str,
) -> T:
    """Call Groq with JSON-mode output parsed into *response_schema*."""
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

    system_with_schema = f"{system}\n\n{_schema_instruction(response_schema)}"

    t0 = time.perf_counter()
    chat = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_with_schema},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    usage = chat.usage
    raw = chat.choices[0].message.content
    result = response_schema.model_validate_json(raw)

    _log_call(LLMCallLog(
        run_id=run_id,
        stage=stage,
        provider="groq",
        model=GROQ_MODEL,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
        success=True,
    ))
    return result


# ---------------------------------------------------------------------------
# Gemini (fallback — also handles vision calls)
# ---------------------------------------------------------------------------

def _get_gemini_client() -> genai.Client:
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=genai_types.HttpOptions(
            # Disable SDK's internal retry — we handle retries ourselves
            # to avoid doubling up and burning rate limit quota
            retryOptions=genai_types.HttpRetryOptions(attempts=1),
        ),
    )


async def _call_gemini_text(
    prompt: str,
    system: str,
    response_schema: type[T],
    run_id: str,
    stage: str,
) -> T:
    """Gemini text-only call. Tries primary model, then fallback on quota errors."""
    client = _get_gemini_client()
    full_prompt = f"{system}\n\n{_schema_instruction(response_schema)}\n\n{prompt}"

    for model in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        try:
            t0 = time.perf_counter()
            response = await client.aio.models.generate_content(
                model=model,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            raw = response.text
            result = response_schema.model_validate_json(raw)

            usage_meta = response.usage_metadata
            _log_call(LLMCallLog(
                run_id=run_id, stage=stage, provider="gemini", model=model,
                input_tokens=getattr(usage_meta, "prompt_token_count", 0),
                output_tokens=getattr(usage_meta, "candidates_token_count", 0),
                latency_ms=latency_ms, success=True,
            ))
            return result
        except Exception as exc:
            logger.warning("Gemini %s text failed (%s), trying next model", model, exc)
            _log_call(LLMCallLog(
                run_id=run_id, stage=stage, provider="gemini", model=model,
                input_tokens=0, output_tokens=0, latency_ms=0,
                success=False, error=str(exc),
            ))
    raise RuntimeError(f"All Gemini text models failed for stage={stage}")


async def _gemini_vision_once(
    client: genai.Client,
    model: str,
    text_part: str,
    image_parts: list[genai_types.Part],
    response_schema: type[T],
    run_id: str,
    stage: str,
) -> T:
    """Single Gemini vision attempt."""
    t0 = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=model,
        contents=[text_part, *image_parts],
        config=genai_types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = response.text
    result = response_schema.model_validate_json(raw)

    usage_meta = response.usage_metadata
    _log_call(LLMCallLog(
        run_id=run_id,
        stage=stage,
        provider="gemini",
        model=model,
        input_tokens=getattr(usage_meta, "prompt_token_count", 0),
        output_tokens=getattr(usage_meta, "candidates_token_count", 0),
        latency_ms=latency_ms,
        success=True,
    ))
    return result


async def call_gemini_vision(
    prompt: str,
    system: str,
    image_parts: list[genai_types.Part],
    response_schema: type[T],
    run_id: str,
    stage: str,
    max_retries: int = 4,
) -> T:
    """Gemini vision call with retry and model fallback.

    Tries GEMINI_MODEL first, then falls back to GEMINI_FALLBACK_MODEL
    if the primary quota is exhausted (separate daily quotas per model).
    """
    client = _get_gemini_client()
    text_part = f"{system}\n\n{_schema_instruction(response_schema)}\n\n{prompt}"

    # Try primary model with retries
    for attempt in range(max_retries):
        try:
            return await _gemini_vision_once(
                client, GEMINI_MODEL, text_part, image_parts,
                response_schema, run_id, stage,
            )
        except Exception as exc:
            wait = _extract_retry_delay(exc) or (15 * (attempt + 1))
            logger.warning(
                "Gemini %s vision attempt %d/%d failed (%s), retrying in %ds...",
                GEMINI_MODEL, attempt + 1, max_retries, type(exc).__name__, wait,
            )
            _log_call(LLMCallLog(
                run_id=run_id, stage=stage, provider="gemini", model=GEMINI_MODEL,
                input_tokens=0, output_tokens=0, latency_ms=0,
                success=False, error=str(exc),
            ))
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)

    # Primary model exhausted — try fallback model
    logger.warning("Primary model %s exhausted, trying fallback %s", GEMINI_MODEL, GEMINI_FALLBACK_MODEL)
    for attempt in range(max_retries):
        try:
            return await _gemini_vision_once(
                client, GEMINI_FALLBACK_MODEL, text_part, image_parts,
                response_schema, run_id, stage,
            )
        except Exception as exc:
            wait = _extract_retry_delay(exc) or (15 * (attempt + 1))
            logger.warning(
                "Gemini %s vision attempt %d/%d failed (%s), retrying in %ds...",
                GEMINI_FALLBACK_MODEL, attempt + 1, max_retries, type(exc).__name__, wait,
            )
            _log_call(LLMCallLog(
                run_id=run_id, stage=stage, provider="gemini", model=GEMINI_FALLBACK_MODEL,
                input_tokens=0, output_tokens=0, latency_ms=0,
                success=False, error=str(exc),
            ))
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)

    raise RuntimeError(f"All Gemini vision attempts exhausted for stage={stage}")


def _extract_retry_delay(exc: Exception) -> int | None:
    """Try to parse a retry delay from Gemini's 429 error message."""
    import re
    msg = str(exc)
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m:
        return int(float(m.group(1))) + 5  # add buffer
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return 30  # default for rate limit
    return None


# ---------------------------------------------------------------------------
# Unified dispatcher — Groq primary, Gemini fallback (text-only calls)
# ---------------------------------------------------------------------------

async def call_llm(
    prompt: str,
    system: str,
    response_schema: type[T],
    run_id: str,
    stage: str,
) -> T:
    """Try Groq first; on any failure fall back to Gemini.

    For vision calls, use ``call_gemini_vision`` directly (Groq has no vision).
    """
    # Try Groq up to 2 times before falling back
    for attempt in range(2):
        try:
            return await _call_groq(prompt, system, response_schema, run_id, stage)
        except Exception as exc:
            logger.warning("Groq attempt %d failed (%s)", attempt + 1, exc)
            _log_call(LLMCallLog(
                run_id=run_id,
                stage=stage,
                provider="groq",
                model=GROQ_MODEL,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                success=False,
                error=str(exc),
            ))
            if attempt == 0:
                await asyncio.sleep(5)

    # All Groq attempts failed — fall back to Gemini with retry
    logger.warning("Groq exhausted, falling back to Gemini")
    for attempt in range(3):
        try:
            return await _call_gemini_text(prompt, system, response_schema, run_id, stage)
        except Exception as exc:
            wait = 2 ** attempt * 10
            logger.warning("Gemini text attempt %d failed (%s), retrying in %ds...", attempt + 1, exc, wait)
            if attempt < 2:
                await asyncio.sleep(wait)
    raise RuntimeError(f"Both Groq and Gemini failed for stage={stage}")
