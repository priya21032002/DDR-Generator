"""Stage 3 — Merge: Fuse inspection + thermal observations by physical area.

Groups observations by area, fuses findings from both sources,
flags conflicts and coverage gaps.
"""

from __future__ import annotations

import json
import logging

from src.llm.client import call_llm
from src.llm.prompts import MERGE_SYSTEM, MERGE_USER_TEMPLATE
from src.models.schemas import MergedFindings, Observation, StructuredDocument

logger = logging.getLogger(__name__)


def _observations_to_json(observations: list[Observation]) -> str:
    """Serialize observations to a readable JSON string for the LLM prompt."""
    return json.dumps(
        [obs.model_dump(mode="json") for obs in observations],
        indent=2,
    )


async def merge_documents(
    inspection: StructuredDocument,
    thermal: StructuredDocument,
    run_id: str,
) -> MergedFindings:
    """Merge inspection and thermal observations into area-grouped findings."""
    inspection_json = _observations_to_json(inspection.observations)
    thermal_json = _observations_to_json(thermal.observations)

    user_prompt = MERGE_USER_TEMPLATE.format(
        inspection_json=inspection_json,
        thermal_json=thermal_json,
    )

    result = await call_llm(
        prompt=user_prompt,
        system=MERGE_SYSTEM,
        response_schema=MergedFindings,
        run_id=run_id,
        stage="merge",
    )

    logger.info(
        "Merged into %d areas, %d coverage gaps",
        len(result.areas), len(result.coverage_gaps),
    )
    return result
