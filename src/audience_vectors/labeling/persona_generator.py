"""Anthropic-backed persona archetype generator.

Generates 5–50 structured viewer archetypes — each is a `Persona`:
short prose story + structured attention_weights + dislikes. The
structured weights are what keeps downstream persona-conditioned
labels from being LLM-fan-fiction (see `[[memory: synthetic_personas]]`
in the design notes).

Strategy: use Claude's tool-use API to force structured output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from audience_vectors.schemas import Persona

logger = logging.getLogger(__name__)


# Weight axes — match the shape downstream labelers expect.
ATTENTION_AXES: tuple[str, ...] = (
    "visual_composition",
    "faces",
    "dialogue",
    "motion",
    "novelty",
    "product_clarity",
    "humor",
    "narrative_arc",
    "audio_quality",
    "emotional_intensity",
)

DISLIKE_AXES: tuple[str, ...] = (
    "hard_sell_ads",
    "low_visual_quality",
    "generic_stock_footage",
    "slow_pacing",
    "overly_polished",
    "shaky_camera",
    "dense_information",
    "irrelevance",
)

# Default starter archetypes. The LLM is asked to produce variants within
# (or beyond) these clusters. Override via the `clusters` arg.
DEFAULT_CLUSTERS: tuple[str, ...] = (
    "fast_scroll_short_form",
    "cinematic_aesthetic",
    "narrative_emotional",
    "skeptical_ad_avoidant",
    "technical_product_evaluator",
    "comedy_sensitive",
    "social_drama",
    "educational_information_seeking",
    "audio_music_driven",
    "sports_action",
)


_PROMPT_HEADER = """\
You are designing synthetic viewer archetypes for a video-response
research project. Each archetype is one row of structured data, NOT a
biography. Vary across genres, ages, and platform habits, but keep the
JSON tight and consistent.

For each persona produce:
- persona_id: short kebab-case unique ID
- cluster: one of the archetype family names provided below
- story: ONE sentence, < 220 chars, no demographic stereotyping
- attention_weights: dict from axis -> float in [0, 1] (higher = more drawn to it)
- dislikes: dict from axis -> float in [0, 1] (higher = stronger turn-off)

Use these attention axes (always include all keys, set 0 where not relevant):
{attention_axes}

Use these dislike axes (always include all keys, set 0 where not relevant):
{dislike_axes}

Clusters to draw from (you may produce multiple personas per cluster):
{clusters}

Generate exactly {n} personas. Pass them through the `emit_personas` tool.
"""


def _build_tool_schema() -> dict[str, Any]:
    """Anthropic input_schema for the `emit_personas` tool."""
    attention_props = {a: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                       for a in ATTENTION_AXES}
    dislike_props = {a: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                     for a in DISLIKE_AXES}
    persona_schema = {
        "type": "object",
        "required": ["persona_id", "cluster", "story", "attention_weights", "dislikes"],
        "properties": {
            "persona_id": {"type": "string"},
            "cluster": {"type": "string"},
            "story": {"type": "string"},
            "attention_weights": {
                "type": "object",
                "required": list(ATTENTION_AXES),
                "properties": attention_props,
            },
            "dislikes": {
                "type": "object",
                "required": list(DISLIKE_AXES),
                "properties": dislike_props,
            },
        },
    }
    return {
        "type": "object",
        "required": ["personas"],
        "properties": {
            "personas": {
                "type": "array",
                "items": persona_schema,
            },
        },
    }


class PersonaGeneratorError(RuntimeError):
    """Raised when the persona generator cannot initialize or parse output."""


class PersonaGenerator:
    """Generate structured viewer archetypes via Anthropic tool use."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise PersonaGeneratorError(
                "Anthropic API key is empty; set ANTHROPIC_API_KEY in .env"
            )
        import anthropic  # type: ignore[import-not-found]

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        n: int = 20,
        clusters: tuple[str, ...] = DEFAULT_CLUSTERS,
    ) -> list[Persona]:
        if n < 1:
            raise ValueError("n must be >= 1")
        prompt = _PROMPT_HEADER.format(
            attention_axes="\n".join(f"- {a}" for a in ATTENTION_AXES),
            dislike_axes="\n".join(f"- {a}" for a in DISLIKE_AXES),
            clusters="\n".join(f"- {c}" for c in clusters),
            n=n,
        )
        tool_schema = _build_tool_schema()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[{
                "name": "emit_personas",
                "description": "Emit structured viewer archetypes.",
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": "emit_personas"},
            messages=[{"role": "user", "content": prompt}],
        )

        payload = self._extract_tool_payload(response)
        personas_raw = payload.get("personas", [])
        if not isinstance(personas_raw, list) or not personas_raw:
            raise PersonaGeneratorError("model returned no personas")

        personas: list[Persona] = []
        seen_ids: set[str] = set()
        for raw in personas_raw:
            pid = str(raw.get("persona_id") or "").strip() or f"p_{len(personas):03d}"
            # Dedupe IDs — model can occasionally repeat them across batches.
            base = pid
            i = 1
            while pid in seen_ids:
                pid = f"{base}_{i}"
                i += 1
            seen_ids.add(pid)
            personas.append(Persona(
                persona_id=pid,
                cluster=str(raw.get("cluster") or "unknown"),
                attention_weights={k: float(v) for k, v in (raw.get("attention_weights") or {}).items()},
                dislikes={k: float(v) for k, v in (raw.get("dislikes") or {}).items()},
                story=str(raw.get("story") or ""),
            ))
        return personas

    def _extract_tool_payload(self, response: Any) -> dict[str, Any]:
        """Pull the JSON args out of the first `tool_use` content block."""
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                inp = getattr(block, "input", None)
                if isinstance(inp, dict):
                    return inp
                if isinstance(inp, str):
                    try:
                        return json.loads(inp)
                    except json.JSONDecodeError as exc:
                        raise PersonaGeneratorError(
                            f"tool_use input was not valid JSON: {exc}"
                        ) from exc
        raise PersonaGeneratorError(
            "no tool_use block in Anthropic response — model didn't call the tool"
        )
