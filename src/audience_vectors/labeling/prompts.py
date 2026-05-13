"""Prompt + structured output schema for synthetic segment labeling.

Keeping the prompt + schema in one module means we can version it as a
single unit. When you change the prompt, bump `PROMPT_VERSION` so
downstream Parquet files can be filtered by which labeler generation
produced them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from audience_vectors.schemas import Persona

PROMPT_VERSION = "v1"


# Dimensions we ask the VLM to score. Keep this list short — adding axes
# costs context per request and dilutes the signal-to-noise ratio. New
# dimensions should land via PR with a short justification.
LABEL_DIMENSIONS: tuple[str, ...] = (
    "attention",
    "memorability",
    "confusion",
    "emotional_intensity",
    "semantic_surprise",
    "narrative_progress",
    "social_salience",
    "visual_salience",
    "audio_salience",
    "rewatch_likelihood",
)


class SegmentLabelOutput(BaseModel):
    """Strict schema returned by the VLM. Every score is bounded [0, 1]."""

    attention: float = Field(ge=0.0, le=1.0)
    memorability: float = Field(ge=0.0, le=1.0)
    confusion: float = Field(ge=0.0, le=1.0)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    semantic_surprise: float = Field(ge=0.0, le=1.0)
    narrative_progress: float = Field(ge=0.0, le=1.0)
    social_salience: float = Field(ge=0.0, le=1.0)
    visual_salience: float = Field(ge=0.0, le=1.0)
    audio_salience: float = Field(ge=0.0, le=1.0)
    rewatch_likelihood: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="One-sentence explanation of the highest-scoring axis.")

    def scores(self) -> dict[str, float]:
        return {dim: getattr(self, dim) for dim in LABEL_DIMENSIONS}


# Gemini response_schema (subset of OpenAPI Schema). Mirror of
# SegmentLabelOutput. Kept hand-written rather than auto-generated so
# the contract is grep-able and review-friendly.
SEGMENT_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{dim: {"type": "number", "minimum": 0.0, "maximum": 1.0} for dim in LABEL_DIMENSIONS},
        "reason": {"type": "string"},
    },
    "required": [*LABEL_DIMENSIONS, "reason"],
}


SEGMENT_LABEL_PROMPT = """\
You are predicting which moments in a video will cause many viewers to
attend to the same thing and remember the moment later. You are NOT
evaluating overall quality.

Score this clip on each axis from 0 (axis not expressed at all) to 1
(axis very strongly expressed). Be precise. Avoid 0.5 unless the axis is
genuinely ambiguous.

Axes:
- attention: how strongly the clip captures and holds viewer attention
- memorability: how likely viewers are to recognize this clip later
- confusion: how likely viewers are to be confused or disoriented
- emotional_intensity: arousal/valence intensity regardless of polarity
- semantic_surprise: how much the content violates the viewer's expectations
- narrative_progress: how much story or causal sequence advances
- social_salience: faces, gaze, social action, emotional expressions
- visual_salience: cuts, motion, contrast, scene change
- audio_salience: speech, music, sudden onsets, sudden silence
- rewatch_likelihood: how likely a viewer is to rewind / replay this moment

Then write a single short sentence in `reason` explaining the
highest-scoring axis.

Return strict JSON matching the schema. No prose outside JSON."""


def _format_weights(d: dict[str, float], k: int = 5) -> str:
    """Top-k axis=weight pairs as a comma-separated string."""
    if not d:
        return "(none)"
    top = sorted(d.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0)


def build_persona_conditioned_prompt(persona: "Persona") -> str:
    """Prepend persona context to the standard label prompt.

    The persona is described via its top attention_weights and top
    dislikes (truncated to the strongest 5 / 3) plus the prose story.
    That keeps the prompt small while still grounding the scores in the
    structured persona profile — without that grounding the LLM tends
    to produce average-viewer scores regardless of the persona.
    """
    top_attn = _format_weights(persona.attention_weights, k=5)
    top_dis = _format_weights(persona.dislikes, k=3)
    persona_block = f"""\
You are scoring this clip as if you were ONE specific viewer, not an
average audience. Score what THIS viewer would feel — attention,
memorability, etc. should reflect their preferences and turn-offs.

Viewer profile:
- id: {persona.persona_id}
- cluster: {persona.cluster}
- about: {persona.story}
- pays strong attention to: {top_attn}
- turn-offs: {top_dis}

A clip can be highly memorable for a cinematic viewer and forgettable
for a fast-scroll viewer. Lean into that — don't average across viewers.

"""
    return persona_block + SEGMENT_LABEL_PROMPT
