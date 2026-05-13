"""Synthetic labeling — generate weak labels for segments via VLMs/LLMs."""

from audience_vectors.labeling.gemini_labeler import GeminiLabeler
from audience_vectors.labeling.persona_generator import (
    DEFAULT_CLUSTERS,
    PersonaGenerator,
)
from audience_vectors.labeling.prompts import (
    SEGMENT_LABEL_PROMPT,
    SEGMENT_LABEL_SCHEMA,
    SegmentLabelOutput,
)

__all__ = [
    "DEFAULT_CLUSTERS",
    "GeminiLabeler",
    "PersonaGenerator",
    "SEGMENT_LABEL_PROMPT",
    "SEGMENT_LABEL_SCHEMA",
    "SegmentLabelOutput",
]
