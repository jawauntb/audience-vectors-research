"""Gemini-backed synthetic labeler.

Wraps `google-genai` to score one video segment at a time, with built-in
async concurrency control, JSON-mode structured output, and graceful
soft failure (returns None on transient errors so a single bad clip
doesn't kill a batch).

Two ways to submit the clip to Gemini:

  - **Inline bytes** for clips under ~18 MB (default). BMD's ~600 KB clips
    fit easily.
  - **File API upload** for larger media. Toggled with `use_file_api=True`.

Both paths go through the same prompt + response schema in
`audience_vectors.labeling.prompts`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from audience_vectors.labeling.prompts import (
    PROMPT_VERSION,
    SEGMENT_LABEL_PROMPT,
    SEGMENT_LABEL_SCHEMA,
    SegmentLabelOutput,
    build_persona_conditioned_prompt,
)
from audience_vectors.schemas import LabelSource, Persona, Segment, SyntheticLabel

logger = logging.getLogger(__name__)

# Gemini inline-data soft cap. The hard limit is ~20 MB per request; leave
# headroom for the prompt + response framing. Above this, switch to the
# File API.
_INLINE_BYTES_CAP = 18 * 1024 * 1024


class GeminiLabelerError(RuntimeError):
    """Raised when the labeler cannot even initialize (no key, bad model)."""


class GeminiLabeler:
    """Score one video segment per call.

    Construct once with an API key + model, call `label_segment` or
    `label_many`. The underlying google-genai client is reused.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_concurrency: int = 4,
        request_timeout_s: float = 60.0,
        use_file_api: bool = False,
    ) -> None:
        if not api_key:
            raise GeminiLabelerError(
                "Gemini API key is empty; set GOOGLE_API_KEY in .env"
            )
        # Inline import — keeps `from audience_vectors.labeling import ...`
        # cheap and lets tests stub the module via sys.modules.
        from google import genai  # type: ignore[import-not-found]

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_concurrency = max(1, max_concurrency)
        self.request_timeout_s = request_timeout_s
        self.use_file_api = use_file_api
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    # -- public API --------------------------------------------------------

    async def label_segment(
        self,
        segment: Segment,
        *,
        persona: Persona | None = None,
    ) -> SyntheticLabel | None:
        """Score one segment. Returns None on soft failure (network, validation).

        When `persona` is provided, prompt is conditioned on that viewer's
        attention_weights + dislikes so scores reflect a persona-specific
        response rather than a global average.
        """
        if not segment.media_path:
            logger.warning("segment %s has no media_path; skipping", segment.sample_id)
            return None
        path = Path(segment.media_path)
        if not path.exists():
            logger.warning("segment %s media missing on disk: %s", segment.sample_id, path)
            return None

        async with self._semaphore:
            try:
                output = await self._call_gemini(path, persona=persona)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Gemini label call failed for %s: %s", segment.sample_id, exc)
                return None

        return SyntheticLabel(
            segment_id=segment.sample_id,
            persona_id=persona.persona_id if persona else None,
            scores=output.scores(),
            reason=output.reason,
            source=LabelSource.SYNTHETIC_VLM,
            prompt_version=f"{PROMPT_VERSION}+persona" if persona else PROMPT_VERSION,
            model_id=self.model,
        )

    async def label_many(
        self,
        segments: list[Segment],
        *,
        persona: Persona | None = None,
    ) -> list[SyntheticLabel]:
        """Run labeling over a list of segments. Concurrency is bounded by
        `max_concurrency`. None results (soft failures) are dropped."""
        coros = [self.label_segment(s, persona=persona) for s in segments]
        results = await asyncio.gather(*coros)
        return [r for r in results if r is not None]

    async def label_persona_grid(
        self,
        segments: list[Segment],
        personas: list[Persona],
    ) -> list[SyntheticLabel]:
        """Score every (persona × segment) pair. ~|personas| × |segments| API calls."""
        coros = [
            self.label_segment(s, persona=p)
            for p in personas
            for s in segments
        ]
        results = await asyncio.gather(*coros)
        return [r for r in results if r is not None]

    # -- internal ---------------------------------------------------------

    async def _call_gemini(
        self,
        path: Path,
        *,
        persona: Persona | None = None,
    ) -> SegmentLabelOutput:
        """Submit the clip + prompt and parse the structured response."""
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None:
            mime = "video/mp4"

        size = path.stat().st_size
        if self.use_file_api or size > _INLINE_BYTES_CAP:
            part = await self._upload_file_part(path, mime)
        else:
            data = path.read_bytes()
            part = self._genai.types.Part.from_bytes(data=data, mime_type=mime)

        config = self._genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SEGMENT_LABEL_SCHEMA,
        )
        prompt = (
            build_persona_conditioned_prompt(persona)
            if persona is not None
            else SEGMENT_LABEL_PROMPT
        )
        contents = [part, prompt]

        start = time.monotonic()
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            ),
            timeout=self.request_timeout_s,
        )
        elapsed = time.monotonic() - start
        logger.debug("Gemini call: %.2fs for %s", elapsed, path.name)

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned no text payload")
        try:
            payload: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini response was not valid JSON: {text[:200]}") from exc
        try:
            return SegmentLabelOutput.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"Gemini response failed schema validation: {exc}") from exc

    async def _upload_file_part(self, path: Path, mime: str):
        """Upload via the File API and return a Part referencing the URI."""
        uploaded = await asyncio.to_thread(
            self.client.files.upload,
            file=str(path),
            config={"mime_type": mime},
        )
        # File API uploads return a File object with a `uri` and `mime_type`.
        return self._genai.types.Part.from_uri(
            file_uri=uploaded.uri,
            mime_type=mime,
        )
