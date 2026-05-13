"""Tests for the Gemini labeler.

Doesn't actually hit the Gemini API — uses a stubbed `google.genai`
module installed into sys.modules so the labeler's full async path
runs against a deterministic fake response.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from audience_vectors.labeling.prompts import LABEL_DIMENSIONS, SegmentLabelOutput
from audience_vectors.schemas import LabelSource, Segment


# ---------------------------------------------------------------------------
# Fake google.genai module — installed before importing the labeler.
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    text: str


class _FakePart:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_bytes(cls, *, data, mime_type):
        return cls(data=data, mime_type=mime_type, kind="bytes")

    @classmethod
    def from_uri(cls, *, file_uri, mime_type):
        return cls(file_uri=file_uri, mime_type=mime_type, kind="uri")


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModels:
    def __init__(self, response_text: str):
        self._text = response_text
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(text=self._text)


class _FakeFiles:
    def __init__(self):
        self.uploads: list[str] = []

    def upload(self, *, file, config):
        self.uploads.append(file)
        return types.SimpleNamespace(uri=f"gs://fake/{Path(file).name}")


class _FakeClient:
    def __init__(self, *, api_key, response_text: str):
        self.api_key = api_key
        self.models = _FakeModels(response_text)
        self.files = _FakeFiles()


def _install_fake_genai(response_payload: dict | str) -> _FakeClient:
    """Install a fake `google.genai` module before the labeler imports it.

    Returns the fake client so tests can inspect call history.
    """
    text = response_payload if isinstance(response_payload, str) else json.dumps(response_payload)

    fake_client_holder = {"client": None}

    class _ClientFactory:
        def __new__(cls, *, api_key):
            client = _FakeClient(api_key=api_key, response_text=text)
            fake_client_holder["client"] = client
            return client

    types_mod = types.ModuleType("google.genai.types")
    types_mod.Part = _FakePart
    types_mod.GenerateContentConfig = _FakeGenerateContentConfig

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _ClientFactory
    genai_mod.types = types_mod

    google_mod = types.ModuleType("google")
    google_mod.genai = genai_mod

    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod
    return fake_client_holder  # client populated after Client(...) call


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _valid_payload(**overrides) -> dict:
    base = {dim: 0.5 for dim in LABEL_DIMENSIONS}
    base.update(overrides)
    base.setdefault("reason", "Strong faces and emotional turn.")
    return base


def test_schema_validates_clean_payload():
    out = SegmentLabelOutput.model_validate(_valid_payload(attention=0.91))
    assert out.attention == 0.91
    scores = out.scores()
    assert set(scores.keys()) == set(LABEL_DIMENSIONS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_schema_rejects_out_of_range():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SegmentLabelOutput.model_validate(_valid_payload(attention=1.5))


def test_labeler_requires_api_key():
    _install_fake_genai(_valid_payload())
    from audience_vectors.labeling import GeminiLabeler
    from audience_vectors.labeling.gemini_labeler import GeminiLabelerError

    with pytest.raises(GeminiLabelerError):
        GeminiLabeler(api_key="")


def test_label_segment_happy_path(tmp_path: Path):
    holder = _install_fake_genai(_valid_payload(attention=0.82, memorability=0.74))
    # Drop a real (empty-ish) .mp4 file so the labeler's existence check passes.
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # bare MP4 magic
    seg = Segment(
        sample_id="seg_0001",
        source_dataset="Synth",
        video_id="vid_001",
        start_time=0.0,
        end_time=3.0,
        duration=3.0,
        media_path=str(media),
    )

    from audience_vectors.labeling import GeminiLabeler

    labeler = GeminiLabeler(api_key="fake", model="gemini-2.0-flash", max_concurrency=2)
    result = asyncio.run(labeler.label_segment(seg))

    assert result is not None
    assert result.segment_id == "seg_0001"
    assert result.scores["attention"] == 0.82
    assert result.scores["memorability"] == 0.74
    assert result.source == LabelSource.SYNTHETIC_VLM
    assert result.model_id == "gemini-2.0-flash"
    assert holder["client"].models.calls, "Gemini.generate_content was not invoked"


def test_label_segment_missing_media_returns_none(tmp_path: Path):
    _install_fake_genai(_valid_payload())
    seg = Segment(
        sample_id="seg_missing",
        source_dataset="Synth",
        video_id="vid_x",
        start_time=0.0, end_time=3.0, duration=3.0,
        media_path=str(tmp_path / "does_not_exist.mp4"),
    )
    from audience_vectors.labeling import GeminiLabeler

    labeler = GeminiLabeler(api_key="fake")
    assert asyncio.run(labeler.label_segment(seg)) is None


def test_label_segment_bad_json_returns_none(tmp_path: Path):
    _install_fake_genai("not even close to JSON")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\x00")
    seg = Segment(
        sample_id="seg_bad",
        source_dataset="Synth",
        video_id="vid_x",
        start_time=0.0, end_time=3.0, duration=3.0,
        media_path=str(media),
    )
    from audience_vectors.labeling import GeminiLabeler

    labeler = GeminiLabeler(api_key="fake")
    assert asyncio.run(labeler.label_segment(seg)) is None


def test_label_many_drops_failures(tmp_path: Path):
    _install_fake_genai(_valid_payload())
    good = tmp_path / "good.mp4"
    good.write_bytes(b"\x00")
    seg_good = Segment(
        sample_id="seg_good",
        source_dataset="Synth", video_id="v",
        start_time=0.0, end_time=3.0, duration=3.0,
        media_path=str(good),
    )
    seg_bad = Segment(
        sample_id="seg_bad",
        source_dataset="Synth", video_id="v",
        start_time=0.0, end_time=3.0, duration=3.0,
        media_path=str(tmp_path / "missing.mp4"),
    )
    from audience_vectors.labeling import GeminiLabeler

    labeler = GeminiLabeler(api_key="fake", max_concurrency=2)
    results = asyncio.run(labeler.label_many([seg_good, seg_bad]))
    assert len(results) == 1
    assert results[0].segment_id == "seg_good"
