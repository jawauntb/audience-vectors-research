"""Smoke tests for canonical schemas. Run with `pytest`."""

from audience_vectors.schemas import (
    AudienceVector,
    LabelSource,
    LabelValue,
    Persona,
    Segment,
)


def test_label_value_roundtrip():
    lv = LabelValue(
        value=0.74,
        raw_value=0.74,
        percentile=0.86,
        confidence=1.0,
        source=LabelSource.HUMAN_ORIGINAL,
        granularity="video",
    )
    assert lv.value == 0.74
    assert lv.source is LabelSource.HUMAN_ORIGINAL
    assert lv.granularity == "video"


def test_segment_id_convention():
    seg = Segment(
        sample_id="videomem_000123_seg_0004",
        source_dataset="VideoMem",
        video_id="videomem_000123",
        start_time=12.0,
        end_time=15.0,
        duration=3.0,
        labels={
            "memorability": LabelValue(
                value=0.74,
                source=LabelSource.HUMAN_ORIGINAL,
                granularity="video",
            ),
            "attention": LabelValue(
                value=0.69,
                source=LabelSource.SYNTHETIC_VLM,
            ),
        },
    )
    assert seg.duration == 3.0
    assert seg.label_sources["memorability"] is LabelSource.HUMAN_ORIGINAL
    assert seg.label_sources["attention"] is LabelSource.SYNTHETIC_VLM


def test_persona_structured_weights():
    p = Persona(
        persona_id="p017",
        cluster="cinematic_aesthetic",
        attention_weights={"visual_composition": 0.9, "humor": 0.3},
        dislikes={"hard_sell_ads": 0.8},
        story="Film-school grad who notices camera movement and symbolism.",
    )
    assert p.attention_weights["visual_composition"] == 0.9
    assert "hard_sell_ads" in p.dislikes


def test_audience_vector_artifact():
    v = AudienceVector(
        vector_id="memorability_v1",
        target="memorability",
        model_id="facebook/tribev2",
        layer="cortical_output",
        direction_uri="./data/models/vectors/memorability_v1.safetensors",
        dim=327684,
        positive_set_size=120,
        negative_set_size=120,
    )
    assert v.target == "memorability"
    assert v.dim > 0
