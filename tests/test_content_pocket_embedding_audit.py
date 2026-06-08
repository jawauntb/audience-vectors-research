from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_audit_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_content_pocket_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_content_pocket_embeddings",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate(
    module,
    *,
    pocket: str,
    label: str,
    vector: list[float],
    score: float,
    seed_idx: int = 0,
):
    arr = module.normalize_vector(np.asarray(vector, dtype=np.float32))
    return module.Candidate(
        task_id=f"{pocket}_{seed_idx}",
        pocket=pocket,
        label=label,
        seed_idx=seed_idx,
        prompt=pocket,
        mean_score=score,
        min_score=score,
        max_score=score,
        n_replicates=1,
        seed_embedding=arr,
        video_embedding=arr,
        vjepa_embedding=arr,
        text_embedding=None,
    )


def synthetic_candidates(module):
    rows = []
    for idx, pocket in enumerate(
        [
            "fresh24_orange_flowers",
            "fresh24_hanging_clothes",
            "fresh24_blue_jellyfish",
            "fresh24_old_car",
        ]
    ):
        rows.append(
            candidate(
                module,
                pocket=pocket,
                label="positive",
                vector=[1.0, 0.1 * idx, 0.0],
                score=2.0 + idx,
                seed_idx=idx,
            )
        )
    for idx, pocket in enumerate(
        [
            "fresh24_aerial_beach",
            "fresh24_city_street",
            "fresh24_storm_beach",
        ]
    ):
        rows.append(
            candidate(
                module,
                pocket=pocket,
                label="negative_control",
                vector=[-1.0, 0.1 * idx, 0.0],
                score=-2.0 - idx,
                seed_idx=idx + 10,
            )
        )
    return rows


def test_pocket_heldout_centroid_margins_separate_synthetic_embeddings():
    module = load_audit_module()
    candidates = synthetic_candidates(module)

    margins = module.pocket_heldout_centroid_margins(
        candidates,
        embedding_name="video_embedding",
    )
    labels = [row.label for row in candidates]
    pos = [margin for margin, label in zip(margins, labels, strict=True) if label == "positive"]
    neg = [
        margin
        for margin, label in zip(margins, labels, strict=True)
        if label == "negative_control"
    ]

    assert min(pos) > 1.5
    assert max(neg) < -1.5


def test_embedding_gate_accepts_strong_descriptor_separator():
    module = load_audit_module()
    candidates = synthetic_candidates(module)

    descriptors = module.descriptor_metrics(candidates)
    gate = module.gate_summary(
        descriptors,
        [],
        min_auc=0.85,
        min_abs_d=1.0,
        min_classifier_auc=0.85,
        min_balanced_accuracy=0.75,
    )

    assert descriptors[0]["separation_auc"] == 1.0
    assert gate["accepted"] is True


def test_leave_pocket_out_classifier_uses_only_available_embedding_family():
    module = load_audit_module()
    candidates = synthetic_candidates(module)

    result = module.leave_pocket_out_classifier(
        candidates,
        embedding_name="vjepa_embedding",
        family="vjepa_video",
    )

    assert result["n_predictions"] == len(candidates)
    assert result["roc_auc"] == 1.0
    assert result["balanced_accuracy"] == 1.0
