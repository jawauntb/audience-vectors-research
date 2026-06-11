"""Modal functions for attention-capture Phase 1 scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from audience_vectors.attention_capture_modal_volume import (
    DEFAULT_MODAL_FEATURES_MOUNT,
    DEFAULT_MODAL_FEATURES_VOLUME_NAME,
    load_roi_masks_npz_bytes,
    run_phase1_modal_volume_features,
)
from audience_vectors.modal_app.app import app, env_secrets
from audience_vectors.modal_app.image_factory import base_image

attention_capture_features_volume = modal.Volume.from_name(
    DEFAULT_MODAL_FEATURES_VOLUME_NAME,
    create_if_missing=False,
)


@app.function(
    image=base_image,
    volumes={DEFAULT_MODAL_FEATURES_MOUNT: attention_capture_features_volume},
    secrets=env_secrets,
    timeout=20 * 60,
)
def score_attention_capture_phase1_modal_volume(
    label_records: list[dict[str, Any]],
    roi_masks_npz: bytes,
    output_prefix: str,
    *,
    manifest_status: str = "real_external_attention_labels",
    dataset: str = "DHF1K",
    ground_truth_name: str = "mean_fixation_density",
    label_audit: dict[str, Any] | None = None,
    require_label_audit: bool = True,
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
    permutations: int = 999,
    seed: int = 17,
    gate_rho: float = 0.40,
    epsilon: float = 1e-6,
    include_rows: bool = True,
) -> dict[str, Any]:
    """Run Phase 1 scoring over feature files already stored in Modal."""

    attention_capture_features_volume.reload()
    roi_masks = load_roi_masks_npz_bytes(roi_masks_npz)
    return run_phase1_modal_volume_features(
        label_records=label_records,
        roi_masks=roi_masks,
        feature_root=Path(DEFAULT_MODAL_FEATURES_MOUNT),
        output_prefix=output_prefix,
        modal_volume_name=DEFAULT_MODAL_FEATURES_VOLUME_NAME,
        modal_mount=DEFAULT_MODAL_FEATURES_MOUNT,
        manifest_status=manifest_status,
        dataset=dataset,
        ground_truth_name=ground_truth_name,
        label_audit=label_audit,
        require_label_audit=require_label_audit,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
        permutations=permutations,
        seed=seed,
        gate_rho=gate_rho,
        epsilon=epsilon,
        include_rows=include_rows,
    )
