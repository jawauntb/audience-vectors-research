"""Run attention-capture ROI scoring as a BOLD Moments control.

This is not a Phase 1 attention-capture validation dataset. It asks whether the
new capture proxy accidentally tracks BOLD Moments memorability in the existing
local TRIBE cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.attention_capture import (
    CaptureRow,
    capture_scores_from_roi_values,
    load_destrieux_roi_masks,
    load_roi_masks_npz,
    load_tribe_feature_mean,
    render_phase1_markdown,
    roi_values_from_feature_vector,
    run_capture_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a BMD memorability control for the capture-score proxy.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/Users/jawaun/isc_mod/data/raw/bold_moments/annotations.json"),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("/Users/jawaun/isc_mod/data/features/tribe"),
    )
    parser.add_argument(
        "--roi-masks",
        type=Path,
        default=None,
        help="Optional frozen ROI mask NPZ. Defaults to live Destrieux masks.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--gate-rho", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _load_bmd_memorability(args.annotations)
    roi_masks = (
        load_roi_masks_npz(args.roi_masks)
        if args.roi_masks is not None
        else load_destrieux_roi_masks()
    )
    rows = []

    for feature_path in sorted(args.feature_dir.glob("bmd_vid_idx*_seg_0000.npz")):
        video_id = feature_path.stem.removeprefix("bmd_").removesuffix("_seg_0000")
        label_key = video_id.removeprefix("vid_idx")
        ground_truth = labels.get(label_key)
        if ground_truth is None:
            continue

        feature_vector = load_tribe_feature_mean(feature_path)
        roi_values = roi_values_from_feature_vector(feature_vector, roi_masks)
        scores = capture_scores_from_roi_values(roi_values)
        rows.append(
            CaptureRow(
                sample_id=feature_path.stem,
                dataset="BOLD_Moments_control",
                ground_truth=ground_truth,
                roi_values=roi_values,
                sensory_mean=scores["sensory_mean"],
                capture_score=scores["capture_score"],
                capture_delta=scores["capture_delta"],
                frontoparietal=scores["frontoparietal"],
                denominator_valid=scores["denominator_valid"],
            )
        )

    report = run_capture_rows(
        rows,
        manifest_path=str(args.annotations),
        manifest_status="real_control_not_attention_capture",
        permutations=args.permutations,
        seed=args.seed,
        gate_rho=args.gate_rho,
        include_rows=False,
    )
    report["control"] = {
        "name": "BOLD Moments memorability control",
        "ground_truth_name": "memorability_score",
        "feature_dir": str(args.feature_dir),
        "roi_source": str(args.roi_masks)
        if args.roi_masks is not None
        else "nilearn fetch_atlas_surf_destrieux exploratory masks",
        "interpretation": (
            "Control-only run. A positive result would show overlap with "
            "memorability labels, not validation of attentional capture."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_phase1_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def _load_bmd_memorability(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or "memorability_score" not in value:
            continue
        out[str(key).zfill(4)] = float(value["memorability_score"])
    return out


if __name__ == "__main__":
    main()
