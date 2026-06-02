"""Score selector candidates with a V-JEPA memorability direction.

This script assumes V-JEPA embeddings have already been extracted for the video
labels in the selector manifest. It projects each embedding onto the BMD-trained
V-JEPA memorability direction and can write an augmented manifest with a
`vjepa_memorability_best` policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_direction(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    direction = np.asarray(payload["direction"], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError(f"direction has near-zero norm: {path}")
    return direction / norm


def load_embedding(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "embedding" not in payload.files:
        raise KeyError(f"{path} has no 'embedding' array")
    embedding = np.asarray(payload["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        return embedding
    return embedding / norm


def candidate_labels(row: dict[str, Any]) -> dict[str, str]:
    labels = row["labels"]
    return {str(policy): str(label) for policy, label in labels.items()}


def score_manifest(
    *,
    manifest: dict[str, Any],
    direction: np.ndarray,
    features_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for row in manifest["rows"]:
        scores_by_label: dict[str, float] = {}
        labels = candidate_labels(row)
        for label in sorted(set(labels.values())):
            feature_path = features_dir / f"{label}.npz"
            if not feature_path.exists():
                missing.append(
                    {
                        "seed": str(row["seed"]),
                        "label": label,
                        "feature_path": str(feature_path),
                    }
                )
                continue
            embedding = load_embedding(feature_path)
            scores_by_label[label] = float(embedding @ direction)

        if scores_by_label:
            best_label = max(scores_by_label, key=lambda label: scores_by_label[label])
            best_score: float | None = scores_by_label[best_label]
        else:
            best_label = None
            best_score = None

        rows.append(
            {
                "seed": row["seed"],
                "vjepa_best_label": best_label,
                "vjepa_best_score": best_score,
                "scores_by_label": scores_by_label,
                "n_scored": len(scores_by_label),
                "n_expected_unique": len(set(labels.values())),
            }
        )

    complete_rows = [row for row in rows if row["n_scored"] == row["n_expected_unique"]]
    return {
        "schema_version": 1,
        "source_manifest": manifest["source"],
        "features_dir": str(features_dir),
        "n_seeds": len(rows),
        "n_complete_seeds": len(complete_rows),
        "n_missing_features": len(missing),
        "rows": rows,
        "missing_features": missing,
    }


def augment_manifest(
    *,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    report_by_seed = {row["seed"]: row for row in report["rows"]}
    out = dict(manifest)
    out["baseline_selectors"] = dict(manifest.get("baseline_selectors", {}))
    out["baseline_selectors"]["vjepa_memorability_best"] = (
        "highest projection on the BMD-trained V-JEPA memorability direction"
    )
    out_rows = []
    for row in manifest["rows"]:
        new_row = dict(row)
        labels = dict(row["labels"])
        video_paths = dict(row["video_paths"])
        scores = dict(row["scores"])
        metadata = dict(row["selector_metadata"])
        report_row = report_by_seed.get(row["seed"])
        if report_row and report_row["vjepa_best_label"]:
            best_label = str(report_row["vjepa_best_label"])
            labels["vjepa_memorability_best"] = best_label
            video_paths["vjepa_memorability_best"] = video_paths.get(
                next(
                    policy
                    for policy, label in labels.items()
                    if label == best_label and policy != "vjepa_memorability_best"
                )
            )
            scores["vjepa_memorability_best"] = report_row["vjepa_best_score"]
            metadata["vjepa_n_scored"] = report_row["n_scored"]
        else:
            labels["vjepa_memorability_best"] = None
            video_paths["vjepa_memorability_best"] = None
            scores["vjepa_memorability_best"] = None
        new_row["labels"] = labels
        new_row["video_paths"] = video_paths
        new_row["scores"] = scores
        new_row["selector_metadata"] = metadata
        out_rows.append(new_row)
    out["rows"] = out_rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_manifest.json"
        ),
    )
    parser.add_argument(
        "--vjepa-vector",
        type=Path,
        default=Path(
            "data/models/vectors/"
            "facebook__vjepa2-vitl-fpc64-256__vjepa_mean_pool__"
            "bmd_memorability_n1026.npz"
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features/vjepa_wan22_selector_pref_weighted_r16_s300"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "vjepa_selector_report.json"
        ),
    )
    parser.add_argument("--augmented-manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    direction = load_direction(args.vjepa_vector)
    report = score_manifest(
        manifest=manifest,
        direction=direction,
        features_dir=args.features_dir,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")

    if args.augmented_manifest is not None:
        augmented = augment_manifest(manifest=manifest, report=report)
        args.augmented_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.augmented_manifest.write_text(json.dumps(augmented, indent=2) + "\n")

    print(f"[done] wrote {args.out_json}")
    print(
        "[done] complete seeds: "
        f"{report['n_complete_seeds']}/{report['n_seeds']}; "
        f"missing features: {report['n_missing_features']}"
    )


if __name__ == "__main__":
    main()
