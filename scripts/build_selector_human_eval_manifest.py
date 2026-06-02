"""Build a human-eval manifest from Wan selector reports.

The manifest is intentionally simple: one row per prompt seed with base video,
single-LoRA video, TRIBE-selected videos, CLIP-only baseline selectors, and final
product-selected video. This is the bridge from proxy-scored selector reports to
blinded human validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def video_path_for_label(
    label: str,
    *,
    single_dir: Path,
    bon_dir: Path,
) -> Path:
    if label.endswith("_base") or label.endswith("_lora"):
        return single_dir / f"{label}.mp4"
    return bon_dir / f"{label}.mp4"


def existing_or_none(path: Path) -> str | None:
    return str(path) if path.exists() else None


def composite_rows_by_label(*reports: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for report in reports:
        for row in report.get("rows", []):
            label = str(row["label"])
            rows[label] = row
    return rows


def add_clip_scores(
    rows_by_label: dict[str, dict[str, Any]],
    *,
    image_weight: float,
    prompt_weight: float,
) -> None:
    seed_values = [float(row["seed_image_cosine"]) for row in rows_by_label.values()]
    prompt_values = [float(row["prompt_clip_cosine"]) for row in rows_by_label.values()]
    seed_mean = sum(seed_values) / len(seed_values)
    prompt_mean = sum(prompt_values) / len(prompt_values)
    seed_std = max(
        (sum((value - seed_mean) ** 2 for value in seed_values) / len(seed_values))
        ** 0.5,
        1e-12,
    )
    prompt_std = max(
        (
            sum((value - prompt_mean) ** 2 for value in prompt_values)
            / len(prompt_values)
        )
        ** 0.5,
        1e-12,
    )

    for row in rows_by_label.values():
        seed_z = (float(row["seed_image_cosine"]) - seed_mean) / seed_std
        prompt_z = (float(row["prompt_clip_cosine"]) - prompt_mean) / prompt_std
        row["clip_seed_image_z"] = seed_z
        row["clip_prompt_z"] = prompt_z
        row["clip_preservation_score"] = (
            image_weight * seed_z + prompt_weight * prompt_z
        )


def rows_for_seed(
    rows_by_label: dict[str, dict[str, Any]], seed: str
) -> list[dict[str, Any]]:
    return [row for row in rows_by_label.values() if row["seed_key"] == seed]


def best_label(seed_rows: list[dict[str, Any]], key: str) -> str:
    if not seed_rows:
        raise ValueError("cannot select from empty seed row list")
    return str(max(seed_rows, key=lambda row: float(row[key]))["label"])


def path_for_label(
    label: str,
    *,
    single_dir: Path,
    bon_dir: Path,
    rows_by_label: dict[str, dict[str, Any]],
) -> Path:
    row = rows_by_label.get(label)
    if row is not None and row.get("video"):
        return Path(str(row["video"]))
    return video_path_for_label(label, single_dir=single_dir, bon_dir=bon_dir)


def label_score(
    label: str,
    rows_by_label: dict[str, dict[str, Any]],
    key: str = "v_mem_projection",
) -> float | None:
    row = rows_by_label.get(label)
    if row is None or key not in row:
        return None
    return float(row[key])


def build_manifest(
    *,
    selector: dict[str, Any],
    single_composite: dict[str, Any],
    bon_composite: dict[str, Any],
    single_dir: Path,
    bon_dir: Path,
    image_weight: float,
    prompt_weight: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    composite_rows = composite_rows_by_label(single_composite, bon_composite)
    add_clip_scores(
        composite_rows,
        image_weight=image_weight,
        prompt_weight=prompt_weight,
    )

    for row in selector["selections"]:
        seed_rows = rows_for_seed(composite_rows, row["seed"])
        labels = {
            "base": row["base_label"],
            "single_lora": row["single_label"],
            "raw_best_of_n": row["raw_best_label"],
            "gated_best_of_n": row["gated_best_label"],
            "product_selected": row["product_label"],
            "clip_seed_image_best": best_label(seed_rows, "seed_image_cosine"),
            "clip_prompt_best": best_label(seed_rows, "prompt_clip_cosine"),
            "clip_preservation_best": best_label(seed_rows, "clip_preservation_score"),
        }
        paths = {
            key: path_for_label(
                label,
                single_dir=single_dir,
                bon_dir=bon_dir,
                rows_by_label=composite_rows,
            )
            for key, label in labels.items()
        }
        for key, path in paths.items():
            if not path.exists():
                missing.append(
                    {
                        "seed": row["seed"],
                        "variant": key,
                        "label": labels[key],
                        "path": str(path),
                    }
                )

        rows.append(
            {
                "seed": row["seed"],
                "labels": labels,
                "video_paths": {
                    key: existing_or_none(path) for key, path in paths.items()
                },
                "scores": {
                    "base": row["base_score"],
                    "single_lora": row["single_score"],
                    "raw_best_of_n": row["raw_best_score"],
                    "gated_best_of_n": row["gated_best_score"],
                    "product_selected": row["product_score"],
                    "clip_seed_image_best": label_score(
                        labels["clip_seed_image_best"], composite_rows
                    ),
                    "clip_prompt_best": label_score(
                        labels["clip_prompt_best"], composite_rows
                    ),
                    "clip_preservation_best": label_score(
                        labels["clip_preservation_best"], composite_rows
                    ),
                    "single_delta": row["single_delta"],
                    "raw_best_delta": row["raw_best_delta"],
                    "gated_best_delta": row["gated_best_delta"],
                    "base_or_gated_lift": row["base_or_gated_lift"],
                },
                "selector_metadata": {
                    "product_variant": row["product_variant"],
                    "gated_passes_preservation_gate": row[
                        "gated_passes_preservation_gate"
                    ],
                    "gated_seed_image_cosine": row["gated_seed_image_cosine"],
                    "gated_prompt_clip_cosine": row["gated_prompt_clip_cosine"],
                    "clip_seed_image_best_cosine": label_score(
                        labels["clip_seed_image_best"],
                        composite_rows,
                        key="seed_image_cosine",
                    ),
                    "clip_prompt_best_cosine": label_score(
                        labels["clip_prompt_best"],
                        composite_rows,
                        key="prompt_clip_cosine",
                    ),
                    "clip_preservation_best_score": label_score(
                        labels["clip_preservation_best"],
                        composite_rows,
                        key="clip_preservation_score",
                    ),
                },
            }
        )

    return {
        "schema_version": 1,
        "source": "wan22_pref_weighted_r16_s300_product_selector",
        "n_seeds": len(rows),
        "summary": selector["summary"],
        "baseline_selectors": {
            "clip_seed_image_best": "highest CLIP cosine between seed image and generated video frames",
            "clip_prompt_best": "highest CLIP cosine between prompt text and generated video frames",
            "clip_preservation_best": (
                f"{image_weight:.2f} * z(seed-image cosine) + "
                f"{prompt_weight:.2f} * z(prompt cosine)"
            ),
        },
        "rows": rows,
        "missing_files": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-report",
        type=Path,
        default=Path(
            "data/reports/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_"
            "bon_24x4_s12_m1p0_product_selector.json"
        ),
    )
    parser.add_argument(
        "--single-composite-report",
        type=Path,
        default=Path(
            "data/reports/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_"
            "s12_composite_gate008.json"
        ),
    )
    parser.add_argument(
        "--bon-composite-report",
        type=Path,
        default=Path(
            "data/reports/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_"
            "bon_24x4_s12_m1p0_composite_gate008.json"
        ),
    )
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=Path(
            "data/generated/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12/"
            "wan22_tribe_proxy_pref_weighted_r16_s300_"
            "wan22_lora_eval_fresh_picsum_24_eval_24x2_s12_m1p0"
        ),
    )
    parser.add_argument(
        "--bon-dir",
        type=Path,
        default=Path(
            "data/generated/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_"
            "bon_24x4_s12_m1p0/"
            "wan22_tribe_proxy_pref_weighted_r16_s300_"
            "wan22_lora_eval_fresh_picsum_24_bon_24x4_s12_m1p0"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_manifest.json"
        ),
    )
    parser.add_argument("--image-weight", type=float, default=0.75)
    parser.add_argument("--prompt-weight", type=float, default=0.25)
    args = parser.parse_args()

    selector = load_json(args.selector_report)
    single_composite = load_json(args.single_composite_report)
    bon_composite = load_json(args.bon_composite_report)
    manifest = build_manifest(
        selector=selector,
        single_composite=single_composite,
        bon_composite=bon_composite,
        single_dir=args.single_dir,
        bon_dir=args.bon_dir,
        image_weight=args.image_weight,
        prompt_weight=args.prompt_weight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    n_missing = len(manifest["missing_files"])
    print(f"[done] wrote {args.out}")
    print(f"[done] seeds: {manifest['n_seeds']}; missing files: {n_missing}")


if __name__ == "__main__":
    main()
