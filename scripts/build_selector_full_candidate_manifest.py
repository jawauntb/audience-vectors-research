"""Build a full Wan selector candidate manifest for representation-frame analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows_by_label: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        for row in payload.get("rows", []):
            label = str(row["label"])
            rows_by_label[label] = {
                "label": label,
                "seed": str(row["seed_key"]),
                "variant": row.get("variant"),
                "video_path": str(row["video"]),
                "seed_image": str(row.get("seed_image", "")),
                "prompt": str(row.get("prompt", "")),
                "source_report": str(path),
            }
    return sorted(rows_by_label.values(), key=lambda row: (row["seed"], row["label"]))


def build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({row["seed"] for row in rows})
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row["label"]
        manifest_rows.append(
            {
                "seed": row["seed"],
                "labels": {"candidate": label},
                "video_paths": {"candidate": row["video_path"]},
                "metadata": {
                    "variant": row["variant"],
                    "seed_image": row["seed_image"],
                    "prompt": row["prompt"],
                    "source_report": row["source_report"],
                },
            }
        )
    return {
        "schema_version": 1,
        "source": "wan22_pref_weighted_r16_s300_full_candidate_pool",
        "n_seeds": len(seeds),
        "n_candidates": len(rows),
        "rows": manifest_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-composite",
        type=Path,
        default=Path(
            "data/reports/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12_"
            "composite_gate008.json"
        ),
    )
    parser.add_argument(
        "--bon-composite",
        type=Path,
        default=Path(
            "data/reports/"
            "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_"
            "bon_24x4_s12_m1p0_composite_gate008.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "full_candidate_pool_manifest.json"
        ),
    )
    args = parser.parse_args()

    rows = load_rows([args.single_composite, args.bon_composite])
    manifest = build_manifest(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] wrote {args.out}")
    print(f"[done] candidates: {manifest['n_candidates']}")
    print(f"[done] seeds: {manifest['n_seeds']}")


if __name__ == "__main__":
    main()
