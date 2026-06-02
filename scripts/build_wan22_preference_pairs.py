"""Build TRIBE-reward preference pairs from scored Wan2.2 generations.

This does not claim human memorability. It prepares the proxy preference data
needed for a reward-distilled LoRA/DPO pass:

- chosen: highest TRIBE/BMD projection variant for a seed
- rejected: median or lowest variant for the same seed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _seed_from_label(label: str) -> str:
    match = re.search(r"vid_idx\d{4}", label)
    if not match:
        raise ValueError(f"could not parse seed from label {label!r}")
    return match.group(0)


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _manifest_by_label(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in report.get("manifest", [])}


def _rows_by_seed(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_seed: dict[str, list[dict[str, Any]]] = {}
    for row in report["scores"]:
        by_seed.setdefault(_seed_from_label(str(row["label"])), []).append(row)
    for rows in by_seed.values():
        rows.sort(key=lambda r: float(r["v_mem_projection"]), reverse=True)
    return dict(sorted(by_seed.items()))


def _pair_payload(
    *,
    seed: str,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    reject_rank: str,
) -> dict[str, Any]:
    chosen_meta = manifest.get(str(chosen["label"]), {})
    rejected_meta = manifest.get(str(rejected["label"]), {})
    chosen_score = float(chosen["v_mem_projection"])
    rejected_score = float(rejected["v_mem_projection"])
    return {
        "pair_id": f"{seed}_top_vs_{reject_rank}",
        "objective": "tribe_bmd_memorability_proxy",
        "seed": seed,
        "prompt": chosen_meta.get("prompt"),
        "seed_image": chosen_meta.get("seed_image"),
        "chosen": {
            "label": chosen["label"],
            "video": chosen_meta.get("local_path"),
            "score": chosen_score,
        },
        "rejected": {
            "label": rejected["label"],
            "video": rejected_meta.get("local_path"),
            "score": rejected_score,
        },
        "margin": chosen_score - rejected_score,
        "bmd_human_memorability_score": chosen_meta.get("bmd_memorability_score"),
        "note": ("Proxy pair from TRIBE/BMD projection, not direct human preference."),
    }


def build_pairs(
    report: dict[str, Any],
    *,
    include_bottom: bool,
    min_margin: float,
) -> list[dict[str, Any]]:
    manifest = _manifest_by_label(report)
    pairs = []
    for seed, rows in _rows_by_seed(report).items():
        if len(rows) < 2:
            continue
        chosen = rows[0]
        scores = np.asarray([float(row["v_mem_projection"]) for row in rows])
        median_value = float(np.median(scores))
        median_row = min(
            rows[1:],
            key=lambda row: abs(float(row["v_mem_projection"]) - median_value),
        )
        median_pair = _pair_payload(
            seed=seed,
            chosen=chosen,
            rejected=median_row,
            manifest=manifest,
            reject_rank="median",
        )
        if median_pair["margin"] >= min_margin:
            pairs.append(median_pair)

        if include_bottom:
            bottom_row = rows[-1]
            bottom_pair = _pair_payload(
                seed=seed,
                chosen=chosen,
                rejected=bottom_row,
                manifest=manifest,
                reject_rank="bottom",
            )
            if bottom_pair["margin"] >= min_margin:
                pairs.append(bottom_pair)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/wan22_replication_matrix_2026-05-20_results.json"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/training/wan22_tribe_reward_preferences.jsonl"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("data/reports/wan22_tribe_reward_preferences_summary.json"),
    )
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--include-bottom", action="store_true")
    args = parser.parse_args()

    report = _load_report(args.report)
    pairs = build_pairs(
        report,
        include_bottom=args.include_bottom,
        min_margin=args.min_margin,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "".join(json.dumps(pair, sort_keys=True) + "\n" for pair in pairs)
    )

    margins = [float(pair["margin"]) for pair in pairs]
    summary = {
        "report": str(args.report),
        "out_jsonl": str(args.out_jsonl),
        "n_pairs": len(pairs),
        "include_bottom": args.include_bottom,
        "min_margin": args.min_margin,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_median": float(np.median(margins)) if margins else None,
        "margin_min": float(np.min(margins)) if margins else None,
        "margin_max": float(np.max(margins)) if margins else None,
        "objective": "tribe_bmd_memorability_proxy",
        "warning": "These are proxy preferences, not human labels.",
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
