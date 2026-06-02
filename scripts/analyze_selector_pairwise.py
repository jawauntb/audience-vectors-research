"""Analyze selector pairwise response JSONs from the Prolific pilot survey."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        if isinstance(data, dict) and isinstance(data.get("responses"), list):
            payloads.append(data)
        elif isinstance(data, list):
            payloads.extend(
                item
                for item in data
                if isinstance(item, dict) and isinstance(item.get("responses"), list)
            )
    return payloads


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


def bootstrap_ci(values: list[float], *, n_boot: int, seed: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_responses(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_comparison: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        participant = str(payload.get("prolific_id") or "unknown")
        for response in payload["responses"]:
            row = dict(response)
            row["participant"] = participant
            by_comparison[str(row["comparison"])].append(row)

    comparison_summaries: dict[str, Any] = {}
    for comparison, rows in sorted(by_comparison.items()):
        hits = sum(1 for row in rows if bool(row["chose_target"]))
        total = len(rows)
        ci_low, ci_high = wilson_ci(hits, total)
        p_value = (
            float(stats.binomtest(hits, total, p=0.5, alternative="two-sided").pvalue)
            if total
            else float("nan")
        )

        by_seed: dict[str, list[float]] = defaultdict(list)
        by_participant: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = 1.0 if bool(row["chose_target"]) else 0.0
            by_seed[str(row["seed"])].append(value)
            by_participant[str(row["participant"])].append(value)

        seed_props = [float(np.mean(values)) for values in by_seed.values()]
        participant_props = [
            float(np.mean(values)) for values in by_participant.values()
        ]
        seed_ci = bootstrap_ci(seed_props, n_boot=20000, seed=20260528)
        participant_ci = bootstrap_ci(
            participant_props,
            n_boot=20000,
            seed=20260529,
        )

        comparison_summaries[comparison] = {
            "hits": hits,
            "total": total,
            "pooled_rate": hits / total if total else None,
            "pooled_wilson_ci": [ci_low, ci_high],
            "pooled_binomial_p_two_sided": p_value,
            "n_seed_clusters": len(seed_props),
            "seed_cluster_mean": float(np.mean(seed_props)) if seed_props else None,
            "seed_cluster_bootstrap_ci": list(seed_ci),
            "n_participant_clusters": len(participant_props),
            "participant_cluster_mean": (
                float(np.mean(participant_props)) if participant_props else None
            ),
            "participant_cluster_bootstrap_ci": list(participant_ci),
        }

    return {
        "schema_version": 1,
        "n_participants": len(payloads),
        "n_responses": sum(len(payload["responses"]) for payload in payloads),
        "comparisons": comparison_summaries,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Selector Pairwise Human Evaluation",
        "",
        f"- Participants: **{summary['n_participants']}**",
        f"- Responses: **{summary['n_responses']}**",
        "",
        "| comparison | target wins | pooled rate | Wilson CI | seed-cluster mean | seed bootstrap CI | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison, row in summary["comparisons"].items():
        wilson = row["pooled_wilson_ci"]
        seed_ci = row["seed_cluster_bootstrap_ci"]
        lines.append(
            f"| {comparison} | {row['hits']}/{row['total']} | "
            f"{row['pooled_rate']:.3f} | [{wilson[0]:.3f}, {wilson[1]:.3f}] | "
            f"{row['seed_cluster_mean']:.3f} | [{seed_ci[0]:.3f}, {seed_ci[1]:.3f}] | "
            f"{row['pooled_binomial_p_two_sided']:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/responses"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "selector_pairwise_analysis.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "selector_pairwise_analysis.md"
        ),
    )
    args = parser.parse_args()

    paths = (
        sorted(args.responses_dir.glob("*.json")) if args.responses_dir.exists() else []
    )
    payloads = load_payloads(paths)
    summary = summarize_responses(payloads)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.out_md)

    print(f"[done] loaded participants: {summary['n_participants']}")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")


if __name__ == "__main__":
    main()
