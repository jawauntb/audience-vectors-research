"""Sweep Wan2.2 product-selector policies on an existing scored candidate pool.

This is a proxy-only tuning pass. It does not prove human preference alignment;
it helps us understand the tradeoff between TRIBE/BMD reward lift and semantic
preservation guardrails before paying for another human eval or training run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    label: str
    seed: str
    v_mem: float
    v_mem_z: float
    seed_z: float
    prompt_z: float
    seed_drop: float
    prompt_drop: float
    seed_cosine: float
    prompt_cosine: float

    def rank_score(self, *, mode: str, image_weight: float, prompt_weight: float) -> float:
        if mode == "v_mem":
            return self.v_mem
        if mode == "composite":
            return self.v_mem_z + image_weight * self.seed_z + prompt_weight * self.prompt_z
        raise ValueError(f"unknown ranking mode: {mode}")


def parse_float_grid(raw: str, *, allow_none: bool) -> list[float | None]:
    values: list[float | None] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null"}:
            if not allow_none:
                raise ValueError(f"None is not allowed in grid: {raw}")
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError(f"empty grid: {raw}")
    return values


def parse_mode_grid(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = set(modes) - {"v_mem", "composite"}
    if unknown:
        raise ValueError(f"unknown rank modes: {sorted(unknown)}")
    return modes


def load_scores_by_label(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["label"]): float(row["v_mem_projection"])
        for row in report["scores"]
    }


def load_base_scores(single_report: dict[str, Any]) -> dict[str, float]:
    scores = load_scores_by_label(single_report)
    out: dict[str, float] = {}
    for label, score in scores.items():
        if label.endswith("_base"):
            out[label.removesuffix("_base")] = score
    if not out:
        raise ValueError("single report did not contain *_base scores")
    return out


def load_candidates(composite_report: dict[str, Any]) -> dict[str, list[Candidate]]:
    by_seed: dict[str, list[Candidate]] = {}
    for row in composite_report["rows"]:
        candidate = Candidate(
            label=str(row["label"]),
            seed=str(row["seed_key"]),
            v_mem=float(row["v_mem_projection"]),
            v_mem_z=float(row["v_mem_z"]),
            seed_z=float(row["seed_image_cosine_z"]),
            prompt_z=float(row["prompt_clip_cosine_z"]),
            seed_drop=float(row["seed_image_cosine_drop_from_seed_best"]),
            prompt_drop=float(row["prompt_clip_cosine_drop_from_seed_best"]),
            seed_cosine=float(row["seed_image_cosine"]),
            prompt_cosine=float(row["prompt_clip_cosine"]),
        )
        by_seed.setdefault(candidate.seed, []).append(candidate)
    if not by_seed:
        raise ValueError("composite report did not contain rows")
    return by_seed


def passes_gate(
    candidate: Candidate,
    *,
    max_seed_drop: float | None,
    max_prompt_drop: float | None,
) -> bool:
    if max_seed_drop is not None and candidate.seed_drop > max_seed_drop:
        return False
    if max_prompt_drop is not None and candidate.prompt_drop > max_prompt_drop:
        return False
    return True


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def maybe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def evaluate_policy(
    *,
    base_scores: dict[str, float],
    candidates_by_seed: dict[str, list[Candidate]],
    rank_mode: str,
    image_weight: float,
    prompt_weight: float,
    max_seed_drop: float | None,
    max_prompt_drop: float | None,
) -> dict[str, Any]:
    candidate_deltas: list[float] = []
    product_lifts: list[float] = []
    selected_seed_drops: list[float] = []
    selected_prompt_drops: list[float] = []
    n_passing = 0
    n_candidates = 0
    n_no_passing = 0
    selections: list[dict[str, Any]] = []

    for seed, base_score in sorted(base_scores.items()):
        candidates = candidates_by_seed.get(seed, [])
        passing = [
            candidate
            for candidate in candidates
            if passes_gate(
                candidate,
                max_seed_drop=max_seed_drop,
                max_prompt_drop=max_prompt_drop,
            )
        ]
        n_candidates += len(candidates)
        n_passing += len(passing)
        if not passing:
            n_no_passing += 1
            candidate_delta = 0.0
            product_lift = 0.0
            selected_label = f"{seed}_base"
            product_label = f"{seed}_base"
            selected_score = base_score
        else:
            selected = max(
                passing,
                key=lambda candidate: candidate.rank_score(
                    mode=rank_mode,
                    image_weight=image_weight,
                    prompt_weight=prompt_weight,
                ),
            )
            candidate_delta = selected.v_mem - base_score
            product_lift = max(0.0, candidate_delta)
            selected_label = selected.label
            product_label = selected.label if candidate_delta > 0.0 else f"{seed}_base"
            selected_score = selected.v_mem
            selected_seed_drops.append(selected.seed_drop)
            selected_prompt_drops.append(selected.prompt_drop)

        candidate_deltas.append(candidate_delta)
        product_lifts.append(product_lift)
        selections.append(
            {
                "seed": seed,
                "base_score": base_score,
                "selected_label": selected_label,
                "selected_score": selected_score,
                "candidate_delta": candidate_delta,
                "product_label": product_label,
                "product_lift": product_lift,
            }
        )

    product_stats = stats(product_lifts)
    candidate_stats = stats(candidate_deltas)
    return {
        "rank_mode": rank_mode,
        "image_weight": image_weight,
        "prompt_weight": prompt_weight,
        "max_seed_drop": max_seed_drop,
        "max_prompt_drop": max_prompt_drop,
        "n_seeds": len(base_scores),
        "n_candidates": n_candidates,
        "n_no_passing_seed": n_no_passing,
        "n_product_improved": sum(value > 1e-9 for value in product_lifts),
        "n_candidate_improved": sum(value > 1e-9 for value in candidate_deltas),
        "n_base_fallback": sum(row["product_label"].endswith("_base") for row in selections),
        "gate_pass_rate": float(n_passing / n_candidates) if n_candidates else 0.0,
        "candidate_delta": candidate_stats,
        "product_lift": product_stats,
        "selected_seed_drop": stats(selected_seed_drops),
        "selected_prompt_drop": stats(selected_prompt_drops),
        "selected_seed_drop_mean": maybe_mean(selected_seed_drops),
        "selected_prompt_drop_mean": maybe_mean(selected_prompt_drops),
        "selections": selections,
    }


def sweep_policies(
    *,
    base_scores: dict[str, float],
    candidates_by_seed: dict[str, list[Candidate]],
    rank_modes: list[str],
    image_weights: list[float | None],
    prompt_weights: list[float | None],
    seed_drops: list[float | None],
    prompt_drops: list[float | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_mode in rank_modes:
        weight_pairs: list[tuple[float, float]]
        if rank_mode == "v_mem":
            weight_pairs = [(0.0, 0.0)]
        else:
            weight_pairs = [
                (float(image_weight), float(prompt_weight))
                for image_weight in image_weights
                for prompt_weight in prompt_weights
                if image_weight is not None and prompt_weight is not None
            ]
        for image_weight, prompt_weight in weight_pairs:
            for max_seed_drop in seed_drops:
                for max_prompt_drop in prompt_drops:
                    rows.append(
                        evaluate_policy(
                            base_scores=base_scores,
                            candidates_by_seed=candidates_by_seed,
                            rank_mode=rank_mode,
                            image_weight=image_weight,
                            prompt_weight=prompt_weight,
                            max_seed_drop=max_seed_drop,
                            max_prompt_drop=max_prompt_drop,
                        )
                    )
    return sorted(
        rows,
        key=lambda row: (
            row["product_lift"]["mean"],
            row["product_lift"]["median"],
            row["n_product_improved"],
        ),
        reverse=True,
    )


def format_optional(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def find_current_policy(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["rank_mode"] == "v_mem"
            and row["max_seed_drop"] == 0.08
            and row["max_prompt_drop"] is None
        ):
            return row
    return None


def write_markdown(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    single_report: Path,
    composite_report: Path,
    top_n: int,
) -> None:
    current = find_current_policy(rows)
    lines = [
        "# Wan2.2 Selector Tuning Sweep",
        "",
        "Proxy-only sweep over ranking scores and preservation gates. Treat this as a selector-design diagnostic, not human validation.",
        "",
        "## Inputs",
        "",
        f"- Single/base report: `{single_report}`",
        f"- Composite candidate report: `{composite_report}`",
        f"- Configs swept: **{len(rows)}**",
        "",
    ]
    if current is not None:
        lift = current["product_lift"]
        lines += [
            "## Current Policy",
            "",
            "Current policy = rank passing candidates by TRIBE score, require seed-image cosine drop <= 0.08, then fall back to base if the selected candidate is worse.",
            "",
            f"- Improved seeds: **{current['n_product_improved']}/{current['n_seeds']}**",
            f"- Product lift mean / median: **{lift['mean']:+.4f} / {lift['median']:+.4f}**",
            f"- Gate pass rate: **{current['gate_pass_rate']:.3f}**",
            "",
        ]

    lines += [
        "## Top Policies By Proxy Product Lift",
        "",
        "| rank | mode | image_w | prompt_w | seed_drop | prompt_drop | improved | mean | median | gate pass | base fallbacks | selected seed drop mean/max |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:top_n], start=1):
        lift = row["product_lift"]
        selected_seed_drop = row["selected_seed_drop"]
        lines.append(
            f"| {idx} | `{row['rank_mode']}` | {row['image_weight']:.2f} | "
            f"{row['prompt_weight']:.2f} | {format_optional(row['max_seed_drop'])} | "
            f"{format_optional(row['max_prompt_drop'])} | "
            f"{row['n_product_improved']}/{row['n_seeds']} | "
            f"{lift['mean']:+.4f} | {lift['median']:+.4f} | "
            f"{row['gate_pass_rate']:.3f} | {row['n_base_fallback']} | "
            f"{selected_seed_drop['mean']:.3f}/{selected_seed_drop['max']:.3f} |"
        )

    lines += [
        "",
        "## Read",
        "",
        "The unconstrained top rows are useful as an upper-bound proxy result, but they are not automatically the safest product selector. Prefer rows that keep most of the lift while limiting selected seed-image drift, then validate those rows with human labels.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-report",
        type=Path,
        default=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12_results.json"
        ),
    )
    parser.add_argument(
        "--composite-report",
        type=Path,
        default=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_composite_gate008.json"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_selector_tuning.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(
            "data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_selector_tuning.md"
        ),
    )
    parser.add_argument("--rank-modes", default="v_mem,composite")
    parser.add_argument(
        "--seed-drop-grid",
        default="0.04,0.06,0.08,0.10,0.12,0.16,none",
    )
    parser.add_argument("--prompt-drop-grid", default="none,0.04,0.08,0.12")
    parser.add_argument("--image-weight-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--prompt-weight-grid", default="0,0.25,0.5")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    single_report = json.loads(args.single_report.read_text())
    composite_report = json.loads(args.composite_report.read_text())
    base_scores = load_base_scores(single_report)
    candidates_by_seed = load_candidates(composite_report)
    rows = sweep_policies(
        base_scores=base_scores,
        candidates_by_seed=candidates_by_seed,
        rank_modes=parse_mode_grid(args.rank_modes),
        image_weights=parse_float_grid(args.image_weight_grid, allow_none=False),
        prompt_weights=parse_float_grid(args.prompt_weight_grid, allow_none=False),
        seed_drops=parse_float_grid(args.seed_drop_grid, allow_none=True),
        prompt_drops=parse_float_grid(args.prompt_drop_grid, allow_none=True),
    )

    payload = {
        "single_report": str(args.single_report),
        "composite_report": str(args.composite_report),
        "n_configs": len(rows),
        "current_policy": find_current_policy(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(
        rows=rows,
        out_path=args.out_md,
        single_report=args.single_report,
        composite_report=args.composite_report,
        top_n=args.top_n,
    )
    print(f"[selector-tune] wrote {args.out_json}", flush=True)
    print(f"[selector-tune] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
