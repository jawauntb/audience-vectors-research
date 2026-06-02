"""Analyze agreement between TRIBE, V-JEPA, CLIP, and human frames.

The current implementation uses the full Wan2.2 candidate pool for which TRIBE
and CLIP-preservation scores already exist. If V-JEPA embeddings are available,
it compares representation geometries and selector orderings across the complete
intersection. Human response integration is scaffolded and becomes active once
V-JEPA-augmented survey responses exist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    result = stats.spearmanr(x, y)
    value = float(result.statistic)
    if np.isnan(value):
        return None
    return value


def linear_cka(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    gram_x = xc @ xc.T
    gram_y = yc @ yc.T
    hsic = float(np.sum(gram_x * gram_y))
    denom = float(np.linalg.norm(gram_x) * np.linalg.norm(gram_y))
    if denom <= 1e-12:
        return None
    return hsic / denom


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe = features / np.maximum(norms, 1e-12)
    return safe @ safe.T


def load_tribe_feature(label: str, dirs: list[Path]) -> np.ndarray | None:
    for directory in dirs:
        path = directory / f"{label}.npz"
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=False)
        frames = np.asarray(payload["frames"], dtype=np.float32)
        feature = frames.mean(axis=0) if frames.ndim == 2 else frames
        return normalize(feature)
    return None


def load_vjepa_feature(label: str, directory: Path) -> np.ndarray | None:
    path = directory / f"{label}.npz"
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    return normalize(np.asarray(payload["embedding"], dtype=np.float32))


def load_vjepa_direction(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    return normalize(np.asarray(payload["direction"], dtype=np.float32))


def load_candidate_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows_by_label: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        for row in payload.get("rows", []):
            label = str(row["label"])
            rows_by_label[label] = dict(row, source_report=str(path))
    return sorted(
        rows_by_label.values(), key=lambda row: (row["seed_key"], row["label"])
    )


def load_human_payloads(responses_dir: Path) -> list[dict[str, Any]]:
    if not responses_dir.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(responses_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("responses"), list):
            payloads.append(data)
        elif isinstance(data, list):
            payloads.extend(
                item
                for item in data
                if isinstance(item, dict) and isinstance(item.get("responses"), list)
            )
    return payloads


def human_preference_scores(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    wins: dict[str, int] = defaultdict(int)
    appearances: dict[str, int] = defaultdict(int)
    for payload in payloads:
        for response in payload["responses"]:
            chosen = response.get("chosen_label")
            other = response.get("other_label")
            if chosen:
                wins[str(chosen)] += 1
                appearances[str(chosen)] += 1
            if other:
                appearances[str(other)] += 1
    scores = {
        label: wins[label] / appearances[label]
        for label in appearances
        if appearances[label] > 0
    }
    return {
        "n_participants": len(payloads),
        "n_responses": sum(len(payload["responses"]) for payload in payloads),
        "scores": scores,
    }


def frame_pair_summary(
    *,
    name_a: str,
    name_b: str,
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> dict[str, Any]:
    sim_a = cosine_similarity_matrix(features_a)
    sim_b = cosine_similarity_matrix(features_b)
    return {
        "frame_a": name_a,
        "frame_b": name_b,
        "rsa_spearman_upper_triangle": spearman(
            upper_triangle(sim_a), upper_triangle(sim_b)
        ),
        "linear_cka": linear_cka(features_a, features_b),
    }


def rank_agreement(
    rows: list[dict[str, Any]],
    score_a: str,
    score_b: str,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[str(row["seed_key"])].append(row)

    for seed, seed_rows in sorted(by_seed.items()):
        valid = [
            row
            for row in seed_rows
            if isinstance(row.get(score_a), (int, float))
            and isinstance(row.get(score_b), (int, float))
        ]
        if len(valid) < 2:
            continue
        a = np.asarray([float(row[score_a]) for row in valid], dtype=np.float32)
        b = np.asarray([float(row[score_b]) for row in valid], dtype=np.float32)
        top_a = max(valid, key=lambda row: float(row[score_a]))
        top_b = max(valid, key=lambda row: float(row[score_b]))
        per_seed.append(
            {
                "seed": seed,
                "n": len(valid),
                "spearman": spearman(a, b),
                "top_a": top_a["label"],
                "top_b": top_b["label"],
                "same_top": top_a["label"] == top_b["label"],
                "top_a_score_a": top_a[score_a],
                "top_a_score_b": top_a[score_b],
                "top_b_score_a": top_b[score_a],
                "top_b_score_b": top_b[score_b],
            }
        )

    corrs = [row["spearman"] for row in per_seed if row["spearman"] is not None]
    return {
        "score_a": score_a,
        "score_b": score_b,
        "n_seed_groups": len(per_seed),
        "mean_seed_spearman": float(np.mean(corrs)) if corrs else None,
        "median_seed_spearman": float(np.median(corrs)) if corrs else None,
        "top1_agreement": (
            float(np.mean([row["same_top"] for row in per_seed])) if per_seed else None
        ),
        "per_seed": per_seed,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Representation-Frame Analysis",
        "",
        "This analysis compares TRIBE, V-JEPA, CLIP-preservation, and available human",
        "response frames on the current Wan2.2 candidate pool.",
        "",
        "## Coverage",
        "",
        f"- Candidate rows: **{report['coverage']['n_candidate_rows']}**",
        f"- Complete TRIBE+V-JEPA+CLIP rows: **{report['coverage']['n_complete_rows']}**",
        f"- Missing TRIBE features: **{report['coverage']['n_missing_tribe']}**",
        f"- Missing V-JEPA features: **{report['coverage']['n_missing_vjepa']}**",
        f"- Human participants loaded: **{report['human']['n_participants']}**",
        f"- Human responses loaded: **{report['human']['n_responses']}**",
        "",
        "## Score Correlations",
        "",
        "| scores | Spearman rho |",
        "|---|---:|",
    ]
    for row in report["score_correlations"]:
        value = row["spearman"]
        value_s = "n/a" if value is None else f"{value:+.3f}"
        lines.append(f"| {row['score_a']} vs {row['score_b']} | {value_s} |")

    lines += [
        "",
        "## Representation Geometry",
        "",
        "| frames | RSA Spearman | linear CKA |",
        "|---|---:|---:|",
    ]
    for row in report["geometry"]:
        rsa = row["rsa_spearman_upper_triangle"]
        cka = row["linear_cka"]
        rsa_s = "n/a" if rsa is None else f"{rsa:+.3f}"
        cka_s = "n/a" if cka is None else f"{cka:+.3f}"
        lines.append(f"| {row['frame_a']} vs {row['frame_b']} | {rsa_s} | {cka_s} |")

    lines += [
        "",
        "## Within-Seed Rank Agreement",
        "",
        "| scores | seed groups | mean rho | median rho | top-1 agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["rank_agreement"]:
        mean_s = (
            "n/a"
            if row["mean_seed_spearman"] is None
            else f"{row['mean_seed_spearman']:+.3f}"
        )
        med_s = (
            "n/a"
            if row["median_seed_spearman"] is None
            else f"{row['median_seed_spearman']:+.3f}"
        )
        top_s = (
            "n/a" if row["top1_agreement"] is None else f"{row['top1_agreement']:.3f}"
        )
        lines.append(
            f"| {row['score_a']} vs {row['score_b']} | {row['n_seed_groups']} | "
            f"{mean_s} | {med_s} | {top_s} |"
        )

    lines += [
        "",
        "## TRIBE vs V-JEPA Top Disagreements",
        "",
        "| seed | TRIBE top | V-JEPA top | TRIBE top V-JEPA score | V-JEPA top TRIBE score |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["top_disagreements"][:20]:
        lines.append(
            f"| `{row['seed']}` | `{row['tribe_top']}` | `{row['vjepa_top']}` | "
            f"{row['tribe_top_vjepa_score']:+.4f} | {row['vjepa_top_tribe_score']:+.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        report["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


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
        "--tribe-feature-dirs",
        type=Path,
        nargs="+",
        default=[
            Path(
                "data/features/tribe_wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12"
            ),
            Path(
                "data/features/"
                "tribe_wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_"
                "bon_24x4_s12_m1p0"
            ),
        ],
    )
    parser.add_argument(
        "--vjepa-features-dir",
        type=Path,
        default=Path("data/features/vjepa_wan22_selector_pref_weighted_r16_s300"),
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
            "representation_frame_analysis.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "representation_frame_analysis.md"
        ),
    )
    args = parser.parse_args()

    candidate_rows = load_candidate_rows([args.single_composite, args.bon_composite])
    vjepa_direction = load_vjepa_direction(args.vjepa_vector)
    enriched: list[dict[str, Any]] = []
    missing_tribe: list[str] = []
    missing_vjepa: list[str] = []

    for row in candidate_rows:
        label = str(row["label"])
        tribe_feature = load_tribe_feature(label, args.tribe_feature_dirs)
        vjepa_feature = load_vjepa_feature(label, args.vjepa_features_dir)
        if tribe_feature is None:
            missing_tribe.append(label)
        if vjepa_feature is None:
            missing_vjepa.append(label)
        clip_vector = np.asarray(
            [float(row["seed_image_cosine"]), float(row["prompt_clip_cosine"])],
            dtype=np.float32,
        )
        new_row = dict(row)
        new_row["tribe_feature_available"] = tribe_feature is not None
        new_row["vjepa_feature_available"] = vjepa_feature is not None
        new_row["vjepa_memorability_score"] = (
            float(vjepa_feature @ vjepa_direction)
            if vjepa_feature is not None
            else None
        )
        new_row["_tribe_feature"] = tribe_feature
        new_row["_vjepa_feature"] = vjepa_feature
        new_row["_clip_feature"] = clip_vector
        new_row["clip_preservation_score"] = float(
            row["composite_score"] - row["v_mem_z"]
        )
        enriched.append(new_row)

    complete = [
        row
        for row in enriched
        if row["_tribe_feature"] is not None and row["_vjepa_feature"] is not None
    ]
    clip_raw = np.stack([row["_clip_feature"] for row in complete]).astype(np.float32)
    clip_features = np.stack(
        [
            zscore(clip_raw[:, 0]),
            zscore(clip_raw[:, 1]),
        ],
        axis=1,
    )
    tribe_features = np.stack([row["_tribe_feature"] for row in complete]).astype(
        np.float32
    )
    vjepa_features = np.stack([row["_vjepa_feature"] for row in complete]).astype(
        np.float32
    )

    for idx, row in enumerate(complete):
        row["clip_preservation_z2_score"] = float(clip_features[idx].sum())

    score_pairs = [
        ("v_mem_projection", "vjepa_memorability_score"),
        ("v_mem_projection", "clip_preservation_score"),
        ("vjepa_memorability_score", "clip_preservation_score"),
        ("v_mem_projection", "clip_preservation_z2_score"),
        ("vjepa_memorability_score", "clip_preservation_z2_score"),
    ]
    score_correlations = []
    for a, b in score_pairs:
        xs = np.asarray([float(row[a]) for row in complete], dtype=np.float32)
        ys = np.asarray([float(row[b]) for row in complete], dtype=np.float32)
        score_correlations.append(
            {"score_a": a, "score_b": b, "spearman": spearman(xs, ys)}
        )

    geometry = [
        frame_pair_summary(
            name_a="TRIBE pooled cortical",
            name_b="V-JEPA video",
            features_a=tribe_features,
            features_b=vjepa_features,
        ),
        frame_pair_summary(
            name_a="TRIBE pooled cortical",
            name_b="CLIP preservation scalars",
            features_a=tribe_features,
            features_b=clip_features,
        ),
        frame_pair_summary(
            name_a="V-JEPA video",
            name_b="CLIP preservation scalars",
            features_a=vjepa_features,
            features_b=clip_features,
        ),
    ]

    rank_pairs = [
        ("v_mem_projection", "vjepa_memorability_score"),
        ("v_mem_projection", "clip_preservation_score"),
        ("vjepa_memorability_score", "clip_preservation_score"),
    ]
    rank_summaries = [
        rank_agreement(complete, score_a=a, score_b=b) for a, b in rank_pairs
    ]

    tribe_vjepa_rank = rank_summaries[0]
    disagreements = []
    for row in tribe_vjepa_rank["per_seed"]:
        if row["same_top"]:
            continue
        tribe_top = next(item for item in complete if item["label"] == row["top_a"])
        vjepa_top = next(item for item in complete if item["label"] == row["top_b"])
        disagreements.append(
            {
                "seed": row["seed"],
                "tribe_top": row["top_a"],
                "vjepa_top": row["top_b"],
                "seed_rank_spearman": row["spearman"],
                "tribe_top_tribe_score": tribe_top["v_mem_projection"],
                "tribe_top_vjepa_score": tribe_top["vjepa_memorability_score"],
                "vjepa_top_tribe_score": vjepa_top["v_mem_projection"],
                "vjepa_top_vjepa_score": vjepa_top["vjepa_memorability_score"],
                "tribe_top_clip_preservation_score": tribe_top[
                    "clip_preservation_score"
                ],
                "vjepa_top_clip_preservation_score": vjepa_top[
                    "clip_preservation_score"
                ],
            }
        )

    human = human_preference_scores(load_human_payloads(args.responses_dir))
    serializable_rows = []
    for row in complete:
        clean = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        serializable_rows.append(clean)

    interpretation = (
        "This is a representation-frame audit, not a human validation. It tells us "
        "how much TRIBE, V-JEPA, and CLIP-preservation agree before collecting the "
        "augmented survey responses. Large top-1 disagreement between TRIBE and "
        "V-JEPA is useful: it means the human study can actually adjudicate between "
        "brain-aligned and self-supervised video frames rather than comparing two "
        "selectors that choose the same clips."
    )
    report = {
        "schema_version": 1,
        "coverage": {
            "n_candidate_rows": len(candidate_rows),
            "n_complete_rows": len(complete),
            "n_missing_tribe": len(missing_tribe),
            "n_missing_vjepa": len(missing_vjepa),
            "missing_tribe": missing_tribe,
            "missing_vjepa": missing_vjepa,
        },
        "score_correlations": score_correlations,
        "geometry": geometry,
        "rank_agreement": rank_summaries,
        "top_disagreements": disagreements,
        "human": human,
        "complete_rows": serializable_rows,
        "interpretation": interpretation,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] complete rows: {len(complete)}/{len(candidate_rows)}")
    print(f"[done] missing V-JEPA: {len(missing_vjepa)}")


if __name__ == "__main__":
    main()
