"""Decode the low-rank persona memorability axes.

The reviewer-corrected persona result says 12 persona directions occupy only a
few effective axes. This script asks what those axes look like by comparing
them against:

1. Global construct directions trained from non-persona VLM labels.
2. Per-persona construct direction similarity matrices (RSA-style).
3. Hand-authored persona metadata weights/dislikes.

This is an exploratory semantic decoding pass, not a definitive cognitive
ontology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

LABEL_AXES = [
    "attention",
    "memorability",
    "confusion",
    "emotional_intensity",
    "semantic_surprise",
    "narrative_progress",
    "social_salience",
    "audio_salience",
    "visual_salience",
    "rewatch_likelihood",
]


def rankdata(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra = rankdata(a)
    rb = rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "frames" in payload.files:
        arr = np.asarray(payload["frames"], dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    return np.asarray(payload["embedding"], dtype=np.float32)


def train_direction(
    features: np.ndarray, scores: np.ndarray, frac: float
) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * frac))
    direction = features[order[-n_each:]].mean(axis=0) - features[order[:n_each]].mean(
        axis=0
    )
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else direction


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def load_bmd_scores(path: Path) -> dict[str, float]:
    ann = json.loads(path.read_text())
    return {
        f"bmd_vid_idx{eid}": float(row["memorability_score"])
        for eid, row in ann.items()
        if "memorability_score" in row
    }


def load_feature_matrix(features_dir: Path) -> tuple[list[str], np.ndarray]:
    feature_files = sorted(features_dir.glob("*.npz"))
    ids = [p.stem for p in feature_files]
    features = np.stack([load_feature(p) for p in feature_files]).astype(np.float32)
    return ids, features


def persona_axis_scores(persona_file: Path, axis: str) -> dict[str, dict[str, float]]:
    df = pl.read_parquet(persona_file)
    scores = df.select("scores").unnest("scores")
    if axis not in scores.columns:
        raise ValueError(f"missing axis {axis!r}; available={scores.columns}")
    rows = (
        df.with_columns(scores[axis].alias("_score"))
        .select(["persona_id", "segment_id", "_score"])
        .drop_nulls()
        .to_dicts()
    )
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        out.setdefault(str(row["persona_id"]), {})[str(row["segment_id"])] = float(
            row["_score"]
        )
    return out


def train_persona_directions(
    ids: list[str],
    features: np.ndarray,
    persona_file: Path,
    axis: str,
    frac: float,
) -> tuple[list[str], np.ndarray]:
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    by_persona = persona_axis_scores(persona_file, axis)
    directions: dict[str, np.ndarray] = {}
    for persona_id, seg_scores in by_persona.items():
        idxs: list[int] = []
        scores: list[float] = []
        for sid, score in seg_scores.items():
            idx = id_to_idx.get(sid)
            if idx is None:
                continue
            idxs.append(idx)
            scores.append(score)
        if len(idxs) < 30:
            continue
        directions[persona_id] = train_direction(
            features[np.asarray(idxs)],
            np.asarray(scores, dtype=np.float32),
            frac,
        )
    persona_ids = sorted(directions)
    return persona_ids, np.stack([directions[p] for p in persona_ids])


def train_global_label_directions(
    ids: list[str],
    features: np.ndarray,
    generic_label_file: Path,
    bmd_scores: dict[str, float],
    frac: float,
) -> dict[str, np.ndarray]:
    directions: dict[str, np.ndarray] = {}

    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    df = pl.read_parquet(generic_label_file)
    scores = df.select("scores").unnest("scores")
    for axis in LABEL_AXES:
        if axis not in scores.columns:
            continue
        rows = (
            df.with_columns(scores[axis].alias("_score"))
            .select(["segment_id", "_score"])
            .drop_nulls()
            .to_dicts()
        )
        idxs: list[int] = []
        ys: list[float] = []
        for row in rows:
            idx = id_to_idx.get(str(row["segment_id"]))
            if idx is None:
                continue
            idxs.append(idx)
            ys.append(float(row["_score"]))
        if len(idxs) >= 30:
            directions[f"global_label::{axis}"] = train_direction(
                features[np.asarray(idxs)],
                np.asarray(ys, dtype=np.float32),
                frac,
            )

    idxs = []
    ys = []
    for i, sid in enumerate(ids):
        vid = sid.split("_seg_")[0]
        if vid in bmd_scores:
            idxs.append(i)
            ys.append(bmd_scores[vid])
    if idxs:
        directions["BMD_human_memorability"] = train_direction(
            features[np.asarray(idxs)],
            np.asarray(ys, dtype=np.float32),
            frac,
        )
    return directions


def load_persona_metadata(
    personas_file: Path, persona_ids: list[str]
) -> dict[str, np.ndarray]:
    df = pl.read_parquet(personas_file)
    weights = df.select("attention_weights").unnest("attention_weights")
    dislikes = df.select("dislikes").unnest("dislikes")
    flat = df.select("persona_id").with_columns(
        [weights[col].alias(f"weight::{col}") for col in weights.columns]
        + [dislikes[col].alias(f"dislike::{col}") for col in dislikes.columns]
    )
    by_id = {row["persona_id"]: row for row in flat.to_dicts()}
    out: dict[str, np.ndarray] = {}
    for col in flat.columns:
        if col == "persona_id":
            continue
        vals = [float(by_id[p][col]) for p in persona_ids if p in by_id]
        if len(vals) == len(persona_ids):
            out[col] = np.asarray(vals, dtype=np.float64)
    return out


def summarize_component(
    component_idx: int,
    axis_vector: np.ndarray,
    loadings: np.ndarray,
    persona_ids: list[str],
    construct_dirs: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
) -> dict[str, object]:
    construct_cos = {
        name: cosine(axis_vector, direction)
        for name, direction in construct_dirs.items()
    }
    meta_corr = {name: spearman(loadings, vals) for name, vals in metadata.items()}
    top_constructs = sorted(
        construct_cos.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:8]
    top_meta = sorted(meta_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    order = np.argsort(loadings)
    return {
        "component": component_idx + 1,
        "positive_personas": [
            {"persona": persona_ids[i], "loading": float(loadings[i])}
            for i in order[-4:][::-1]
        ],
        "negative_personas": [
            {"persona": persona_ids[i], "loading": float(loadings[i])}
            for i in order[:4]
        ],
        "top_construct_cosines": [
            {"construct": name, "cosine": float(value)}
            for name, value in top_constructs
        ],
        "top_metadata_spearman": [
            {"metadata": name, "rho": float(value)} for name, value in top_meta
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=Path("data/features/tribe")
    )
    parser.add_argument(
        "--persona-file",
        type=Path,
        default=Path("data/labels/synthetic_persona_haiku_clean.parquet"),
    )
    parser.add_argument(
        "--generic-label-file",
        type=Path,
        default=Path("data/labels/synthetic_gemini.parquet"),
    )
    parser.add_argument(
        "--personas-file", type=Path, default=Path("data/labels/personas.parquet")
    )
    parser.add_argument(
        "--bmd-annotations",
        type=Path,
        default=Path("data/raw/bold_moments/annotations.json"),
    )
    parser.add_argument("--top-k-frac", type=float, default=0.30)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/reports/persona_axis_decode.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("data/reports/persona_axis_decode.md"),
    )
    args = parser.parse_args()

    print("[decode] loading TRIBE features")
    ids, features = load_feature_matrix(args.features_dir)
    print(f"[decode] features: n={len(ids)} dim={features.shape[1]}")

    print("[decode] training persona memorability directions")
    persona_ids, mem_dirs = train_persona_directions(
        ids, features, args.persona_file, "memorability", args.top_k_frac
    )
    print(f"[decode] persona directions: {len(persona_ids)}")

    print("[decode] training global construct directions")
    bmd_scores = load_bmd_scores(args.bmd_annotations)
    construct_dirs = train_global_label_directions(
        ids, features, args.generic_label_file, bmd_scores, args.top_k_frac
    )
    print(f"[decode] global construct directions: {len(construct_dirs)}")

    print("[decode] PCA/SVD on persona memorability directions")
    u, s, vt = np.linalg.svd(mem_dirs, full_matrices=False)
    variance = s**2
    variance_ratio = variance / variance.sum()
    components = []
    metadata = load_persona_metadata(args.personas_file, persona_ids)
    n_components = min(args.n_components, len(persona_ids))
    for k in range(n_components):
        axis_vector = vt[k].copy()
        loadings = (u[:, k] * s[k]).copy()
        bmd_cos = construct_dirs.get("BMD_human_memorability")
        if bmd_cos is not None and cosine(axis_vector, bmd_cos) < 0:
            axis_vector *= -1
            loadings *= -1
        elif bmd_cos is None and loadings.sum() < 0:
            axis_vector *= -1
            loadings *= -1
        summary = summarize_component(
            k,
            axis_vector,
            loadings,
            persona_ids,
            construct_dirs,
            metadata,
        )
        summary["variance_ratio"] = float(variance_ratio[k])
        components.append(summary)

    print("[decode] RSA against other persona construct matrices")
    mem_abs = upper_triangle(np.abs(mem_dirs @ mem_dirs.T))
    rsa = []
    for axis in LABEL_AXES:
        axis_personas, dirs = train_persona_directions(
            ids, features, args.persona_file, axis, args.top_k_frac
        )
        if axis_personas != persona_ids:
            continue
        sim = upper_triangle(np.abs(dirs @ dirs.T))
        rsa.append(
            {
                "construct_axis": axis,
                "spearman_vs_memorability_abs_cos": spearman(mem_abs, sim),
                "mean_abs_cos": float(sim.mean()),
                "median_abs_cos": float(np.median(sim)),
            }
        )
    rsa.sort(key=lambda row: abs(row["spearman_vs_memorability_abs_cos"]), reverse=True)

    payload = {
        "n_personas": len(persona_ids),
        "persona_ids": persona_ids,
        "feature_dim": int(features.shape[1]),
        "svd_variance_ratio": [float(x) for x in variance_ratio.tolist()],
        "svd_cumulative_variance_ratio": [
            float(x) for x in np.cumsum(variance_ratio).tolist()
        ],
        "components": components,
        "rsa": rsa,
        "available_constructs": sorted(construct_dirs),
        "note": (
            "Exploratory decode. Valence is not directly measured in current labels; "
            "emotional_intensity is an arousal-like proxy, and semantic_surprise is "
            "the available surprise proxy."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Persona Latent Axis Decode",
        "",
        "Exploratory decode of the low-rank persona memorability result.",
        "",
        "Valence is not directly labeled in the current dataset; "
        "`emotional_intensity` is the closest arousal-like proxy, and "
        "`semantic_surprise` is the surprise proxy.",
        "",
        "## SVD",
        "",
        "| component | variance | cumulative |",
        "|---|---:|---:|",
    ]
    cumulative = np.cumsum(variance_ratio)
    for k in range(n_components):
        lines.append(f"| PC{k + 1} | {variance_ratio[k]:.3f} | {cumulative[k]:.3f} |")

    lines += ["", "## Component Decodes", ""]
    for comp in components:
        lines.append(f"### PC{comp['component']}")
        lines.append(f"- variance explained: {comp['variance_ratio']:.3f}")
        pos = ", ".join(
            f"{row['persona']} ({row['loading']:+.2f})"
            for row in comp["positive_personas"]
        )
        neg = ", ".join(
            f"{row['persona']} ({row['loading']:+.2f})"
            for row in comp["negative_personas"]
        )
        lines.append(f"- positive personas: {pos}")
        lines.append(f"- negative personas: {neg}")
        constructs = ", ".join(
            f"{row['construct'].replace('global_label::', '')} ({row['cosine']:+.2f})"
            for row in comp["top_construct_cosines"][:5]
        )
        metadata_bits = ", ".join(
            f"{row['metadata']} ({row['rho']:+.2f})"
            for row in comp["top_metadata_spearman"][:5]
        )
        lines.append(f"- construct cosines: {constructs}")
        lines.append(f"- persona metadata correlations: {metadata_bits}")
        lines.append("")

    lines += [
        "## RSA-Style Matrix Decode",
        "",
        "Spearman correlation between the memorability persona |cos| matrix and each",
        "other per-persona construct |cos| matrix.",
        "",
        "| construct | RSA rho | mean |cos| | median |cos| |",
        "|---|---:|---:|---:|",
    ]
    for row in rsa:
        lines.append(
            f"| {row['construct_axis']} | "
            f"{row['spearman_vs_memorability_abs_cos']:+.3f} | "
            f"{row['mean_abs_cos']:.3f} | {row['median_abs_cos']:.3f} |"
        )

    args.md_out.write_text("\n".join(lines) + "\n")
    print(f"[decode] wrote {args.json_out}")
    print(f"[decode] wrote {args.md_out}")


if __name__ == "__main__":
    main()
