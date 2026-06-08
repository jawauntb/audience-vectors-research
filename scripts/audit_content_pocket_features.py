"""Audit visual descriptors for restored SVD content pockets.

The pocket-regime audit established that several restored seed-content pockets
stay positive under local alpha/guidance perturbations while hard controls stay
negative. This script asks a narrower follow-up question: can lightweight visual
descriptors of the seed images and generated videos separate the positive
pockets from the hard negative controls without using the TRIBE score itself?
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from audience_vectors.bo_prompt_manifests import build_replay_seed_pool

POSITIVE_POCKETS = {
    "fresh24_orange_flowers",
    "fresh24_hanging_clothes",
    "fresh24_blue_jellyfish",
    "fresh24_old_car",
}
NEGATIVE_CONTROLS = {
    "fresh24_aerial_beach",
    "fresh24_city_street",
    "fresh24_storm_beach",
}

DEFAULT_REPLAY_REPORT = Path(
    "data/reports/"
    "bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_"
    "20260608.json"
)
DEFAULT_OUT_JSON = (
    Path("research_program")
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
    / "content_pocket_feature_audit_summary_20260608.json"
)
DEFAULT_OUT_MD = (
    Path("research_program")
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
    / "content_pocket_feature_audit_result_20260608.md"
)


def logical_path(path: Path) -> str:
    """Return a stable repo/data-lake-looking path for reports."""
    parts = path.resolve().parts
    for anchor in ("data", "research_program", "scripts", "src", "tests"):
        if anchor in parts:
            return str(Path(*parts[parts.index(anchor) :]))
    return str(path)


def as_array(image: Image.Image) -> np.ndarray:
    """Convert an image to an RGB numpy array."""
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def load_image_descriptor(path: Path) -> dict[str, float]:
    """Compute descriptors for one image file."""
    with Image.open(path) as image:
        return visual_descriptors(as_array(image))


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB to HSV for arrays scaled to 0..1."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    chroma = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = chroma > 1e-8
    r_is_max = (maxc == r) & nonzero
    g_is_max = (maxc == g) & nonzero
    b_is_max = (maxc == b) & nonzero
    hue[r_is_max] = ((g[r_is_max] - b[r_is_max]) / chroma[r_is_max]) % 6
    hue[g_is_max] = ((b[g_is_max] - r[g_is_max]) / chroma[g_is_max]) + 2
    hue[b_is_max] = ((r[b_is_max] - g[b_is_max]) / chroma[b_is_max]) + 4
    hue = hue / 6.0

    saturation = np.zeros_like(maxc)
    saturation[maxc > 1e-8] = chroma[maxc > 1e-8] / maxc[maxc > 1e-8]
    return hue, saturation, maxc


def visual_descriptors(frame: np.ndarray) -> dict[str, float]:
    """Compute lightweight color, edge, and composition descriptors."""
    rgb = np.asarray(frame, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 frame, got shape {rgb.shape}")
    rgb = rgb[..., :3]
    if float(rgb.max(initial=0.0)) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    gray = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    hue, saturation, value = rgb_to_hsv(rgb)

    grad_y = np.diff(gray, axis=0)
    grad_x = np.diff(gray, axis=1)
    edge_mean = float(np.mean(np.abs(grad_y)) + np.mean(np.abs(grad_x)))

    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        lap_var = float(np.var(lap))
    else:
        lap_var = 0.0

    height, width = gray.shape
    y0, y1 = height // 4, (height * 3) // 4
    x0, x1 = width // 4, (width * 3) // 4
    center = gray[y0:y1, x0:x1]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[y0:y1, x0:x1] = False

    rg = r - g
    yb = ((r + g) / 2.0) - b
    colorfulness = math.sqrt(float(np.var(rg)) + float(np.var(yb))) + 0.3 * math.sqrt(
        float(np.mean(rg) ** 2) + float(np.mean(yb) ** 2)
    )

    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    probs = hist.astype(np.float64)
    probs = probs[probs > 0] / probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum())

    def fraction(mask: np.ndarray) -> float:
        return float(np.mean(mask.astype(np.float32)))

    orange = (hue >= 0.055) & (hue <= 0.16) & (saturation > 0.25) & (value > 0.2)
    yellow = (hue > 0.16) & (hue <= 0.22) & (saturation > 0.25) & (value > 0.2)
    green = (hue > 0.22) & (hue <= 0.45) & (saturation > 0.2) & (value > 0.2)
    cyan_blue = (hue > 0.45) & (hue <= 0.68) & (saturation > 0.2) & (value > 0.2)
    red = ((hue <= 0.04) | (hue >= 0.94)) & (saturation > 0.25) & (value > 0.2)
    neutral = saturation < 0.12

    return {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "saturation_mean": float(np.mean(saturation)),
        "saturation_std": float(np.std(saturation)),
        "value_mean": float(np.mean(value)),
        "rgb_r_mean": float(np.mean(r)),
        "rgb_g_mean": float(np.mean(g)),
        "rgb_b_mean": float(np.mean(b)),
        "warm_minus_cool": float(np.mean(r - b)),
        "green_excess": float(np.mean(g - np.maximum(r, b))),
        "colorfulness": float(colorfulness),
        "edge_mean": edge_mean,
        "laplacian_var": lap_var,
        "gray_entropy": entropy,
        "center_brightness_delta": float(np.mean(center) - np.mean(gray[border_mask])),
        "orange_fraction": fraction(orange),
        "yellow_fraction": fraction(yellow),
        "green_fraction": fraction(green),
        "cyan_blue_fraction": fraction(cyan_blue),
        "red_fraction": fraction(red),
        "neutral_fraction": fraction(neutral),
        "bright_fraction": fraction(value > 0.75),
        "dark_fraction": fraction(value < 0.2),
    }


def load_video_descriptor(path: Path, *, max_frames: int) -> dict[str, float]:
    """Compute descriptors over a small deterministic frame sample."""
    import imageio.v3 as iio  # noqa: PLC0415

    frames = np.asarray(iio.imread(path))
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise ValueError(f"expected video frame stack, got shape {frames.shape}")

    n_frames = frames.shape[0]
    n_sample = min(max_frames, n_frames)
    indices = np.linspace(0, n_frames - 1, n_sample).round().astype(int)
    frame_descriptors = [visual_descriptors(frames[idx]) for idx in indices]
    keys = sorted(frame_descriptors[0])
    averaged = {
        key: float(np.mean([descriptor[key] for descriptor in frame_descriptors]))
        for key in keys
    }
    averaged["sampled_frames"] = float(n_sample)
    return averaged


def load_report(path: Path) -> dict[str, Any]:
    """Load a replay report."""
    return json.loads(path.read_text())


def resolve_report_path(path: Path) -> Path:
    """Resolve a report path without requiring it to exist in clean worktrees."""
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{path} does not exist. Pass --replay-report pointing at the local "
        "data-lake copy of the pocket-regime audit report."
    )


def repo_root_for_report(report_path: Path) -> Path:
    """Infer the worktree root from a data/reports report path."""
    if report_path.parent.name == "reports" and report_path.parent.parent.name == "data":
        return report_path.parent.parent.parent
    return Path.cwd()


def score_rows(
    report: dict[str, Any],
    *,
    seed_root: Path,
    video_root: Path,
    max_video_frames: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    """Join report rows to seed slots and descriptors."""
    pool_size = int(report["replay_seed_pool_size"])
    seed_pool = build_replay_seed_pool(seed_root, pool_size=pool_size)
    seed_by_slot = {slot.slot: slot for slot in seed_pool}
    seed_descriptors = {
        slot: load_image_descriptor(seed.image_path) for slot, seed in seed_by_slot.items()
    }

    rows: list[dict[str, Any]] = []
    for raw_row in report["rows"]:
        score = raw_row.get("replay_tribe_score")
        if score is None:
            continue
        trial = raw_row["trial"]
        seed_idx = int(trial["seed_idx"])
        seed_slot = seed_by_slot[seed_idx]
        pocket = seed_slot.bmd_name
        if pocket not in POSITIVE_POCKETS and pocket not in NEGATIVE_CONTROLS:
            continue

        video_descriptor: dict[str, float] = {}
        local_video_path = raw_row.get("local_video_path")
        if local_video_path:
            video_path = video_root / str(local_video_path)
            if video_path.exists():
                video_descriptor = load_video_descriptor(
                    video_path,
                    max_frames=max_video_frames,
                )

        rows.append(
            {
                "task_id": str(trial["task_id"]),
                "pocket": pocket,
                "label": "positive"
                if pocket in POSITIVE_POCKETS
                else "negative_control",
                "seed_idx": seed_idx,
                "alpha": float(trial["alpha"]),
                "guidance": float(trial["guidance"]),
                "noise_seed": int(raw_row["noise_seed"]),
                "replay_tribe_score": float(score),
                "visual_first_status": str(raw_row.get("visual_first_status")),
                "seed_features": seed_descriptors[seed_idx],
                "video_features": video_descriptor,
            }
        )
    return rows, seed_descriptors


def aggregate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate replicate rows to candidate/task level."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)

    candidates: list[dict[str, Any]] = []
    for task_id, group in sorted(grouped.items()):
        first = group[0]
        scores = [float(row["replay_tribe_score"]) for row in group]
        video_feature_keys = sorted(
            set().union(*(row["video_features"].keys() for row in group))
        )
        video_features = {
            key: float(
                np.mean(
                    [
                        row["video_features"][key]
                        for row in group
                        if key in row["video_features"]
                    ]
                )
            )
            for key in video_feature_keys
        }
        candidates.append(
            {
                "task_id": task_id,
                "pocket": first["pocket"],
                "label": first["label"],
                "seed_idx": first["seed_idx"],
                "alpha": first["alpha"],
                "guidance": first["guidance"],
                "n_replicates": len(group),
                "mean_replay_tribe_score": float(np.mean(scores)),
                "min_replay_tribe_score": float(np.min(scores)),
                "max_replay_tribe_score": float(np.max(scores)),
                "seed_features": first["seed_features"],
                "video_features": video_features,
            }
        )
    return candidates


def aggregate_by_pocket(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate candidate rows to pocket-level summaries."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["pocket"])].append(candidate)

    summaries: list[dict[str, Any]] = []
    for pocket, group in sorted(grouped.items()):
        scores = [float(row["mean_replay_tribe_score"]) for row in group]
        summaries.append(
            {
                "pocket": pocket,
                "label": group[0]["label"],
                "n_candidates": len(group),
                "mean_score": float(np.mean(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "positive_candidates": int(sum(score > 0 for score in scores)),
            }
        )
    return summaries


def finite_values(values: list[float]) -> bool:
    """Return true if all values are finite and nonconstant."""
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)) and np.std(arr) > 1e-12)


def pearson(x: list[float], y: list[float]) -> float | None:
    """Small Pearson correlation helper."""
    if not finite_values(x) or not finite_values(y):
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def cohen_d(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Compute signed Cohen's d."""
    if len(pos) < 2 or len(neg) < 2:
        return None
    pos_var = float(np.var(pos, ddof=1))
    neg_var = float(np.var(neg, ddof=1))
    pooled = math.sqrt(((len(pos) - 1) * pos_var + (len(neg) - 1) * neg_var) / (len(pos) + len(neg) - 2))
    if pooled <= 1e-12:
        return None
    return float((np.mean(pos) - np.mean(neg)) / pooled)


def analyze_feature_family(
    candidates: list[dict[str, Any]],
    *,
    family: str,
) -> list[dict[str, Any]]:
    """Score how well one descriptor family separates positives and controls."""
    feature_rows = [row for row in candidates if row[f"{family}_features"]]
    if not feature_rows:
        return []
    feature_keys = sorted(set().union(*(row[f"{family}_features"].keys() for row in feature_rows)))
    labels = [1 if row["label"] == "positive" else 0 for row in feature_rows]
    scores = [float(row["mean_replay_tribe_score"]) for row in feature_rows]

    metrics: list[dict[str, Any]] = []
    for key in feature_keys:
        values = [float(row[f"{family}_features"][key]) for row in feature_rows]
        if not finite_values(values):
            continue
        pos = np.asarray([value for value, label in zip(values, labels, strict=True) if label == 1])
        neg = np.asarray([value for value, label in zip(values, labels, strict=True) if label == 0])
        auc = float(roc_auc_score(labels, values))
        signed_d = cohen_d(pos, neg)
        pos_mean = float(np.mean(pos))
        neg_mean = float(np.mean(neg))
        metrics.append(
            {
                "feature": key,
                "family": family,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "direction": "higher_for_positive" if auc >= 0.5 else "lower_for_positive",
                "roc_auc_raw": auc,
                "separation_auc": max(auc, 1.0 - auc),
                "cohen_d": signed_d,
                "abs_cohen_d": abs(signed_d) if signed_d is not None else None,
                "pearson_with_mean_score": pearson(values, scores),
            }
        )

    return sorted(
        metrics,
        key=lambda row: (
            float(row["separation_auc"]),
            float(row["abs_cohen_d"] or 0.0),
        ),
        reverse=True,
    )


def gate_summary(
    seed_metrics: list[dict[str, Any]],
    video_metrics: list[dict[str, Any]],
    *,
    min_auc: float,
    min_abs_d: float,
) -> dict[str, Any]:
    """Decide whether descriptor-level separation clears the audit gate."""
    candidates = seed_metrics + video_metrics
    passing = [
        row
        for row in candidates
        if float(row["separation_auc"]) >= min_auc
        and float(row["abs_cohen_d"] or 0.0) >= min_abs_d
    ]
    best = passing[0] if passing else candidates[0] if candidates else None
    return {
        "accepted": bool(passing),
        "rule": f"separation_auc >= {min_auc:.2f} and abs_cohen_d >= {min_abs_d:.2f}",
        "n_passing_features": len(passing),
        "best_feature": best,
    }


def fmt(value: Any, digits: int = 4) -> str:
    """Format optional floats for markdown."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_feature_table(rows: list[dict[str, Any]], *, limit: int = 8) -> str:
    """Render top descriptor rows."""
    lines = [
        "| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            "| {family} | {feature} | {direction} | {positive_mean} | "
            "{negative_mean} | {auc} | {abs_d} | {corr} |".format(
                family=row["family"],
                feature=row["feature"],
                direction=row["direction"],
                positive_mean=fmt(float(row["positive_mean"])),
                negative_mean=fmt(float(row["negative_mean"])),
                auc=fmt(float(row["separation_auc"])),
                abs_d=fmt(row["abs_cohen_d"]),
                corr=fmt(row["pearson_with_mean_score"]),
            )
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the human-readable audit result."""
    gate = summary["gate"]
    score_rows = sorted(
        summary["score_by_pocket"],
        key=lambda row: float(row["mean_score"]),
        reverse=True,
    )
    lines = [
        "# Content-Pocket Feature Audit Result - 2026-06-08",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: are the stable positive pockets explainable by lightweight "
        "visual descriptors, or are they only TRIBE score islands?",
        "",
        "Current regime:",
        "",
        "- Artifact types: restored seed images, generated SVD replay videos, "
        "TRIBE replay scores, visual-gate status, seed/video descriptors, and "
        "positive/control pocket labels.",
        "- Operations: join the pocket-regime replay report to restored seed "
        "images, compute visual descriptors on seed images and sampled generated "
        "video frames, then compare positive pockets with hard negative controls.",
        "- Gates/verifiers: descriptor separation must not use TRIBE score as an "
        "input feature; acceptance requires the pre-registered AUC and effect-size "
        "threshold in the gate below.",
        "",
        "Action class: search inside the current compute-proxy regime. It becomes "
        "discovery-relevant only if the descriptor becomes an accepted verifier or "
        "artifact class for the content-pocket regime.",
        "",
        "## Inputs",
        "",
        f"- Replay report: `{summary['source_replay_report']}`",
        f"- Seed root: `{summary['seed_root']}`",
        f"- Candidates: {summary['n_candidates']} task-level candidates from "
        f"{summary['n_rows']} scored replicate rows.",
        "- Positive targets: orange flowers, hanging clothes, blue jellyfish, old car.",
        "- Negative controls: aerial beach, city street, storm beach.",
        "",
        "## Score By Pocket",
        "",
        "| pocket | label | candidates | mean | min | max | positive candidates |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            "| {pocket} | {label} | {n} | {mean} | {min_score} | {max_score} | {pos} |".format(
                pocket=row["pocket"],
                label=row["label"],
                n=row["n_candidates"],
                mean=fmt(float(row["mean_score"])),
                min_score=fmt(float(row["min_score"])),
                max_score=fmt(float(row["max_score"])),
                pos=row["positive_candidates"],
            )
        )

    top_metrics = summary["top_seed_features"] + summary["top_video_features"]
    top_metrics = sorted(
        top_metrics,
        key=lambda row: (
            float(row["separation_auc"]),
            float(row["abs_cohen_d"] or 0.0),
        ),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Best Descriptor Separators",
            "",
            render_feature_table(top_metrics, limit=10),
            "",
            "## Gate",
            "",
            f"Acceptance rule: {gate['rule']}.",
            "",
            f"Gate result: **{'accepted' if gate['accepted'] else 'not accepted'}**.",
            "",
        ]
    )
    if gate["best_feature"]:
        best = gate["best_feature"]
        lines.extend(
            [
                "Best feature:",
                "",
                "- family: `{}`".format(best["family"]),
                "- feature: `{}`".format(best["feature"]),
                "- direction: {}".format(best["direction"]),
                "- separation AUC: {}".format(fmt(float(best["separation_auc"]))),
                "- absolute Cohen d: {}".format(fmt(best["abs_cohen_d"])),
                "- correlation with mean TRIBE score: {}".format(
                    fmt(best["pearson_with_mean_score"])
                ),
                "",
            ]
        )

    if gate["accepted"]:
        interpretation = (
            "The audit finds a descriptor-level explanation strong enough to keep "
            "testing: positive pockets are visually separable from the hard "
            "negative controls without using the memorability score directly. "
            "This does not prove human memorability, but it does upgrade the next "
            "work item from blind pocket replication to descriptor-conditioned "
            "content-pocket consolidation."
        )
        next_move = (
            "Run the orange-flowers and hanging-clothes stochastic replication, "
            "and track the accepted descriptor family as a covariate. If the "
            "descriptor predicts which new stochastic variants stay positive, it "
            "can become a stronger content-pocket verifier."
        )
    else:
        interpretation = (
            "The audit does not find a lightweight descriptor that clears the "
            "gate. C-017 should remain a black-box compute-proxy pocket finding "
            "until a stronger embedding or human/BMD verifier explains it."
        )
        next_move = (
            "Try a stronger embedding audit, such as CLIP/V-JEPA video embeddings, "
            "before spending more replication budget."
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Next Move",
            "",
            next_move,
            "",
        ]
    )
    return "\n".join(lines)


def build_summary(
    *,
    replay_report_path: Path,
    seed_root: Path | None,
    max_video_frames: int,
    min_auc: float,
    min_abs_d: float,
) -> dict[str, Any]:
    """Build the audit summary."""
    report_path = resolve_report_path(replay_report_path)
    report = load_report(report_path)
    resolved_seed_root = seed_root or Path(str(report["seed_root"]))
    video_root = repo_root_for_report(report_path)
    rows, _seed_descriptors = score_rows(
        report,
        seed_root=resolved_seed_root,
        video_root=video_root,
        max_video_frames=max_video_frames,
    )
    candidates = aggregate_candidates(rows)
    score_by_pocket = aggregate_by_pocket(candidates)
    seed_metrics = analyze_feature_family(candidates, family="seed")
    video_metrics = analyze_feature_family(candidates, family="video")
    gate = gate_summary(
        seed_metrics,
        video_metrics,
        min_auc=min_auc,
        min_abs_d=min_abs_d,
    )
    return {
        "schema_version": 1,
        "kind": "content_pocket_feature_audit",
        "source_replay_report": logical_path(report_path),
        "seed_root": logical_path(resolved_seed_root),
        "positive_pockets": sorted(POSITIVE_POCKETS),
        "negative_controls": sorted(NEGATIVE_CONTROLS),
        "max_video_frames": max_video_frames,
        "n_rows": len(rows),
        "n_candidates": len(candidates),
        "score_by_pocket": score_by_pocket,
        "top_seed_features": seed_metrics[:20],
        "top_video_features": video_metrics[:20],
        "gate": gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY_REPORT)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--max-video-frames", type=int, default=5)
    parser.add_argument("--min-auc", type=float, default=0.85)
    parser.add_argument("--min-abs-d", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        replay_report_path=args.replay_report,
        seed_root=args.seed_root,
        max_video_frames=args.max_video_frames,
        min_auc=args.min_auc,
        min_abs_d=args.min_abs_d,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    args.out_md.write_text(render_markdown(summary))
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "n_rows": summary["n_rows"],
                "n_candidates": summary["n_candidates"],
                "gate": summary["gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
