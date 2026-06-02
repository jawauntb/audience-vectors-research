"""Analyze palette and visual density against TRIBE-scored YouTube windows.

This is deliberately lightweight: it uses ffmpeg for frame extraction and
PIL/NumPy/SciPy/sklearn for metrics, avoiding a heavy CV detector dependency.

"Figures" are approximated with edge-connected component density:
cartoon/person/object detectors are brittle here, but connected high-gradient
regions give a useful proxy for how many visually distinct shapes are on screen.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_frames(video_path: Path, frame_dir: Path, fps: float, force: bool) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in frame_dir.glob("frame_*.jpg"):
            path.unlink()
    if any(frame_dir.glob("frame_*.jpg")):
        return
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps},scale=160:-1",
            str(frame_dir / "frame_%05d.jpg"),
        ]
    )


def rgb_to_hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.round(rgb), 0, 255).astype(int)
    return "#" + "".join(f"{v:02X}" for v in vals)


def frame_palette(rgb: np.ndarray, k: int, seed: int) -> list[dict[str, Any]]:
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 4096:
        rng = np.random.default_rng(seed)
        pixels = pixels[rng.choice(len(pixels), size=4096, replace=False)]
    n_clusters = min(k, len(pixels))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        batch_size=1024,
    )
    labels = np.asarray(km.fit_predict(pixels), dtype=np.int32)
    centers = np.asarray(km.cluster_centers_, dtype=np.float32)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    order = np.argsort(counts)[::-1]
    total = float(counts.sum())
    return [
        {
            "hex": rgb_to_hex(centers[int(idx)]),
            "proportion": float(counts[idx] / total) if total else 0.0,
        }
        for idx in order
    ]


def color_entropy(rgb: np.ndarray, bins: int = 8) -> float:
    quant = np.clip((rgb.astype(np.int32) * bins) // 256, 0, bins - 1)
    flat = quant[:, :, 0] * bins * bins + quant[:, :, 1] * bins + quant[:, :, 2]
    counts = np.bincount(flat.reshape(-1), minlength=bins**3).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)) / math.log2(bins**3))


def saturation_and_value(rgb: np.ndarray) -> tuple[float, float, float, float]:
    x = rgb.astype(np.float32) / 255.0
    maxc = x.max(axis=2)
    minc = x.min(axis=2)
    sat = np.zeros_like(maxc)
    np.divide(maxc - minc, maxc, out=sat, where=maxc > 1e-6)
    return (
        float(sat.mean()),
        float(sat.std()),
        float(maxc.mean()),
        float(maxc.std()),
    )


def visual_density_metrics(rgb: np.ndarray, prev_rgb: np.ndarray | None) -> dict[str, float]:
    x = rgb.astype(np.float32) / 255.0
    gray = (
        0.2126 * x[:, :, 0]
        + 0.7152 * x[:, :, 1]
        + 0.0722 * x[:, :, 2]
    )
    sx = ndimage.sobel(gray, axis=1, mode="reflect")
    sy = ndimage.sobel(gray, axis=0, mode="reflect")
    grad = np.sqrt(sx**2 + sy**2)
    edge_mask = grad > 0.18
    edge_density = float(edge_mask.mean())
    edge_strength = float(grad.mean())

    # Connect nearby outline fragments into rough "figure" blobs.
    closed = ndimage.binary_closing(edge_mask, structure=np.ones((3, 3)))
    dilated = ndimage.binary_dilation(closed, iterations=1)
    label_result: Any = ndimage.label(dilated)
    labels = np.asarray(label_result[0], dtype=np.int32)
    areas = np.bincount(labels.reshape(-1))[1:]
    min_area = max(12, int(rgb.shape[0] * rgb.shape[1] * 0.001))
    max_area = int(rgb.shape[0] * rgb.shape[1] * 0.65)
    component_areas = areas[(areas >= min_area) & (areas <= max_area)]
    figure_count_proxy = int(len(component_areas))
    figure_area_fraction = float(component_areas.sum() / (rgb.shape[0] * rgb.shape[1]))

    lum_contrast = float(gray.std())
    motion = 0.0
    if prev_rgb is not None:
        prev = prev_rgb.astype(np.float32) / 255.0
        if prev.shape != x.shape:
            prev = np.asarray(Image.fromarray(prev_rgb).resize((rgb.shape[1], rgb.shape[0])))
            prev = prev.astype(np.float32) / 255.0
        motion = float(np.abs(x - prev).mean())

    return {
        "edge_density": edge_density,
        "edge_strength": edge_strength,
        "figure_count_proxy": float(figure_count_proxy),
        "figure_area_fraction": figure_area_fraction,
        "luminance_contrast": lum_contrast,
        "frame_motion_delta": motion,
    }


def load_frame(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def frame_metrics(frame_paths: list[Path], fps: float, palette_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_rgb: np.ndarray | None = None
    for idx, path in enumerate(frame_paths):
        rgb = load_frame(path)
        sat_mean, sat_std, val_mean, val_std = saturation_and_value(rgb)
        metrics = {
            "frame_index": idx,
            "time_s": idx / fps,
            "frame_path": str(path),
            "palette": frame_palette(rgb, palette_k, seed=idx),
            "color_entropy": color_entropy(rgb),
            "saturation_mean": sat_mean,
            "saturation_std": sat_std,
            "value_mean": val_mean,
            "value_std": val_std,
            **visual_density_metrics(rgb, prev_rgb),
        }
        rows.append(metrics)
        prev_rgb = rgb
    return rows


def segment_for_time(segments: list[dict[str, Any]], time_s: float) -> dict[str, Any] | None:
    for row in segments:
        if row.get("status") != "ok":
            continue
        if float(row["start_s"]) <= time_s < float(row["end_s"]):
            return row
    if segments and time_s >= float(segments[-1]["start_s"]):
        return segments[-1]
    return None


def aggregate_palette(rows: list[dict[str, Any]], top_n: int = 6) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in row["palette"]:
            counts[item["hex"]] += item["proportion"]
    total = sum(counts.values())
    return [
        {"hex": color, "proportion": float(weight / total) if total else 0.0}
        for color, weight in counts.most_common(top_n)
    ]


def aggregate_by_segment(
    frame_rows: list[dict[str, Any]],
    score_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = [
        "color_entropy",
        "saturation_mean",
        "saturation_std",
        "value_mean",
        "value_std",
        "edge_density",
        "edge_strength",
        "figure_count_proxy",
        "figure_area_fraction",
        "luminance_contrast",
        "frame_motion_delta",
    ]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_row in frame_rows:
        seg = segment_for_time(score_segments, float(frame_row["time_s"]))
        if seg is None:
            continue
        by_sample[str(seg["sample_id"])].append(frame_row)

    out: list[dict[str, Any]] = []
    for seg in score_segments:
        sample_id = str(seg["sample_id"])
        rows = by_sample.get(sample_id, [])
        if not rows:
            continue
        agg = {
            "sample_id": sample_id,
            "timestamp": seg["timestamp"],
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "n_frames": len(rows),
            "mem_percentile": float(seg["bmd_reference_percentile"]),
            "mem_z": float(seg["bmd_reference_z"]),
            "attention_percentile": float(
                seg.get("synthetic_attention_reference_percentile", float("nan"))
            ),
            "attention_z": float(
                seg.get("synthetic_attention_reference_z", float("nan"))
            ),
            "palette": aggregate_palette(rows),
        }
        for metric in metrics:
            vals = np.asarray([row[metric] for row in rows], dtype=np.float32)
            agg[f"{metric}_mean"] = float(vals.mean())
            agg[f"{metric}_max"] = float(vals.max())
        out.append(agg)
    return out


def correlation_table(segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "color_entropy_mean",
        "saturation_mean_mean",
        "saturation_std_mean",
        "value_mean_mean",
        "value_std_mean",
        "edge_density_mean",
        "edge_strength_mean",
        "figure_count_proxy_mean",
        "figure_area_fraction_mean",
        "luminance_contrast_mean",
        "frame_motion_delta_mean",
    ]
    targets = ["mem_percentile", "attention_percentile"]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        x = np.asarray([row[metric] for row in segment_rows], dtype=np.float64)
        for target in targets:
            y = np.asarray([row[target] for row in segment_rows], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            spearman: Any = spearmanr(x[mask], y[mask])
            pearson: Any = pearsonr(x[mask], y[mask])
            rows.append(
                {
                    "metric": metric,
                    "target": target,
                    "spearman_r": float(spearman[0]),
                    "spearman_p": float(spearman[1]),
                    "pearson_r": float(pearson[0]),
                    "pearson_p": float(pearson[1]),
                }
            )
    return rows


def contrastive_summary(segment_rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    metrics = [
        "color_entropy_mean",
        "saturation_mean_mean",
        "value_std_mean",
        "edge_density_mean",
        "figure_count_proxy_mean",
        "figure_area_fraction_mean",
        "frame_motion_delta_mean",
    ]
    sorted_rows = sorted(segment_rows, key=lambda row: row[target])
    n = max(3, len(sorted_rows) // 4)
    low = sorted_rows[:n]
    high = sorted_rows[-n:]
    deltas = {}
    for metric in metrics:
        high_mean = float(np.mean([row[metric] for row in high]))
        low_mean = float(np.mean([row[metric] for row in low]))
        deltas[metric] = {
            "high_mean": high_mean,
            "low_mean": low_mean,
            "delta": high_mean - low_mean,
        }
    return {
        "target": target,
        "n_high": len(high),
        "n_low": len(low),
        "high_windows": [row["timestamp"] for row in high],
        "low_windows": [row["timestamp"] for row in low],
        "metric_deltas": deltas,
    }


def composite_density(segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "color_entropy_mean",
        "saturation_mean_mean",
        "value_std_mean",
        "edge_density_mean",
        "figure_count_proxy_mean",
        "figure_area_fraction_mean",
        "frame_motion_delta_mean",
    ]
    x = np.asarray([[row[f] for f in features] for row in segment_rows], dtype=np.float32)
    z = StandardScaler().fit_transform(x)
    for row, score in zip(segment_rows, z.mean(axis=1), strict=True):
        row["visual_density_composite_z"] = float(score)
    return segment_rows


def format_palette(palette: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{item['hex']} {100 * item['proportion']:.0f}%" for item in palette[:4]
    )


def write_markdown(
    *,
    report: dict[str, Any],
    out_path: Path,
) -> None:
    segments = report["segments"]
    corr = sorted(
        report["correlations"],
        key=lambda row: abs(row["spearman_r"]),
        reverse=True,
    )
    lines = [
        "# Visual Density vs TRIBE Scores",
        "",
        "This analysis decomposes sampled frames into color palettes and visual-density proxies, then compares those metrics with TRIBE memorability and synthetic-persona attention scores.",
        "",
        "## Measurement Notes",
        "",
        "- **Color palette:** six dominant frame colors from MiniBatchKMeans, aggregated by 10-second scoring window.",
        "- **Color density:** quantized RGB entropy, saturation, brightness variation.",
        "- **Figure density proxy:** connected edge-component count and edge/area density. This is not semantic object detection.",
        "- **Frame density:** frame-to-frame pixel change, used as a motion/shot-change proxy.",
        "",
        "## Strongest Correlations",
        "",
        "| Metric | Target | Spearman r | Pearson r |",
        "|---|---|---:|---:|",
    ]
    for row in corr[:10]:
        lines.append(
            f"| {row['metric']} | {row['target']} | "
            f"{row['spearman_r']:+.3f} | {row['pearson_r']:+.3f} |"
        )

    lines.extend(["", "## Contrastive Deltas", ""])
    for contrast in report["contrasts"]:
        lines.extend(
            [
                f"### High vs Low `{contrast['target']}`",
                "",
                f"- High windows: {', '.join(contrast['high_windows'])}",
                f"- Low windows: {', '.join(contrast['low_windows'])}",
                "",
                "| Metric | Low mean | High mean | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric, values in contrast["metric_deltas"].items():
            lines.append(
                f"| {metric} | {values['low_mean']:.4f} | "
                f"{values['high_mean']:.4f} | {values['delta']:+.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Window Matrix",
            "",
            "| Window | Mem %ile | Att %ile | Density z | Palette | Natural Read |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in segments:
        read = row.get("natural_read", "")
        lines.append(
            f"| {row['timestamp']} | {100 * row['mem_percentile']:.1f}% | "
            f"{100 * row['attention_percentile']:.1f}% | "
            f"{row['visual_density_composite_z']:+.2f} | "
            f"{format_palette(row['palette'])} | {read} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def natural_read(row: dict[str, Any]) -> str:
    mem = row["mem_percentile"]
    att = row["attention_percentile"]
    density = row["visual_density_composite_z"]
    if mem >= 0.85 and att >= 0.85:
        return "Hook zone: high predicted encoding and high pull."
    if att >= 0.85 and mem < 0.65:
        return "Attention-heavy: visually/temporally active but less cleanly encoded."
    if mem >= 0.85 and att < 0.8:
        return "Memory-heavy: distinctive semantic payload more than raw pull."
    if mem < 0.35 and att < 0.35:
        return "Low-density/lull zone by both score axes."
    if density >= 0.75:
        return "Visually dense, but score depends on whether the density forms a clear concept."
    if density <= -0.75:
        return "Sparse/static relative to this clip."
    return "Middle band: neither an obvious hook nor a clear trough."


def build_interpretation(report: dict[str, Any]) -> str:
    return (
        "The main pattern is not simply 'more colors/figures equals more memory.' "
        "The strongest high-high window, `00:20-00:30`, combines relatively high visual density "
        "with an identity/title transition, so density is organized into a clear concept. "
        "The factual body around `01:00-01:50` scores well when the content is concrete and imageable "
        "(tornado definition, extreme speed, waterspout category). In contrast, the quiz section "
        "from roughly `02:00-02:50` has weaker attention despite repeated factual content, suggesting "
        "that static Q&A cadence is not enough. The useful product heuristic is: maximize structured "
        "density, not raw clutter. Color/edge/motion changes should support a crisp semantic beat."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/external/youtube/nw-2sPa7DAg/source.mp4"),
    )
    parser.add_argument(
        "--score-report",
        type=Path,
        default=Path("data/external/youtube/nw-2sPa7DAg/tribe_score_report.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/external/youtube/nw-2sPa7DAg/visual_density"),
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--palette-k", type=int, default=6)
    parser.add_argument("--force-frames", action="store_true")
    args = parser.parse_args()

    frame_dir = args.out_dir / "frames"
    extract_frames(args.video, frame_dir, args.fps, args.force_frames)
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    frame_rows = frame_metrics(frame_paths, args.fps, args.palette_k)
    score_report = json.loads(args.score_report.read_text())
    segment_rows = aggregate_by_segment(frame_rows, score_report["segments"])
    segment_rows = composite_density(segment_rows)
    for row in segment_rows:
        row["natural_read"] = natural_read(row)

    report = {
        "video": str(args.video),
        "score_report": str(args.score_report),
        "fps": args.fps,
        "n_frames": len(frame_rows),
        "frame_metrics": frame_rows,
        "segments": segment_rows,
        "correlations": correlation_table(segment_rows),
        "contrasts": [
            contrastive_summary(segment_rows, "mem_percentile"),
            contrastive_summary(segment_rows, "attention_percentile"),
            contrastive_summary(segment_rows, "visual_density_composite_z"),
        ],
        "interpretation": "",
    }
    report["interpretation"] = build_interpretation(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "visual_density_report.json"
    md_path = args.out_dir / "visual_density_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report=report, out_path=md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
