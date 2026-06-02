"""Spectral positional probe for open video-encoder memorability directions.

This tests a reviewer-style hypothesis: maybe the memorability direction is
partly a positional/frequency artifact rather than a clean semantic axis. We use
an inspectable CLIP ViT frame encoder, keep hidden patch activations as
time x height x width x channel tensors, learn a contrastive memorability
direction, and decompose that direction with an FFT across the positional axes.

The result is not a TRIBE-internal mechanistic proof. It is a cheap open-model
probe that can support or weaken a simple positional-frequency explanation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_FEATURE_DIR = Path("data/features/clip_spectral_positional_probe")
DEFAULT_JSON = Path("data/reports/spectral_positional_memorability_probe.json")
DEFAULT_MD = Path("data/reports/spectral_positional_memorability_probe.md")


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    annotation_key: str
    path: Path
    score: float


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    denom = float(np.std(ra) * np.std(rb))
    if denom <= 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def unit(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("near-zero vector")
    return np.asarray(x / norm, dtype=np.float32)


def load_records(data_root: Path) -> list[VideoRecord]:
    annotations = data_root / "raw" / "bold_moments" / "annotations.json"
    video_dir = data_root / "raw" / "bold_moments" / "videos"
    payload = json.loads(annotations.read_text())
    records: list[VideoRecord] = []
    for key, row in payload.items():
        if "memorability_score" not in row:
            continue
        path = video_dir / f"vid_idx{key}.mp4"
        if path.exists():
            records.append(
                VideoRecord(
                    video_id=f"bmd_vid_idx{key}",
                    annotation_key=str(key),
                    path=path,
                    score=float(row["memorability_score"]),
                )
            )
    if not records:
        raise FileNotFoundError(f"no scored videos found in {video_dir}")
    return records


def select_records(records: Sequence[VideoRecord], max_videos: int) -> list[VideoRecord]:
    ordered = sorted(records, key=lambda r: r.score)
    if max_videos <= 0 or max_videos >= len(ordered):
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=max_videos, dtype=int)
    return [ordered[int(i)] for i in indices]


def split_indices(n: int, *, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(8, int(round(n * test_frac)))
    n_test = min(n_test, n - 8)
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test


def train_direction(
    features: np.ndarray,
    scores: np.ndarray,
    train_idx: np.ndarray,
    *,
    top_frac: float,
    min_tail: int,
) -> np.ndarray:
    train_scores = scores[train_idx]
    n_each = max(min_tail, int(round(len(train_idx) * top_frac)))
    if n_each * 2 > len(train_idx):
        raise ValueError(f"tails too large for n_train={len(train_idx)}")
    order = train_idx[np.argsort(train_scores)]
    low = features[order[:n_each]].mean(axis=0)
    high = features[order[-n_each:]].mean(axis=0)
    return unit(high - low)


def standardize_train(
    features: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-4)
    return np.asarray((features - mean) / std, dtype=np.float32), mean, std


def optional_modules() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    modules: list[Any] = []
    for name in ("torch", "transformers", "imageio"):
        try:
            modules.append(importlib.import_module(name))
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise ModuleNotFoundError(
            "missing local dependencies: "
            + ", ".join(missing)
            + ". Install the repo's ml extras or use the project .venv."
        )
    transformers = modules[1]
    return (
        modules[0],
        transformers.CLIPImageProcessor,
        transformers.CLIPVisionModelWithProjection,
        modules[2],
    )


def resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_frames(imageio: Any, path: Path, n_frames: int) -> list[Image.Image]:
    reader = imageio.get_reader(str(path))
    try:
        try:
            n_total = int(reader.count_frames())
        except Exception:  # noqa: BLE001
            meta = reader.get_meta_data()
            n_total = int(meta.get("fps", 24) * meta.get("duration", 3.0))
        indices = np.linspace(0, max(n_total - 1, 0), num=n_frames, dtype=int)
        return [
            Image.fromarray(np.asarray(reader.get_data(int(i)), dtype=np.uint8))
            for i in indices
        ]
    finally:
        reader.close()


def cache_path(
    feature_dir: Path,
    *,
    model_id: str,
    layer_index: int,
    frames_per_video: int,
    max_videos: int,
) -> Path:
    slug = model_id.replace("/", "__")
    return feature_dir / (
        f"{slug}_layer{layer_index}_f{frames_per_video}_n{max_videos}.npz"
    )


def load_or_extract_features(
    *,
    data_root: Path,
    feature_dir: Path,
    model_id: str,
    layer_index: int,
    frames_per_video: int,
    max_videos: int,
    batch_videos: int,
    device_name: str,
    allow_downloads: bool,
    overwrite: bool,
) -> tuple[list[VideoRecord], np.ndarray, dict[str, Any]]:
    records = select_records(load_records(data_root), max_videos=max_videos)
    out_path = cache_path(
        feature_dir,
        model_id=model_id,
        layer_index=layer_index,
        frames_per_video=frames_per_video,
        max_videos=max_videos,
    )
    if out_path.exists() and not overwrite:
        payload = np.load(out_path, allow_pickle=True)
        video_ids = [str(x) for x in payload["video_ids"].tolist()]
        scores = np.asarray(payload["scores"], dtype=np.float32)
        by_id = {record.video_id: record for record in records}
        cached_records = [
            VideoRecord(
                video_id=video_id,
                annotation_key=by_id[video_id].annotation_key,
                path=by_id[video_id].path,
                score=float(score),
            )
            for video_id, score in zip(video_ids, scores, strict=True)
        ]
        meta = json.loads(str(payload["meta"].item()))
        return cached_records, np.asarray(payload["features"], dtype=np.float32), meta

    torch, processor_cls, model_cls, imageio = optional_modules()
    device = resolve_device(torch, device_name)
    print(f"[spectral-pos] loading {model_id} on {device}", flush=True)
    processor = processor_cls.from_pretrained(
        model_id,
        local_files_only=not allow_downloads,
    )
    model = model_cls.from_pretrained(
        model_id,
        local_files_only=not allow_downloads,
    ).to(device)
    model.eval()

    all_features: list[np.ndarray] = []
    patch_grid: int | None = None
    with torch.inference_mode():
        for start in range(0, len(records), batch_videos):
            batch_records = records[start : start + batch_videos]
            frames: list[Image.Image] = []
            for record in batch_records:
                frames.extend(sample_frames(imageio, record.path, frames_per_video))
            inputs = processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            output = model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states = output.hidden_states
            layer = layer_index if layer_index >= 0 else len(hidden_states) + layer_index
            hidden = hidden_states[layer].detach().float().cpu().numpy()
            patches = hidden[:, 1:, :]
            grid = int(round(math.sqrt(patches.shape[1])))
            if grid * grid != patches.shape[1]:
                raise ValueError(f"non-square patch grid: {patches.shape[1]}")
            patch_grid = grid
            patches = patches.reshape(len(batch_records), frames_per_video, grid, grid, -1)
            all_features.append(np.asarray(patches, dtype=np.float32))
            print(
                f"[spectral-pos] extracted {min(start + len(batch_records), len(records))}/{len(records)}",
                flush=True,
            )

    features = np.concatenate(all_features, axis=0)
    meta = {
        "model_id": model_id,
        "layer_index": layer_index,
        "frames_per_video": frames_per_video,
        "patch_grid": patch_grid,
        "feature_shape": list(features.shape),
        "local_files_only": not allow_downloads,
    }
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features,
        scores=np.asarray([record.score for record in records], dtype=np.float32),
        video_ids=np.asarray([record.video_id for record in records], dtype=object),
        meta=np.asarray(json.dumps(meta)),
    )
    print(f"[spectral-pos] wrote {out_path}", flush=True)
    return records, features, meta


def fft_energy(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.fftn(direction, axes=(0, 1, 2), norm="ortho")
    energy = np.abs(spectrum) ** 2
    energy_pos = energy.sum(axis=-1)
    return spectrum, np.asarray(energy_pos, dtype=np.float64)


def frequency_grids(t: int, h: int, w: int) -> dict[str, np.ndarray]:
    ft = np.abs(np.fft.fftfreq(t))
    fy = np.abs(np.fft.fftfreq(h))
    fx = np.abs(np.fft.fftfreq(w))
    tt, yy, xx = np.meshgrid(ft, fy, fx, indexing="ij")
    spatial_radius = np.sqrt(xx**2 + yy**2)
    max_spatial = float(spatial_radius.max()) or 1.0
    max_temporal = float(tt.max()) or 1.0
    return {
        "temporal": tt,
        "spatial_radius": spatial_radius,
        "temporal_rel": tt / max_temporal,
        "spatial_rel": spatial_radius / max_spatial,
    }


def masks(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    grids = frequency_grids(*shape)
    temporal = grids["temporal"]
    temporal_rel = grids["temporal_rel"]
    spatial_rel = grids["spatial_rel"]
    return {
        "temporal_dc": temporal == 0,
        "temporal_low_nonzero": (temporal > 0) & (temporal_rel <= 0.5),
        "temporal_high": temporal_rel > 0.5,
        "spatial_dc": spatial_rel == 0,
        "spatial_low": (spatial_rel > 0) & (spatial_rel <= 0.25),
        "spatial_mid": (spatial_rel > 0.25) & (spatial_rel <= 0.50),
        "spatial_high": spatial_rel > 0.50,
        "low_temporal_low_spatial": (temporal_rel <= 0.5) & (spatial_rel <= 0.25),
        "nonzero_temporal": temporal > 0,
        "all": np.ones(shape, dtype=bool),
    }


def direction_from_mask(spectrum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = spectrum * mask[..., None]
    out = np.fft.ifftn(masked, axes=(0, 1, 2), norm="ortho").real
    return np.asarray(out, dtype=np.float32)


def project(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.tensordot(features, direction, axes=features.ndim - 1)


def band_results(
    *,
    features: np.ndarray,
    scores: np.ndarray,
    test_idx: np.ndarray,
    direction: np.ndarray,
    spectrum: np.ndarray,
    energy: np.ndarray,
) -> list[dict[str, Any]]:
    total_energy = float(energy.sum())
    out: list[dict[str, Any]] = []
    for name, mask in masks(direction.shape[:3]).items():
        band_direction = direction_from_mask(spectrum, mask)
        band_energy = float(energy[mask].sum())
        if float(np.linalg.norm(band_direction)) <= 1e-12:
            rho = 0.0
            norm = 0.0
        else:
            band_direction = unit(band_direction)
            pred = project(features[test_idx], band_direction)
            rho = spearman(pred, scores[test_idx])
            norm = float(np.linalg.norm(band_direction))
        out.append(
            {
                "band": name,
                "energy_fraction": band_energy / total_energy if total_energy else 0.0,
                "test_rho_band_only": rho,
                "direction_norm_after_unit": norm,
            }
        )
    return sorted(out, key=lambda row: row["energy_fraction"], reverse=True)


def positional_embedding_spectrum(
    *,
    model_id: str,
    allow_downloads: bool,
) -> dict[str, Any] | None:
    torch, _processor_cls, model_cls, _imageio = optional_modules()
    model = model_cls.from_pretrained(model_id, local_files_only=not allow_downloads)
    embeddings = model.vision_model.embeddings
    if not hasattr(embeddings, "position_embedding"):
        return None
    weight = embeddings.position_embedding.weight.detach().float().cpu().numpy()
    if weight.shape[0] <= 1:
        return None
    patch = weight[1:]
    grid = int(round(math.sqrt(patch.shape[0])))
    if grid * grid != patch.shape[0]:
        return None
    patch = patch.reshape(grid, grid, -1)
    spectrum = np.fft.fftn(patch, axes=(0, 1), norm="ortho")
    energy = np.abs(spectrum).sum(axis=-1) ** 2
    spatial_rel = frequency_grids(1, grid, grid)["spatial_rel"][0]
    total = float(energy.sum())
    bins = {
        "spatial_dc": spatial_rel == 0,
        "spatial_low": (spatial_rel > 0) & (spatial_rel <= 0.25),
        "spatial_mid": (spatial_rel > 0.25) & (spatial_rel <= 0.50),
        "spatial_high": spatial_rel > 0.50,
    }
    return {
        "shape": list(weight.shape),
        "patch_grid": grid,
        "energy_fractions": {
            name: float(energy[mask].sum() / total) if total else 0.0
            for name, mask in bins.items()
        },
    }


def summarize_energy(energy: np.ndarray) -> dict[str, float]:
    total = float(energy.sum())
    out: dict[str, float] = {}
    for name, mask in masks(energy.shape).items():
        if name == "all":
            continue
        out[name] = float(energy[mask].sum() / total) if total else 0.0
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Spectral Positional Memorability Probe",
        "",
        "Open-encoder probe of Spencer's positional-frequency critique. This uses CLIP ViT hidden patch states as an inspectable frame-level video encoder, learns a contrastive memorability direction, and decomposes that direction across time/space FFT bands.",
        "",
        "## Summary",
        "",
        f"- Model: `{report['config']['model_id']}`",
        f"- Layer: `{report['config']['layer_index']}`",
        f"- Videos: **{report['data']['n_videos']}**",
        f"- Feature tensor: `{report['data']['feature_shape']}`",
        f"- Full tensor direction test rho: **{report['readout']['full_test_rho']:+.3f}**",
        f"- Mean-pooled direction test rho: **{report['readout']['mean_pooled_test_rho']:+.3f}**",
        "",
        "## Direction Energy By FFT Band",
        "",
        "| band | energy fraction | band-only test rho |",
        "|---|---:|---:|",
    ]
    for row in report["band_results"]:
        lines.append(
            f"| `{row['band']}` | {row['energy_fraction']:.3f} | {row['test_rho_band_only']:+.3f} |"
        )
    lines += [
        "",
        "## Positional-Embedding Spectrum",
        "",
    ]
    pos = report.get("positional_embedding_spectrum")
    if pos is None:
        lines.append("No learned positional embedding table was available for FFT analysis.")
    else:
        lines.append(f"- Patch grid: **{pos['patch_grid']}x{pos['patch_grid']}**")
        for name, value in pos["energy_fractions"].items():
            lines.append(f"- `{name}`: {value:.3f}")
    lines += [
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "Caveat: this is an open CLIP frame-encoder proxy, not TRIBE internals and not Wan2.2 RoPE patching. It answers whether a simple positional-frequency explanation is plausible in an inspectable encoder.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def interpretation_from(report: dict[str, Any]) -> str:
    bands = {row["band"]: row for row in report["band_results"]}
    full = float(report["readout"]["full_test_rho"])
    temporal_dc = float(bands["temporal_dc"]["test_rho_band_only"])
    spatial_dc = float(bands["spatial_dc"]["test_rho_band_only"])
    low = float(bands["spatial_low"]["test_rho_band_only"])
    high = float(bands["spatial_high"]["test_rho_band_only"])
    temporal = float(bands["nonzero_temporal"]["test_rho_band_only"])
    energy = report["direction_energy"]
    if abs(full) < 0.15:
        return (
            "The open-encoder tensor direction is weak on held-out videos, so the spectral decomposition should be treated as exploratory. This does not yet support a strong mechanistic claim."
        )
    if (
        max(abs(temporal_dc), abs(spatial_dc)) >= abs(full) * 0.8
        and max(abs(high), abs(temporal)) < abs(full) * 0.5
    ):
        return (
            "Most of the held-out signal is recoverable from temporally stable or spatially pooled components, while nonzero temporal and high-spatial-frequency bands are weak. This does not look like a high-frequency motion/position artifact; it looks more like global content, layout, or salience in this open encoder."
        )
    if max(abs(low), abs(high), abs(temporal)) >= abs(full) * 0.8:
        return (
            "A restricted positional-frequency band preserves a large fraction of the held-out signal. That makes Spencer's critique live: memorability may partly ride on positional or motion-frequency structure in this open encoder."
        )
    if energy.get("spatial_high", 0.0) + energy.get("nonzero_temporal", 0.0) < 0.25:
        return (
            "Most direction energy sits in low/position-stable modes, and no narrow frequency band clearly recovers the full readout. This weakens a simple high-frequency motion/position artifact explanation, though it does not rule out nonlinear entanglement."
        )
    return (
        "The signal is distributed across frequency bands. This is consistent with an entangled basis: positional-frequency structure matters, but no single Fourier band cleanly explains the memorability direction."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer-index", type=int, default=18)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=184)
    parser.add_argument("--batch-videos", type=int, default=4)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--min-tail", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records, raw_features, feature_meta = load_or_extract_features(
        data_root=args.data_root,
        feature_dir=args.feature_dir,
        model_id=args.model_id,
        layer_index=args.layer_index,
        frames_per_video=args.frames_per_video,
        max_videos=args.max_videos,
        batch_videos=args.batch_videos,
        device_name=args.device,
        allow_downloads=args.allow_downloads,
        overwrite=args.overwrite,
    )
    scores = np.asarray([record.score for record in records], dtype=np.float32)
    train_idx, test_idx = split_indices(len(records), test_frac=args.test_frac, seed=args.seed)
    features, _mean, _std = standardize_train(raw_features, train_idx)
    direction = train_direction(
        features,
        scores,
        train_idx,
        top_frac=args.top_frac,
        min_tail=args.min_tail,
    )
    full_pred = project(features[test_idx], direction)
    full_rho = spearman(full_pred, scores[test_idx])

    mean_features = features.mean(axis=(1, 2, 3))
    mean_direction = train_direction(
        mean_features,
        scores,
        train_idx,
        top_frac=args.top_frac,
        min_tail=args.min_tail,
    )
    mean_pred = mean_features[test_idx] @ mean_direction
    mean_rho = spearman(mean_pred, scores[test_idx])

    spectrum, energy = fft_energy(direction)
    report: dict[str, Any] = {
        "config": {
            "model_id": args.model_id,
            "layer_index": args.layer_index,
            "frames_per_video": args.frames_per_video,
            "max_videos": args.max_videos,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "top_frac": args.top_frac,
            "min_tail": args.min_tail,
        },
        "data": {
            "n_videos": len(records),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "feature_shape": list(raw_features.shape),
            "feature_meta": feature_meta,
            "video_ids": [record.video_id for record in records],
            "scores": [record.score for record in records],
        },
        "readout": {
            "full_test_rho": full_rho,
            "mean_pooled_test_rho": mean_rho,
        },
        "direction_energy": summarize_energy(energy),
        "band_results": band_results(
            features=features,
            scores=scores,
            test_idx=test_idx,
            direction=direction,
            spectrum=spectrum,
            energy=energy,
        ),
        "positional_embedding_spectrum": positional_embedding_spectrum(
            model_id=args.model_id,
            allow_downloads=args.allow_downloads,
        ),
    }
    report["interpretation"] = interpretation_from(report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[spectral-pos] wrote {args.out_json}", flush=True)
    print(f"[spectral-pos] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
