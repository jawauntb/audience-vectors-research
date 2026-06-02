"""Fold-safe TRIBE hidden-direction patch experiment.

This is a held-out version of the layerwise TRIBE direction patch lane. It
trains both the output memorability readout and each hidden high-minus-low
direction on a balanced train split, then applies the hidden intervention only
to held-out evaluation clips.

The script deliberately fails into a status report when the local Modal service
does not expose hidden-direction patching. That keeps the runbook runnable on a
fresh branch without importing unscoped service edits.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audience_vectors.services.tribe_service import TribeService, TribeValidationError

DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_FEATURE_DIR = Path("data/features/tribe")
DEFAULT_HIDDEN_DIR = Path("data/features/tribe_layerwise_encoder")
DEFAULT_OUTPUT_DIR = Path("data/features/tribe_foldsafe_direction_patch")
DEFAULT_JSON = Path("data/reports/tribe_foldsafe_direction_patch.json")
DEFAULT_MD = Path("data/reports/tribe_foldsafe_direction_patch.md")


@dataclass(frozen=True)
class BmdRecord:
    sample_id: str
    video_id: str
    score: float
    volume_path: str | None
    url: str | None


@dataclass(frozen=True)
class LayerTarget:
    label: str
    hook_module: str


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train: list[BmdRecord]
    eval: list[BmdRecord]


@dataclass(frozen=True)
class HiddenDirection:
    target: LayerTarget
    direction: np.ndarray
    payload: bytes
    summary: dict[str, Any]


def default_targets() -> list[LayerTarget]:
    return [
        *[
            LayerTarget(
                label=f"attn{layer:02d}_post_resid",
                hook_module=f"_model.encoder.layers.{layer}.2",
            )
            for layer in range(0, 16, 2)
        ],
        LayerTarget(label="final_encoder", hook_module="_model.encoder"),
    ]


def parse_targets(text: str | None) -> list[LayerTarget]:
    if not text:
        return default_targets()
    targets = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            label, hook = part.split("=", 1)
            targets.append(LayerTarget(label=label.strip(), hook_module=hook.strip()))
        else:
            label = (
                part.removeprefix("_model.")
                .replace("encoder.layers.", "layer")
                .replace(".", "_")
            )
            targets.append(LayerTarget(label=label, hook_module=part))
    return targets


def parse_alphas(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def alpha_label(alpha: float) -> str:
    text = f"{alpha:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def canonical_hidden(arr: np.ndarray) -> np.ndarray:
    hidden = np.asarray(arr, dtype=np.float32)
    while hidden.ndim > 2 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.ndim == 1:
        return hidden[:, None]
    if hidden.ndim == 2:
        return hidden
    return hidden.reshape(-1, hidden.shape[-1])


def frequency_energy(seq_by_dim: np.ndarray) -> dict[str, float]:
    spectrum = np.fft.fft(seq_by_dim, axis=0, norm="ortho")
    energy = (np.abs(spectrum) ** 2).sum(axis=1)
    total = float(energy.sum())
    if total <= 1e-12:
        return {"dc": 0.0, "low_nonzero": 0.0, "mid": 0.0, "high": 0.0}
    freqs = np.abs(np.fft.fftfreq(seq_by_dim.shape[0]))
    rel = freqs / (float(freqs.max()) or 1.0)
    return {
        "dc": float(energy[freqs == 0].sum() / total),
        "low_nonzero": float(energy[(freqs > 0) & (rel <= 0.25)].sum() / total),
        "mid": float(energy[(rel > 0.25) & (rel <= 0.50)].sum() / total),
        "high": float(energy[rel > 0.50].sum() / total),
    }


def source_candidates(record: BmdRecord, prefer_url: bool) -> list[str]:
    sources = [record.url] if prefer_url else [record.volume_path, record.url]
    return [source for source in sources if source]


def hidden_path(hidden_dir: Path, target: LayerTarget, record: BmdRecord) -> Path:
    return hidden_dir / "hidden" / target.label / f"{record.sample_id}.npz"


def patch_path(
    output_dir: Path,
    fold: int,
    target: LayerTarget,
    alpha: float,
    record: BmdRecord,
) -> Path:
    return (
        output_dir
        / f"fold_{fold:02d}"
        / f"direction_alpha_{alpha_label(alpha)}"
        / target.label
        / f"{record.sample_id}.npz"
    )


def direction_path(output_dir: Path, fold: int, target: LayerTarget) -> Path:
    return output_dir / f"fold_{fold:02d}" / "directions" / f"{target.label}.npz"


def load_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "frames" in payload.files:
        frames = np.asarray(payload["frames"], dtype=np.float32)
        return frames.mean(axis=0) if frames.ndim == 2 else frames
    if "embedding" in payload.files:
        return np.asarray(payload["embedding"], dtype=np.float32)
    raise KeyError(f"{path} has neither frames nor embedding")


def load_records(
    annotations: Path,
    feature_dir: Path,
) -> tuple[list[BmdRecord], np.ndarray, np.ndarray]:
    with annotations.open() as fh:
        annotations_json = json.load(fh)
    records: list[BmdRecord] = []
    features = []
    scores = []
    for video_id, row in annotations_json.items():
        if "memorability_score" not in row:
            continue
        sample_id = f"bmd_vid_idx{video_id}_seg_0000"
        feature_path = feature_dir / f"{sample_id}.npz"
        if not feature_path.exists():
            continue
        record = BmdRecord(
            sample_id=sample_id,
            video_id=str(video_id),
            score=float(row["memorability_score"]),
            volume_path=f"/bmd-videos/videos/vid_idx{video_id}.mp4",
            url=row.get("MiT_url") or row.get("url") or row.get("video_url"),
        )
        records.append(record)
        features.append(load_feature(feature_path))
        scores.append(record.score)
    if not records:
        raise FileNotFoundError(
            f"no scored TRIBE features found in {feature_dir} using {annotations}"
        )
    return (
        records,
        np.stack(features).astype(np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def make_fold_splits(
    records: list[BmdRecord],
    *,
    n_train_each: int,
    n_eval_each: int,
    folds: int,
    seed: int,
) -> list[FoldSplit]:
    order = np.argsort([record.score for record in records])
    low = [records[int(i)] for i in order[: len(order) // 2]]
    high = [records[int(i)] for i in order[len(order) // 2 :]]
    needed = n_train_each + n_eval_each
    if len(low) < needed or len(high) < needed:
        raise ValueError(
            "not enough scored clips for requested fold-safe split: "
            f"need {needed} per tail, found low={len(low)} high={len(high)}"
        )

    splits = []
    rng = np.random.default_rng(seed)
    for fold in range(1, folds + 1):
        low_perm = rng.permutation(len(low))
        high_perm = rng.permutation(len(high))
        train = [
            *[low[int(i)] for i in low_perm[:n_train_each]],
            *[high[int(i)] for i in high_perm[:n_train_each]],
        ]
        eval_records = [
            *[low[int(i)] for i in low_perm[n_train_each:needed]],
            *[high[int(i)] for i in high_perm[n_train_each:needed]],
        ]
        splits.append(FoldSplit(fold=fold, train=train, eval=eval_records))
    return splits


def hidden_cache_eligible_records(
    records: list[BmdRecord],
    *,
    targets: list[LayerTarget],
    hidden_dir: Path,
) -> list[BmdRecord]:
    return [
        record
        for record in records
        if all(hidden_path(hidden_dir, target, record).exists() for target in targets)
    ]


def hidden_target_counts(
    records: list[BmdRecord],
    *,
    targets: list[LayerTarget],
    hidden_dir: Path,
) -> dict[str, int]:
    return {
        target.label: sum(
            hidden_path(hidden_dir, target, record).exists() for record in records
        )
        for target in targets
    }


def balanced_hidden_cache_coverage(
    records: list[BmdRecord],
    *,
    targets: list[LayerTarget],
    hidden_dir: Path,
    n_each: int,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record.score)
    if n_each * 2 > len(ordered):
        raise ValueError(f"n_each={n_each} too large for n={len(ordered)}")
    low = ordered[:n_each]
    high = ordered[-n_each:]

    def has_all(record: BmdRecord) -> bool:
        return all(hidden_path(hidden_dir, target, record).exists() for target in targets)

    low_missing = [record.sample_id for record in low if not has_all(record)]
    high_missing = [record.sample_id for record in high if not has_all(record)]
    return {
        "n_each_tail_required": n_each,
        "low_ready": n_each - len(low_missing),
        "high_ready": n_each - len(high_missing),
        "low_missing": len(low_missing),
        "high_missing": len(high_missing),
        "total_ready": (n_each * 2) - len(low_missing) - len(high_missing),
        "total_missing": len(low_missing) + len(high_missing),
        "low_missing_sample_ids": low_missing,
        "high_missing_sample_ids": high_missing,
    }


def train_direction(
    features: np.ndarray, scores: np.ndarray, top_frac: float
) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * top_frac))
    low = features[order[:n_each]].mean(axis=0)
    high = features[order[-n_each:]].mean(axis=0)
    return unit(high - low)


def load_hidden_stack(
    *,
    records: list[BmdRecord],
    target: LayerTarget,
    hidden_dir: Path,
) -> np.ndarray:
    rows = []
    for record in records:
        path = hidden_path(hidden_dir, target, record)
        if not path.exists():
            raise FileNotFoundError(
                f"missing hidden cache {path}; run layerwise hidden capture first"
            )
        rows.append(canonical_hidden(np.load(path, allow_pickle=False)["hidden"]))
    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"{target.label} hidden shapes differ: {sorted(shapes)}")
    return np.stack(rows).astype(np.float32)


def train_hidden_direction(
    *,
    records: list[BmdRecord],
    target: LayerTarget,
    hidden_dir: Path,
    output_dir: Path,
    fold: int,
) -> HiddenDirection:
    hidden = load_hidden_stack(records=records, target=target, hidden_dir=hidden_dir)
    scores = np.asarray([record.score for record in records], dtype=np.float32)
    order = np.argsort(scores)
    n_each = len(records) // 2
    low = hidden[order[:n_each]]
    high = hidden[order[-n_each:]]
    direction = unit(
        high.reshape(high.shape[0], -1).mean(axis=0)
        - low.reshape(low.shape[0], -1).mean(axis=0)
    ).reshape(hidden.shape[1], hidden.shape[2])
    projection = hidden.reshape(hidden.shape[0], -1) @ direction.reshape(-1)
    energy = frequency_energy(direction)
    summary = {
        "hidden_shape": list(hidden.shape[1:]),
        "train_spearman_vs_memorability": spearman(projection, scores),
        "direction_frequency_energy": energy,
        "non_dc_energy": 1.0 - float(energy["dc"]),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, direction=direction.astype(np.float16))
    path = direction_path(output_dir, fold, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        direction=direction.astype(np.float16),
        target_label=np.array(target.label),
        hook_module=np.array(target.hook_module),
        hidden_shape=np.asarray(hidden.shape[1:], dtype=np.int32),
    )
    return HiddenDirection(
        target=target,
        direction=direction,
        payload=buffer.getvalue(),
        summary=summary,
    )


def result_frames(result: Any) -> tuple[np.ndarray, float]:
    return np.asarray(result["frames"], dtype=np.float32), float(
        result["duration_seconds"]
    )


async def patch_one(
    *,
    service: TribeService,
    record: BmdRecord,
    hidden_direction: HiddenDirection,
    alpha: float,
    output_dir: Path,
    fold: int,
    timeout: float,
    prefer_url: bool,
) -> dict[str, Any]:
    path = patch_path(output_dir, fold, hidden_direction.target, alpha, record)
    if path.exists():
        return {
            "ok": True,
            "cached": True,
            "sample_id": record.sample_id,
            "target": hidden_direction.target.label,
            "alpha": alpha,
        }
    errors: list[str] = []
    patch_method = getattr(service, "predict_video_hidden_direction_patch")
    for source in source_candidates(record, prefer_url):
        try:
            result = await asyncio.wait_for(
                patch_method(
                    source,
                    hook_module=hidden_direction.target.hook_module,
                    direction_npz=hidden_direction.payload,
                    patch_alpha=alpha,
                ),
                timeout=timeout,
            )
        except TribeValidationError as exc:
            errors.append(f"{source}: validation {exc}")
            continue
        except TimeoutError:
            errors.append(f"{source}: timeout")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        if result is None:
            errors.append(f"{source}: empty result")
            continue
        frames, duration = result_frames(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frames=frames,
            duration_seconds=np.array(duration, dtype=np.float32),
            sample_id=np.array(record.sample_id),
            memorability_score=np.array(record.score, dtype=np.float32),
            target_label=np.array(hidden_direction.target.label),
            hook_module=np.array(hidden_direction.target.hook_module),
            patch_alpha=np.array(alpha, dtype=np.float32),
            source=np.array(source),
        )
        return {
            "ok": True,
            "cached": False,
            "sample_id": record.sample_id,
            "target": hidden_direction.target.label,
            "alpha": alpha,
        }
    return {
        "ok": False,
        "sample_id": record.sample_id,
        "target": hidden_direction.target.label,
        "alpha": alpha,
        "errors": errors,
    }


async def run_patches(
    *,
    records: list[BmdRecord],
    directions: list[HiddenDirection],
    alphas: list[float],
    output_dir: Path,
    fold: int,
    concurrency: int,
    timeout: float,
    prefer_url: bool,
) -> list[dict[str, Any]]:
    service = TribeService()
    if not hasattr(service, "predict_video_hidden_direction_patch"):
        raise AttributeError(
            "TribeService lacks predict_video_hidden_direction_patch; "
            "deploy/import the TRIBE hidden-direction patch service lane first"
        )
    sem = asyncio.Semaphore(concurrency)

    async def guarded(
        hidden_direction: HiddenDirection,
        alpha: float,
        record: BmdRecord,
    ) -> dict[str, Any]:
        async with sem:
            print(
                f"[foldsafe-direction-patch] fold={fold} "
                f"{hidden_direction.target.label} alpha={alpha:+.3f} {record.sample_id}",
                flush=True,
            )
            return await patch_one(
                service=service,
                record=record,
                hidden_direction=hidden_direction,
                alpha=alpha,
                output_dir=output_dir,
                fold=fold,
                timeout=timeout,
                prefer_url=prefer_url,
            )

    return await asyncio.gather(
        *[
            guarded(hidden_direction, alpha, record)
            for hidden_direction in directions
            for alpha in alphas
            for record in records
        ]
    )


def load_patch_features(
    *,
    records: list[BmdRecord],
    target: LayerTarget,
    alpha: float,
    output_dir: Path,
    fold: int,
) -> np.ndarray:
    rows = []
    for record in records:
        frames = np.asarray(
            np.load(
                patch_path(output_dir, fold, target, alpha, record),
                allow_pickle=False,
            )["frames"],
            dtype=np.float32,
        )
        rows.append(frames.mean(axis=0))
    return np.stack(rows).astype(np.float32)


def summarize_patch(
    *,
    features: np.ndarray,
    baseline_features: np.ndarray,
    scores: np.ndarray,
    v_mem: np.ndarray,
    n_each: int,
) -> dict[str, Any]:
    baseline_projection = baseline_features @ v_mem
    patch_projection = features @ v_mem
    order = np.argsort(scores)
    low_idx = order[:n_each]
    high_idx = order[-n_each:]
    baseline_gap = float(
        baseline_projection[high_idx].mean() - baseline_projection[low_idx].mean()
    )
    patch_gap = float(
        patch_projection[high_idx].mean() - patch_projection[low_idx].mean()
    )
    delta = patch_projection - baseline_projection
    return {
        "eval_baseline_spearman_vs_memorability": spearman(
            baseline_projection,
            scores,
        ),
        "eval_patch_spearman_vs_memorability": spearman(patch_projection, scores),
        "eval_patch_spearman_vs_baseline_projection": spearman(
            patch_projection,
            baseline_projection,
        ),
        "eval_patch_pearson_vs_baseline_projection": float(
            np.corrcoef(patch_projection, baseline_projection)[0, 1]
        ),
        "eval_projection_delta_in_baseline_std": float(
            np.abs(delta).mean() / max(float(baseline_projection.std()), 1e-12)
        ),
        "eval_baseline_high_minus_low_gap": baseline_gap,
        "eval_patch_high_minus_low_gap": patch_gap,
        "eval_patch_gap_ratio_vs_baseline": patch_gap / baseline_gap
        if abs(baseline_gap) > 1e-12
        else None,
    }


def index_features(
    records: list[BmdRecord],
    all_records: list[BmdRecord],
    all_features: np.ndarray,
) -> np.ndarray:
    by_id = {record.sample_id: idx for idx, record in enumerate(all_records)}
    return np.stack(
        [all_features[by_id[record.sample_id]] for record in records]
    ).astype(np.float32)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# TRIBE Fold-Safe Hidden Direction Patch",
        "",
        "## Status",
        "",
        f"- Status: **{report['status']}**.",
    ]
    if report.get("blocker"):
        lines.append(f"- Blocker: {report['blocker']}")
    config = report["config"]
    summary = report.get("summary", {})
    balanced = summary.get("balanced_hidden_cache_coverage", {})
    lines += [
        "",
        "## Setup",
        "",
        f"- Scored TRIBE feature clips found: **{summary.get('n_scored_features', 'n/a')}**.",
        "- Clips with all requested layerwise hidden caches: "
        f"**{summary.get('n_hidden_cache_eligible', 'n/a')}**.",
        "- Balanced hidden-cache coverage: "
        f"**{balanced.get('total_ready', 'n/a')} / {config['n_train_each'] * 2 + config['n_eval_each'] * 2}** "
        f"({balanced.get('low_ready', 'n/a')} low + {balanced.get('high_ready', 'n/a')} high ready; "
        f"missing {balanced.get('low_missing', 'n/a')} low + {balanced.get('high_missing', 'n/a')} high).",
        f"- Train clips per fold: **{config['n_train_each'] * 2}** "
        f"({config['n_train_each']} low + {config['n_train_each']} high).",
        f"- Held-out eval clips per fold: **{config['n_eval_each'] * 2}** "
        f"({config['n_eval_each']} low + {config['n_eval_each']} high).",
        f"- Folds: **{config['folds']}**.",
        f"- Alphas: `{', '.join(str(alpha) for alpha in config['alphas'])}`.",
        "- Fold-safe rule: hidden direction, output readout, and reported patch "
        "metrics use disjoint train/eval clips within each fold.",
    ]
    if report.get("summary", {}).get("layers"):
        lines += [
            "",
            "## Layerwise Results",
            "",
            "| fold | target | alpha | train hidden rho | eval baseline rho | eval patch rho | eval gap ratio | |Δproj| / std |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in report["summary"]["layers"]:
            patch = row["patch"]
            gap_ratio = patch["eval_patch_gap_ratio_vs_baseline"]
            gap_text = "n/a" if gap_ratio is None else f"{float(gap_ratio):+.3f}"
            lines.append(
                f"| {row['fold']} | `{row['label']}` | {float(row['alpha']):+.3f} | "
                f"{row['hidden']['train_spearman_vs_memorability']:+.3f} | "
                f"{patch['eval_baseline_spearman_vs_memorability']:+.3f} | "
                f"{patch['eval_patch_spearman_vs_memorability']:+.3f} | "
                f"{gap_text} | "
                f"{patch['eval_projection_delta_in_baseline_std']:.3f} |"
            )
    lines += [
        "",
        "## Hidden Cache Expansion",
        "",
        "```bash",
        "uv run python scripts/tribe_layerwise_encoder_localization.py \\",
        "  --annotations data/raw/bold_moments/annotations.json \\",
        "  --feature-dir data/features/tribe \\",
        "  --output-dir data/features/tribe_layerwise_encoder \\",
        f"  --n-each {config['n_train_each'] + config['n_eval_each']} \\",
        "  --capture-only \\",
        "  --capture-concurrency 4 --timeout 300 \\",
        "  --out-json data/reports/tribe_layerwise_encoder_hidden_capture_104.json \\",
        "  --out-md data/reports/tribe_layerwise_encoder_hidden_capture_104.md",
        "```",
        "",
        "This expanded the hidden cache to the completed fold-safe requirement of "
        f"{config['n_train_each'] + config['n_eval_each']} low + "
        f"{config['n_train_each'] + config['n_eval_each']} high clips.",
        "",
        "## Fold-Safe Patch Rerun",
        "",
        "```bash",
        "uv run python scripts/tribe_foldsafe_direction_patch.py \\",
        "  --annotations data/raw/bold_moments/annotations.json \\",
        "  --feature-dir data/features/tribe \\",
        "  --hidden-dir data/features/tribe_layerwise_encoder \\",
        "  --n-train-each 40 --n-eval-each 12 --folds 5 \\",
        "  --alphas 1.0 --concurrency 6",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_status(
    *,
    args: argparse.Namespace,
    status: str,
    blocker: str | None,
    summary: dict[str, Any],
) -> None:
    report = {
        "status": status,
        "blocker": blocker,
        "config": {
            "annotations": str(args.annotations),
            "feature_dir": str(args.feature_dir),
            "hidden_dir": str(args.hidden_dir),
            "output_dir": str(args.output_dir),
            "targets": args.targets,
            "alphas": parse_alphas(args.alphas),
            "n_train_each": args.n_train_each,
            "n_eval_each": args.n_eval_each,
            "folds": args.folds,
            "top_frac": args.top_frac,
            "seed": args.seed,
            "concurrency": args.concurrency,
        },
        "summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[foldsafe-direction-patch] wrote {args.out_json}", flush=True)
    print(f"[foldsafe-direction-patch] wrote {args.out_md}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    targets = parse_targets(args.targets)
    alphas = parse_alphas(args.alphas)
    all_records, all_features, _all_scores = load_records(
        args.annotations, args.feature_dir
    )
    hidden_ready = hidden_cache_eligible_records(
        all_records,
        targets=targets,
        hidden_dir=args.hidden_dir,
    )
    needed_total = 2 * (args.n_train_each + args.n_eval_each)
    balanced_coverage = balanced_hidden_cache_coverage(
        all_records,
        targets=targets,
        hidden_dir=args.hidden_dir,
        n_each=args.n_train_each + args.n_eval_each,
    )
    target_counts = hidden_target_counts(
        all_records,
        targets=targets,
        hidden_dir=args.hidden_dir,
    )
    if len(hidden_ready) < needed_total:
        write_status(
            args=args,
            status="blocked",
            blocker=(
                "insufficient layerwise hidden cache for requested fold-safe split: "
                f"need at least {needed_total} clips with all requested targets, "
                f"found {len(hidden_ready)}. Balanced expansion is missing "
                f"{balanced_coverage['low_missing']} low-tail and "
                f"{balanced_coverage['high_missing']} high-tail clips."
            ),
            summary={
                "n_scored_features": len(all_records),
                "n_hidden_cache_eligible": len(hidden_ready),
                "needed_hidden_cache_clips": needed_total,
                "target_cache_counts": target_counts,
                "balanced_hidden_cache_coverage": balanced_coverage,
                "targets": [
                    {"label": target.label, "hook_module": target.hook_module}
                    for target in targets
                ],
                "layers": [],
            },
        )
        return
    splits = make_fold_splits(
        hidden_ready,
        n_train_each=args.n_train_each,
        n_eval_each=args.n_eval_each,
        folds=args.folds,
        seed=args.seed,
    )
    layers: list[dict[str, Any]] = []
    try:
        for split in splits:
            train_features = index_features(split.train, all_records, all_features)
            train_scores = np.asarray(
                [record.score for record in split.train], dtype=np.float32
            )
            eval_features = index_features(split.eval, all_records, all_features)
            eval_scores = np.asarray(
                [record.score for record in split.eval], dtype=np.float32
            )
            v_mem = train_direction(train_features, train_scores, args.top_frac)
            directions = [
                train_hidden_direction(
                    records=split.train,
                    target=target,
                    hidden_dir=args.hidden_dir,
                    output_dir=args.output_dir,
                    fold=split.fold,
                )
                for target in targets
            ]
            results = await run_patches(
                records=split.eval,
                directions=directions,
                alphas=alphas,
                output_dir=args.output_dir,
                fold=split.fold,
                concurrency=args.concurrency,
                timeout=args.timeout,
                prefer_url=args.use_urls,
            )
            failures = [row for row in results if not row.get("ok")]
            if failures:
                raise RuntimeError(f"{len(failures)} TRIBE patch calls failed")
            by_target = {direction.target.label: direction for direction in directions}
            for target in targets:
                hidden_direction = by_target[target.label]
                for alpha in alphas:
                    patch_features = load_patch_features(
                        records=split.eval,
                        target=target,
                        alpha=alpha,
                        output_dir=args.output_dir,
                        fold=split.fold,
                    )
                    layers.append(
                        {
                            "fold": split.fold,
                            "label": target.label,
                            "hook_module": target.hook_module,
                            "alpha": alpha,
                            "hidden": hidden_direction.summary,
                            "patch": summarize_patch(
                                features=patch_features,
                                baseline_features=eval_features,
                                scores=eval_scores,
                                v_mem=v_mem,
                                n_each=args.n_eval_each,
                            ),
                        }
                    )
    except Exception as exc:  # noqa: BLE001
        write_status(
            args=args,
            status="blocked",
            blocker=f"{type(exc).__name__}: {exc}",
            summary={
                "n_scored_features": len(all_records),
                "n_hidden_cache_eligible": len(hidden_ready),
                "needed_hidden_cache_clips": needed_total,
                "target_cache_counts": target_counts,
                "balanced_hidden_cache_coverage": balanced_coverage,
                "targets": [
                    {"label": target.label, "hook_module": target.hook_module}
                    for target in targets
                ],
                "layers": layers,
            },
        )
        return

    write_status(
        args=args,
        status="complete",
        blocker=None,
        summary={
            "n_scored_features": len(all_records),
            "n_hidden_cache_eligible": len(hidden_ready),
            "needed_hidden_cache_clips": needed_total,
            "target_cache_counts": target_counts,
            "balanced_hidden_cache_coverage": balanced_coverage,
            "targets": [
                {"label": target.label, "hook_module": target.hook_module}
                for target in targets
            ],
            "layers": layers,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--hidden-dir", type=Path, default=DEFAULT_HIDDEN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--n-train-each", type=int, default=40)
    parser.add_argument("--n-eval-each", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--use-urls",
        action="store_true",
        help="Use annotation URLs directly instead of Modal bmd-videos volume paths.",
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
