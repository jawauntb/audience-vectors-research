"""Layerwise TRIBE encoder localization for the memorability sequence signal.

Targets the post-attention residual modules for encoder layers 0, 2, ..., 14,
plus the full encoder output. For each target, the script:

1. Captures hidden states on a balanced BMD high/low memorability subset.
2. Trains a high-minus-low hidden direction and Fourier-decomposes it over the
   token sequence.
3. Reruns TRIBE while removing non-DC sequence variation at that target, then
   checks whether the output-space BMD/TRIBE memorability projection survives.
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

from scripts.tribe_hidden_position_patch_probe import (
    canonical_hidden,
    frequency_energy,
    result_frames,
    source_candidates,
)
from scripts.tribe_timepos_patch_probe import (
    BmdRecord,
    load_records,
    select_balanced,
    spearman,
    train_v_mem,
    unit,
)

from audience_vectors.services.tribe_service import TribeService, TribeValidationError

DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_FEATURE_DIR = Path("data/features/tribe")
DEFAULT_OUTPUT_DIR = Path("data/features/tribe_layerwise_encoder")
DEFAULT_JSON = Path("data/reports/tribe_layerwise_encoder_localization.json")
DEFAULT_MD = Path("data/reports/tribe_layerwise_encoder_localization.md")


@dataclass(frozen=True)
class LayerTarget:
    label: str
    hook_module: str


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


def hidden_path(output_dir: Path, target: LayerTarget, record: BmdRecord) -> Path:
    return output_dir / "hidden" / target.label / f"{record.sample_id}.npz"


def patch_path(output_dir: Path, target: LayerTarget, record: BmdRecord) -> Path:
    return output_dir / "patch_non_dc_x0" / target.label / f"{record.sample_id}.npz"


def baseline_path(output_dir: Path, record: BmdRecord) -> Path:
    return output_dir / "baseline" / f"{record.sample_id}.npz"


def has_all_hidden(
    *, output_dir: Path, targets: list[LayerTarget], record: BmdRecord
) -> bool:
    return all(hidden_path(output_dir, target, record).exists() for target in targets)


async def capture_one(
    *,
    service: TribeService,
    record: BmdRecord,
    targets: list[LayerTarget],
    output_dir: Path,
    timeout: float,
    prefer_url: bool,
) -> dict[str, Any]:
    if has_all_hidden(output_dir=output_dir, targets=targets, record=record):
        return {"ok": True, "cached": True, "sample_id": record.sample_id}
    errors: list[str] = []
    hook_modules = [target.hook_module for target in targets]
    for source in source_candidates(record, prefer_url):
        try:
            result = await asyncio.wait_for(
                service.capture_video_hiddens(source, hook_modules),
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
        base_path = baseline_path(output_dir, record)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            base_path,
            frames=frames,
            duration_seconds=np.array(duration, dtype=np.float32),
            sample_id=np.array(record.sample_id),
            memorability_score=np.array(record.score, dtype=np.float32),
            source=np.array(source),
        )
        captures = result["captures"]
        for target in targets:
            capture = captures[target.hook_module]
            payload = np.load(io.BytesIO(capture["hidden_npz"]), allow_pickle=False)
            path = hidden_path(output_dir, target, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                hidden=np.asarray(payload["hidden"], dtype=np.float16),
                sample_id=np.array(record.sample_id),
                memorability_score=np.array(record.score, dtype=np.float32),
                hook_module=np.array(target.hook_module),
                target_label=np.array(target.label),
                hidden_shape=np.asarray(capture.get("hidden_shape", []), dtype=np.int32),
                sequence_axis=np.array(capture.get("sequence_axis", -1), dtype=np.int32),
            )
        return {"ok": True, "cached": False, "sample_id": record.sample_id}
    return {"ok": False, "sample_id": record.sample_id, "errors": errors}


async def patch_one(
    *,
    service: TribeService,
    record: BmdRecord,
    target: LayerTarget,
    output_dir: Path,
    timeout: float,
    prefer_url: bool,
) -> dict[str, Any]:
    path = patch_path(output_dir, target, record)
    if path.exists():
        return {"ok": True, "cached": True, "sample_id": record.sample_id}
    errors: list[str] = []
    for source in source_candidates(record, prefer_url):
        try:
            result = await asyncio.wait_for(
                service.predict_video_hidden_patch(
                    source,
                    hook_module=target.hook_module,
                    patch_mode="non_dc_scale",
                    patch_scale=0.0,
                    rotary_inv_freq_scale=1.0,
                    capture_hidden=False,
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
            hook_module=np.array(target.hook_module),
            target_label=np.array(target.label),
            patch_mode=np.array("non_dc_scale"),
            patch_scale=np.array(0.0, dtype=np.float32),
            source=np.array(source),
        )
        return {"ok": True, "cached": False, "sample_id": record.sample_id}
    return {
        "ok": False,
        "sample_id": record.sample_id,
        "target": target.label,
        "errors": errors,
    }


async def run_captures(
    *,
    records: list[BmdRecord],
    targets: list[LayerTarget],
    output_dir: Path,
    concurrency: int,
    timeout: float,
    prefer_url: bool,
) -> list[dict[str, Any]]:
    service = TribeService()
    sem = asyncio.Semaphore(concurrency)

    async def guarded(record: BmdRecord) -> dict[str, Any]:
        async with sem:
            print(f"[layerwise] capture {record.sample_id}", flush=True)
            return await capture_one(
                service=service,
                record=record,
                targets=targets,
                output_dir=output_dir,
                timeout=timeout,
                prefer_url=prefer_url,
            )

    return await asyncio.gather(*[guarded(record) for record in records])


async def run_patches(
    *,
    records: list[BmdRecord],
    targets: list[LayerTarget],
    output_dir: Path,
    concurrency: int,
    timeout: float,
    prefer_url: bool,
) -> list[dict[str, Any]]:
    service = TribeService()
    sem = asyncio.Semaphore(concurrency)

    async def guarded(target: LayerTarget, record: BmdRecord) -> dict[str, Any]:
        async with sem:
            print(f"[layerwise] patch {target.label} {record.sample_id}", flush=True)
            return await patch_one(
                service=service,
                record=record,
                target=target,
                output_dir=output_dir,
                timeout=timeout,
                prefer_url=prefer_url,
            )

    return await asyncio.gather(
        *[guarded(target, record) for target in targets for record in records]
    )


def load_hidden(records: list[BmdRecord], target: LayerTarget, output_dir: Path) -> np.ndarray:
    rows = []
    for record in records:
        rows.append(
            canonical_hidden(
                np.load(hidden_path(output_dir, target, record), allow_pickle=False)[
                    "hidden"
                ]
            )
        )
    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"{target.label} hidden shapes differ: {sorted(shapes)}")
    return np.stack(rows).astype(np.float32)


def load_patch_features(
    records: list[BmdRecord], target: LayerTarget, output_dir: Path
) -> np.ndarray:
    rows = []
    for record in records:
        frames = np.asarray(
            np.load(patch_path(output_dir, target, record), allow_pickle=False)["frames"],
            dtype=np.float32,
        )
        rows.append(frames.mean(axis=0))
    return np.stack(rows).astype(np.float32)


def summarize_hidden(
    *, hidden: np.ndarray, scores: np.ndarray, n_each: int
) -> dict[str, Any]:
    order = np.argsort(scores)
    low = hidden[order[:n_each]]
    high = hidden[order[-n_each:]]
    full_features = hidden.reshape(hidden.shape[0], -1)
    full_dir = unit(
        high.reshape(high.shape[0], -1).mean(axis=0)
        - low.reshape(low.shape[0], -1).mean(axis=0)
    )
    full_projection = full_features @ full_dir
    mean_features = hidden.mean(axis=1)
    mean_dir = unit(
        mean_features[order[-n_each:]].mean(axis=0)
        - mean_features[order[:n_each]].mean(axis=0)
    )
    mean_projection = mean_features @ mean_dir
    energy = frequency_energy(full_dir.reshape(hidden.shape[1], hidden.shape[2]))
    return {
        "hidden_shape": list(hidden.shape[1:]),
        "full_sequence_spearman_vs_memorability": spearman(full_projection, scores),
        "mean_pooled_spearman_vs_memorability": spearman(mean_projection, scores),
        "direction_frequency_energy": energy,
        "non_dc_energy": 1.0 - float(energy["dc"]),
    }


def summarize_patch(
    *,
    patch_features: np.ndarray,
    baseline_features: np.ndarray,
    scores: np.ndarray,
    v_mem: np.ndarray,
    n_each: int,
) -> dict[str, Any]:
    baseline_projection = baseline_features @ v_mem
    patch_projection = patch_features @ v_mem
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
        "spearman_vs_memorability": spearman(patch_projection, scores),
        "spearman_vs_baseline_projection": spearman(
            patch_projection, baseline_projection
        ),
        "pearson_vs_baseline_projection": float(
            np.corrcoef(patch_projection, baseline_projection)[0, 1]
        ),
        "projection_delta_in_baseline_std": float(
            np.abs(delta).mean() / max(float(baseline_projection.std()), 1e-12)
        ),
        "high_minus_low_gap": patch_gap,
        "high_minus_low_gap_ratio_vs_baseline": patch_gap / baseline_gap
        if abs(baseline_gap) > 1e-12
        else None,
    }


def emergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strongest = min(
        rows,
        key=lambda row: float(row["patch"]["high_minus_low_gap_ratio_vs_baseline"]),
    )
    below_half = [
        row
        for row in rows
        if float(row["patch"]["high_minus_low_gap_ratio_vs_baseline"]) < 0.5
    ]
    return {
        "strongest_non_dc_dependency": {
            "label": strongest["label"],
            "hook_module": strongest["hook_module"],
            "gap_ratio": strongest["patch"]["high_minus_low_gap_ratio_vs_baseline"],
            "patch_rho": strongest["patch"]["spearman_vs_memorability"],
        },
        "first_gap_ratio_below_0p5": None
        if not below_half
        else {
            "label": below_half[0]["label"],
            "hook_module": below_half[0]["hook_module"],
            "gap_ratio": below_half[0]["patch"][
                "high_minus_low_gap_ratio_vs_baseline"
            ],
            "patch_rho": below_half[0]["patch"]["spearman_vs_memorability"],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    emergence = summary["emergence"]
    first = emergence["first_gap_ratio_below_0p5"]
    strongest = emergence["strongest_non_dc_dependency"]
    first_text = (
        "none"
        if first is None
        else f"`{first['label']}` (gap ratio {float(first['gap_ratio']):+.3f})"
    )
    lines = [
        "# TRIBE Layerwise Encoder Localization",
        "",
        "## Takeaway",
        "",
        f"- Strongest non-DC dependency: `{strongest['label']}` "
        f"(patch rho {float(strongest['patch_rho']):+.3f}, "
        f"gap ratio {float(strongest['gap_ratio']):+.3f}).",
        f"- First target with gap ratio < 0.5: {first_text}.",
        "",
        "## Setup",
        "",
        f"- Clips: **{summary['n_clips']}** balanced BMD clips "
        f"({summary['n_each_tail']} low + {summary['n_each_tail']} high).",
        "- Hidden patch: remove non-DC sequence variation at the target module.",
        "- Output readout: cached BMD/TRIBE memorability direction.",
        "",
        "## Layerwise Results",
        "",
        "| target | hidden shape | hidden rho | mean-pooled hidden rho | DC energy | low nonzero | high | patch rho | rho vs baseline | gap ratio | |Δproj| / std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["layers"]:
        hidden = row["hidden"]
        patch = row["patch"]
        energy = hidden["direction_frequency_energy"]
        lines.append(
            f"| `{row['label']}` | `{hidden['hidden_shape']}` | "
            f"{hidden['full_sequence_spearman_vs_memorability']:+.3f} | "
            f"{hidden['mean_pooled_spearman_vs_memorability']:+.3f} | "
            f"{energy['dc']:.3f} | {energy['low_nonzero']:.3f} | "
            f"{energy['high']:.3f} | "
            f"{patch['spearman_vs_memorability']:+.3f} | "
            f"{patch['spearman_vs_baseline_projection']:+.3f} | "
            f"{float(patch['high_minus_low_gap_ratio_vs_baseline']):+.3f} | "
            f"{patch['projection_delta_in_baseline_std']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Earlier layers can carry a hidden high-vs-low direction without being causally",
        "load-bearing for the final output readout. The load-bearing point is where",
        "non-DC sequence removal first damages the downstream memorability projection.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_capture_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# TRIBE Layerwise Encoder Hidden Capture",
        "",
        "## Status",
        "",
        f"- Status: **{report['status']}**.",
        f"- Clips requested: **{summary['n_clips']}** balanced BMD clips "
        f"({summary['n_each_tail']} low + {summary['n_each_tail']} high).",
        f"- Cached clips reused: **{summary['n_cached']}**.",
        f"- Newly captured clips: **{summary['n_captured']}**.",
        "",
        "## Target Cache Counts",
        "",
        "| target | cached selected clips |",
        "|---|---:|",
    ]
    for target in summary["targets"]:
        lines.append(
            f"| `{target['label']}` | {summary['target_cache_counts'][target['label']]} |"
        )
    lines += [
        "",
        "## Next Step",
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


def write_capture_status(
    *,
    args: argparse.Namespace,
    selected: list[BmdRecord],
    targets: list[LayerTarget],
    capture_results: list[dict[str, Any]],
) -> None:
    target_cache_counts = {
        target.label: sum(
            hidden_path(args.output_dir, target, record).exists()
            for record in selected
        )
        for target in targets
    }
    report = {
        "status": "complete",
        "config": {
            "annotations": str(args.annotations),
            "feature_dir": str(args.feature_dir),
            "output_dir": str(args.output_dir),
            "targets": args.targets,
            "n_each": args.n_each,
            "capture_concurrency": args.capture_concurrency,
            "timeout": args.timeout,
            "use_urls": args.use_urls,
            "capture_only": args.capture_only,
        },
        "summary": {
            "n_clips": len(selected),
            "n_each_tail": args.n_each,
            "n_cached": sum(1 for row in capture_results if row.get("cached")),
            "n_captured": sum(
                1
                for row in capture_results
                if row.get("ok") and not row.get("cached")
            ),
            "targets": [
                {"label": target.label, "hook_module": target.hook_module}
                for target in targets
            ],
            "target_cache_counts": target_cache_counts,
            "selected_sample_ids": [record.sample_id for record in selected],
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_capture_markdown(report, args.out_md)
    print(f"[layerwise] wrote {args.out_json}", flush=True)
    print(f"[layerwise] wrote {args.out_md}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    targets = parse_targets(args.targets)
    all_records, all_features, all_scores = load_records(
        annotations=args.annotations,
        feature_dir=args.feature_dir,
    )
    v_mem = train_v_mem(all_features, all_scores, top_frac=args.top_frac)
    selected = select_balanced(all_records, n_each=args.n_each)
    selected_index = {record.sample_id: idx for idx, record in enumerate(all_records)}
    baseline_features = np.stack(
        [all_features[selected_index[record.sample_id]] for record in selected]
    ).astype(np.float32)
    scores = np.asarray([record.score for record in selected], dtype=np.float32)

    capture_results = await run_captures(
        records=selected,
        targets=targets,
        output_dir=args.output_dir,
        concurrency=args.capture_concurrency,
        timeout=args.timeout,
        prefer_url=args.use_urls,
    )
    capture_failures = [row for row in capture_results if not row.get("ok")]
    if capture_failures:
        failure_path = args.out_json.with_suffix(".failures.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(capture_failures, indent=2) + "\n")
        raise RuntimeError(
            f"{len(capture_failures)} TRIBE hidden captures failed; wrote {failure_path}"
        )
    if args.capture_only:
        write_capture_status(
            args=args,
            selected=selected,
            targets=targets,
            capture_results=capture_results,
        )
        return

    patch_results = await run_patches(
        records=selected,
        targets=targets,
        output_dir=args.output_dir,
        concurrency=args.patch_concurrency,
        timeout=args.timeout,
        prefer_url=args.use_urls,
    )
    failures = [
        row for row in [*capture_results, *patch_results] if not row.get("ok")
    ]
    if failures:
        failure_path = args.out_json.with_suffix(".failures.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} TRIBE calls failed; wrote {failure_path}")

    layer_rows = []
    for target in targets:
        hidden = load_hidden(selected, target, args.output_dir)
        patch_features = load_patch_features(selected, target, args.output_dir)
        layer_rows.append(
            {
                "label": target.label,
                "hook_module": target.hook_module,
                "hidden": summarize_hidden(
                    hidden=hidden, scores=scores, n_each=args.n_each
                ),
                "patch": summarize_patch(
                    patch_features=patch_features,
                    baseline_features=baseline_features,
                    scores=scores,
                    v_mem=v_mem,
                    n_each=args.n_each,
                ),
            }
        )

    summary = {
        "n_clips": len(selected),
        "n_each_tail": args.n_each,
        "targets": [
            {"label": target.label, "hook_module": target.hook_module}
            for target in targets
        ],
        "layers": layer_rows,
        "emergence": emergence_summary(layer_rows),
    }
    report = {
        "config": {
            "n_each": args.n_each,
            "top_frac": args.top_frac,
            "output_dir": str(args.output_dir),
            "capture_concurrency": args.capture_concurrency,
            "patch_concurrency": args.patch_concurrency,
            "targets": args.targets,
        },
        "summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[layerwise] wrote {args.out_json}", flush=True)
    print(f"[layerwise] wrote {args.out_md}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--n-each", type=int, default=12)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--capture-concurrency", type=int, default=4)
    parser.add_argument("--patch-concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture layerwise hidden caches and skip non-DC patch calls.",
    )
    parser.add_argument(
        "--use-urls",
        action="store_true",
        help="Use MiT URLs directly instead of the Modal bmd-videos volume path.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
