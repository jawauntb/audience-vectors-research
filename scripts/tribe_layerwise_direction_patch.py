"""Patch the learned hidden memorability direction at each TRIBE encoder layer.

This is the sharper follow-up to the non-DC sequence ablation. Instead of
collapsing all sequence variation, it trains a high-minus-low memorability
direction in each captured hidden space and removes only that one direction
during TRIBE inference.
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
from scripts.tribe_layerwise_encoder_localization import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LAYERWISE_HIDDEN_DIR,
)
from scripts.tribe_layerwise_encoder_localization import (
    LayerTarget,
    default_targets,
    hidden_path,
    parse_targets,
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
DEFAULT_OUTPUT_DIR = Path("data/features/tribe_layerwise_direction_patch")
DEFAULT_JSON = Path("data/reports/tribe_layerwise_direction_patch.json")
DEFAULT_MD = Path("data/reports/tribe_layerwise_direction_patch.md")


@dataclass(frozen=True)
class HiddenDirection:
    target: LayerTarget
    direction: np.ndarray
    hidden_summary: dict[str, Any]
    payload: bytes


def alpha_label(alpha: float) -> str:
    text = f"{alpha:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def patch_path(
    output_dir: Path,
    target: LayerTarget,
    alpha: float,
    record: BmdRecord,
) -> Path:
    return (
        output_dir
        / f"direction_alpha_{alpha_label(alpha)}"
        / target.label
        / f"{record.sample_id}.npz"
    )


def direction_path(output_dir: Path, target: LayerTarget) -> Path:
    return output_dir / "directions" / f"{target.label}.npz"


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
                f"missing hidden cache {path}; run tribe_layerwise_encoder_localization.py first"
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
        "spearman_vs_memorability": spearman(projection, scores),
        "direction_frequency_energy": energy,
        "non_dc_energy": 1.0 - float(energy["dc"]),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, direction=direction.astype(np.float16))
    payload = buffer.getvalue()
    path = direction_path(output_dir, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        direction=direction.astype(np.float16),
        target_label=np.array(target.label),
        hook_module=np.array(target.hook_module),
        hidden_shape=np.asarray(hidden.shape[1:], dtype=np.int32),
        spearman_vs_memorability=np.array(summary["spearman_vs_memorability"], dtype=np.float32),
    )
    return HiddenDirection(
        target=target,
        direction=direction,
        hidden_summary=summary,
        payload=payload,
    )


async def patch_one(
    *,
    service: TribeService,
    record: BmdRecord,
    hidden_direction: HiddenDirection,
    alpha: float,
    output_dir: Path,
    timeout: float,
    prefer_url: bool,
) -> dict[str, Any]:
    path = patch_path(output_dir, hidden_direction.target, alpha, record)
    if path.exists():
        return {
            "ok": True,
            "cached": True,
            "sample_id": record.sample_id,
            "target": hidden_direction.target.label,
            "alpha": alpha,
        }
    errors: list[str] = []
    for source in source_candidates(record, prefer_url):
        try:
            result = await asyncio.wait_for(
                service.predict_video_hidden_direction_patch(
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
            hidden_shape=np.asarray(result.get("hidden_shape", []), dtype=np.int32),
            sequence_axis=np.array(result.get("sequence_axis", -1), dtype=np.int32),
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
    concurrency: int,
    timeout: float,
    prefer_url: bool,
) -> list[dict[str, Any]]:
    service = TribeService()
    sem = asyncio.Semaphore(concurrency)

    async def guarded(
        hidden_direction: HiddenDirection,
        alpha: float,
        record: BmdRecord,
    ) -> dict[str, Any]:
        async with sem:
            print(
                f"[direction-patch] {hidden_direction.target.label} "
                f"alpha={alpha:+.3f} {record.sample_id}",
                flush=True,
            )
            return await patch_one(
                service=service,
                record=record,
                hidden_direction=hidden_direction,
                alpha=alpha,
                output_dir=output_dir,
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
) -> np.ndarray:
    rows = []
    for record in records:
        frames = np.asarray(
            np.load(patch_path(output_dir, target, alpha, record), allow_pickle=False)[
                "frames"
            ],
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


def emergence_summary(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    alpha_rows = [row for row in rows if float(row["alpha"]) == float(alpha)]
    strongest = min(
        alpha_rows,
        key=lambda row: float(row["patch"]["high_minus_low_gap_ratio_vs_baseline"]),
    )
    below_half = [
        row
        for row in alpha_rows
        if float(row["patch"]["high_minus_low_gap_ratio_vs_baseline"]) < 0.5
    ]
    return {
        "alpha": alpha,
        "strongest_direction_dependency": {
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
    emergence = summary["emergence"][0]
    first = emergence["first_gap_ratio_below_0p5"]
    strongest = emergence["strongest_direction_dependency"]
    first_text = (
        "none"
        if first is None
        else f"`{first['label']}` (gap ratio {float(first['gap_ratio']):+.3f})"
    )
    lines = [
        "# TRIBE Layerwise Hidden Direction Patch",
        "",
        "## Takeaway",
        "",
        f"- Strongest learned-direction dependency: `{strongest['label']}` "
        f"(patch rho {float(strongest['patch_rho']):+.3f}, "
        f"gap ratio {float(strongest['gap_ratio']):+.3f}).",
        f"- First target with gap ratio < 0.5: {first_text}.",
        "",
        "## Setup",
        "",
        f"- Clips: **{summary['n_clips']}** balanced BMD clips "
        f"({summary['n_each_tail']} low + {summary['n_each_tail']} high).",
        "- Patch: remove only the high-minus-low hidden memorability direction.",
        "- Output readout: cached BMD/TRIBE memorability direction.",
        "",
        "## Layerwise Results",
        "",
        "| target | hidden rho | DC energy | non-DC energy | patch alpha | patch rho | rho vs baseline | gap ratio | |Δproj| / std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["layers"]:
        hidden = row["hidden"]
        energy = hidden["direction_frequency_energy"]
        patch = row["patch"]
        lines.append(
            f"| `{row['label']}` | {hidden['spearman_vs_memorability']:+.3f} | "
            f"{energy['dc']:.3f} | {hidden['non_dc_energy']:.3f} | "
            f"{float(row['alpha']):+.3f} | "
            f"{patch['spearman_vs_memorability']:+.3f} | "
            f"{patch['spearman_vs_baseline_projection']:+.3f} | "
            f"{float(patch['high_minus_low_gap_ratio_vs_baseline']):+.3f} | "
            f"{patch['projection_delta_in_baseline_std']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is the sharper causal test: unlike non-DC removal, it patches only one",
        "learned high-minus-low hidden direction. If the gap survives, the earlier",
        "non-DC result mostly reflected broad sequence dependence. If it collapses,",
        "the learned memorability direction itself is causally load-bearing.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_alphas(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


async def main_async(args: argparse.Namespace) -> None:
    targets = parse_targets(args.targets) if args.targets else default_targets()
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
    directions = [
        train_hidden_direction(
            records=selected,
            target=target,
            hidden_dir=args.hidden_dir,
            output_dir=args.output_dir,
        )
        for target in targets
    ]
    alphas = parse_alphas(args.alphas)
    results = await run_patches(
        records=selected,
        directions=directions,
        alphas=alphas,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        timeout=args.timeout,
        prefer_url=args.use_urls,
    )
    failures = [row for row in results if not row.get("ok")]
    if failures:
        failure_path = args.out_json.with_suffix(".failures.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} TRIBE calls failed; wrote {failure_path}")

    layer_rows = []
    by_target = {direction.target.label: direction for direction in directions}
    for target in targets:
        hidden_direction = by_target[target.label]
        for alpha in alphas:
            patch_features = load_patch_features(
                records=selected,
                target=target,
                alpha=alpha,
                output_dir=args.output_dir,
            )
            layer_rows.append(
                {
                    "label": target.label,
                    "hook_module": target.hook_module,
                    "alpha": alpha,
                    "hidden": hidden_direction.hidden_summary,
                    "patch": summarize_patch(
                        features=patch_features,
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
        "alphas": alphas,
        "layers": layer_rows,
        "emergence": [emergence_summary(layer_rows, alpha) for alpha in alphas],
    }
    report = {
        "config": {
            "n_each": args.n_each,
            "top_frac": args.top_frac,
            "hidden_dir": str(args.hidden_dir),
            "output_dir": str(args.output_dir),
            "alphas": alphas,
            "targets": args.targets,
            "concurrency": args.concurrency,
        },
        "summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[direction-patch] wrote {args.out_json}", flush=True)
    print(f"[direction-patch] wrote {args.out_md}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--hidden-dir", type=Path, default=DEFAULT_LAYERWISE_HIDDEN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--n-each", type=int, default=12)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--use-urls",
        action="store_true",
        help="Use MiT URLs directly instead of the Modal bmd-videos volume path.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
