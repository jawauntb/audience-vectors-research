"""Internal TRIBE hidden/rotary position probe.

This extends the learned `time_pos_embed` patch by intervening at the TRIBE
encoder output, downstream of the rotary attention stack and upstream of the
fMRI prediction head.

The probe asks three reviewer-facing questions:

1. Can encoder hidden states linearly separate high vs low BMD memorability?
2. Is that hidden memorability direction mostly temporal/sequence DC or
   high-frequency positional structure?
3. Do output-space memorability projections collapse when we keep only the
   encoder-output sequence mean, or when we zero rotary frequencies?
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

from scripts.tribe_timepos_patch_probe import (
    BmdRecord,
    load_records,
    scale_label,
    select_balanced,
    spearman,
    train_v_mem,
    unit,
)

from audience_vectors.services.tribe_service import TribeService, TribeValidationError

DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_FEATURE_DIR = Path("data/features/tribe")
DEFAULT_OUTPUT_DIR = Path("data/features/tribe_hidden_position_patch")
DEFAULT_JSON = Path("data/reports/tribe_hidden_position_patch_probe.json")
DEFAULT_MD = Path("data/reports/tribe_hidden_position_patch_probe.md")


@dataclass(frozen=True)
class PatchCondition:
    name: str
    patch_mode: str
    patch_scale: float
    rotary_inv_freq_scale: float
    capture_hidden: bool = False


def condition_label(condition: PatchCondition) -> str:
    return condition.name.replace(".", "p").replace("-", "m")


def prediction_path(output_dir: Path, condition: PatchCondition, record: BmdRecord) -> Path:
    return output_dir / "predictions" / condition_label(condition) / f"{record.sample_id}.npz"


def hidden_path(output_dir: Path, hook_module: str, record: BmdRecord) -> Path:
    hook = hook_module.replace("_model.", "").replace(".", "_")
    return output_dir / "hidden" / hook / f"{record.sample_id}.npz"


def result_frames(result: Any) -> tuple[np.ndarray, float]:
    frames = np.asarray(result["frames"], dtype=np.float32)
    return frames, float(result["duration_seconds"])


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


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return values or [1.0]


def build_conditions(args: argparse.Namespace) -> list[PatchCondition]:
    conditions = [
        PatchCondition(
            name="baseline",
            patch_mode="none",
            patch_scale=1.0,
            rotary_inv_freq_scale=1.0,
            capture_hidden=True,
        )
    ]
    for scale in parse_float_list(args.hidden_non_dc_scales):
        if scale == 1.0:
            continue
        conditions.append(
            PatchCondition(
                name=f"hidden_non_dc_x{scale_label(scale)}",
                patch_mode="non_dc_scale",
                patch_scale=scale,
                rotary_inv_freq_scale=1.0,
            )
        )
    for scale in parse_float_list(args.rotary_scales):
        if scale == 1.0:
            continue
        conditions.append(
            PatchCondition(
                name=f"rotary_inv_freq_x{scale_label(scale)}",
                patch_mode="none",
                patch_scale=1.0,
                rotary_inv_freq_scale=scale,
            )
        )
    return conditions


def source_candidates(record: BmdRecord, prefer_url: bool) -> list[str]:
    sources = [record.url] if prefer_url else [record.volume_path, record.url]
    return [source for source in sources if source]


async def fetch_one(
    *,
    service: TribeService,
    record: BmdRecord,
    condition: PatchCondition,
    hook_module: str,
    output_dir: Path,
    timeout: float,
    prefer_url: bool,
) -> dict[str, Any]:
    out_path = prediction_path(output_dir, condition, record)
    h_path = hidden_path(output_dir, hook_module, record)
    if out_path.exists() and (not condition.capture_hidden or h_path.exists()):
        return {"ok": True, "cached": True, "sample_id": record.sample_id}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for source in source_candidates(record, prefer_url):
        try:
            result = await asyncio.wait_for(
                service.predict_video_hidden_patch(
                    source,
                    hook_module=hook_module,
                    patch_mode=condition.patch_mode,
                    patch_scale=condition.patch_scale,
                    rotary_inv_freq_scale=condition.rotary_inv_freq_scale,
                    capture_hidden=condition.capture_hidden,
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
        np.savez_compressed(
            out_path,
            frames=frames,
            duration_seconds=np.array(duration, dtype=np.float32),
            sample_id=np.array(record.sample_id),
            memorability_score=np.array(record.score, dtype=np.float32),
            condition=np.array(condition.name),
            hook_module=np.array(hook_module),
            patch_mode=np.array(condition.patch_mode),
            patch_scale=np.array(condition.patch_scale, dtype=np.float32),
            rotary_inv_freq_scale=np.array(
                condition.rotary_inv_freq_scale, dtype=np.float32
            ),
            source=np.array(source),
            hidden_shape=np.asarray(result.get("hidden_shape", []), dtype=np.int32),
            sequence_axis=np.array(result.get("sequence_axis", -1), dtype=np.int32),
        )
        if condition.capture_hidden and "hidden_npz" in result:
            h_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_payload = np.load(io.BytesIO(result["hidden_npz"]), allow_pickle=False)
            np.savez_compressed(
                h_path,
                hidden=np.asarray(hidden_payload["hidden"], dtype=np.float16),
                sample_id=np.array(record.sample_id),
                memorability_score=np.array(record.score, dtype=np.float32),
                hook_module=np.array(hook_module),
            )
        return {"ok": True, "cached": False, "sample_id": record.sample_id}
    return {"ok": False, "sample_id": record.sample_id, "errors": errors}


async def run_predictions(
    *,
    records: list[BmdRecord],
    conditions: list[PatchCondition],
    hook_module: str,
    output_dir: Path,
    concurrency: int,
    timeout: float,
    prefer_url: bool,
) -> list[dict[str, Any]]:
    service = TribeService()
    sem = asyncio.Semaphore(concurrency)

    async def guarded(record: BmdRecord, condition: PatchCondition) -> dict[str, Any]:
        async with sem:
            print(f"[hidden-pos] {condition.name} {record.sample_id}", flush=True)
            return await fetch_one(
                service=service,
                record=record,
                condition=condition,
                hook_module=hook_module,
                output_dir=output_dir,
                timeout=timeout,
                prefer_url=prefer_url,
            )

    tasks = [guarded(record, condition) for condition in conditions for record in records]
    return await asyncio.gather(*tasks)


def load_condition_features(
    records: list[BmdRecord],
    conditions: list[PatchCondition],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {condition.name: [] for condition in conditions}
    for condition in conditions:
        for record in records:
            path = prediction_path(output_dir, condition, record)
            frames = np.asarray(np.load(path, allow_pickle=False)["frames"], dtype=np.float32)
            rows[condition.name].append(frames.mean(axis=0))
    return {name: np.stack(values).astype(np.float32) for name, values in rows.items()}


def load_hidden(records: list[BmdRecord], hook_module: str, output_dir: Path) -> np.ndarray:
    rows = []
    for record in records:
        path = hidden_path(output_dir, hook_module, record)
        hidden = np.load(path, allow_pickle=False)["hidden"]
        rows.append(canonical_hidden(hidden))
    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"hidden shapes differ across clips: {sorted(shapes)}")
    return np.stack(rows).astype(np.float32)


def summarize_hidden(
    *,
    hidden: np.ndarray,
    scores: np.ndarray,
    n_each: int,
) -> dict[str, Any]:
    order = np.argsort(scores)
    low = hidden[order[:n_each]]
    high = hidden[order[-n_each:]]
    full_features = hidden.reshape(hidden.shape[0], -1)
    full_dir = unit(high.reshape(high.shape[0], -1).mean(axis=0) - low.reshape(low.shape[0], -1).mean(axis=0))
    full_projection = full_features @ full_dir
    mean_features = hidden.mean(axis=1)
    mean_dir = unit(mean_features[order[-n_each:]].mean(axis=0) - mean_features[order[:n_each]].mean(axis=0))
    mean_projection = mean_features @ mean_dir
    full_dir_by_seq = full_dir.reshape(hidden.shape[1], hidden.shape[2])
    return {
        "hidden_shape": list(hidden.shape[1:]),
        "full_sequence_spearman_vs_memorability": spearman(full_projection, scores),
        "mean_pooled_spearman_vs_memorability": spearman(mean_projection, scores),
        "full_direction_frequency_energy": frequency_energy(full_dir_by_seq),
        "mean_pool_direction_norm": float(np.linalg.norm(mean_dir)),
        "full_direction_norm": float(np.linalg.norm(full_dir)),
    }


def summarize_output_conditions(
    *,
    condition_features: dict[str, np.ndarray],
    conditions: list[PatchCondition],
    scores: np.ndarray,
    v_mem: np.ndarray,
    n_each: int,
) -> dict[str, Any]:
    baseline_projection = condition_features["baseline"] @ v_mem
    order = np.argsort(scores)
    low_idx = order[:n_each]
    high_idx = order[-n_each:]
    baseline_gap = float(
        baseline_projection[high_idx].mean() - baseline_projection[low_idx].mean()
    )
    rows: dict[str, Any] = {}
    for condition in conditions:
        projection = condition_features[condition.name] @ v_mem
        delta = projection - baseline_projection
        high_low_gap = float(projection[high_idx].mean() - projection[low_idx].mean())
        rows[condition.name] = {
            "patch_mode": condition.patch_mode,
            "patch_scale": condition.patch_scale,
            "rotary_inv_freq_scale": condition.rotary_inv_freq_scale,
            "spearman_vs_memorability": spearman(projection, scores),
            "spearman_vs_baseline_projection": spearman(
                projection, baseline_projection
            ),
            "pearson_vs_baseline_projection": float(
                np.corrcoef(projection, baseline_projection)[0, 1]
            ),
            "mean_abs_projection_delta_vs_baseline": float(np.abs(delta).mean()),
            "projection_delta_in_baseline_std": float(
                np.abs(delta).mean() / max(float(baseline_projection.std()), 1e-12)
            ),
            "high_minus_low_gap": high_low_gap,
            "high_minus_low_gap_ratio_vs_baseline": high_low_gap / baseline_gap
            if abs(baseline_gap) > 1e-12
            else None,
        }
    return rows


def interpretation(summary: dict[str, Any]) -> str:
    output = summary["output_conditions"]
    hidden_dc = summary["hidden"]["full_direction_frequency_energy"]["dc"]
    non_dc = output.get("hidden_non_dc_xp0")
    rotary_zero = output.get("rotary_inv_freq_xp0")
    pieces = []
    if non_dc is not None:
        pieces.append(
            "encoder non-DC removal leaves rho "
            f"{non_dc['spearman_vs_memorability']:+.3f} and gap ratio "
            f"{float(non_dc['high_minus_low_gap_ratio_vs_baseline']):+.3f}"
        )
    if rotary_zero is not None:
        pieces.append(
            "rotary-frequency zeroing leaves rho "
            f"{rotary_zero['spearman_vs_memorability']:+.3f} and gap ratio "
            f"{float(rotary_zero['high_minus_low_gap_ratio_vs_baseline']):+.3f}"
        )
    if hidden_dc >= 0.5:
        pieces.append(f"the hidden memorability direction is mostly sequence-DC ({hidden_dc:.3f})")
    else:
        pieces.append(f"the hidden memorability direction has substantial non-DC energy (DC {hidden_dc:.3f})")
    return "; ".join(pieces) + "."


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# TRIBE Hidden/Rotary Position Patch Probe",
        "",
        report["interpretation"],
        "",
        "## Setup",
        "",
        f"- Clips: **{summary['n_clips']}** balanced top/bottom BMD memorability clips.",
        f"- Hook module: `{summary['hook_module']}`.",
        "- Readout: output projection onto cached BMD/TRIBE memorability direction.",
        "",
        "## Hidden Direction",
        "",
        f"- Captured hidden shape per clip: `{summary['hidden']['hidden_shape']}`.",
        f"- Full hidden direction rho vs BMD: **{summary['hidden']['full_sequence_spearman_vs_memorability']:+.3f}**.",
        f"- Mean-pooled hidden direction rho vs BMD: **{summary['hidden']['mean_pooled_spearman_vs_memorability']:+.3f}**.",
        "",
        "| band | direction energy |",
        "|---|---:|",
    ]
    for band, value in summary["hidden"]["full_direction_frequency_energy"].items():
        lines.append(f"| {band} | {float(value):.3f} |")
    lines += [
        "",
        "## Output Patch Results",
        "",
        "| condition | rho vs BMD | rho vs baseline | mean |Δproj| / baseline std | high-low gap ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["output_conditions"].items():
        ratio = row["high_minus_low_gap_ratio_vs_baseline"]
        ratio_text = "n/a" if ratio is None else f"{float(ratio):+.3f}"
        lines.append(
            f"| `{name}` | {row['spearman_vs_memorability']:+.3f} | "
            f"{row['spearman_vs_baseline_projection']:+.3f} | "
            f"{row['projection_delta_in_baseline_std']:.3f} | {ratio_text} |"
        )
    lines += [
        "",
        "## Caveat",
        "",
        "This patches the encoder output and the rotary frequency buffer, but it does not yet patch every intermediate attention-layer hidden state independently.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


async def main_async(args: argparse.Namespace) -> None:
    all_records, all_features, all_scores = load_records(
        annotations=args.annotations,
        feature_dir=args.feature_dir,
    )
    v_mem = train_v_mem(all_features, all_scores, top_frac=args.top_frac)
    selected = select_balanced(all_records, n_each=args.n_each)
    scores = np.asarray([record.score for record in selected], dtype=np.float32)
    conditions = build_conditions(args)
    results = await run_predictions(
        records=selected,
        conditions=conditions,
        hook_module=args.hook_module,
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
    hidden = load_hidden(selected, args.hook_module, args.output_dir)
    condition_features = load_condition_features(selected, conditions, args.output_dir)
    summary = {
        "n_clips": len(selected),
        "n_each_tail": args.n_each,
        "hook_module": args.hook_module,
        "hidden": summarize_hidden(hidden=hidden, scores=scores, n_each=args.n_each),
        "output_conditions": summarize_output_conditions(
            condition_features=condition_features,
            conditions=conditions,
            scores=scores,
            v_mem=v_mem,
            n_each=args.n_each,
        ),
    }
    report: dict[str, Any] = {
        "config": {
            "n_each": args.n_each,
            "top_frac": args.top_frac,
            "hook_module": args.hook_module,
            "hidden_non_dc_scales": args.hidden_non_dc_scales,
            "rotary_scales": args.rotary_scales,
            "output_dir": str(args.output_dir),
        },
        "summary": summary,
    }
    report["interpretation"] = interpretation(summary)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[hidden-pos] wrote {args.out_json}", flush=True)
    print(f"[hidden-pos] wrote {args.out_md}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--hook-module", default="_model.encoder")
    parser.add_argument("--n-each", type=int, default=12)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--hidden-non-dc-scales", default="1.0,0.0")
    parser.add_argument("--rotary-scales", default="1.0,0.0")
    parser.add_argument("--concurrency", type=int, default=4)
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
