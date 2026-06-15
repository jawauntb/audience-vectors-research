"""Screen confirmatory Seedance candidate videos before proxy selection.

This is a launch-readiness and selection-readiness artifact. It checks the
generated MP4 bytes, runs the lightweight sampled-frame visual gate, computes
simple visual descriptors, writes contact sheets, and records which candidates
are eligible for the later TRIBE/V-JEPA/CLIP selector pass.

It does not choose final selector-top videos and does not claim human
memorability. Final selection remains blocked until the preregistered proxy
scores are attached for these exact MP4 bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from audience_vectors.visual_artifact_gate import (
    frame_contrast,
    frame_sharpness,
    luminance,
    sample_video_frames,
    summarize_frames,
)

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_MANIFEST = (
    EXPERIMENT_DIR / "seedance_candidate_generation_manifest_improved_v1_20260615.json"
)
DEFAULT_OUT_JSON = (
    EXPERIMENT_DIR / "seedance_candidate_visual_screening_improved_v1_20260615.json"
)
DEFAULT_OUT_MD = (
    EXPERIMENT_DIR / "seedance_candidate_visual_screening_improved_v1_20260615.md"
)
DEFAULT_SHEET_DIR = (
    EXPERIMENT_DIR / "seedance_candidate_visual_sheets_improved_v1_20260615"
)
TEXT_RISK_FAMILIES = {
    "city_street",
    "old_car",
    "street_food_grill",
}
REVIEW_ONLY_FLAGS = {
    "low_sampled_sharpness",
    "manual_text_review_recommended",
    "tail_sharpness_collapse",
}

RGBFrame = NDArray[np.uint8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root-prefix", type=Path, default=Path.cwd())
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--overview-cols", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def frame_count(path: Path) -> int:
    return sum(1 for _ in iio.imiter(path))


def colorfulness(frame: RGBFrame) -> float:
    rgb = frame.astype(np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(std_root + 0.3 * mean_root)


def motion_magnitude(frames: list[RGBFrame]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = []
    for left, right in zip(frames, frames[1:]):
        diffs.append(float(np.mean(np.abs(luminance(right) - luminance(left)))))
    return float(mean(diffs))


def rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def load_fonts() -> tuple[Any, Any]:
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def fit_image(frame: RGBFrame, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, "white")
    out.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return out


def sheet_tile(row: dict[str, Any], frames: list[RGBFrame], font: Any) -> Image.Image:
    frame_size = (144, 81)
    text_height = 72
    tile = Image.new("RGB", (frame_size[0] * len(frames), frame_size[1] + text_height), "white")
    draw = ImageDraw.Draw(tile)
    for index, frame in enumerate(frames):
        tile.paste(fit_image(frame, frame_size), (index * frame_size[0], text_height))
    draw.rectangle((0, 0, tile.width - 1, tile.height - 1), outline=(190, 190, 190), width=1)
    title = f"{row['family_id']} v{row['candidate_index']:02d}"
    metrics = (
        f"q={row['visual_quality_proxy']:+.2f} "
        f"mot={row['descriptors']['motion_magnitude']:.3f} "
        f"col={row['descriptors']['colorfulness']:.1f}"
    )
    flags = ",".join(row["screening_flags"]) or "auto-pass"
    draw.text((8, 5), title[:84], font=font, fill=(20, 20, 20))
    draw.text((8, 25), metrics[:84], font=font, fill=(70, 70, 70))
    draw.text((8, 45), flags[:84], font=font, fill=(120, 50, 50))
    return tile


def write_contact_sheet(
    rows: list[dict[str, Any]],
    frames_by_job: dict[str, list[RGBFrame]],
    *,
    out_path: Path,
    title: str,
    cols: int,
) -> None:
    rows_with_frames = [row for row in rows if row["job_id"] in frames_by_job]
    if not rows_with_frames:
        return
    title_font, small_font = load_fonts()
    frame_count_per_tile = len(next(iter(frames_by_job.values())))
    tile_size = (144 * frame_count_per_tile, 153)
    top_height = 44
    n_rows = (len(rows_with_frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size[0], top_height + n_rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=title_font, fill=(10, 10, 10))
    for index, row in enumerate(rows_with_frames):
        tile = sheet_tile(row, frames_by_job[row["job_id"]], small_font)
        sheet.paste(tile, ((index % cols) * tile_size[0], top_height + (index // cols) * tile_size[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def technical_flags(
    *,
    metadata: dict[str, Any] | None,
    duration_seconds: float | None,
    file_size_bytes: int | None,
) -> list[str]:
    flags = []
    if metadata is None:
        return ["ffprobe_failed"]
    streams = metadata.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        flags.append("missing_video_stream")
        return flags
    stream = video_streams[0]
    if int(stream.get("width") or 0) != 1280 or int(stream.get("height") or 0) != 720:
        flags.append("unexpected_resolution")
    if duration_seconds is None or not (4.5 <= duration_seconds <= 5.6):
        flags.append("unexpected_duration")
    if file_size_bytes is None or file_size_bytes <= 0:
        flags.append("empty_file")
    return flags


def descriptor_flags(descriptors: dict[str, float]) -> list[str]:
    flags = []
    if descriptors["mean_brightness"] < 0.08:
        flags.append("very_dark")
    if descriptors["mean_brightness"] > 0.92:
        flags.append("very_bright")
    if descriptors["mean_contrast"] < 0.035:
        flags.append("low_contrast")
    if descriptors["mean_sharpness"] < 0.00035:
        flags.append("low_sampled_sharpness")
    if descriptors["motion_magnitude"] < 0.003:
        flags.append("low_sampled_motion")
    return flags


def hard_screening_flags(flags: list[str]) -> list[str]:
    """Return flags that block proxy scoring before manual review."""
    hard = []
    for flag in flags:
        if flag in REVIEW_ONLY_FLAGS:
            continue
        if flag.startswith("ffprobe_error:") or flag.startswith("frame_sampling_error:"):
            hard.append(flag)
            continue
        hard.append(flag)
    return hard


def visual_quality_proxy(row: dict[str, Any]) -> float:
    descriptors = row["descriptors"]
    gate = row["visual_gate"] or {}
    score = (
        4.0 * descriptors["mean_contrast"]
        + 90.0 * descriptors["mean_sharpness"]
        + 0.015 * descriptors["colorfulness"]
        + 2.0 * descriptors["motion_magnitude"]
        - 1.5 * float(gate.get("collapse_score") or 0.0)
    )
    if "manual_text_review_recommended" in row["screening_flags"]:
        score -= 0.05
    return float(score)


def screen_job(
    job: dict[str, Any],
    *,
    video_root_prefix: Path,
    samples: int,
) -> tuple[dict[str, Any], list[RGBFrame] | None]:
    output_path = Path(str(job["output_video"]["path"]))
    video_path = video_root_prefix / output_path
    row: dict[str, Any] = {
        "job_id": job["job_id"],
        "family_id": job["family_id"],
        "candidate_index": job["candidate_index"],
        "prior_role": job["prior_role"],
        "prompt_set": job.get("prompt_set"),
        "variant_direction": job["variant_direction"],
        "prompt": job["prompt"],
        "manifest_output_path": str(output_path),
        "source_absolute_path": str(video_path),
        "exists": video_path.exists(),
        "sha256": sha256_file(video_path),
        "metadata": None,
        "duration_seconds": None,
        "file_size_bytes": None,
        "frame_count": None,
        "frame_sample_ok": False,
        "visual_gate": None,
        "descriptors": None,
        "visual_quality_proxy": None,
        "screening_flags": [],
        "hard_screening_flags": [],
        "automated_visual_gate_passed": False,
        "manual_review_required": False,
    }
    if not video_path.exists():
        row["screening_flags"].append("missing_file")
        row["hard_screening_flags"] = hard_screening_flags(row["screening_flags"])
        row["manual_review_required"] = True
        return row, None

    try:
        metadata = ffprobe(video_path)
        row["metadata"] = metadata
        row["duration_seconds"] = float(metadata["format"]["duration"])
        row["file_size_bytes"] = int(metadata["format"]["size"])
        row["screening_flags"].extend(
            technical_flags(
                metadata=metadata,
                duration_seconds=row["duration_seconds"],
                file_size_bytes=row["file_size_bytes"],
            )
        )
    except Exception as exc:  # noqa: BLE001
        row["screening_flags"].append(f"ffprobe_error:{type(exc).__name__}")

    try:
        frames = sample_video_frames(video_path, samples=samples)
        row["frame_sample_ok"] = True
        row["frame_count"] = frame_count(video_path)
        gate = summarize_frames(frames)
        row["visual_gate"] = gate
        row["screening_flags"].extend(gate["artifact_flags"])
        descriptors = {
            "mean_brightness": float(mean(float(np.mean(luminance(frame))) for frame in frames)),
            "mean_contrast": float(mean(frame_contrast(frame) for frame in frames)),
            "mean_sharpness": float(mean(frame_sharpness(frame) for frame in frames)),
            "colorfulness": float(mean(colorfulness(frame) for frame in frames)),
            "motion_magnitude": motion_magnitude(frames),
        }
        row["descriptors"] = {
            key: rounded(value) for key, value in descriptors.items()
        }
        row["screening_flags"].extend(descriptor_flags(descriptors))
        if row["family_id"] in TEXT_RISK_FAMILIES:
            row["screening_flags"].append("manual_text_review_recommended")
        row["hard_screening_flags"] = hard_screening_flags(row["screening_flags"])
        row["manual_review_required"] = bool(row["screening_flags"])
        row["automated_visual_gate_passed"] = not row["hard_screening_flags"]
        row["visual_quality_proxy"] = rounded(visual_quality_proxy(row), 6)
        return row, frames
    except Exception as exc:  # noqa: BLE001
        row["screening_flags"].append(f"frame_sampling_error:{type(exc).__name__}")
        row["hard_screening_flags"] = hard_screening_flags(row["screening_flags"])
        row["manual_review_required"] = True
        return row, None


def rank_rows_for_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            not row["automated_visual_gate_passed"],
            -float(row["visual_quality_proxy"] or -999.0),
            int(row["candidate_index"]),
        ),
    )


def family_selection_readiness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family_id"]].append(row)
    out = []
    for family_id, family_rows in sorted(by_family.items()):
        eligible = [
            row
            for row in family_rows
            if row["exists"] and row["frame_sample_ok"] and row["automated_visual_gate_passed"]
        ]
        ranked = rank_rows_for_review(eligible)
        out.append(
            {
                "family_id": family_id,
                "n_candidates": len(family_rows),
                "n_automated_visual_eligible": len(eligible),
                "n_manual_review_required": sum(
                    1 for row in family_rows if row["manual_review_required"]
                ),
                "proxy_selection_status": (
                    "blocked_pending_tribe_vjepa_clip_scores"
                    if len(eligible) >= 2
                    else "blocked_insufficient_visual_candidates"
                ),
                "visual_review_order": [
                    {
                        "job_id": row["job_id"],
                        "candidate_index": row["candidate_index"],
                        "visual_quality_proxy": row["visual_quality_proxy"],
                        "screening_flags": row["screening_flags"],
                    }
                    for row in ranked
                ],
            }
        )
    return out


def build_report(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    video_root_prefix: Path,
    sheet_dir: Path,
    samples: int,
    overview_cols: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    frames_by_job: dict[str, list[RGBFrame]] = {}
    for job in manifest["jobs"]:
        row, frames = screen_job(job, video_root_prefix=video_root_prefix, samples=samples)
        rows.append(row)
        if frames is not None:
            frames_by_job[row["job_id"]] = frames

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family_id"]].append(row)

    contact_sheets = []
    for family_id, family_rows in sorted(by_family.items()):
        family_rows.sort(key=lambda row: int(row["candidate_index"]))
        out_path = sheet_dir / f"{family_id}_sampled_frames.jpg"
        write_contact_sheet(
            family_rows,
            frames_by_job,
            out_path=out_path,
            title=f"{family_id} improved-v1 candidates",
            cols=2,
        )
        contact_sheets.append(
            {"family_id": family_id, "n_candidates": len(family_rows), "path": str(out_path)}
        )

    overview_rows = [
        sorted(group, key=lambda row: int(row["candidate_index"]))[0]
        for _, group in sorted(by_family.items())
    ]
    overview_path = sheet_dir / "family_v00_overview.jpg"
    write_contact_sheet(
        overview_rows,
        frames_by_job,
        out_path=overview_path,
        title="Improved-v1 family overview: v00 sampled frames",
        cols=overview_cols,
    )

    flags = Counter(flag for row in rows for flag in row["screening_flags"])
    hard_flags = Counter(flag for row in rows for flag in row["hard_screening_flags"])
    family_counts = Counter(str(row["family_id"]) for row in rows)
    automated_failures = [
        row
        for row in rows
        if not row["automated_visual_gate_passed"]
    ]
    manual_review = [row for row in rows if row["manual_review_required"]]
    selection_readiness = family_selection_readiness(rows)
    if automated_failures:
        status = "visual_gate_has_hard_failures"
    elif manual_review:
        status = "visual_gate_passed_manual_review_needed_proxy_selection_blocked"
    else:
        status = "visual_gate_passed_proxy_selection_blocked"
    summary = {
        "n_candidates": len(rows),
        "hard_visual_failures": len(automated_failures),
        "manual_review_required": len(manual_review),
        "n_families": len(family_counts),
        "n_families_with_two_or_more_visual_candidates": sum(
            1
            for family in selection_readiness
            if family["n_automated_visual_eligible"] >= 2
        ),
    }
    return {
        "schema_version": "confirmatory_seedance_candidate_visual_screening.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "summary": summary,
        "source_manifest": str(manifest_path),
        "manifest_prompt_set": manifest.get("prompt_set"),
        "video_root_prefix": str(video_root_prefix),
        "samples_per_video": samples,
        "n_candidates": summary["n_candidates"],
        "counts": {
            "by_family": dict(sorted(family_counts.items())),
            "hard_visual_failures": summary["hard_visual_failures"],
            "manual_review_required": summary["manual_review_required"],
            "screening_flags": dict(sorted(flags.items())),
            "hard_screening_flags": dict(sorted(hard_flags.items())),
        },
        "contact_sheets": [
            {"family_id": "overview", "n_candidates": len(overview_rows), "path": str(overview_path)},
            *contact_sheets,
        ],
        "selection_readiness": selection_readiness,
        "claim_boundary": [
            "This is visual/technical screening, not human memorability evidence.",
            "Do not freeze selector_top or quality_matched_control until TRIBE/BMD, exact V-JEPA, and CLIP/proxy scores are attached.",
            "low_sampled_sharpness, tail_sharpness_collapse, and manual_text_review_recommended are review cues, not automated proxy-scoring blockers.",
            "Preserve all generated candidates and rejected/withheld reasons.",
        ],
        "next_actions": [
            "Review contact sheets and MP4s flagged for manual text/content review.",
            "Attach TRIBE/BMD, exact V-JEPA, CLIP, and optional saliency scores to the 96 exact MP4s.",
            "Then select one selector_top and one quality_matched_control per family from visually eligible candidates.",
            "Generate lures only after selected old videos are frozen.",
        ],
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Improved-v1 Seedance Candidate Visual Screening",
        "",
        f"Date: `{report['created_at_utc']}`",
        f"Status: `{report['status']}`",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: are the 96 improved-v1 Seedance candidate old videos visually and technically eligible for proxy scoring and later selector/control assignment?",
        "",
        "Current regime:",
        "",
        "- Artifact types: Seedance MP4 candidates, prompt manifest rows, sampled-frame gates, contact sheets, low-level descriptors, later proxy-score tables.",
        "- Operations: local MP4 byte/hash/metadata checks, sampled-frame artifact gate, visual descriptor extraction, manual review queueing.",
        "- Gates/verifiers: all 12 families have 8 candidates; MP4s are playable 1280x720 around 5 seconds; sampled frames avoid collapse/low contrast; text/signage risk is queued for manual review.",
        "- Known limitation: this gate cannot read OCR reliably and cannot choose memory selector winners without TRIBE/V-JEPA/CLIP scores.",
        "",
        "Action class: production search inside the confirmatory recognition-study regime.",
        "",
        "## Summary",
        "",
        f"- Candidates screened: `{report['n_candidates']}`",
        f"- Samples per video: `{report['samples_per_video']}`",
        f"- Hard visual failures: `{report['counts']['hard_visual_failures']}`",
        f"- Manual review required: `{report['counts']['manual_review_required']}`",
        f"- Source manifest: `{report['source_manifest']}`",
        f"- Video root prefix: `{report['video_root_prefix']}`",
        "",
        "## Family Counts",
        "",
        "| family | candidates |",
        "|---|---:|",
    ]
    for family_id, count in report["counts"]["by_family"].items():
        lines.append(f"| `{family_id}` | {count} |")

    lines.extend(["", "## Screening Flags", "", "| flag | count |", "|---|---:|"])
    if report["counts"]["screening_flags"]:
        for flag, count in report["counts"]["screening_flags"].items():
            lines.append(f"| `{flag}` | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Hard Blockers", "", "| flag | count |", "|---|---:|"])
    if report["counts"]["hard_screening_flags"]:
        for flag, count in report["counts"]["hard_screening_flags"].items():
            lines.append(f"| `{flag}` | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Contact Sheets", "", "| family | candidates | sheet |", "|---|---:|---|"])
    for sheet in report["contact_sheets"]:
        lines.append(
            f"| `{sheet['family_id']}` | {sheet['n_candidates']} | `{sheet['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Selection Readiness",
            "",
            "| family | visually eligible | manual review | status | visual review head |",
            "|---|---:|---:|---|---|",
        ]
    )
    for family in report["selection_readiness"]:
        head = ", ".join(
            item["job_id"] for item in family["visual_review_order"][:3]
        )
        lines.append(
            "| `{family_id}` | {eligible} | {manual} | `{status}` | {head} |".format(
                family_id=family["family_id"],
                eligible=family["n_automated_visual_eligible"],
                manual=family["n_manual_review_required"],
                status=family["proxy_selection_status"],
                head=head,
            )
        )

    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    report = build_report(
        manifest=manifest,
        manifest_path=args.manifest,
        video_root_prefix=args.video_root_prefix,
        sheet_dir=args.sheet_dir,
        samples=args.samples,
        overview_cols=args.overview_cols,
    )
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_candidates": report["n_candidates"],
                "hard_visual_failures": report["counts"]["hard_visual_failures"],
                "manual_review_required": report["counts"][
                    "manual_review_required"
                ],
                "output_json": str(args.out_json),
                "output_md": str(args.out_md),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
