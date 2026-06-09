"""Generate and screen recognition-memory SVD MP4s from the production manifest.

The production manifest already freezes seed images, alpha/guidance recipes,
noise seeds, and output paths. This script materializes those MP4s with the
same Modal SVDGenerator used by the BO replay path, then writes an agent
sampled-frame screening report and contact sheets for video review.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audience_vectors.visual_artifact_gate import (
    ArtifactThresholds,
    sample_video_frames,
    summarize_frames,
)

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_MANIFEST = (
    ARTIFACT_DIR / "content_pocket_recognition_stimulus_production_manifest_20260608.json"
)
DEFAULT_GENERATION_JSON = (
    ARTIFACT_DIR / "content_pocket_recognition_video_generation_result_20260608.json"
)
DEFAULT_GENERATION_MD = (
    ARTIFACT_DIR / "content_pocket_recognition_video_generation_result_20260608.md"
)
DEFAULT_SCREENING_JSON = (
    ARTIFACT_DIR / "content_pocket_recognition_video_screening_20260608.json"
)
DEFAULT_SCREENING_MD = (
    ARTIFACT_DIR / "content_pocket_recognition_video_screening_20260608.md"
)
DEFAULT_SHEET_DIR = (
    ARTIFACT_DIR / "content_pocket_recognition_video_screening_sheets_20260608"
)
DEFAULT_LAUNCH_ASSETS = (
    ARTIFACT_DIR / "content_pocket_recognition_launch_assets_20260608.json"
)


@dataclass(frozen=True)
class GenerationConfig:
    """Stable Video Diffusion settings for one recognition production pass."""

    app_name: str
    num_frames: int
    num_inference_steps: int
    motion_bucket_id: int
    noise_aug_strength: float
    fps: int
    timeout_seconds: int


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


def file_inventory(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "bytes": path.stat().st_size if exists else None,
    }


def image_bytes(path: Path) -> bytes:
    """Load a seed image as 1024x576 PNG bytes for Modal SVD."""
    image = Image.open(path).convert("RGB").resize((1024, 576))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_steering_vector(path: Path, *, key: str) -> list[float]:
    """Load and normalize a steering vector from a `.pt`, `.npz`, or `.npy` file."""
    if path.suffix == ".pt":
        import torch  # noqa: PLC0415

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if key not in payload:
            available = ", ".join(sorted(str(item) for item in payload))
            raise ValueError(f"{path} missing {key!r}; available: {available}")
        vector = np.asarray(payload[key], dtype=np.float32)
    elif path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if key not in payload:
            available = ", ".join(payload.files)
            raise ValueError(f"{path} missing {key!r}; available: {available}")
        vector = np.asarray(payload[key], dtype=np.float32)
    elif path.suffix == ".npy":
        vector = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raise ValueError(f"unsupported steering artifact suffix: {path.suffix}")

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("near-zero steering vector")
    return (vector / norm).astype(np.float32).tolist()


def selected_jobs(
    manifest: dict[str, Any],
    *,
    roles: set[str] | None,
    job_ids: set[str] | None,
    limit: int | None,
    only_missing: bool,
) -> list[dict[str, Any]]:
    jobs = list(manifest["generation_jobs"])
    if roles:
        jobs = [job for job in jobs if str(job["role"]) in roles]
    if job_ids:
        jobs = [job for job in jobs if str(job["job_id"]) in job_ids]
    if only_missing:
        jobs = [
            job
            for job in jobs
            if not Path(str(job["output_video"]["path"])).exists()
        ]
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def job_row(job: dict[str, Any]) -> dict[str, Any]:
    seed_path = Path(str(job["seed_image"]["path"]))
    output_path = Path(str(job["output_video"]["path"]))
    return {
        "job_id": job["job_id"],
        "role": job["role"],
        "matched_id": job.get("matched_id"),
        "source_pocket": job.get("source_pocket"),
        "old_target_id": job.get("old_target_id"),
        "prompt": job["prompt"],
        "alpha": job["alpha"],
        "guidance": job["guidance"],
        "noise_seed": job["noise_seed"],
        "seed_image": file_inventory(seed_path),
        "output_video": file_inventory(output_path),
        "generation_seconds": None,
        "status": None,
        "error": None,
    }


def generate_videos_on_modal(
    *,
    jobs: list[dict[str, Any]],
    steering_vector: list[float],
    config: GenerationConfig,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Spawn Modal SVD jobs and write MP4 bytes to the manifest output paths."""
    import modal  # type: ignore[import-not-found]  # noqa: PLC0415

    generator_cls = modal.Cls.from_name(config.app_name, "SVDGenerator")
    generator = generator_cls()
    pending = []
    rows: list[dict[str, Any]] = []
    for job in jobs:
        row = job_row(job)
        out_path = Path(str(job["output_video"]["path"]))
        seed_path = Path(str(job["seed_image"]["path"]))
        if out_path.exists() and not overwrite:
            row["status"] = "already_present"
            rows.append(row)
            continue
        if not seed_path.exists():
            row["status"] = "failed"
            row["error"] = f"missing seed image: {seed_path}"
            rows.append(row)
            continue
        started = time.monotonic()
        call = generator.generate.spawn(
            image_bytes(seed_path),
            steering_vector=steering_vector,
            alpha=float(job["alpha"]),
            guidance_scale=float(job["guidance"]),
            num_frames=config.num_frames,
            num_inference_steps=config.num_inference_steps,
            motion_bucket_id=config.motion_bucket_id,
            noise_aug_strength=config.noise_aug_strength,
            fps=config.fps,
            seed=int(job["noise_seed"]),
            output_label=str(job["job_id"]),
            persist_output=False,
        )
        pending.append((job, row, out_path, started, call))

    for job, row, out_path, started, call in pending:
        try:
            video = call.get(timeout=config.timeout_seconds)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_bytes(video)
            tmp_path.replace(out_path)
            row["status"] = "generated"
            row["generation_seconds"] = time.monotonic() - started
            row["output_video"] = file_inventory(out_path)
        except Exception as exc:  # noqa: BLE001 - preserve generation failures.
            row["status"] = "failed"
            row["error"] = repr(exc)
        rows.append(row)
    return rows


def frame_count(path: Path) -> int:
    return sum(1 for _ in iio.imiter(path))


def load_fonts() -> tuple[Any, Any]:
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def fit_image(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, "white")
    out.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return out


def sheet_tile(row: dict[str, Any], frames: list[np.ndarray], small_font: Any) -> Image.Image:
    frame_size = (160, 90)
    text_height = 76
    tile = Image.new("RGB", (frame_size[0] * 3, frame_size[1] + text_height), "white")
    draw = ImageDraw.Draw(tile)
    for index, frame in enumerate(frames):
        tile.paste(fit_image(frame, frame_size), (frame_size[0] * index, text_height))
    draw.rectangle((0, 0, tile.width - 1, tile.height - 1), outline=(190, 190, 190), width=1)
    title = f"{row['source_pocket'] or 'filler'} / {row['role']}"
    params = f"a={float(row['alpha']):+.2f} g={float(row['guidance']):.2f} seed={row['noise_seed']}"
    draw.text((8, 5), title[:62], font=small_font, fill=(20, 20, 20))
    draw.text((8, 25), str(row["job_id"])[:62], font=small_font, fill=(70, 70, 70))
    draw.text((8, 45), params[:62], font=small_font, fill=(90, 90, 90))
    return tile


def slug(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def write_contact_sheet(
    rows: list[dict[str, Any]],
    frames_by_job: dict[str, list[np.ndarray]],
    *,
    out_path: Path,
    title: str,
    cols: int = 2,
) -> dict[str, Any]:
    if not rows:
        return {"path": str(out_path), "exists": False, "items": 0}
    title_font, small_font = load_fonts()
    tile_size = (480, 166)
    top_height = 42
    n_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size[0], top_height + n_rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=title_font, fill=(10, 10, 10))
    for index, row in enumerate(rows):
        tile = sheet_tile(row, frames_by_job[row["job_id"]], small_font)
        sheet.paste(tile, ((index % cols) * tile_size[0], top_height + (index // cols) * tile_size[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return {
        "path": str(out_path),
        "exists": True,
        "items": len(rows),
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
    }


def screen_video_row(
    row: dict[str, Any],
    *,
    samples: int,
    thresholds: ArtifactThresholds,
) -> tuple[dict[str, Any], list[np.ndarray] | None]:
    out_path = Path(str(row["output_video"]["path"]))
    screened = {
        **row,
        "exists": out_path.exists(),
        "actual_sha256": sha256_file(out_path),
        "frame_count": None,
        "frame_sample_ok": False,
        "visual_gate": None,
        "screening_flags": [],
    }
    if not out_path.exists():
        screened["screening_flags"].append("missing_file")
        return screened, None
    try:
        frames = sample_video_frames(out_path, samples=samples)
        screened["frame_sample_ok"] = True
        screened["frame_count"] = frame_count(out_path)
        gate = summarize_frames(frames, thresholds=thresholds)
        screened["visual_gate"] = gate
        screened["screening_flags"].extend(gate["artifact_flags"])
    except Exception as exc:  # noqa: BLE001 - preserve screening failure reason.
        screened["screening_flags"].append(f"frame_sampling_error:{type(exc).__name__}")
        screened["frame_sampling_error"] = str(exc)
        return screened, None
    return screened, frames


def launch_assets_ready(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    payload = load_json(path)
    counts = payload.get("counts", {})
    return bool(
        payload.get("status") == "hosted_launch_assets_ready"
        and counts.get("missing_videos") == 0
    )


def screening_launch_blockers(*, launch_ready: bool) -> list[str]:
    blockers = [
        "This is an agent sampled-frame pre-screen, not final IRB/faculty sign-off.",
    ]
    if launch_ready:
        blockers.append(
            "Final Prolific project configuration, completion codes, and response "
            "endpoint are not recorded."
        )
    else:
        blockers.extend(
            [
                "Stable HTTPS hosted video URLs are still required before launch.",
                "Two-session Prolific wiring and response collection remain open.",
            ]
        )
    blockers.append("Human recognition-memory validation has not run.")
    return blockers


def build_screening_report(
    *,
    generation_report: dict[str, Any],
    sheet_dir: Path,
    samples: int,
    thresholds: ArtifactThresholds,
    agent_review_note: str,
    launch_assets_result: Path | None = None,
) -> tuple[dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    frames_by_job: dict[str, list[np.ndarray]] = {}
    for row in generation_report["rows"]:
        screened, frames = screen_video_row(row, samples=samples, thresholds=thresholds)
        rows.append(screened)
        if frames is not None:
            frames_by_job[str(screened["job_id"])] = frames

    rows_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["job_id"]) in frames_by_job:
            rows_by_role[str(row["role"])].append(row)

    contact_sheets: list[dict[str, Any]] = []
    for role, group in sorted(rows_by_role.items()):
        group.sort(key=lambda item: str(item["job_id"]))
        sheet = write_contact_sheet(
            group,
            frames_by_job,
            out_path=sheet_dir / f"{slug(role)}.jpg",
            title=f"{role} sampled frames",
        )
        contact_sheets.append({"role": role, **sheet})

    failures = [row for row in rows if row["screening_flags"]]
    visual_failures = [
        row
        for row in rows
        if row.get("visual_gate") is None
        or row["visual_gate"].get("passes_visual_gate") is not True
    ]
    status = (
        "agent_video_screen_passed_for_hosting"
        if not failures
        else "agent_video_screen_requires_review"
    )
    ready_launch_assets = launch_assets_ready(launch_assets_result)
    report = {
        "schema_version": "content_pocket_recognition_video_screening.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "accepted_for_hosting": not failures,
        "source_generation_result": generation_report["result_path"],
        "source_production_manifest": generation_report["source_production_manifest"],
        "samples_per_video": samples,
        "thresholds": thresholds.__dict__,
        "counts": {
            "videos_screened": len(rows),
            "screening_failures": len(failures),
            "visual_gate_failures": len(visual_failures),
            "contact_sheets": len(contact_sheets),
        },
        "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
        "contact_sheets": contact_sheets,
        "agent_contact_sheet_review": agent_review_note,
        "rows": rows,
        "launch_assets_ready": ready_launch_assets,
        "launch_blockers": screening_launch_blockers(
            launch_ready=ready_launch_assets
        ),
    }
    return report, render_screening_markdown(report)


def generation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["status"] for row in rows)
    return {
        "requested": len(rows),
        "generated": int(counts.get("generated", 0)),
        "already_present": int(counts.get("already_present", 0)),
        "failed": int(counts.get("failed", 0)),
        "present_after_run": sum(
            1 for row in rows if row["output_video"]["exists"]
        ),
    }


def render_generation_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Content-Pocket Recognition Video Generation Result",
        "",
        f"Date: {report['created_at_utc']}",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: can the recognition-memory production manifest's screened",
        "seed images be converted into SVD MP4s for old-vs-lure validation?",
        "",
        "Current regime:",
        "",
        "- Artifact types: screened seed PNGs, SVD generation jobs, generated",
        "  MP4s, video hashes, sampled-frame visual screening records.",
        "- Operations: Modal SVD-XT generation from frozen seed images and",
        "  fixed alpha/guidance/noise settings.",
        "- Gates/verifiers: every requested MP4 present, video-level visual",
        "  screening before hosting, and human recognition data before any",
        "  memorability claim.",
        "- Known limitation: this production result is not human-memory evidence.",
        "",
        "Action class: production search inside the accepted recognition-memory",
        "validation regime.",
        "",
        "## Counts",
        "",
        f"- Requested: {counts['requested']}",
        f"- Generated: {counts['generated']}",
        f"- Already present: {counts['already_present']}",
        f"- Failed: {counts['failed']}",
        f"- Present after run: {counts['present_after_run']}",
        "",
        "## Claim Boundary",
        "",
        "- Generated recognition MP4s are launch-prep artifacts, not human",
        "  memorability evidence.",
        "- Failed or visually rejected videos must be retained with reasons.",
        "- The recognition-memory claim remains blocked until the two-session",
        "  human gate clears.",
        "",
        "## Next Action",
        "",
        "Use the paired video screening result and contact sheets to decide which",
        "MP4s can move to hosting and final human/IRB-facing review.",
        "",
    ]
    return "\n".join(lines)


def render_screening_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    status = (
        "No automated video screening failures were found."
        if counts["screening_failures"] == 0
        else f"{counts['screening_failures']} video screening failures require review."
    )
    lines = [
        "# Content-Pocket Recognition Video Screening",
        "",
        f"Date: {report['created_at_utc']}",
        "",
        "## Status",
        "",
        "Agent sampled-frame screening only. This artifact checks generated",
        "recognition MP4 availability and sampled-frame visual stability. It does",
        "not launch a study and does not validate human memorability.",
        "",
        f"Result: {status}",
        "",
        "## Summary",
        "",
        f"- Videos screened: {counts['videos_screened']}",
        f"- Sampled frames per video: {report['samples_per_video']}",
        f"- Contact sheets: {counts['contact_sheets']}",
        f"- Automated screening failures: {counts['screening_failures']}",
        f"- Visual-gate failures: {counts['visual_gate_failures']}",
        f"- Accepted for hosting prep: {str(report['accepted_for_hosting']).lower()}",
        f"- Agent contact-sheet review: {report['agent_contact_sheet_review']}",
        "",
        "## Role Counts",
        "",
        "| role | videos |",
        "|---|---:|",
    ]
    for role, count in report["role_counts"].items():
        lines.append(f"| `{role}` | {count} |")

    lines.extend(["", "## Contact Sheets", "", "| role | videos | sheet |", "|---|---:|---|"])
    for sheet in report["contact_sheets"]:
        lines.append(f"| `{sheet['role']}` | {sheet['items']} | `{sheet['path']}` |")

    failures = [row for row in report["rows"] if row["screening_flags"]]
    lines.extend(["", "## Screening Flags", ""])
    if failures:
        lines.extend(["| role | job | flags |", "|---|---|---|"])
        for row in failures:
            lines.append(
                "| `{role}` | `{job}` | `{flags}` |".format(
                    role=row["role"],
                    job=row["job_id"],
                    flags=", ".join(row["screening_flags"]),
                )
            )
    else:
        lines.append("None from sampled-frame video screening.")

    if report.get("launch_assets_ready"):
        next_action = [
            "Complete final human/IRB-facing content review, configure the",
            "two Prolific sessions with completion codes and the response",
            "endpoint, then run the delayed recognition-memory study.",
        ]
    else:
        next_action = [
            "Review the contact sheets and generated MP4s, host the accepted videos",
            "at stable HTTPS URLs, then wire those URLs into the two-session",
            "recognition study.",
        ]

    lines.extend(
        [
            "",
            "## Launch Blockers",
            "",
            *[f"- {blocker}" for blocker in report["launch_blockers"]],
            "",
            "## Next Action",
            "",
            *next_action,
            "",
        ]
    )
    return "\n".join(lines)


def build_generation_report(
    *,
    manifest_path: Path,
    result_path: Path,
    jobs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config: GenerationConfig,
    steering_artifact: Path,
    steering_key: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "content_pocket_recognition_video_generation.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "result_path": str(result_path),
        "source_production_manifest": str(manifest_path),
        "status": "dry_run" if dry_run else "generation_complete",
        "dry_run": dry_run,
        "config": {
            "app_name": config.app_name,
            "num_frames": config.num_frames,
            "num_inference_steps": config.num_inference_steps,
            "motion_bucket_id": config.motion_bucket_id,
            "noise_aug_strength": config.noise_aug_strength,
            "fps": config.fps,
            "timeout_seconds": config.timeout_seconds,
            "steering_artifact": str(steering_artifact),
            "steering_key": steering_key,
        },
        "counts": generation_counts(rows),
        "role_counts": dict(sorted(Counter(job["role"] for job in jobs).items())),
        "rows": rows,
        "claim_boundary": [
            "Generated recognition MP4s are not human-memory evidence.",
            "Video-level screening and final human/IRB-facing review are required before launch.",
            "Failed or visually rejected videos must be retained with reasons.",
        ],
    }


def run_generation(
    *,
    manifest_path: Path,
    steering_artifact: Path,
    steering_key: str,
    roles: set[str] | None,
    job_ids: set[str] | None,
    limit: int | None,
    only_missing: bool,
    overwrite: bool,
    dry_run: bool,
    config: GenerationConfig,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    jobs = selected_jobs(
        manifest,
        roles=roles,
        job_ids=job_ids,
        limit=limit,
        only_missing=only_missing,
    )
    if dry_run:
        rows = [{**job_row(job), "status": "dry_run"} for job in jobs]
    else:
        steering_vector = load_steering_vector(steering_artifact, key=steering_key)
        rows = generate_videos_on_modal(
            jobs=jobs,
            steering_vector=steering_vector,
            config=config,
            overwrite=overwrite,
        )
    return build_generation_report(
        manifest_path=manifest_path,
        result_path=DEFAULT_GENERATION_JSON,
        jobs=jobs,
        rows=rows,
        config=config,
        steering_artifact=steering_artifact,
        steering_key=steering_key,
        dry_run=dry_run,
    )


def maybe_populate_svd_cache(*, app_name: str, enabled: bool) -> None:
    if not enabled:
        return
    import modal  # type: ignore[import-not-found]  # noqa: PLC0415

    function = modal.Function.from_name(app_name, "populate_svd_weights")
    function.remote()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_GENERATION_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_GENERATION_MD)
    parser.add_argument("--screening-json", type=Path, default=DEFAULT_SCREENING_JSON)
    parser.add_argument("--screening-md", type=Path, default=DEFAULT_SCREENING_MD)
    parser.add_argument("--sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--launch-assets-result", type=Path, default=DEFAULT_LAUNCH_ASSETS)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--populate-svd-cache", action="store_true")
    parser.add_argument(
        "--steering-artifact",
        type=Path,
        default=Path(os.environ["BO_MEM_STEERING_ARTIFACT"])
        if os.environ.get("BO_MEM_STEERING_ARTIFACT")
        else Path("data/reports/adapter_tribe_to_clip_h.pt"),
    )
    parser.add_argument("--steering-key", default="v_mem_clip_h_via_adapter")
    parser.add_argument("--app-name", default=os.environ.get("MODAL_APP_NAME", "audience-vectors-dev"))
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--svd-num-frames", type=int, default=25)
    parser.add_argument("--svd-motion-bucket-id", type=int, default=5)
    parser.add_argument("--svd-noise-aug-strength", type=float, default=0.0)
    parser.add_argument("--svd-fps", type=int, default=7)
    parser.add_argument("--generation-timeout", type=int, default=20 * 60)
    parser.add_argument("--screening-samples", type=int, default=3)
    parser.add_argument("--visual-min-tail-sharpness-ratio", type=float, default=0.35)
    parser.add_argument("--visual-min-tail-contrast-ratio", type=float, default=0.55)
    parser.add_argument("--visual-min-tail-contrast", type=float, default=0.04)
    parser.add_argument(
        "--agent-review-note",
        default=(
            "Contact sheets generated for review; final human/IRB-facing "
            "screening still required."
        ),
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    roles = set(args.role) if args.role else None
    job_ids = set(args.job_id) if args.job_id else None
    config = GenerationConfig(
        app_name=args.app_name,
        num_frames=args.svd_num_frames,
        num_inference_steps=args.num_inference_steps,
        motion_bucket_id=args.svd_motion_bucket_id,
        noise_aug_strength=args.svd_noise_aug_strength,
        fps=args.svd_fps,
        timeout_seconds=args.generation_timeout,
    )
    maybe_populate_svd_cache(app_name=args.app_name, enabled=args.populate_svd_cache)
    generation_report = run_generation(
        manifest_path=args.manifest,
        steering_artifact=args.steering_artifact,
        steering_key=args.steering_key,
        roles=roles,
        job_ids=job_ids,
        limit=args.limit,
        only_missing=not args.include_existing,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        config=config,
    )
    generation_report["result_path"] = str(args.out_json)
    write_json(args.out_json, generation_report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_generation_markdown(generation_report), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] generation counts: {generation_report['counts']}")

    if args.dry_run:
        return 0

    thresholds = ArtifactThresholds(
        min_tail_sharpness_ratio=args.visual_min_tail_sharpness_ratio,
        min_tail_contrast_ratio=args.visual_min_tail_contrast_ratio,
        min_tail_contrast=args.visual_min_tail_contrast,
    )
    screening_report, screening_markdown = build_screening_report(
        generation_report=generation_report,
        sheet_dir=args.sheet_dir,
        samples=args.screening_samples,
        thresholds=thresholds,
        agent_review_note=args.agent_review_note,
        launch_assets_result=args.launch_assets_result,
    )
    write_json(args.screening_json, screening_report)
    args.screening_md.parent.mkdir(parents=True, exist_ok=True)
    args.screening_md.write_text(screening_markdown, encoding="utf-8")
    print(f"[done] wrote {args.screening_json}")
    print(f"[done] wrote {args.screening_md}")
    print(f"[done] wrote {len(screening_report['contact_sheets'])} contact sheets")
    print(f"[done] screening counts: {screening_report['counts']}")
    return 0 if generation_report["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
