"""Build the production manifest for recognition-memory stimuli.

The recognition-memory design freezes the old targets and defines same-category
lure slots. This builder turns that design into an operational manifest for the
next step: acquire distinct seed images, generate SVD MP4s, screen them, and
only then freeze the launchable recognition set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_DESIGN = ARTIFACT_DIR / "content_pocket_recognition_memory_design_20260608.json"
DEFAULT_OUT_JSON = (
    ARTIFACT_DIR / "content_pocket_recognition_stimulus_production_manifest_20260608.json"
)
DEFAULT_OUT_MD = (
    ARTIFACT_DIR / "content_pocket_recognition_stimulus_production_manifest_20260608.md"
)
DEFAULT_SEED_SCREENING_RESULT = (
    ARTIFACT_DIR / "content_pocket_recognition_seed_screening_result_20260608.json"
)
DEFAULT_SEED_ROOT = Path("data/recognition_memory_seed_images_20260608")
DEFAULT_VIDEO_OUT_DIR = Path("data/generated/content_pocket_recognition_memory_20260608")

DEFAULT_FILLER_OLD_COUNT = 25
DEFAULT_FILLER_RECOGNITION_COUNT = 20
NEUTRAL_FILLER_ALPHA = 0.0
NEUTRAL_FILLER_GUIDANCE = 7.5
FILLER_OLD_NOISE_SEED_BASE = 880_000
FILLER_LURE_NOISE_SEED_BASE = 890_000


@dataclass(frozen=True)
class FillerTemplate:
    """One unrelated filler category template."""

    pocket: str
    label: str
    old_prompt: str
    lure_prompt: str
    old_requirements: tuple[str, ...]
    lure_requirements: tuple[str, ...]


FILLER_TEMPLATES: tuple[FillerTemplate, ...] = (
    FillerTemplate(
        pocket="fresh24_golden_grass",
        label="golden grass",
        old_prompt=(
            "Golden grass in warm sunlight with shallow depth of field. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different patch of golden grass in warm sunlight, with distinct "
            "plant shapes and camera angle. Natural realistic short video, "
            "clear central subject, continuous motion, stable composition, no "
            "text, no watermark."
        ),
        old_requirements=("unrelated to flowers, clothes, beaches, and streets",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_red_mailbox",
        label="red mailbox",
        old_prompt=(
            "A red mailbox mounted near a building facade. Natural realistic "
            "short video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different red mailbox in a distinct setting and camera angle. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_suspension_bridge",
        label="suspension bridge",
        old_prompt=(
            "A suspension bridge viewed from a distinctive angle. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different suspension bridge view with distinct cables and "
            "framing. Natural realistic short video, clear central subject, "
            "continuous motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_lighthouse",
        label="lighthouse",
        old_prompt=(
            "A lighthouse standing under dramatic sky. Natural realistic short "
            "video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different lighthouse with distinct surroundings and horizon. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_coastal_tracks",
        label="coastal tracks",
        old_prompt=(
            "Railroad tracks beside a coastline with rocks and pale sky. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_prompt=(
            "Different coastal railroad tracks with a distinct shoreline and "
            "track perspective. Natural realistic short video, clear central "
            "subject, continuous motion, stable composition, no text, no "
            "watermark."
        ),
        old_requirements=("not visually confusable with beach analysis controls",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_ocean_cliffs",
        label="ocean cliffs",
        old_prompt=(
            "Ocean cliffs above a pale surf line. Natural realistic short video, "
            "clear central subject, continuous motion, stable composition, no "
            "text, no watermark."
        ),
        lure_prompt=(
            "Different ocean cliffs with a distinct cliff face, surf line, and "
            "framing. Natural realistic short video, clear central subject, "
            "continuous motion, stable composition, no text, no watermark."
        ),
        old_requirements=("not visually confusable with storm beach controls",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_concert_stage",
        label="concert stage",
        old_prompt=(
            "A small concert stage with musicians under lights. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different concert stage with distinct lighting and performer "
            "layout. Natural realistic short video, clear central subject, "
            "continuous motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_misty_woods",
        label="misty woods",
        old_prompt=(
            "Misty woods with tall trunks and soft light. Natural realistic "
            "short video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different misty woodland scene with distinct tree spacing and "
            "light. Natural realistic short video, clear central subject, "
            "continuous motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_wheat_closeup",
        label="wheat closeup",
        old_prompt=(
            "A close view of wheat heads in a field. Natural realistic short "
            "video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different close view of wheat heads with distinct spacing and "
            "background blur. Natural realistic short video, clear central "
            "subject, continuous motion, stable composition, no text, no "
            "watermark."
        ),
        old_requirements=("unrelated to flowers, clothes, beaches, and streets",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_cloud_mountain",
        label="cloud mountain",
        old_prompt=(
            "Clouds passing across a mountain ridge. Natural realistic short "
            "video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different mountain ridge with clouds and distinct rock shapes. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_mountain_fog",
        label="mountain fog",
        old_prompt=(
            "A foggy mountain face with muted natural light. Natural realistic "
            "short video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different foggy mountain face with distinct ridge geometry. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_forest_canopy",
        label="forest canopy",
        old_prompt=(
            "A forest canopy viewed upward through tall trunks. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different upward forest canopy view with distinct trunks and sky "
            "openings. Natural realistic short video, clear central subject, "
            "continuous motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_dewy_grass",
        label="dewy grass",
        old_prompt=(
            "Dewy grass blades with low camera angle and soft sun. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different low-angle dewy grass scene with distinct blade shapes "
            "and highlights. Natural realistic short video, clear central "
            "subject, continuous motion, stable composition, no text, no "
            "watermark."
        ),
        old_requirements=("unrelated to flowers, clothes, beaches, and streets",),
        lure_requirements=("same broad category as the filler old target",),
    ),
    FillerTemplate(
        pocket="fresh24_sparse_forest",
        label="sparse forest",
        old_prompt=(
            "A sparse forest with bare trunks and pale sky. Natural realistic "
            "short video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_prompt=(
            "A different sparse forest with distinct trunk spacing and horizon. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        old_requirements=("unrelated to primary and hard-negative analysis arms",),
        lure_requirements=("same broad category as the filler old target",),
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "bytes": path.stat().st_size if exists else None,
    }


def seed_request(
    *,
    request_id: str,
    role: str,
    seed_path: Path,
    prompt: str,
    requirements: list[str],
    matched_id: str | None = None,
    source_pocket: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "role": role,
        "matched_id": matched_id,
        "source_pocket": source_pocket,
        "seed_image": file_status(seed_path),
        "prompt": prompt,
        "requirements": requirements,
        "must_not_optimize_for_memorability": True,
        "status": "present" if seed_path.exists() else "missing_seed_image",
    }


def generation_job(
    *,
    job_id: str,
    role: str,
    seed_path: Path,
    output_path: Path,
    prompt: str,
    alpha: float,
    guidance: float,
    noise_seed: int,
    matched_id: str | None = None,
    source_pocket: str | None = None,
    old_target_id: str | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "role": role,
        "matched_id": matched_id,
        "source_pocket": source_pocket,
        "old_target_id": old_target_id,
        "seed_image": file_status(seed_path),
        "output_video": file_status(output_path),
        "generator": "current image-conditioned SVD runner",
        "prompt": prompt,
        "alpha": alpha,
        "guidance": guidance,
        "noise_seed": noise_seed,
        "must_screen_before_use": True,
        "must_not_optimize_for_memorability": True,
    }


def build_analysis_lure_artifacts(
    design: dict[str, Any],
    *,
    seed_root: Path,
    video_out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_requests: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for request in design["lure_generation_requests"]:
        lure_id = str(request["lure_id"])
        seed_path = seed_root / "analysis_lures" / f"{lure_id}.png"
        output_path = video_out_dir / "analysis_lures" / f"{lure_id}.mp4"
        requirements = [
            *request["distinctiveness_requirements"],
            "must be visually distinct from the matched old target",
            "must be manually screened before SVD generation",
        ]
        seed_requests.append(
            seed_request(
                request_id=lure_id,
                role="analysis_lure_seed",
                seed_path=seed_path,
                prompt=str(request["prompt"]),
                requirements=requirements,
                matched_id=str(request["target_id"]),
                source_pocket=str(request["pocket"]),
            )
        )
        generation = request["generation_request"]
        jobs.append(
            generation_job(
                job_id=lure_id,
                role="analysis_lure_video",
                seed_path=seed_path,
                output_path=output_path,
                prompt=str(request["prompt"]),
                alpha=float(generation["alpha"]),
                guidance=float(generation["guidance"]),
                noise_seed=int(generation["suggested_noise_seed"]),
                matched_id=str(request["target_id"]),
                source_pocket=str(request["pocket"]),
                old_target_id=str(request["target_id"]),
            )
        )
    return seed_requests, jobs


def filler_template(index: int) -> FillerTemplate:
    return FILLER_TEMPLATES[index % len(FILLER_TEMPLATES)]


def filler_old_id(index: int) -> str:
    return f"filler_old_v{index:02d}"


def filler_lure_id(index: int) -> str:
    return f"filler_lure_v{index:02d}"


def build_filler_artifacts(
    *,
    seed_root: Path,
    video_out_dir: Path,
    filler_old_count: int,
    filler_recognition_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if filler_recognition_count > filler_old_count:
        raise ValueError("filler recognition count cannot exceed old filler count")

    seed_requests: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for index in range(filler_old_count):
        template = filler_template(index)
        request_id = filler_old_id(index)
        seed_path = seed_root / "fillers" / "old" / f"{request_id}.png"
        output_path = video_out_dir / "fillers" / "old" / f"{request_id}.mp4"
        requirements = [
            *template.old_requirements,
            "must not be reused as an analysis lure",
            "must pass manual image screening before SVD generation",
        ]
        seed_requests.append(
            seed_request(
                request_id=request_id,
                role="filler_old_seed",
                seed_path=seed_path,
                prompt=template.old_prompt,
                requirements=requirements,
                source_pocket=template.pocket,
            )
        )
        jobs.append(
            generation_job(
                job_id=request_id,
                role="filler_old_video",
                seed_path=seed_path,
                output_path=output_path,
                prompt=template.old_prompt,
                alpha=NEUTRAL_FILLER_ALPHA,
                guidance=NEUTRAL_FILLER_GUIDANCE,
                noise_seed=FILLER_OLD_NOISE_SEED_BASE + index,
                source_pocket=template.pocket,
            )
        )

    for index in range(filler_recognition_count):
        template = filler_template(index)
        request_id = filler_lure_id(index)
        matched_id = filler_old_id(index)
        seed_path = seed_root / "fillers" / "lures" / f"{request_id}.png"
        output_path = video_out_dir / "fillers" / "lures" / f"{request_id}.mp4"
        requirements = [
            *template.lure_requirements,
            "must be visually distinct from the matched filler old target",
            "must pass manual image screening before SVD generation",
        ]
        seed_requests.append(
            seed_request(
                request_id=request_id,
                role="filler_lure_seed",
                seed_path=seed_path,
                prompt=template.lure_prompt,
                requirements=requirements,
                matched_id=matched_id,
                source_pocket=template.pocket,
            )
        )
        jobs.append(
            generation_job(
                job_id=request_id,
                role="filler_lure_video",
                seed_path=seed_path,
                output_path=output_path,
                prompt=template.lure_prompt,
                alpha=NEUTRAL_FILLER_ALPHA,
                guidance=NEUTRAL_FILLER_GUIDANCE,
                noise_seed=FILLER_LURE_NOISE_SEED_BASE + index,
                matched_id=matched_id,
                source_pocket=template.pocket,
            )
        )
    return seed_requests, jobs


def artifact_counts(
    seed_requests: list[dict[str, Any]],
    generation_jobs: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "seed_image_requests": len(seed_requests),
        "seed_images_present": sum(
            1 for request in seed_requests if request["seed_image"]["exists"]
        ),
        "seed_images_missing": sum(
            1 for request in seed_requests if not request["seed_image"]["exists"]
        ),
        "generation_jobs": len(generation_jobs),
        "output_videos_present": sum(
            1 for job in generation_jobs if job["output_video"]["exists"]
        ),
        "output_videos_missing": sum(
            1 for job in generation_jobs if not job["output_video"]["exists"]
        ),
    }


def load_optional_screening_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "status": None,
            "accepted_for_svd_generation": False,
        }
    payload = load_json(path)
    return {
        "path": str(path),
        "exists": True,
        "status": payload.get("status"),
        "accepted_for_svd_generation": bool(payload.get("accepted_for_svd_generation")),
    }


def production_status(
    counts: dict[str, int],
    *,
    seed_screening: dict[str, Any],
) -> str:
    if counts["seed_images_missing"]:
        return "missing_seed_images_not_ready_for_generation"
    if not seed_screening["accepted_for_svd_generation"]:
        return "seed_images_present_screening_required"
    if counts["output_videos_missing"]:
        return "seed_images_screened_ready_for_svd_generation"
    return "generated_videos_present_screening_required"


def launch_blockers(
    counts: dict[str, int],
    *,
    seed_screening: dict[str, Any],
) -> list[str]:
    blockers = []
    if counts["seed_images_missing"]:
        blockers.append(
            f"{counts['seed_images_missing']} seed images are missing or not materialized."
        )
    if counts["output_videos_missing"]:
        blockers.append(f"{counts['output_videos_missing']} SVD output MP4s are missing.")
    if not seed_screening["accepted_for_svd_generation"]:
        blockers.append("Manual image distinctiveness screening has not been recorded.")
    blockers.extend(
        [
            "Generated MP4 visual screening/contact sheets have not been recorded.",
            "Hosted HTTPS URLs and two-session Prolific wiring are not complete.",
        ]
    )
    return blockers


def build_manifest(
    *,
    design_path: Path,
    seed_root: Path,
    video_out_dir: Path,
    seed_screening_result: Path,
    filler_old_count: int,
    filler_recognition_count: int,
) -> tuple[dict[str, Any], str]:
    design = load_json(design_path)
    analysis_seed_requests, analysis_jobs = build_analysis_lure_artifacts(
        design,
        seed_root=seed_root,
        video_out_dir=video_out_dir,
    )
    filler_seed_requests, filler_jobs = build_filler_artifacts(
        seed_root=seed_root,
        video_out_dir=video_out_dir,
        filler_old_count=filler_old_count,
        filler_recognition_count=filler_recognition_count,
    )
    seed_requests = analysis_seed_requests + filler_seed_requests
    generation_jobs = analysis_jobs + filler_jobs
    counts = artifact_counts(seed_requests, generation_jobs)
    seed_screening = load_optional_screening_status(seed_screening_result)
    manifest = {
        "schema_version": "content_pocket_recognition_stimulus_production.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": production_status(counts, seed_screening=seed_screening),
        "source_recognition_design": str(design_path),
        "source_task_payload_sha256": design["source_task_payload_sha256"],
        "seed_root": str(seed_root),
        "video_output_dir": str(video_out_dir),
        "seed_image_screening": seed_screening,
        "question": (
            "Can the recognition-memory validation set be materialized without "
            "using near-duplicate same-category lures?"
        ),
        "current_regime_reading": (
            "Production search inside the accepted content-pocket recognition "
            "memory regime. This does not change claims until human recognition "
            "or BMD-grounded validation clears."
        ),
        "artifact_counts": counts,
        "seed_image_requests": seed_requests,
        "generation_jobs": generation_jobs,
        "screening_gate": {
            "image_screening": [
                "all seed images present",
                "analysis lures match broad category but are not near-duplicates",
                "filler images are unrelated to primary/hard-negative analysis arms",
                "no text, watermark, obvious artifacts, or attention-check leakage",
            ],
            "video_screening": [
                "all SVD MP4s generated",
                "visual subject retained",
                "no frame collapse, text, watermark, or broken playback",
                "all rejected/missing clips retained with reasons",
            ],
        },
        "launch_blockers": launch_blockers(counts, seed_screening=seed_screening),
        "claim_boundary": [
            "This artifact is a production manifest, not human evidence.",
            "Do not claim actual memorability until the recognition gate clears.",
            "Do not use near-duplicate lures to rescue an underpowered recognition result.",
        ],
    }
    return manifest, render_markdown(manifest)


def render_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest["artifact_counts"]
    if manifest["seed_image_screening"]["accepted_for_svd_generation"]:
        next_action = [
            "Generate SVD MP4s from the screened seed images and generation",
            "jobs, then screen/contact-sheet every MP4 before freezing the",
            "launchable recognition set.",
        ]
    else:
        next_action = [
            "Materialize the listed seed images under the manifest seed root,",
            "review their contact sheet for category match and distinctiveness,",
            "generate SVD MP4s from the generation jobs, then screen/contact-sheet",
            "those MP4s before freezing the launchable recognition set.",
        ]
    lines = [
        "# Content-Pocket Recognition Stimulus Production Manifest",
        "",
        f"Date: {manifest['created_at_utc']}",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: can the recognition-memory validation set be materialized",
        "without using near-duplicate same-category lures?",
        "",
        "Current regime:",
        "",
        "- Artifact types: frozen old MP4s, seed-image requests, SVD generation",
        "  jobs, generated lure/filler MP4s, visual screening records, hosted URLs,",
        "  two-session recognition responses.",
        "- Operations: acquire distinct seed images, generate SVD MP4s, screen",
        "  images and videos, freeze launchable old-vs-lure recognition forms.",
        "- Gates/verifiers: seed-image distinctiveness, MP4 visual validity,",
        "  complete-candidate retention, old-vs-lure human recognition accuracy.",
        "- Known limitation: seed images and generated recognition lures are",
        "  not accepted for launch until their corresponding screening records",
        "  exist.",
        "",
        "Action class: production search inside the accepted recognition-memory",
        "validation regime.",
        "",
        "## Status",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Seed image requests: {counts['seed_image_requests']}",
        f"- Seed images present: {counts['seed_images_present']}",
        f"- Seed images missing: {counts['seed_images_missing']}",
        f"- SVD generation jobs: {counts['generation_jobs']}",
        f"- Output MP4s present: {counts['output_videos_present']}",
        f"- Output MP4s missing: {counts['output_videos_missing']}",
        "",
        "## Required Production Blocks",
        "",
        "| block | count | purpose |",
        "|---|---:|---|",
        "| analysis lures | 15 | same-category old-vs-lure trials for primary and hard-negative arms |",
        "| filler old targets | 25 | Session 1 unrelated filler exposures |",
        "| filler lures | 20 | Session 2 unrelated filler recognition trials |",
        "",
        "## Launch Blockers",
        "",
        *[f"- {blocker}" for blocker in manifest["launch_blockers"]],
        "",
        "## Claim Boundary",
        "",
        *[f"- {rule}" for rule in manifest["claim_boundary"]],
        "",
        "## Next Action",
        "",
        *next_action,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--video-out-dir", type=Path, default=DEFAULT_VIDEO_OUT_DIR)
    parser.add_argument(
        "--seed-screening-result",
        type=Path,
        default=DEFAULT_SEED_SCREENING_RESULT,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--filler-old-count", type=int, default=DEFAULT_FILLER_OLD_COUNT)
    parser.add_argument(
        "--filler-recognition-count",
        type=int,
        default=DEFAULT_FILLER_RECOGNITION_COUNT,
    )
    args = parser.parse_args()

    manifest, markdown = build_manifest(
        design_path=args.design,
        seed_root=args.seed_root,
        video_out_dir=args.video_out_dir,
        seed_screening_result=args.seed_screening_result,
        filler_old_count=args.filler_old_count,
        filler_recognition_count=args.filler_recognition_count,
    )
    write_json(args.out_json, manifest)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] status: {manifest['status']}")
    print(f"[done] seed requests: {manifest['artifact_counts']['seed_image_requests']}")
    print(f"[done] generation jobs: {manifest['artifact_counts']['generation_jobs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
