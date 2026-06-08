"""Extract exact V-JEPA features for the SVD pocket-regime replay videos.

The content-pocket embedding audit can consume V-JEPA features only when they
correspond to the exact generated MP4s in the pocket-regime replay report. This
script reads that report, uploads each local MP4 by bytes to the Modal
`VjepaPredictor`, and saves one `.npz` per video stem so
`audit_content_pocket_embeddings.py --vjepa-features-dir ...` can integrate the
feature family without mixing in mismatched Wan/BMD artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.services.vjepa_service import VjepaService, VjepaValidationError

DEFAULT_REPLAY_REPORT = Path(
    "data/reports/"
    "bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_"
    "20260608.json"
)
DEFAULT_OUT_DIR = Path("data/features/vjepa_pocket_regime_audit_20260608")
DEFAULT_SUMMARY_JSON = (
    Path("research_program")
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
    / "content_pocket_vjepa_extraction_summary_20260608.json"
)
DEFAULT_SUMMARY_MD = (
    Path("research_program")
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
    / "content_pocket_vjepa_extraction_result_20260608.md"
)


@dataclass(frozen=True)
class VideoJob:
    """One exact generated replay MP4 to encode."""

    sample_id: str
    task_id: str
    seed_idx: int
    replicate: int
    local_video_path: str
    absolute_video_path: Path
    output_path: Path


def logical_path(path: Path) -> str:
    """Return a stable repo/data-lake-looking path for reports."""
    parts = path.resolve().parts
    for anchor in ("data", "research_program", "scripts", "src", "tests"):
        if anchor in parts:
            return str(Path(*parts[parts.index(anchor) :]))
    return str(path)


def repo_root_for_report(report_path: Path) -> Path:
    """Infer the worktree root from a data/reports report path."""
    if report_path.parent.name == "reports" and report_path.parent.parent.name == "data":
        return report_path.parent.parent.parent
    return Path.cwd()


def load_report(path: Path) -> dict[str, Any]:
    """Load a replay report JSON."""
    if not path.exists():
        raise FileNotFoundError(f"replay report not found: {path}")
    return json.loads(path.read_text())


def parse_replicate(path: Path) -> int:
    """Parse the trailing `_repNN` replicate id from a generated video stem."""
    marker = "_rep"
    if marker not in path.stem:
        return -1
    tail = path.stem.rsplit(marker, 1)[1]
    try:
        return int(tail)
    except ValueError:
        return -1


def build_video_jobs(
    *,
    report: dict[str, Any],
    report_path: Path,
    output_dir: Path,
) -> list[VideoJob]:
    """Build one V-JEPA extraction job per scored replay-video row."""
    video_root = repo_root_for_report(report_path)
    jobs_by_sample_id: dict[str, VideoJob] = {}
    for row in report["rows"]:
        if row.get("replay_tribe_score") is None:
            continue
        local_video_path = row.get("local_video_path")
        if not local_video_path:
            continue
        absolute_video_path = video_root / str(local_video_path)
        if not absolute_video_path.exists():
            raise FileNotFoundError(f"missing generated video: {absolute_video_path}")
        sample_id = absolute_video_path.stem
        trial = row["trial"]
        job = VideoJob(
            sample_id=sample_id,
            task_id=str(trial["task_id"]),
            seed_idx=int(trial["seed_idx"]),
            replicate=parse_replicate(absolute_video_path),
            local_video_path=str(local_video_path),
            absolute_video_path=absolute_video_path,
            output_path=output_dir / f"{sample_id}.npz",
        )
        jobs_by_sample_id[sample_id] = job
    return sorted(jobs_by_sample_id.values(), key=lambda job: job.sample_id)


def result_to_arrays(result: Any) -> tuple[np.ndarray, float, int]:
    """Normalize a V-JEPA Modal result into arrays/scalars."""
    if hasattr(result, "embedding"):
        embedding = np.asarray(result.embedding, dtype=np.float32)
        duration = float(result.duration_seconds)
        n_frames = int(result.n_frames)
    else:
        embedding = np.asarray(result["embedding"], dtype=np.float32)
        duration = float(result["duration_seconds"])
        n_frames = int(result["n_frames"])
    return embedding.reshape(-1), duration, n_frames


async def extract_one(
    *,
    job: VideoJob,
    service: VjepaService,
    semaphore: asyncio.Semaphore,
    force: bool,
) -> dict[str, Any]:
    """Extract and save one V-JEPA feature file."""
    if job.output_path.exists() and job.output_path.stat().st_size > 0 and not force:
        return {
            "sample_id": job.sample_id,
            "status": "cached",
            "feature_path": logical_path(job.output_path),
            "video_path": logical_path(job.absolute_video_path),
            "task_id": job.task_id,
            "seed_idx": job.seed_idx,
            "replicate": job.replicate,
        }

    async with semaphore:
        try:
            result = await service.predict_video_bytes(
                job.absolute_video_path.read_bytes(),
                suffix=job.absolute_video_path.suffix or ".mp4",
            )
        except VjepaValidationError as exc:
            return {
                "sample_id": job.sample_id,
                "status": "rejected",
                "error": str(exc),
                "video_path": logical_path(job.absolute_video_path),
                "task_id": job.task_id,
                "seed_idx": job.seed_idx,
                "replicate": job.replicate,
            }

    if result is None:
        return {
            "sample_id": job.sample_id,
            "status": "failed",
            "error": "predict_video_bytes returned None",
            "video_path": logical_path(job.absolute_video_path),
            "task_id": job.task_id,
            "seed_idx": job.seed_idx,
            "replicate": job.replicate,
        }

    embedding, duration, n_frames = result_to_arrays(result)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        job.output_path,
        embedding=embedding.astype(np.float32),
        duration_seconds=np.array(duration, dtype=np.float32),
        n_frames=np.array(n_frames, dtype=np.int32),
        sample_id=np.array(job.sample_id),
        source_video_path=np.array(str(job.absolute_video_path)),
    )
    return {
        "sample_id": job.sample_id,
        "status": "written",
        "feature_path": logical_path(job.output_path),
        "video_path": logical_path(job.absolute_video_path),
        "task_id": job.task_id,
        "seed_idx": job.seed_idx,
        "replicate": job.replicate,
        "embedding_dim": int(embedding.shape[0]),
        "duration_seconds": duration,
        "n_frames": n_frames,
    }


async def extract_many(
    *,
    jobs: list[VideoJob],
    output_dir: Path,
    app_name: str | None,
    max_concurrency: int,
    force: bool,
) -> list[dict[str, Any]]:
    """Run bounded concurrent V-JEPA extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    service = VjepaService(app_name=app_name)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    return await asyncio.gather(
        *[
            extract_one(
                job=job,
                service=service,
                semaphore=semaphore,
                force=force,
            )
            for job in jobs
        ]
    )


def summarize(
    *,
    report_path: Path,
    output_dir: Path,
    app_name: str | None,
    max_concurrency: int,
    jobs: list[VideoJob],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a serializable extraction summary."""
    by_status: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        by_status[status] = by_status.get(status, 0) + 1
    feature_paths = [
        row["feature_path"]
        for row in rows
        if row["status"] in {"written", "cached"} and "feature_path" in row
    ]
    return {
        "schema_version": 1,
        "kind": "content_pocket_vjepa_extraction",
        "source_replay_report": logical_path(report_path),
        "output_dir": logical_path(output_dir),
        "app_name": app_name,
        "max_concurrency": max_concurrency,
        "n_jobs": len(jobs),
        "n_features_available": len(feature_paths),
        "coverage_complete": len(feature_paths) == len(jobs),
        "status_counts": by_status,
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise human-readable extraction note."""
    status_counts = ", ".join(
        f"{key}: {value}" for key, value in sorted(summary["status_counts"].items())
    )
    lines = [
        "# Content-Pocket V-JEPA Extraction Result - 2026-06-08",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: can the missing exact V-JEPA artifact family be populated for "
        "the pocket-regime replay videos, so the embedding audit can test "
        "V-JEPA without using mismatched features?",
        "",
        "Current regime:",
        "",
        "- Artifact types: pocket-regime replay report rows, exact generated MP4s, "
        "V-JEPA `.npz` feature files, extraction status rows, and embedding-audit "
        "inputs.",
        "- Operations: upload each exact local MP4 by bytes to the Modal "
        "`VjepaPredictor`, save one feature file by generated-video stem, and "
        "report exact feature coverage.",
        "- Gates/verifiers: coverage is complete only if every scored replay-video "
        "row has a cached or newly written feature file.",
        "",
        "## Result",
        "",
        f"- Replay report: `{summary['source_replay_report']}`",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Jobs: {summary['n_jobs']}",
        f"- Features available: {summary['n_features_available']}",
        f"- Coverage complete: **{summary['coverage_complete']}**",
        f"- Status counts: {status_counts or 'none'}",
        "",
        "## Next Move",
        "",
        "Rerun `scripts/audit_content_pocket_embeddings.py` with "
        "`--vjepa-features-dir` pointing at this output directory. Only then may "
        "the claim ledger say whether V-JEPA passed or failed for the exact "
        "pocket replay videos.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--app-name")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Process only the first N jobs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.replay_report
    report = load_report(report_path)
    jobs = build_video_jobs(
        report=report,
        report_path=report_path,
        output_dir=args.out_dir,
    )
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    rows = asyncio.run(
        extract_many(
            jobs=jobs,
            output_dir=args.out_dir,
            app_name=args.app_name,
            max_concurrency=args.max_concurrency,
            force=args.force,
        )
    )
    summary = summarize(
        report_path=report_path,
        output_dir=args.out_dir,
        app_name=args.app_name,
        max_concurrency=args.max_concurrency,
        jobs=jobs,
        rows=rows,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    args.summary_md.write_text(render_markdown(summary))
    print(
        json.dumps(
            {
                "summary_json": str(args.summary_json),
                "summary_md": str(args.summary_md),
                "output_dir": str(args.out_dir),
                "n_jobs": summary["n_jobs"],
                "n_features_available": summary["n_features_available"],
                "coverage_complete": summary["coverage_complete"],
                "status_counts": summary["status_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
