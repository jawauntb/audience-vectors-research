"""Build Seedance candidate-generation jobs for the confirmatory study."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_CONFIG = EXPERIMENT_DIR / "confirmatory_study_config_20260615.json"
DEFAULT_OUT_JSON = EXPERIMENT_DIR / "seedance_candidate_generation_manifest_20260615.json"
DEFAULT_OUT_MD = EXPERIMENT_DIR / "seedance_candidate_generation_manifest_20260615.md"
DEFAULT_VIDEO_ROOT = Path(
    "data/generated/content_pocket_confirmatory_recognition_20260615/candidate_old_videos"
)

VARIANT_DIRECTIONS = (
    "close framing, clear central subject, gentle natural motion",
    "medium framing, visible surroundings, gentle camera drift",
    "wider contextual framing, stable composition, subtle environmental motion",
    "side-angle composition, distinct background geometry, continuous motion",
    "foreground detail with readable background separation, stable camera",
    "slight push-in camera movement, clear subject identity, no abrupt cuts",
    "static camera with subject motion, natural lighting, coherent scene",
    "alternate viewpoint with distinct layout, realistic motion, clean edges",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate_prompt(template: str, variant_direction: str) -> str:
    return (
        f"{template} Variant requirements: {variant_direction}; audio-free; "
        "no captions, no subtitles, no logos, no watermarks, no text overlays."
    )


def build_manifest(config: dict[str, Any], *, video_root: Path) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    generator = config["generator"]
    candidates_per_family = int(
        config["stimulus_counts"]["candidate_old_videos_per_family"]
    )
    if candidates_per_family != len(VARIANT_DIRECTIONS):
        raise ValueError(
            "candidate_old_videos_per_family must match VARIANT_DIRECTIONS"
        )

    for family in config["content_families"]:
        family_id = str(family["family_id"])
        template = str(family["prompt_template"])
        for candidate_index, variant_direction in enumerate(VARIANT_DIRECTIONS):
            candidate_id = f"{family_id}_candidate_old_v{candidate_index:02d}"
            output_path = video_root / family_id / f"{candidate_id}.mp4"
            jobs.append(
                {
                    "job_id": candidate_id,
                    "role": "candidate_old_video",
                    "family_id": family_id,
                    "candidate_index": candidate_index,
                    "prior_role": family["prior_role"],
                    "prompt": candidate_prompt(template, variant_direction),
                    "variant_direction": variant_direction,
                    "generator": generator,
                    "output_video": {
                        "path": str(output_path),
                        "sha256": None,
                        "bytes": None,
                        "status": "not_generated",
                    },
                    "retain_if_failed": True,
                }
            )

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": config["experiment_id"],
        "source_config": str(DEFAULT_CONFIG),
        "status": "manifest_only_no_api_calls",
        "job_count": len(jobs),
        "jobs": jobs,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Seedance Candidate Generation Manifest",
        "",
        f"Created: `{manifest['created_at_utc']}`",
        f"Experiment: `{manifest['experiment_id']}`",
        f"Status: `{manifest['status']}`",
        f"Jobs: `{manifest['job_count']}`",
        "",
        "This manifest contains candidate old-video jobs only. It does not call",
        "Seedance and does not generate lures, fillers, or human-study assets.",
        "",
        "## Families",
        "",
    ]
    by_family: dict[str, int] = {}
    for job in manifest["jobs"]:
        by_family[job["family_id"]] = by_family.get(job["family_id"], 0) + 1
    for family_id, count in sorted(by_family.items()):
        lines.append(f"- `{family_id}`: `{count}` candidate jobs")
    lines.extend(["", "## First Jobs", ""])
    for job in manifest["jobs"][:12]:
        lines.extend(
            [
                f"### {job['job_id']}",
                "",
                f"- Family: `{job['family_id']}`",
                f"- Output: `{job['output_video']['path']}`",
                f"- Prompt: {job['prompt']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    manifest = build_manifest(config, video_root=args.video_root)
    write_json(args.out_json, manifest)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
