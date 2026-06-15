"""Build Seedance candidate-generation jobs for the confirmatory study."""

from __future__ import annotations

import argparse
import json
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
IMPROVED_OUT_JSON = (
    EXPERIMENT_DIR / "seedance_candidate_generation_manifest_improved_v1_20260615.json"
)
IMPROVED_OUT_MD = (
    EXPERIMENT_DIR / "seedance_candidate_generation_manifest_improved_v1_20260615.md"
)
IMPROVED_VIDEO_ROOT = Path(
    "data/generated/content_pocket_confirmatory_recognition_20260615/"
    "candidate_old_videos_improved_v1"
)

BASE_PROMPT_RULES = (
    "single continuous realistic camera shot; five seconds; natural light; "
    "clear focal subject; coherent physical motion; no scene cuts; no montage; "
    "no captions, subtitles, logos, brands, watermarks, signs, readable text, "
    "UI elements, or artificial borders; silent video."
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

IMPROVED_VARIANTS = (
    "subject-forward composition with crisp details and one easy-to-name visual anchor",
    "medium shot with subject plus a distinctive surrounding object or background shape",
    "wider establishing shot that still keeps the subject easy to recognize",
    "side-angle shot with parallax or gentle camera drift and stable composition",
    "foreground subject with layered background separation and readable texture",
    "slow push-in toward the subject without abrupt movement or cuts",
    "locked-off camera where the subject motion carries the clip",
    "alternate viewpoint with different layout, lighting direction, and spatial landmarks",
)

IMPROVED_FAMILY_PROMPTS = {
    "orange_flowers": (
        "real garden footage of vivid orange marigold or poppy-like flowers, "
        "petals sharply defined, green stems and leaves visible, a light breeze "
        "moving petals naturally, warm daylight, high detail, no people"
    ),
    "hanging_clothes": (
        "real outdoor laundry line with colorful shirts and sheets hanging from "
        "clothespins, fabric folds and stitching visible, clothes moving gently "
        "in wind, clear domestic background, no faces or readable labels"
    ),
    "blue_jellyfish": (
        "realistic aquarium or underwater footage of luminous blue jellyfish, "
        "translucent bell and trailing tentacles visible, slow drifting motion, "
        "dark blue water background, no divers or text"
    ),
    "old_car": (
        "realistic footage of a distinctive vintage car parked on a quiet street "
        "or rural roadside, chrome details and rounded body shape visible, slight "
        "camera movement or moving reflections, no license plate text"
    ),
    "aerial_beach": (
        "realistic drone-style beach footage from above, shoreline curve, foamy "
        "waves rolling onto sand, visible water-sand boundary, smooth camera "
        "glide, no people as focal subjects"
    ),
    "city_street": (
        "realistic city street footage with buildings, crosswalk geometry, "
        "parked cars or traffic motion, strong perspective lines, overcast or "
        "daylight realism, no readable storefront text"
    ),
    "storm_beach": (
        "realistic stormy beach footage with dark clouds, choppy waves, wet sand, "
        "wind-driven spray, dramatic but natural lighting, stable horizon"
    ),
    "butterflies_on_flowers": (
        "realistic garden footage of butterflies landing on bright flowers, wings "
        "opening and closing, flower heads and stems clear, natural daylight, no people"
    ),
    "street_food_grill": (
        "realistic street food grill close to the cooking surface, skewers or "
        "vegetables sizzling, steam or flame motion, metal grill texture visible, "
        "no vendor face or readable menu text"
    ),
    "rain_on_window": (
        "realistic close view of raindrops sliding down a window, blurred city or "
        "greenery background, droplets merging and streaking, soft daylight, no text"
    ),
    "hands_pottery_wheel": (
        "realistic close footage of hands shaping wet clay on a spinning pottery "
        "wheel, clay texture and circular motion visible, studio table background, "
        "no face or text"
    ),
    "candle_flame_table": (
        "realistic tabletop candle footage, flame flickering, wax and wick visible, "
        "nearby simple objects softly out of focus, warm natural shadows, no text"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--prompt-set",
        choices=("baseline", "improved_v1"),
        default="baseline",
        help="Prompt regime to materialize in the manifest.",
    )
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


def improved_candidate_prompt(family_id: str, variant_direction: str) -> str:
    family_prompt = IMPROVED_FAMILY_PROMPTS.get(family_id)
    if family_prompt is None:
        raise ValueError(f"Missing improved prompt for family: {family_id}")
    return (
        f"{family_prompt}. Variant requirements: {variant_direction}. "
        f"Global requirements: {BASE_PROMPT_RULES}"
    )


def build_manifest(
    config: dict[str, Any], *, video_root: Path, prompt_set: str
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    generator = config["generator"]
    candidates_per_family = int(
        config["stimulus_counts"]["candidate_old_videos_per_family"]
    )
    variant_directions = (
        IMPROVED_VARIANTS if prompt_set == "improved_v1" else VARIANT_DIRECTIONS
    )
    if candidates_per_family != len(variant_directions):
        raise ValueError(
            "candidate_old_videos_per_family must match selected variant directions"
        )

    for family in config["content_families"]:
        family_id = str(family["family_id"])
        template = str(family["prompt_template"])
        for candidate_index, variant_direction in enumerate(variant_directions):
            candidate_id = f"{family_id}_candidate_old_v{candidate_index:02d}"
            output_path = video_root / family_id / f"{candidate_id}.mp4"
            prompt = (
                improved_candidate_prompt(family_id, variant_direction)
                if prompt_set == "improved_v1"
                else candidate_prompt(template, variant_direction)
            )
            jobs.append(
                {
                    "job_id": candidate_id,
                    "role": "candidate_old_video",
                    "family_id": family_id,
                    "candidate_index": candidate_index,
                    "prior_role": family["prior_role"],
                    "prompt": prompt,
                    "prompt_set": prompt_set,
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
        "created_at_utc": config["manifest_created_at_utc"],
        "experiment_id": config["experiment_id"],
        "source_config": str(DEFAULT_CONFIG),
        "status": "manifest_only_no_api_calls",
        "prompt_set": prompt_set,
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
        f"Prompt set: `{manifest['prompt_set']}`",
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
    if args.prompt_set == "improved_v1":
        if args.out_json == DEFAULT_OUT_JSON:
            args.out_json = IMPROVED_OUT_JSON
        if args.out_md == DEFAULT_OUT_MD:
            args.out_md = IMPROVED_OUT_MD
        if args.video_root == DEFAULT_VIDEO_ROOT:
            args.video_root = IMPROVED_VIDEO_ROOT
    config = load_json(args.config)
    manifest = build_manifest(config, video_root=args.video_root, prompt_set=args.prompt_set)
    write_json(args.out_json, manifest)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
