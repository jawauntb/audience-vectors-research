"""Screen frozen content-pocket validation MP4s before human/BMD launch.

This script is intentionally a launch-readiness artifact, not a new scientific
verifier. It checks that frozen MP4 bytes still match the stimulus manifest,
re-runs the lightweight sampled-frame visual gate, and writes contact sheets for
agent/human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audience_vectors.visual_artifact_gate import sample_video_frames, summarize_frames

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_MANIFEST = (
    ARTIFACT_DIR / "content_pocket_validation_stimuli_manifest_20260608.json"
)
DEFAULT_OUT_JSON = ARTIFACT_DIR / "content_pocket_validation_mp4_screening_20260608.json"
DEFAULT_OUT_MD = ARTIFACT_DIR / "content_pocket_validation_mp4_screening_20260608.md"
DEFAULT_SHEET_DIR = ARTIFACT_DIR / "content_pocket_validation_screening_sheets_20260608"
DEFAULT_URL_MAP = (
    ARTIFACT_DIR / "content_pocket_validation_hosted_video_url_map_template_20260608.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    text_height = 70
    tile = Image.new("RGB", (frame_size[0] * 3, frame_size[1] + text_height), "white")
    draw = ImageDraw.Draw(tile)
    for index, frame in enumerate(frames):
        tile.paste(fit_image(frame, frame_size), (frame_size[0] * index, text_height))
    draw.rectangle((0, 0, tile.width - 1, tile.height - 1), outline=(190, 190, 190), width=1)
    title = f"{row['pocket']} {row['recipe_index']} rep{row['replicate_index']:02d}"
    subtitle = f"{row['role']} score={row['replay_tribe_score']:+.2f}"
    label = str(row["label"])
    draw.text((8, 5), title[:62], font=small_font, fill=(20, 20, 20))
    draw.text((8, 25), subtitle[:62], font=small_font, fill=(70, 70, 70))
    draw.text((8, 45), label[:62], font=small_font, fill=(90, 90, 90))
    return tile


def write_contact_sheet(
    rows: list[dict[str, Any]],
    frames_by_path: dict[str, list[np.ndarray]],
    *,
    out_path: Path,
    title: str,
    cols: int = 2,
) -> None:
    title_font, small_font = load_fonts()
    tile_size = (480, 160)
    top_height = 42
    n_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size[0], top_height + n_rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=title_font, fill=(10, 10, 10))
    for index, row in enumerate(rows):
        tile = sheet_tile(row, frames_by_path[row["local_video_path"]], small_font)
        sheet.paste(tile, ((index % cols) * tile_size[0], top_height + (index // cols) * tile_size[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def slug(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def screen_stimulus(
    stimulus: dict[str, Any],
    *,
    samples: int,
) -> tuple[dict[str, Any], list[np.ndarray] | None]:
    path = Path(str(stimulus["source_absolute_path"]))
    expected_sha = stimulus.get("sha256")
    actual_sha = sha256_bytes(path)
    row: dict[str, Any] = {
        "analysis_tier": stimulus["analysis_tier"],
        "role": stimulus["role"],
        "pocket": stimulus["pocket"],
        "task_id": stimulus["task_id"],
        "recipe_index": stimulus["recipe_index"],
        "replicate_index": stimulus["replicate_index"],
        "label": stimulus["label"],
        "local_video_path": stimulus["local_video_path"],
        "source_absolute_path": str(path),
        "replay_tribe_score": stimulus["replay_tribe_score"],
        "exists": path.exists(),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha256_matches_manifest": actual_sha == expected_sha,
        "frame_sample_ok": False,
        "frame_count": None,
        "visual_gate": None,
        "screening_flags": [],
    }
    if not path.exists():
        row["screening_flags"].append("missing_file")
        return row, None
    if actual_sha != expected_sha:
        row["screening_flags"].append("sha256_mismatch")
    try:
        frames = sample_video_frames(path, samples=samples)
        row["frame_sample_ok"] = True
        row["frame_count"] = frame_count(path)
        gate = summarize_frames(frames)
        row["visual_gate"] = gate
        row["screening_flags"].extend(gate["artifact_flags"])
    except Exception as exc:  # noqa: BLE001 - screening must preserve the failure reason.
        row["screening_flags"].append(f"frame_sampling_error:{type(exc).__name__}")
        row["frame_sampling_error"] = str(exc)
        return row, None
    return row, frames


def build_screening(
    *,
    manifest_path: Path,
    sheet_dir: Path,
    samples: int,
    agent_review_note: str,
) -> tuple[dict[str, Any], str]:
    manifest = load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    frames_by_path: dict[str, list[np.ndarray]] = {}
    for stimulus in manifest["stimuli"]:
        row, frames = screen_stimulus(stimulus, samples=samples)
        rows.append(row)
        if frames is not None:
            frames_by_path[row["local_video_path"]] = frames

    rows_by_sheet: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_sheet[(str(row["analysis_tier"]), str(row["role"]))].append(row)

    contact_sheets: list[dict[str, Any]] = []
    for (tier, role), group in sorted(rows_by_sheet.items()):
        group.sort(
            key=lambda item: (
                str(item["pocket"]),
                int(item["recipe_index"]),
                int(item["replicate_index"]),
            )
        )
        path = sheet_dir / f"{slug(tier)}_{slug(role)}.jpg"
        write_contact_sheet(
            group,
            frames_by_path,
            out_path=path,
            title=f"{tier} / {role} sampled frames",
        )
        contact_sheets.append(
            {
                "analysis_tier": tier,
                "role": role,
                "n_stimuli": len(group),
                "path": str(path),
            }
        )

    failures = [row for row in rows if row["screening_flags"]]
    visual_failures = [
        row
        for row in rows
        if row.get("visual_gate") is None or row["visual_gate"]["passes_visual_gate"] is not True
    ]
    comparison_counts = Counter(task["comparison"] for task in manifest["tasks"])
    report = {
        "schema_version": "content_pocket_mp4_screening.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "agent_prelaunch_screening",
        "source_manifest": str(manifest_path),
        "source_task_payload_sha256": manifest["task_pool"]["task_payload_sha256"],
        "samples_per_video": samples,
        "n_stimuli": len(rows),
        "n_contact_sheets": len(contact_sheets),
        "n_screening_failures": len(failures),
        "n_visual_gate_failures": len(visual_failures),
        "comparison_counts": dict(sorted(comparison_counts.items())),
        "contact_sheets": contact_sheets,
        "agent_contact_sheet_review": agent_review_note,
        "rows": rows,
        "launch_blockers": [
            "This is an agent sampled-frame pre-screen, not final IRB/faculty sign-off.",
            "Stable HTTPS hosted video URLs are still required before launch.",
            "Participant-facing consent, compensation, and response collection remain open.",
            "Human/BMD validation has not run; all content-pocket claims remain proxy-selected.",
        ],
    }
    return report, render_markdown(report)


def render_markdown(report: dict[str, Any]) -> str:
    status = (
        "No automated screening failures were found."
        if report["n_screening_failures"] == 0
        else f"{report['n_screening_failures']} automated screening failures require review."
    )
    lines = [
        "# Content-Pocket Validation MP4 Screening",
        "",
        f"Date: {report['created_at_utc']}",
        "",
        "## Status",
        "",
        "Agent prelaunch sampled-frame screening only. This artifact checks frozen",
        "MP4 availability, byte hashes, sampled-frame visual stability, and contact",
        "sheet review readiness. It does not launch a study and does not validate",
        "human memorability or measured-BMD grounding.",
        "",
        f"Result: {status}",
        "",
        "## Summary",
        "",
        f"- Frozen task payload SHA-256: `{report['source_task_payload_sha256']}`",
        f"- Stimuli screened: {report['n_stimuli']}",
        f"- Sampled frames per video: {report['samples_per_video']}",
        f"- Contact sheets: {report['n_contact_sheets']}",
        f"- Automated screening failures: {report['n_screening_failures']}",
        f"- Visual-gate failures: {report['n_visual_gate_failures']}",
        f"- Agent contact-sheet review: {report['agent_contact_sheet_review']}",
        "",
        "## Task Counts",
        "",
        "| comparison | tasks |",
        "|---|---:|",
    ]
    for comparison, count in report["comparison_counts"].items():
        lines.append(f"| `{comparison}` | {count} |")

    lines.extend(
        [
            "",
            "## Contact Sheets",
            "",
            "| tier | role | stimuli | sheet |",
            "|---|---|---:|---|",
        ]
    )
    for sheet in report["contact_sheets"]:
        lines.append(
            "| {tier} | {role} | {n} | `{path}` |".format(
                tier=sheet["analysis_tier"],
                role=sheet["role"],
                n=sheet["n_stimuli"],
                path=sheet["path"],
            )
        )

    failures = [row for row in report["rows"] if row["screening_flags"]]
    lines.extend(["", "## Screening Flags", ""])
    if failures:
        lines.extend(["| pocket | label | flags |", "|---|---|---|"])
        for row in failures:
            lines.append(
                "| `{pocket}` | `{label}` | `{flags}` |".format(
                    pocket=row["pocket"],
                    label=row["label"],
                    flags=", ".join(row["screening_flags"]),
                )
            )
    else:
        lines.append("None from byte/hash/frame-gate screening.")

    lines.extend(
        [
            "",
            "## Launch Blockers",
            "",
            *[f"- {blocker}" for blocker in report["launch_blockers"]],
            "",
            "## Next Action",
            "",
            "Review the contact sheets and selected MP4s, fill the hosted-video URL",
            "map template for the frozen task JSON, and mark hosted videos screened",
            "only after final human/IRB-facing content review.",
            "",
        ]
    )
    return "\n".join(lines)


def build_url_map_template(report: dict[str, Any]) -> dict[str, Any]:
    rows_by_path = {row["local_video_path"]: row for row in report["rows"]}
    videos = []
    for local_path, row in sorted(rows_by_path.items()):
        videos.append(
            {
                "local_path": local_path,
                "source_absolute_path": row["source_absolute_path"],
                "sha256": row["actual_sha256"],
                "hosted_url": "",
                "agent_prescreened": row["screening_flags"] == [],
                "screened": False,
                "screening_notes": (
                    "Agent sampled-frame pre-screen passed; set screened=true "
                    "only after final human/IRB-facing content review."
                ),
                "analysis_tier": row["analysis_tier"],
                "role": row["role"],
                "pocket": row["pocket"],
            }
        )
    return {
        "schema_version": "content_pocket_hosted_video_url_map_template.v1",
        "prepared_for": "content_pocket_validation_prolific_survey_20260608.html",
        "source_manifest": report["source_manifest"],
        "source_task_payload_sha256": report["source_task_payload_sha256"],
        "instructions": [
            "Upload each source_absolute_path MP4 to stable HTTPS hosting.",
            "Fill hosted_url for every local_path.",
            "Keep screened=false until final human/IRB-facing content review is complete.",
            "After all entries have hosted_url and screened=true, replace local paths in the survey HTML.",
        ],
        "videos": videos,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--out-url-map", type=Path, default=DEFAULT_URL_MAP)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--agent-review-note",
        default=(
            "Contact sheets generated for review; final human/IRB-facing "
            "screening still required."
        ),
    )
    args = parser.parse_args()

    report, markdown = build_screening(
        manifest_path=args.manifest,
        sheet_dir=args.sheet_dir,
        samples=args.samples,
        agent_review_note=args.agent_review_note,
    )
    write_json(args.out_json, report)
    write_json(args.out_url_map, build_url_map_template(report))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] wrote {args.out_url_map}")
    print(f"[done] wrote {len(report['contact_sheets'])} contact sheets to {args.sheet_dir}")
    print(f"[done] screening failures: {report['n_screening_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
