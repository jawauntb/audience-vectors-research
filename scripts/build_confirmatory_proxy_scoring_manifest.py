"""Build the proxy-scoring intake manifest for confirmatory Seedance candidates.

The visual gate freezes which exact MP4 bytes are eligible for TRIBE/BMD,
V-JEPA, and CLIP scoring. This script converts that visual-screening report into
a compact scoring manifest with one row per candidate, stable paths and hashes,
and explicit empty score fields. It does not run Modal or select study stimuli.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_VISUAL_REPORT = (
    EXPERIMENT_DIR / "seedance_candidate_visual_screening_improved_v1_20260615.json"
)
DEFAULT_OUT_JSON = (
    EXPERIMENT_DIR / "seedance_candidate_proxy_scoring_manifest_improved_v1_20260615.json"
)
DEFAULT_OUT_MD = (
    EXPERIMENT_DIR / "seedance_candidate_proxy_scoring_manifest_improved_v1_20260615.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-report", type=Path, default=DEFAULT_VISUAL_REPORT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_slots() -> dict[str, Any]:
    """Return explicit null score slots for later scorer outputs."""
    return {
        "tribe_bmd_projection": None,
        "vjepa_feature_path": None,
        "vjepa_centroid_margin": None,
        "clip_video_feature_path": None,
        "clip_seed_video_preservation": None,
        "saliency_or_quality_optional": None,
        "composite_proxy_score": None,
        "selection_role": None,
    }


def candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "family_id": row["family_id"],
        "candidate_index": row["candidate_index"],
        "prior_role": row.get("prior_role"),
        "prompt_set": row.get("prompt_set"),
        "variant_direction": row.get("variant_direction"),
        "prompt": row.get("prompt"),
        "manifest_output_path": row["manifest_output_path"],
        "source_absolute_path": row["source_absolute_path"],
        "sha256": row["sha256"],
        "duration_seconds": row["duration_seconds"],
        "frame_count": row["frame_count"],
        "automated_visual_gate_passed": row["automated_visual_gate_passed"],
        "manual_review_required": row["manual_review_required"],
        "screening_flags": row["screening_flags"],
        "hard_screening_flags": row["hard_screening_flags"],
        "visual_quality_proxy": row["visual_quality_proxy"],
        "descriptors": row["descriptors"],
        "proxy_scores": score_slots(),
    }


def build_family_queue(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_family[row["family_id"]].append(row)

    queue = []
    for family_id, family_rows in sorted(by_family.items()):
        eligible = [
            row
            for row in family_rows
            if row["automated_visual_gate_passed"] and not row["hard_screening_flags"]
        ]
        ranked = sorted(
            eligible,
            key=lambda row: (
                row["manual_review_required"],
                -float(row["visual_quality_proxy"] or -999.0),
                int(row["candidate_index"]),
            ),
        )
        queue.append(
            {
                "family_id": family_id,
                "n_candidates": len(family_rows),
                "n_score_eligible": len(eligible),
                "n_manual_review_required": sum(
                    1 for row in family_rows if row["manual_review_required"]
                ),
                "status": (
                    "ready_for_proxy_scoring"
                    if len(eligible) >= 2
                    else "blocked_insufficient_visual_candidates"
                ),
                "recommended_scoring_order": [row["job_id"] for row in ranked],
            }
        )
    return queue


def build_manifest(visual_report: dict[str, Any], visual_report_path: Path) -> dict[str, Any]:
    candidates = [
        candidate_row(row)
        for row in visual_report["rows"]
        if row["automated_visual_gate_passed"] and not row["hard_screening_flags"]
    ]
    flags = Counter(flag for row in candidates for flag in row["screening_flags"])
    families = Counter(row["family_id"] for row in candidates)
    family_queue = build_family_queue(candidates)
    return {
        "schema_version": "confirmatory_seedance_proxy_scoring_manifest.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": (
            "ready_for_proxy_scoring"
            if all(family["status"] == "ready_for_proxy_scoring" for family in family_queue)
            else "blocked_before_proxy_scoring"
        ),
        "source_visual_report": str(visual_report_path),
        "visual_report_status": visual_report["status"],
        "summary": {
            "n_candidates": len(candidates),
            "n_families": len(families),
            "n_manual_review_required": sum(
                1 for row in candidates if row["manual_review_required"]
            ),
            "screening_flags": dict(sorted(flags.items())),
            "families": dict(sorted(families.items())),
        },
        "score_contract": {
            "required_before_selection": [
                "tribe_bmd_projection",
                "vjepa_centroid_margin",
                "clip_seed_video_preservation",
            ],
            "optional_before_selection": ["saliency_or_quality_optional"],
            "selection_rule": (
                "Within each family, select one selector_top and one "
                "quality_matched_control only after required proxy scores and "
                "manual text/content review are complete."
            ),
            "claim_boundary": (
                "Proxy scoring can rank generated candidates, but it is not "
                "human memorability evidence until the delayed-recognition "
                "study clears its preregistered gate."
            ),
        },
        "next_scoring_steps": [
            "Run TRIBE/BMD projection on each exact source_absolute_path MP4.",
            "Extract exact V-JEPA embeddings for each exact MP4 and compute prospective centroid margins.",
            "Extract CLIP video/seed-image preservation features for each exact MP4.",
            "Attach scores back to this manifest without changing candidate hashes.",
            "Freeze selector/control roles per family only after manual review and proxy scores are complete.",
        ],
        "family_scoring_queue": family_queue,
        "candidates": sorted(
            candidates,
            key=lambda row: (str(row["family_id"]), int(row["candidate_index"])),
        ),
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Improved-v1 Seedance Proxy-Scoring Manifest",
        "",
        f"Date: `{manifest['created_at_utc']}`",
        f"Status: `{manifest['status']}`",
        "",
        "## Purpose",
        "",
        "This freezes the exact visually eligible MP4 byte targets for TRIBE/BMD, V-JEPA, and CLIP scoring. It is not a selection result and not human memorability evidence.",
        "",
        "## Summary",
        "",
        f"- Candidate MP4s queued: `{manifest['summary']['n_candidates']}`",
        f"- Families queued: `{manifest['summary']['n_families']}`",
        f"- Manual review cues retained: `{manifest['summary']['n_manual_review_required']}`",
        f"- Source visual report: `{manifest['source_visual_report']}`",
        "",
        "## Required Score Contract",
        "",
    ]
    for name in manifest["score_contract"]["required_before_selection"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Family Queue", "", "| family | queued | manual review | status | head |", "|---|---:|---:|---|---|"])
    for family in manifest["family_scoring_queue"]:
        head = ", ".join(family["recommended_scoring_order"][:3])
        lines.append(
            "| `{family_id}` | {queued} | {manual} | `{status}` | {head} |".format(
                family_id=family["family_id"],
                queued=family["n_score_eligible"],
                manual=family["n_manual_review_required"],
                status=family["status"],
                head=head,
            )
        )
    lines.extend(["", "## Screening Flags Retained", "", "| flag | count |", "|---|---:|"])
    for flag, count in manifest["summary"]["screening_flags"].items():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(["", "## Next Scoring Steps", ""])
    lines.extend(f"- {step}" for step in manifest["next_scoring_steps"])
    lines.extend(["", "## Claim Boundary", "", manifest["score_contract"]["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    visual_report = load_json(args.visual_report)
    manifest = build_manifest(visual_report, args.visual_report)
    write_json(args.out_json, manifest)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "n_candidates": manifest["summary"]["n_candidates"],
                "n_families": manifest["summary"]["n_families"],
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
