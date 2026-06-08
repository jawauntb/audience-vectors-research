"""Build a recognition-memory validation packet for content pockets.

The existing frozen content-pocket survey tests perceived memorability. This
builder creates the next, stronger design: sparse exposure plus old-vs-lure
recognition trials, with new same-category lures required before launch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_STIMULUS_MANIFEST = (
    ARTIFACT_DIR / "content_pocket_validation_stimuli_manifest_20260608.json"
)
DEFAULT_OUT_JSON = ARTIFACT_DIR / "content_pocket_recognition_memory_design_20260608.json"
DEFAULT_OUT_MD = ARTIFACT_DIR / "content_pocket_recognition_memory_packet_20260608.md"
DEFAULT_LURE_SEED_DIR = ARTIFACT_DIR / "recognition_lure_seed_requests_20260608"


@dataclass(frozen=True)
class TargetArm:
    """One sparse-exposure analysis arm."""

    arm_id: str
    pocket: str
    source_tier: str
    source_role: str
    analysis_group: str
    old_target_sort_descending: bool
    lure_prompt: str
    lure_requirements: tuple[str, ...]


TARGET_ARMS: tuple[TargetArm, ...] = (
    TargetArm(
        arm_id="orange_flowers",
        pocket="fresh24_orange_flowers",
        source_tier="primary",
        source_role="candidate",
        analysis_group="primary_positive",
        old_target_sort_descending=True,
        lure_prompt=(
            "A different cluster of orange flowers in a wider garden setting, "
            "with a distinct camera angle, background, and petal layout. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        lure_requirements=(
            "same broad category as orange flowers",
            "not a close-up near-duplicate of the frozen target clip",
            "different flower arrangement, background, and camera angle",
            "orange remains visible but color layout differs from old target",
        ),
    ),
    TargetArm(
        arm_id="hanging_clothes",
        pocket="fresh24_hanging_clothes",
        source_tier="primary",
        source_role="candidate",
        analysis_group="primary_positive",
        old_target_sort_descending=True,
        lure_prompt=(
            "Colorful clothes hang outdoors on a clothesline in a different "
            "composition, with fabric shifting gently in natural light. Natural "
            "realistic short video, clear central subject, continuous motion, "
            "stable composition, no text, no watermark."
        ),
        lure_requirements=(
            "same broad category as hanging clothes",
            "not a rack-centered near-duplicate of the frozen target clip",
            "different setting, hanger/clothesline geometry, and camera angle",
            "fabric colors differ enough to support exact old-vs-lure memory",
        ),
    ),
    TargetArm(
        arm_id="aerial_beach",
        pocket="fresh24_aerial_beach",
        source_tier="primary",
        source_role="control",
        analysis_group="hard_negative_control",
        old_target_sort_descending=False,
        lure_prompt=(
            "A different wide aerial beach with shallow shoreline water and tiny "
            "figures, using a distinct coastline shape and camera framing. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        lure_requirements=(
            "same broad category as aerial beach",
            "different shoreline geometry and beach layout from old target",
            "avoid near-identical haze/color banding",
        ),
    ),
    TargetArm(
        arm_id="city_street",
        pocket="fresh24_city_street",
        source_tier="primary",
        source_role="control",
        analysis_group="hard_negative_control",
        old_target_sort_descending=False,
        lure_prompt=(
            "A different city street between tall buildings with trees and "
            "distant motion, using a distinct street angle and building layout. "
            "Natural realistic short video, clear central subject, continuous "
            "motion, stable composition, no text, no watermark."
        ),
        lure_requirements=(
            "same broad category as city street",
            "different street geometry, building facades, and tree placement",
            "avoid near-identical centerline perspective",
        ),
    ),
    TargetArm(
        arm_id="storm_beach",
        pocket="fresh24_storm_beach",
        source_tier="primary",
        source_role="control",
        analysis_group="hard_negative_control",
        old_target_sort_descending=False,
        lure_prompt=(
            "A different stormy beach under heavy clouds with waves and cliffs, "
            "using a distinct shoreline and cliff arrangement. Natural realistic "
            "short video, clear central subject, continuous motion, stable "
            "composition, no text, no watermark."
        ),
        lure_requirements=(
            "same broad category as storm beach",
            "different cliffs, wave line, and horizon composition",
            "avoid near-identical dark shoreline framing",
        ),
    ),
)

FILLER_SEED_POCKETS = (
    "fresh24_golden_grass",
    "fresh24_red_mailbox",
    "fresh24_suspension_bridge",
    "fresh24_lighthouse",
    "fresh24_coastal_tracks",
    "fresh24_ocean_cliffs",
    "fresh24_concert_stage",
    "fresh24_misty_woods",
    "fresh24_wheat_closeup",
    "fresh24_cloud_mountain",
    "fresh24_mountain_fog",
    "fresh24_forest_canopy",
    "fresh24_dewy_grass",
    "fresh24_sparse_forest",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_report_row(stimulus: dict[str, Any]) -> dict[str, Any]:
    source_report = stimulus.get("source_report")
    if not source_report:
        return {
            "lookup_status": "missing_source_report_field",
            "alpha": None,
            "guidance": None,
            "prompt": None,
            "source_noise_seed": None,
        }
    report_path = Path(str(source_report))
    if not report_path.exists():
        return {
            "lookup_status": "source_report_not_available",
            "alpha": None,
            "guidance": None,
            "prompt": None,
            "source_noise_seed": None,
        }
    report = load_json(report_path)
    label = stimulus["label"]
    for row in report.get("rows", []):
        if row.get("label") != label:
            continue
        trial = row.get("trial") or {}
        return {
            "lookup_status": "found",
            "alpha": trial.get("alpha"),
            "guidance": trial.get("guidance"),
            "prompt": trial.get("prompt"),
            "source_noise_seed": row.get("noise_seed"),
        }
    return {
        "lookup_status": "source_report_row_not_found",
        "alpha": None,
        "guidance": None,
        "prompt": None,
        "source_noise_seed": None,
    }


def target_record(stimulus: dict[str, Any], *, arm: TargetArm, variant_index: int) -> dict[str, Any]:
    source_trial = load_source_report_row(stimulus)
    return {
        "target_id": f"{arm.arm_id}_old_v{variant_index:02d}",
        "arm_id": arm.arm_id,
        "analysis_group": arm.analysis_group,
        "pocket": arm.pocket,
        "variant_index": variant_index,
        "old_video": {
            "label": stimulus["label"],
            "local_video_path": stimulus["local_video_path"],
            "source_absolute_path": stimulus["source_absolute_path"],
            "sha256": stimulus["sha256"],
            "recipe_index": stimulus["recipe_index"],
            "replicate_index": stimulus["replicate_index"],
            "replay_tribe_score": stimulus["replay_tribe_score"],
            "source_report": stimulus.get("source_report"),
            "source_trial": source_trial,
        },
    }


def select_old_targets(
    manifest: dict[str, Any],
    *,
    variants_per_arm: int,
) -> dict[str, list[dict[str, Any]]]:
    targets_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in TARGET_ARMS:
        candidates = [
            stimulus
            for stimulus in manifest["stimuli"]
            if stimulus["analysis_tier"] == arm.source_tier
            and stimulus["role"] == arm.source_role
            and stimulus["pocket"] == arm.pocket
        ]
        candidates.sort(
            key=lambda stimulus: (
                float(stimulus["replay_tribe_score"]),
                -int(stimulus["recipe_index"]),
                -int(stimulus["replicate_index"]),
            ),
            reverse=arm.old_target_sort_descending,
        )
        if len(candidates) < variants_per_arm:
            raise ValueError(
                f"{arm.arm_id} has only {len(candidates)} old targets; "
                f"need {variants_per_arm}"
            )
        targets_by_arm[arm.arm_id] = [
            target_record(stimulus, arm=arm, variant_index=index)
            for index, stimulus in enumerate(candidates[:variants_per_arm])
        ]
    return targets_by_arm


def lure_request(
    *,
    arm: TargetArm,
    target: dict[str, Any],
    lure_seed_dir: Path,
    noise_seed_base: int,
) -> dict[str, Any]:
    variant_index = int(target["variant_index"])
    old_video = target["old_video"]
    lure_id = f"{arm.arm_id}_lure_v{variant_index:02d}"
    matched_recipe_index = int(old_video["recipe_index"])
    return {
        "lure_id": lure_id,
        "target_id": target["target_id"],
        "arm_id": arm.arm_id,
        "analysis_group": arm.analysis_group,
        "pocket": arm.pocket,
        "seed_image_required_path": str(lure_seed_dir / f"{lure_id}.png"),
        "seed_image_status": "required_not_committed",
        "prompt": arm.lure_prompt,
        "distinctiveness_requirements": list(arm.lure_requirements),
        "matched_old_video": old_video,
        "generation_request": {
            "generator": "current image-conditioned SVD runner",
            "matched_recipe_index": matched_recipe_index,
            "alpha": old_video["source_trial"]["alpha"],
            "guidance": old_video["source_trial"]["guidance"],
            "suggested_noise_seed": noise_seed_base + matched_recipe_index * 10 + variant_index,
            "must_screen_before_use": True,
            "must_not_optimize_lure_for_memorability": True,
        },
    }


def build_lure_requests(
    targets_by_arm: dict[str, list[dict[str, Any]]],
    *,
    lure_seed_dir: Path,
    noise_seed_base: int,
) -> list[dict[str, Any]]:
    arms_by_id = {arm.arm_id: arm for arm in TARGET_ARMS}
    requests: list[dict[str, Any]] = []
    for arm_id, targets in targets_by_arm.items():
        arm = arms_by_id[arm_id]
        requests.extend(
            lure_request(
                arm=arm,
                target=target,
                lure_seed_dir=lure_seed_dir,
                noise_seed_base=noise_seed_base,
            )
            for target in targets
        )
    return requests


def build_forms(
    targets_by_arm: dict[str, list[dict[str, Any]]],
    lure_requests: list[dict[str, Any]],
    *,
    n_forms: int,
    session1_filler_count: int,
    session2_filler_recognition_trials: int,
) -> list[dict[str, Any]]:
    lures_by_target = {request["target_id"]: request for request in lure_requests}
    forms: list[dict[str, Any]] = []
    n_variants = min(len(targets) for targets in targets_by_arm.values())
    for form_index in range(n_forms):
        encoding_targets = []
        recognition_trials = []
        for arm_index, arm in enumerate(TARGET_ARMS):
            target = targets_by_arm[arm.arm_id][(form_index + arm_index) % n_variants]
            lure = lures_by_target[target["target_id"]]
            old_side = "left" if (form_index + arm_index) % 2 == 0 else "right"
            encoding_targets.append(
                {
                    "arm_id": arm.arm_id,
                    "analysis_group": arm.analysis_group,
                    "target_id": target["target_id"],
                    "old_video_path": target["old_video"]["local_video_path"],
                }
            )
            recognition_trials.append(
                {
                    "trial_id": f"form{form_index:02d}_{target['target_id']}_recognition",
                    "arm_id": arm.arm_id,
                    "analysis_group": arm.analysis_group,
                    "target_id": target["target_id"],
                    "old_video_path": target["old_video"]["local_video_path"],
                    "lure_id": lure["lure_id"],
                    "lure_seed_required_path": lure["seed_image_required_path"],
                    "old_side": old_side,
                    "correct_choice": old_side,
                    "question": "Which clip did you see in the first part?",
                }
            )
        forms.append(
            {
                "form_id": f"recognition_form_{form_index:02d}",
                "assignment": "assign by participant-id hash modulo n_forms",
                "session1": {
                    "analysis_encoding_targets": encoding_targets,
                    "required_unrelated_filler_targets": session1_filler_count,
                    "cover_task": "rate visual clarity or pleasantness after each clip",
                },
                "session2": {
                    "analysis_recognition_trials": recognition_trials,
                    "required_unrelated_filler_recognition_trials": session2_filler_recognition_trials,
                    "delay": "24-48 hours preferred; 10-20 minute short-term variant is weaker",
                },
            }
        )
    return forms


def build_design(
    *,
    stimulus_manifest_path: Path,
    variants_per_arm: int,
    n_forms: int,
    session1_filler_count: int,
    session2_filler_recognition_trials: int,
    lure_seed_dir: Path,
    noise_seed_base: int,
) -> tuple[dict[str, Any], str]:
    manifest = load_json(stimulus_manifest_path)
    targets_by_arm = select_old_targets(manifest, variants_per_arm=variants_per_arm)
    lure_requests = build_lure_requests(
        targets_by_arm,
        lure_seed_dir=lure_seed_dir,
        noise_seed_base=noise_seed_base,
    )
    forms = build_forms(
        targets_by_arm,
        lure_requests,
        n_forms=n_forms,
        session1_filler_count=session1_filler_count,
        session2_filler_recognition_trials=session2_filler_recognition_trials,
    )
    analysis_counts = Counter(arm.analysis_group for arm in TARGET_ARMS)
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    design = {
        "schema_version": "content_pocket_recognition_memory_design.v1",
        "created_at_utc": now,
        "status": "design_not_launchable_until_lures_generated_screened_and_hosted",
        "source_stimulus_manifest": str(stimulus_manifest_path),
        "source_task_payload_sha256": manifest["task_pool"]["task_payload_sha256"],
        "question": (
            "Do humans actually recognize primary content-pocket clips better "
            "than matched hard-negative controls?"
        ),
        "current_regime_reading": (
            "This upgrades the validation endpoint from perceived memorability "
            "preference to old-vs-lure recognition memory. It still remains "
            "unlaunched until new lures and fillers are generated and screened."
        ),
        "target_arms": [
            {
                "arm_id": arm.arm_id,
                "pocket": arm.pocket,
                "analysis_group": arm.analysis_group,
                "source_tier": arm.source_tier,
                "source_role": arm.source_role,
            }
            for arm in TARGET_ARMS
        ],
        "old_targets_by_arm": targets_by_arm,
        "lure_generation_requests": lure_requests,
        "filler_requirements": {
            "session1_unrelated_filler_targets_per_participant": session1_filler_count,
            "session2_unrelated_filler_recognition_trials_per_participant": session2_filler_recognition_trials,
            "candidate_filler_seed_pockets": list(FILLER_SEED_POCKETS),
            "rule": (
                "Fillers must be unrelated to primary and hard-negative analysis "
                "arms and must not be reused as analysis lures."
            ),
        },
        "session_forms": forms,
        "participant_assignment": {
            "method": "hash Prolific participant ID modulo n_forms",
            "form_count": n_forms,
            "requirements": [
                "Persist the assigned form_id from Session 1 to Session 2.",
                "Do not let a participant complete more than one form.",
                "Keep all randomization logs, including side assignment and trial order.",
            ],
        },
        "response_collection_schema": {
            "session1_fields": [
                "participant_id",
                "form_id",
                "trial_id",
                "target_id",
                "arm_id",
                "analysis_group",
                "video_url",
                "cover_task_rating",
                "exposure_completed",
                "started_at",
                "completed_at",
            ],
            "session2_fields": [
                "participant_id",
                "form_id",
                "trial_id",
                "target_id",
                "arm_id",
                "analysis_group",
                "old_video_url",
                "lure_video_url",
                "old_side",
                "choice_side",
                "correct_choice",
                "is_correct",
                "response_time_ms",
                "started_at",
                "completed_at",
            ],
        },
        "exclusion_rules": [
            "Exclude participants who do not complete both sessions with a matched participant_id.",
            "Exclude participants with incomplete old-target exposure in Session 1.",
            "Exclude trials where either old or lure MP4 fails to load.",
            "Pre-register response-time and attention-check exclusions before launch.",
            "Keep excluded rows in the exported dataset with exclusion reasons.",
        ],
        "prolific_setup_requirements": {
            "study_type": "two-session delayed recognition memory study",
            "session_gap": "24-48 hours preferred",
            "technical_dry_run": "allowed for plumbing only; exclude dry-run rows from the evidence gate",
            "evidence_gate_sample": (
                "target 300 usable Session 2 participants; minimum 200 usable "
                "Session 2 participants before interpreting the primary gate"
            ),
        },
        "sample_size_plan": {
            "recommended_session1_slots": 350,
            "target_session2_usable": 300,
            "minimum_session2_usable": 200,
            "rationale": (
                "Two-session attrition is expected; 300 usable delayed "
                "participants gives 600 primary-positive and 900 control "
                "recognition outcomes before item-level modeling."
            ),
        },
        "primary_gate": {
            "endpoint": "old-vs-lure 2AFC recognition accuracy",
            "chance": 0.5,
            "primary_success": [
                "pooled primary_positive recognition accuracy exceeds hard_negative_control accuracy",
                "fresh24_orange_flowers effect direction is positive",
                "fresh24_hanging_clothes effect direction is positive",
                "same-category lure false familiarity does not collapse accuracy to chance",
            ],
            "recommended_model": (
                "mixed-effects logistic regression with fixed effect for "
                "analysis_group and random intercepts for participant and target_id"
            ),
            "fallback_tests": [
                "participant-level paired contrast between primary_positive and hard_negative_control accuracy",
                "item-level bootstrap over target_id and participant",
            ],
        },
        "withheld_or_rejected_rules": [
            "Do not launch if lure seed images are near-duplicates of old targets.",
            "Do not launch if generated lures fail visual screening or contain text/watermark artifacts.",
            "Do not pool exploratory boundary arms into the primary gate.",
            "Do not claim actual human memorability until the recognition gate clears.",
        ],
        "analysis_counts_per_form": dict(sorted(analysis_counts.items())),
        "launch_blockers": [
            "Need distinct same-category lure seed images for every old target variant.",
            "Need generated lure MP4s from those seed images.",
            "Need unrelated filler targets and filler lures.",
            "Need MP4 screening/contact sheets for generated lures and fillers.",
            "Need hosted HTTPS URLs and Prolific two-session setup.",
        ],
    }
    return design, render_markdown(design)


def render_markdown(design: dict[str, Any]) -> str:
    lines = [
        "# Content-Pocket Recognition-Memory Validation Packet",
        "",
        f"Date: {design['created_at_utc']}",
        "",
        "## Purpose",
        "",
        "Build the direct human-memory validation study for the accepted",
        "content-pocket candidates. Unlike the current forced-choice survey, this",
        "design asks whether participants can later recognize the exact clip they",
        "saw, against a newly generated same-category lure.",
        "",
        "This packet is not launchable yet. It defines the old targets, the lure",
        "generation requirements, the sparse-exposure form structure, and the",
        "pre-registered success gate.",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: do primary SVD content pockets produce actual human recognition",
        "memory gains, not just perceived memorability preferences?",
        "",
        "Current regime:",
        "",
        "- Artifact types: frozen old MP4s, pocket labels, TRIBE/V-JEPA verifier",
        "  status, generated lure requests, recognition form templates, hosted",
        "  survey tasks, recognition responses.",
        "- Operations: sparse exposure, same-category lure generation, visual",
        "  screening, delayed old-vs-lure recognition, mixed-effects analysis.",
        "- Gates/verifiers: old-target freeze integrity, lure distinctiveness, visual",
        "  screening, 2AFC recognition accuracy, primary-pocket positive direction.",
        "- Known limitation: no human/BMD recognition result has run yet.",
        "",
        "Action class: discovery-transition design. The validation endpoint changes",
        "from a preference readout to an actual memory-behavior readout.",
        "",
        "## Design Summary",
        "",
        "- Session 1: each participant sees one old target from each analysis arm,",
        "  plus unrelated fillers, and performs a light cover task.",
        "- Session 2: 24-48 hours later, each analysis trial shows the old target",
        "  against a newly generated same-category lure.",
        "- Sparse exposure rule: no participant sees more than one old clip from the",
        "  same analysis arm.",
            "- Primary endpoint: old-vs-lure 2AFC recognition accuracy.",
            "- Participant assignment: hash the Prolific participant ID to one",
            "  of the six sparse forms and persist that form into Session 2.",
            "",
            "## Analysis Arms",
            "",
        "| arm | pocket | group | source |",
        "|---|---|---|---|",
    ]
    for arm in design["target_arms"]:
        lines.append(
            "| `{arm_id}` | `{pocket}` | `{group}` | {tier}/{role} |".format(
                arm_id=arm["arm_id"],
                pocket=arm["pocket"],
                group=arm["analysis_group"],
                tier=arm["source_tier"],
                role=arm["source_role"],
            )
        )

    lines.extend(
        [
            "",
            "## Old Target Variants",
            "",
            "| arm | variants | selected labels |",
            "|---|---:|---|",
        ]
    )
    for arm_id, targets in design["old_targets_by_arm"].items():
        labels = ", ".join(f"`{target['old_video']['label']}`" for target in targets)
        lines.append(f"| `{arm_id}` | {len(targets)} | {labels} |")

    lines.extend(
        [
            "",
            "## Lure Generation",
            "",
            f"Required same-category lures: {len(design['lure_generation_requests'])}",
            "",
            "Lure seed images must be visually distinct from the frozen old target",
            "clip while preserving the broad category. Prompt rewrites alone are not",
            "sufficient in the current image-conditioned SVD path.",
            "",
            "## Sample Size",
            "",
            f"- Recommended Session 1 slots: {design['sample_size_plan']['recommended_session1_slots']}",
            f"- Target Session 2 usable participants: {design['sample_size_plan']['target_session2_usable']}",
            f"- Minimum Session 2 usable participants: {design['sample_size_plan']['minimum_session2_usable']}",
            "- Small dry runs are only for plumbing and must be excluded from",
            "  the evidence gate.",
            "",
            "## Response Capture",
            "",
            "Session 1 rows must retain participant ID, form ID, target ID, arm,",
            "analysis group, video URL, cover-task rating, exposure-completion",
            "status, and timestamps.",
            "",
            "Session 2 rows must retain participant ID, form ID, target ID, arm,",
            "analysis group, old/lure URLs, old side, choice side, correctness,",
            "response time, and timestamps.",
            "",
            "Excluded participants and failed-load trials must remain in the",
            "exported dataset with explicit exclusion reasons.",
            "",
            "## Primary Gate",
            "",
        ]
    )
    for rule in design["primary_gate"]["primary_success"]:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Launch Blockers",
            "",
            *[f"- {blocker}" for blocker in design["launch_blockers"]],
            "",
            "## Next Action",
            "",
            "Acquire or generate the required distinct lure seed images, generate lure",
            "and filler MP4s under matched SVD settings, screen them, then freeze the",
            "complete recognition stimulus set before creating the two-session",
            "Prolific study.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimulus-manifest", type=Path, default=DEFAULT_STIMULUS_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--lure-seed-dir", type=Path, default=DEFAULT_LURE_SEED_DIR)
    parser.add_argument("--variants-per-arm", type=int, default=3)
    parser.add_argument("--n-forms", type=int, default=6)
    parser.add_argument("--session1-filler-count", type=int, default=25)
    parser.add_argument("--session2-filler-recognition-trials", type=int, default=20)
    parser.add_argument("--noise-seed-base", type=int, default=760000)
    args = parser.parse_args()

    design, markdown = build_design(
        stimulus_manifest_path=args.stimulus_manifest,
        variants_per_arm=args.variants_per_arm,
        n_forms=args.n_forms,
        session1_filler_count=args.session1_filler_count,
        session2_filler_recognition_trials=args.session2_filler_recognition_trials,
        lure_seed_dir=args.lure_seed_dir,
        noise_seed_base=args.noise_seed_base,
    )
    write_json(args.out_json, design)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] old target variants: {sum(len(v) for v in design['old_targets_by_arm'].values())}")
    print(f"[done] lure requests: {len(design['lure_generation_requests'])}")
    print(f"[done] session forms: {len(design['session_forms'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
