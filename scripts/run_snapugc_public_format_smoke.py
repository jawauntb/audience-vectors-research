"""Smoke-test the public SnapUGC/VQualA CSV format without making claims."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from audit_attention_capture_retention_baselines import audit_retention_baselines
from audit_attention_capture_retention_labels import audit_retention_labels
from build_snapugc_retention_labels import build_snapugc_retention_labels

DEFAULT_METADATA_URL = (
    "https://raw.githubusercontent.com/dasongli1/SnapUGC_Engagement/main/"
    "ECR_inference/dataset/val_data_sample.csv"
)
DEFAULT_SCORES_URL = (
    "https://raw.githubusercontent.com/dasongli1/SnapUGC_Engagement/main/"
    "ECR_inference/submission_baseline.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv-or-url", default=DEFAULT_METADATA_URL)
    parser.add_argument("--scores-csv-or-url", default=DEFAULT_SCORES_URL)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--dataset", default="SnapUGC")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-distinct-ecr", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_public_format_smoke(
        metadata_csv_or_url=args.metadata_csv_or_url,
        scores_csv_or_url=args.scores_csv_or_url,
        dataset=args.dataset,
        min_samples=args.min_samples,
        min_distinct_ecr=args.min_distinct_ecr,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_public_format_smoke_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["smoke_passed"]:
        raise SystemExit(1)


def run_public_format_smoke(
    *,
    metadata_csv_or_url: str,
    scores_csv_or_url: str,
    dataset: str = "SnapUGC",
    min_samples: int = 10,
    min_distinct_ecr: int = 3,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="snapugc_public_format_smoke_") as tmp:
        tmpdir = Path(tmp)
        metadata_csv = materialize_csv(
            metadata_csv_or_url,
            tmpdir / "metadata.csv",
            fallback_name="metadata.csv",
        )
        scores_csv = materialize_csv(
            scores_csv_or_url,
            tmpdir / "submission_baseline.csv",
            fallback_name="scores.csv",
        )
        canonical_csv = tmpdir / "canonical.csv"
        builder = build_snapugc_retention_labels(
            metadata_csv=metadata_csv,
            scores_csv=scores_csv,
            output_csv=canonical_csv,
            dataset=dataset,
            allow_prediction_score_file=True,
            min_samples=min_samples,
            min_distinct_ecr=min_distinct_ecr,
        )
        label_audit = audit_retention_labels(
            labels_csv=canonical_csv,
            dataset=dataset,
            sample_id_column="sample_id",
            ground_truth_column="ecr",
            media_path_column="video_path",
            ground_truth_name="ecr",
            min_samples=min_samples,
            min_distinct_ground_truth=min_distinct_ecr,
        )
        baseline = audit_retention_baselines(
            labels_csv=canonical_csv,
            dataset=dataset,
            sample_id_column="sample_id",
            ground_truth_column="ecr",
            media_path_column="video_path",
            ground_truth_name="ecr",
            min_samples=min_samples,
            min_distinct_ground_truth=min_distinct_ecr,
            max_control_abs_rho=1.1,
        )
        return {
            "schema_version": 1,
            "experiment": "snapugc_public_format_smoke",
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "metadata_source": metadata_csv_or_url,
            "scores_source": scores_csv_or_url,
            "metadata_sha256": sha256(metadata_csv.read_bytes()).hexdigest(),
            "scores_sha256": sha256(scores_csv.read_bytes()).hexdigest(),
            "canonical_sha256": sha256(canonical_csv.read_bytes()).hexdigest(),
            "public_prediction_score_file": True,
            "claim_blocked": True,
            "claim_blocking_reason": (
                "The default scores source is a public baseline/prediction file, "
                "not granted behavioral ECR labels."
            ),
            "builder_ready": bool(builder["ready_for_retention_audit"]),
            "label_audit_ready": bool(label_audit["ready_for_manifest_alignment"]),
            "baseline_ready": bool(baseline["ready_for_modal_slice"]),
            "smoke_passed": bool(
                builder["ready_for_retention_audit"]
                and label_audit["ready_for_manifest_alignment"]
                and baseline["ready_for_modal_slice"]
            ),
            "counts": {
                "metadata_rows": builder["n_metadata_rows"],
                "score_rows": builder["n_score_rows"],
                "canonical_rows": builder["n_output_rows"],
                "label_audit_rows": label_audit["n_rows"],
                "baseline_feature_metrics": baseline["n_feature_metrics"],
            },
            "best_baseline_feature": baseline["best_feature"],
            "negative_controls": baseline["negative_controls"],
            "blocking_reasons": {
                "builder": builder["blocking_reasons"],
                "label_audit": label_audit["blocking_reasons"],
                "baseline": baseline["blocking_reasons"],
            },
            "warnings": baseline["warnings"],
            "next_actions": [
                "Do not use this smoke report for claims.",
                (
                    "Replace the scores source with granted behavioral ECR labels, "
                    "then rerun build_snapugc_retention_labels.py without "
                    "--allow-prediction-score-file."
                ),
                (
                    "If the real-label audits pass, run a small full-mode Modal "
                    "TRIBE slice from the canonical CSV."
                ),
            ],
        }


def materialize_csv(source: str, default_path: Path, *, fallback_name: str) -> Path:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        suffix = Path(parsed.path).name or fallback_name
        output_path = default_path.with_name(suffix)
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "audience-vectors/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            output_path.write_bytes(response.read())
        return output_path
    input_path = Path(source).expanduser().resolve()
    if input_path == default_path:
        return input_path
    shutil.copyfile(input_path, default_path)
    return default_path


def render_public_format_smoke_markdown(report: dict[str, Any]) -> str:
    best = report["best_baseline_feature"] or {}
    lines = [
        "# SnapUGC Public-Format Smoke",
        "",
        "## Verdict",
        "",
        f"- Smoke passed: {report['smoke_passed']}",
        f"- Claim blocked: {report['claim_blocked']}",
        f"- Claim blocking reason: {report['claim_blocking_reason']}",
        f"- Builder ready: {report['builder_ready']}",
        f"- Label audit ready: {report['label_audit_ready']}",
        f"- Baseline ready: {report['baseline_ready']}",
        f"- Canonical rows: {report['counts']['canonical_rows']}",
        f"- Best baseline feature: `{best.get('feature', 'n/a')}`",
        f"- Best baseline rho: {format_optional_float(best.get('rho'))}",
        "",
        "## Sources",
        "",
        f"- Metadata: {report['metadata_source']}",
        f"- Scores: {report['scores_source']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    any_blockers = False
    for section, reasons in blockers.items():
        for reason in reasons:
            any_blockers = True
            lines.append(f"- {section}: {reason}")
    if not any_blockers:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = report["warnings"]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append(
        "- none"
    )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in report["next_actions"])
    return "\n".join(lines) + "\n"


def format_optional_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
