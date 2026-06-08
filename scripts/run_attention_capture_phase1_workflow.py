"""Run the guarded Phase 1 attention-capture workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.attention_capture import (
    render_phase1_workflow_markdown,
    run_phase1_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--primary-label", default="primary")
    parser.add_argument("--roi-masks", type=Path, default=None)
    parser.add_argument(
        "--sensitivity-roi-masks",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional sensitivity mask NPZ. May be passed multiple times.",
    )
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ground-truth", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-rho", type=float, default=0.40)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--omit-rows", action="store_true")
    parser.add_argument(
        "--score-claim-blocked",
        action="store_true",
        help=(
            "Allow synthetic/control manifests to be scored for diagnostics. "
            "Claim validation remains blocked."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_phase1_workflow(
        args.manifest,
        primary_label=args.primary_label,
        primary_roi_masks_path=args.roi_masks,
        sensitivity_roi_masks=parse_sensitivity_specs(args.sensitivity_roi_masks),
        min_samples=args.min_samples,
        min_distinct_ground_truth=args.min_distinct_ground_truth,
        permutations=args.permutations,
        seed=args.seed,
        gate_rho=args.gate_rho,
        epsilon=args.epsilon,
        include_rows=not args.omit_rows,
        score_claim_blocked=args.score_claim_blocked,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_phase1_workflow_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["score_decision"]["scoring_executed"]:
        raise SystemExit(1)


def parse_sensitivity_specs(raw_specs: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"expected LABEL=PATH, got {raw_spec!r}")
        label, raw_path = raw_spec.split("=", 1)
        if not label:
            raise ValueError(f"sensitivity label cannot be empty: {raw_spec!r}")
        specs[label] = Path(raw_path)
    return specs


if __name__ == "__main__":
    main()
