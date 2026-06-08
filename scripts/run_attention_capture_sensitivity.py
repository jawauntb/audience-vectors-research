"""Run primary and sensitivity ROI-mask Phase 1 capture-score comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.attention_capture import (
    render_sensitivity_markdown,
    run_phase1_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-label", default="disjoint")
    parser.add_argument("--primary-roi-masks", type=Path, required=True)
    parser.add_argument(
        "--sensitivity-roi-masks",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional ROI masks to run as sensitivity checks.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-rho", type=float, default=0.40)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--include-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_phase1_sensitivity(
        args.manifest,
        primary_label=args.primary_label,
        primary_roi_masks_path=args.primary_roi_masks,
        sensitivity_roi_masks=parse_mask_specs(args.sensitivity_roi_masks),
        permutations=args.permutations,
        seed=args.seed,
        gate_rho=args.gate_rho,
        epsilon=args.epsilon,
        include_rows=args.include_rows,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_sensitivity_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def parse_mask_specs(values: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        specs[label] = Path(path)
    return specs


if __name__ == "__main__":
    main()
