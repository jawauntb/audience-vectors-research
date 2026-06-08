"""Run Phase 1 attention-capture validation from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.attention_capture import (
    render_phase1_markdown,
    run_phase1_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TRIBE ROI capture-score Phase 1 dry run.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--roi-masks", type=Path, default=None)
    parser.add_argument(
        "--omit-rows",
        action="store_true",
        help="Do not include per-sample rows in the JSON report.",
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-rho", type=float, default=0.40)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_phase1_manifest(
        args.manifest,
        roi_masks_path=args.roi_masks,
        permutations=args.permutations,
        seed=args.seed,
        gate_rho=args.gate_rho,
        epsilon=args.epsilon,
        include_rows=not args.omit_rows,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_phase1_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
