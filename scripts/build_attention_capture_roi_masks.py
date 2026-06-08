"""Freeze exploratory Destrieux ROI masks for attention-capture dry runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audience_vectors.attention_capture import (
    CONTROL_ROI,
    SENSORY_ROIS,
    load_destrieux_roi_selection,
    render_roi_mask_audit_markdown,
    roi_mask_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build auditable Destrieux ROI masks for capture-score runs.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("allow", "drop_shared"),
        default="allow",
        help=(
            "How to handle vertices selected by more than one ROI. "
            "drop_shared removes shared vertices from every ROI."
        ),
    )
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = load_destrieux_roi_selection(overlap_policy=args.overlap_policy)
    audit = roi_mask_audit(selection)
    audit["mask_npz_path"] = str(args.output_npz)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output_npz,
        V1=selection.masks[SENSORY_ROIS[0]],
        PPA=selection.masks[SENSORY_ROIS[1]],
        language=selection.masks[SENSORY_ROIS[2]],
        frontoparietal=selection.masks[CONTROL_ROI],
    )
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    args.output_md.write_text(render_roi_mask_audit_markdown(audit), encoding="utf-8")
    print(f"wrote {args.output_npz}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
