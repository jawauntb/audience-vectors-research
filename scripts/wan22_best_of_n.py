"""Generate a Wan2.2 best-of-N set on Modal B200.

This is the fastest next experiment after SVD:

1. Generate N Wan2.2 variants from one prompt, optionally seeded by an image.
2. Save mp4s locally and in the Wan Modal output volume.
3. Score the outputs with the existing TRIBE pipeline in a separate step.

Examples:

    uv run python scripts/wan22_best_of_n.py \
      --prompt "a cinematic product shot of..." \
      --image data/seed.png \
      --n 8

    uv run python scripts/wan22_best_of_n.py \
      --task i2v-A14B --size '1280*720' \
      --prompt "..." --image data/seed.png --n 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_APP = "audience-vectors-dev"


def _read_image(path: Path | None) -> bytes | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-name", default=DEFAULT_APP)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/generated/wan22_best_of_n")
    )
    parser.add_argument(
        "--task", default="ti2v-5B", choices=["ti2v-5B", "i2v-A14B", "t2v-A14B"]
    )
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=240000)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum concurrent Modal generations. 0 means spawn all variants.",
    )
    parser.add_argument("--frame-num", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--sample-guide-scale", type=float)
    parser.add_argument("--sample-shift", type=float)
    parser.add_argument(
        "--offload-model",
        action="store_true",
        help="Allow Wan to offload model weights to CPU. Default keeps weights on B200.",
    )
    parser.add_argument("--label-prefix", default="wan22")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = _read_image(args.image)

    import modal  # noqa: PLC0415

    Generator = modal.Cls.from_name(args.app_name, "Wan22Generator")
    gen = Generator()

    max_in_flight = args.max_in_flight or args.n
    if max_in_flight < 1:
        raise ValueError("--max-in-flight must be at least 1")

    pending: list[tuple[int, int, str, Any]] = []

    def spawn_one(idx: int) -> None:
        seed = args.seed_base + idx
        label = f"{args.label_prefix}_{args.task.replace('-', '_')}_n{idx:02d}"
        fc = gen.generate.spawn(
            args.prompt,
            image_bytes=image_bytes,
            task=args.task,
            size=args.size,
            frame_num=args.frame_num,
            sample_steps=args.sample_steps,
            sample_guide_scale=args.sample_guide_scale,
            sample_shift=args.sample_shift,
            seed=seed,
            offload_model=args.offload_model,
            output_label=label,
        )
        pending.append((idx, seed, label, fc))

    manifest = []
    next_idx = 0
    while next_idx < args.n or pending:
        while next_idx < args.n and len(pending) < max_in_flight:
            spawn_one(next_idx)
            next_idx += 1
        print(
            f"[wan22] in-flight={len(pending)} completed={len(manifest)} "
            f"total={args.n} on {args.app_name}",
            flush=True,
        )

        idx, seed, label, fc = pending.pop(0)
        try:
            result = fc.get(timeout=4 * 60 * 60)
        except Exception as exc:  # noqa: BLE001
            print(f"  x {label}: {exc!r}", flush=True)
            manifest.append(
                {"idx": idx, "seed": seed, "label": label, "error": repr(exc)}
            )
            continue

        video_bytes = result.pop("video_bytes")
        local_path = args.out_dir / f"{label}.mp4"
        local_path.write_bytes(video_bytes)
        row = {
            "idx": idx,
            "seed": seed,
            "label": label,
            "local_path": str(local_path),
            **result,
        }
        manifest.append(row)
        print(f"  ok {label}: {len(video_bytes) / 1024 / 1024:.2f} MB", flush=True)

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[wan22] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
