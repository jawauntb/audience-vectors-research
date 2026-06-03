#!/usr/bin/env python3
"""F3/F4: Run the multi-objective BO loop.

Requires:
  - v_mem.npy and persona_directions.npy (from Jawaun)
  - tribe_clip_adapter.pt (from Jawaun)
  - GPU with CUDA or MPS
  - HuggingFace token for SVD-XT and TRIBE v2

Usage:
    uv run python scripts/run_bo_loop.py \\
        --image path/to/conditioning_frame.png \\
        --prompt "A chef preparing a meal in a kitchen" \\
        --n-iter 10 \\
        --batch-size 4
"""

import argparse
from pathlib import Path

from bo_mem.config import Config
from bo_mem.loop.bo_loop import BOLoop
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="Conditioning frame (PNG/JPG)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bo_run"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory of reference videos for FVD (defaults to SVD best-of-N)",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.n_iter:
        cfg.bo.n_iterations = args.n_iter
    if args.batch_size:
        cfg.bo.batch_size = args.batch_size

    ref_dir = args.reference_dir or cfg.paths.generated_svd_dir
    reference_paths = sorted(ref_dir.glob("*.mp4"))[:50]
    if not reference_paths:
        console.print("[yellow]No reference videos found; FVD constraint disabled.")

    loop = BOLoop(
        cfg=cfg,
        conditioning_image_path=args.image,
        prompt=args.prompt,
        reference_video_paths=reference_paths,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )

    console.rule("[bold green]Starting BO Loop")
    state = loop.run()
    console.print(f"\n[bold green]Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
