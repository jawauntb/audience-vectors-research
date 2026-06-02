"""Conditioning-space steering sweep on CogVideoX-5B via Modal.

For each base prompt × α ∈ {-2, -1, 0, +1, +2}:
  1. Call CogVideoXGenerator.generate(prompt, steering_vector=v_mem_t5, alpha=α)
  2. Save the returned mp4 to data/generated/cogvideox/<prompt>_a<α>.mp4
Then push every generated clip through TRIBE, project onto v_mem (TRIBE space),
and verify the projection scales monotonically with α.

Runs in parallel across Modal containers (configured max_containers=20).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


PROMPTS = [
    "A chef preparing a meal in a busy kitchen.",
    "A small animal eating fruit from a person's hand.",
    "Construction workers building a wooden frame.",
    "A person sewing fabric on a domestic sewing machine.",
    "A scientist running an experiment with glassware.",
    "Children playing on a playground.",
    "A potter shaping clay on a wheel.",
    "An artisan making pottery by hand.",
    "Machinery operating in an industrial setting.",
    "A man typing at a desk in an office.",
]
ALPHAS = [-2.0, -1.0, 0.0, +1.0, +2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/cogvideox"))
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--direction", default="v_mem_t5_native",
                        choices=["v_mem_t5_native", "v_mem_t5_via_adapter"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load the v_mem direction in T5 space.
    # We use v_mem_t5_native (direct contrastive on caption T5 embeddings) for the
    # first sweep — it's the cleanest "does text-encoder steering work?" test.
    # The adapter-derived direction had cos(-0.19) with this one, so they encode
    # different things; we can test the adapter version in a follow-up sweep.
    ckpt = torch.load("data/reports/adapter_tribe_to_t5.pt", weights_only=False, map_location="cpu")
    v_mem_t5 = np.asarray(ckpt[args.direction], dtype=np.float32)
    print(f"[sweep] direction = {args.direction}")
    print(f"[sweep] v_mem_t5 dim={v_mem_t5.shape[0]} norm={np.linalg.norm(v_mem_t5):.4f}")
    print(f"[sweep] cos(text-native, adapter-derived) = {ckpt['cos_alignment']:+.4f}")

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "CogVideoXGenerator")
    gen = Generator()

    prompts = PROMPTS[: args.max_prompts]
    calls = []
    for pi, prompt in enumerate(prompts):
        for ai, alpha in enumerate(ALPHAS):
            label = f"p{pi:02d}_a{ai:+d}"
            calls.append((prompt, alpha, label))

    print(f"[sweep] dispatching {len(calls)} generations (concurrency = up to 20 B200/H100s)")

    # Parallel via .spawn / .get — Modal class methods don't support .map with
    # kwargs cleanly, but spawn returns a FunctionCall we can join.
    print(f"[sweep] spawning {len(calls)} jobs …")
    pending = []
    v_list = v_mem_t5.tolist()
    for pi, (prompt, alpha, label) in enumerate(calls):
        fc = gen.generate.spawn(
            prompt,
            steering_vector=v_list,
            alpha=alpha,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + pi,
            output_label=label,
        )
        pending.append((label, prompt, alpha, fc))
    print(f"[sweep] spawned, awaiting results...")

    results = []
    for label, prompt, alpha, fc in pending:
        try:
            video_bytes = fc.get(timeout=20 * 60)
        except Exception as exc:  # noqa: BLE001
            print(f"  [sweep] ✗ {label}: {exc!r}")
            video_bytes = None
        results.append(video_bytes)

    n_ok = 0
    for (prompt, alpha, label), video_bytes in zip(calls, results):
        if video_bytes is None:
            print(f"  [sweep] ✗ {label}")
            continue
        path = args.out_dir / f"{label}.mp4"
        path.write_bytes(video_bytes)
        n_ok += 1
        print(f"  [sweep] ✓ {label} ({len(video_bytes) / 1024:.1f} KB)")

    print(f"\n[done] {n_ok}/{len(calls)} clips generated")
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps([
        {"prompt": p, "alpha": a, "label": l, "path": str(args.out_dir / f"{l}.mp4")}
        for p, a, l in calls
    ], indent=2))
    print(f"[done] manifest at {manifest}")


if __name__ == "__main__":
    main()
