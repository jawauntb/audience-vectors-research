"""Best-of-N on Veo 3 Fast — test whether the brain-direction quality filter
generalizes off SVD onto a different (closed) T2V model.

For each of K prompts, generate N variants via parallel Veo calls (Veo's
generation is stochastic, so N parallel calls give N different videos).
Push through TRIBE, project on v_mem, report per-prompt within-seed lift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


PROMPTS = [
    "A chef in a kitchen preparing a meal, wide static shot.",
    "An animal eating food, normal speed, ambient lighting.",
    "A potter shaping clay on a wheel, calm studio.",
    "A scientist running an experiment with glassware on a lab bench.",
    "A person typing at a wooden desk in a sunlit office.",
]
N_PER_PROMPT = 8


async def _generate_via_veo(client, prompt: str, out_path: Path, model: str,
                             timeout_s: float = 240.0) -> bool:
    try:
        from google.genai import types
        config = types.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            duration_seconds=4,
            negative_prompt="static, blurry, frozen, slideshow",
        )
        op = await asyncio.to_thread(
            client.models.generate_videos,
            model=model, prompt=prompt, config=config,
        )
        t0 = time.monotonic()
        while not op.done:
            await asyncio.sleep(8)
            op = await asyncio.to_thread(client.operations.get, op)
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"Veo gen timed out after {timeout_s:.0f}s")
        if not op.response or not op.response.generated_videos:
            return False
        video = op.response.generated_videos[0]
        await asyncio.to_thread(client.files.download, file=video.video)
        await asyncio.to_thread(video.video.save, str(out_path))
        print(f"  ✓ {out_path.name}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {out_path.name}: {type(exc).__name__}: {exc}", flush=True)
        return False


async def main_async(args: argparse.Namespace) -> None:
    # bootstrap env
    for line in Path(".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    pending = []
    for pi, prompt in enumerate(PROMPTS):
        for ni in range(N_PER_PROMPT):
            label = f"p{pi:02d}_n{ni:02d}"
            out = args.out_dir / f"{label}.mp4"
            if out.exists():
                print(f"  [skip] {label}")
                continue
            async def one(prompt=prompt, out=out, label=label):
                async with sem:
                    print(f"  [veo] {label}: {prompt[:60]!r}", flush=True)
                    await _generate_via_veo(client, prompt, out, args.model)
            pending.append(one())

    await asyncio.gather(*pending)

    n_ok = len(list(args.out_dir.glob("*.mp4")))
    print(f"\n[done] {n_ok}/{len(PROMPTS) * N_PER_PROMPT} clips at {args.out_dir}")
    (args.out_dir / "prompts.json").write_text(json.dumps(
        [{"prompt_idx": i, "prompt": p} for i, p in enumerate(PROMPTS)], indent=2,
    ))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/veo_best_of_n"))
    parser.add_argument("--model", default="veo-3.0-fast-generate-001")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
