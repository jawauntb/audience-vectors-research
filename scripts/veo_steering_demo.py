"""End-to-end brain-direction-conditioned generation demo via Veo 3.

Generates 10 prompt pairs (memorable-styled vs neutral), pushes each through
TRIBE, projects onto the BMD memorability direction, and reports paired stats.

This is the prompt-level steering version. The 'memorable' prompts are
constructed by injecting the signature phrases discovered in Approach A
(machine, cooking, sizzling, precise, mechanical, eating, framing) into a
base scene description; the 'neutral' prompts describe the same scene
without those modifiers.

This validates the framework end-to-end: brain-derived memorability direction
→ caption signature (§6.1) → T2V prompting → generated clips → projection
onto the brain-derived direction → measured score shift.
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


PROMPT_PAIRS = [
    {
        "base": "a chef in a kitchen preparing a meal",
        "memorable": "Tight close-up of a chef's hand precisely working a sizzling cooking machine, mechanical motion, food eating into the pan, dynamic framing.",
        "neutral": "A chef in a kitchen preparing a meal, wide static shot.",
    },
    {
        "base": "an animal eating food",
        "memorable": "Extreme close-up of a small animal precisely eating from a spoon, unusual sizzling sound, mechanical jaw motion.",
        "neutral": "An animal eating food, normal speed, ambient lighting.",
    },
    {
        "base": "construction work outdoors",
        "memorable": "Sizzling sparks fly as a precise mechanical welding machine cuts metal, dynamic active framing, close on the cutting action.",
        "neutral": "A construction worker doing his job outdoors, normal angle.",
    },
    {
        "base": "a person sewing fabric",
        "memorable": "Mechanical sewing machine, precise needle motion, fabric framing the active hands, eating thread through the loops.",
        "neutral": "A person sewing fabric on a normal sewing machine.",
    },
    {
        "base": "person at a desk working",
        "memorable": "Active mechanical typewriter sizzling under precise finger strikes, paper feeding through, tight framing.",
        "neutral": "Person sitting at a desk doing work on a computer.",
    },
    {
        "base": "an animal in nature",
        "memorable": "Sizzling-hot precise close-up of an unusual animal eating insects with mechanical jaw motion, vivid framing.",
        "neutral": "An animal walking in a forest, gentle nature footage.",
    },
    {
        "base": "machinery operating",
        "memorable": "Precise mechanical 3D printer sizzling layers into shape, active extruder framing the build plate.",
        "neutral": "A piece of machinery operating in a factory.",
    },
    {
        "base": "kids playing",
        "memorable": "Kids precisely working a mechanical pinball machine, sizzling lights, active framing on hands.",
        "neutral": "Children playing together in a park.",
    },
    {
        "base": "an experiment",
        "memorable": "Precise sizzling chemistry, mechanical pipette eating drops into a beaker, dynamic framing of the reaction.",
        "neutral": "A scientist running an experiment in a laboratory.",
    },
    {
        "base": "a craft being made",
        "memorable": "Mechanical pottery wheel sizzling as precise hands shape clay, active framing on the rotating piece.",
        "neutral": "An artisan making a pottery piece by hand.",
    },
]


async def _generate_via_veo(client, prompt: str, out_path: Path, model: str, timeout_s: float = 240.0) -> bool:
    """Generate one clip via Google Veo, save to out_path, return success."""
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
            model=model,
            prompt=prompt,
            config=config,
        )
        t0 = time.monotonic()
        while not op.done:
            await asyncio.sleep(8)
            op = await asyncio.to_thread(client.operations.get, op)
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"Veo generation timed out after {timeout_s:.0f}s")
        if not op.response or not op.response.generated_videos:
            print(f"  [veo] no video in response for prompt: {prompt[:60]!r}")
            return False
        video = op.response.generated_videos[0]
        await asyncio.to_thread(client.files.download, file=video.video)
        await asyncio.to_thread(video.video.save, str(out_path))
        print(f"  [veo] ✓ {out_path.name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [veo] ✗ failed for {out_path.name}: {exc!r}")
        return False


async def _generate_all(prompts: list[tuple[str, str]], out_dir: Path, model: str, concurrency: int = 6) -> list[dict]:
    """Generate every prompt in parallel, return list of {label, prompt, path, ok}."""
    for line in Path(".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    sem = asyncio.Semaphore(concurrency)

    async def one(label: str, prompt: str) -> dict:
        out_path = out_dir / f"{label}.mp4"
        if out_path.exists():
            print(f"  [skip] {label} (exists)")
            return {"label": label, "prompt": prompt, "path": str(out_path), "ok": True}
        async with sem:
            print(f"  [veo] starting {label}: {prompt[:80]!r}")
            ok = await _generate_via_veo(client, prompt, out_path, model)
        return {"label": label, "prompt": prompt, "path": str(out_path), "ok": ok}

    results = await asyncio.gather(*[one(label, prompt) for label, prompt in prompts])
    return list(results)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="veo-3.0-fast-generate-001")
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/veo"))
    parser.add_argument("--max-pairs", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts: list[tuple[str, str]] = []
    for i, pair in enumerate(PROMPT_PAIRS[: args.max_pairs]):
        prompts.append((f"pair{i:02d}_mem", pair["memorable"]))
        prompts.append((f"pair{i:02d}_neu", pair["neutral"]))

    print(f"[veo-demo] generating {len(prompts)} clips via {args.model}")
    print(f"[veo-demo] out dir: {args.out_dir}")
    results = asyncio.run(_generate_all(prompts, args.out_dir, args.model))

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[veo-demo] generation complete: {n_ok}/{len(prompts)} clips")
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2))
    print(f"[veo-demo] manifest: {manifest}")


if __name__ == "__main__":
    main()
