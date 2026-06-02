"""Bigger Veo sweep: 5 more diverse prompts × N=8 = 40 more clips.

We already have 4 prompts. Adds prompts p04-p08 (different scene types) so we
have N≥7 statistical power on the generator-agnostic best-of-N claim."""

from __future__ import annotations

import argparse, asyncio, os, time
from pathlib import Path

NEW_PROMPTS = [
    "A close-up of hands kneading dough on a marble countertop, soft natural light.",
    "A child blowing soap bubbles in a sunny backyard, slow motion.",
    "A barista pouring milk into espresso, creating latte art, overhead shot.",
    "A drone flying over a coastline at sunset, smooth cinematic motion.",
    "A black cat jumping onto a wooden table, golden hour light.",
]
N_PER_PROMPT = 8
OFFSET = 4  # we already have p00-p03


async def _generate_via_veo(client, prompt, out_path, model, timeout_s=240.0):
    try:
        from google.genai import types
        config = types.GenerateVideosConfig(
            aspect_ratio="16:9", number_of_videos=1, duration_seconds=4,
            negative_prompt="static, blurry, frozen",
        )
        op = await asyncio.to_thread(client.models.generate_videos,
                                      model=model, prompt=prompt, config=config)
        t0 = time.monotonic()
        while not op.done:
            await asyncio.sleep(8)
            op = await asyncio.to_thread(client.operations.get, op)
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"timeout {timeout_s}s")
        if not op.response or not op.response.generated_videos:
            return False
        video = op.response.generated_videos[0]
        await asyncio.to_thread(client.files.download, file=video.video)
        await asyncio.to_thread(video.video.save, str(out_path))
        print(f"  ✓ {out_path.name}", flush=True)
        return True
    except Exception as e:
        print(f"  ✗ {out_path.name}: {type(e).__name__}", flush=True)
        return False


async def main_async(args):
    for line in Path(".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    pending = []
    for pi_off, prompt in enumerate(NEW_PROMPTS):
        pi = OFFSET + pi_off
        for ni in range(N_PER_PROMPT):
            label = f"p{pi:02d}_n{ni:02d}"
            out = args.out_dir / f"{label}.mp4"
            if out.exists(): continue
            async def one(prompt=prompt, out=out):
                async with sem:
                    await _generate_via_veo(client, prompt, out, args.model)
            pending.append(one())
    print(f"[veo more] {len(pending)} jobs")
    await asyncio.gather(*pending)
    # update prompts.json
    import json
    pj = args.out_dir / "prompts.json"
    existing = json.loads(pj.read_text()) if pj.exists() else []
    for pi_off, prompt in enumerate(NEW_PROMPTS):
        pi = OFFSET + pi_off
        if not any(p.get("prompt_idx") == pi for p in existing):
            existing.append({"prompt_idx": pi, "prompt": prompt})
    pj.write_text(json.dumps(existing, indent=2))
    print(f"[done] {args.out_dir}, prompts.json updated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/veo_best_of_n"))
    parser.add_argument("--model", default="veo-3.0-fast-generate-001")
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
