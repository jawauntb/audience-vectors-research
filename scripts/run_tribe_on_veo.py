"""Direct loop: call TRIBE per Veo clip, save .npz to data/features/tribe_veo/.

Simpler than the full extractor wrapper — avoids whatever's hanging there.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from audience_vectors.services.tribe_service import TribeService  # noqa: E402


async def one(svc: TribeService, mp4: Path, out_dir: Path, sem: asyncio.Semaphore) -> tuple[str, bool]:
    out_path = out_dir / f"{mp4.stem}.npz"
    if out_path.exists():
        return mp4.stem, True
    modal_path = f"/bmd-videos/generated/{mp4.name}"
    async with sem:
        print(f"  [tribe] -> {mp4.stem}", flush=True)
        try:
            result = await asyncio.wait_for(svc.predict_video(modal_path), timeout=180.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  [tribe] ✗ {mp4.stem}: {exc!r}", flush=True)
            return mp4.stem, False
    if result is None or getattr(result, "frames", None) is None:
        print(f"  [tribe] ✗ {mp4.stem}: empty result", flush=True)
        return mp4.stem, False
    frames = np.asarray(result.frames, dtype=np.float32)
    duration = float(getattr(result, "duration_seconds", 4.0) or 4.0)
    np.savez_compressed(out_path, frames=frames, duration_seconds=duration, sample_id=np.asarray([mp4.stem]))
    print(f"  [tribe] ✓ {mp4.stem} frames={frames.shape}", flush=True)
    return mp4.stem, True


async def main_async() -> None:
    veo_dir = Path("data/generated/veo")
    out_dir = Path("data/features/tribe_veo")
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(veo_dir.glob("*.mp4"))
    print(f"[run-tribe] {len(mp4s)} clips, concurrency=6, timeout=180s each")

    svc = TribeService()
    sem = asyncio.Semaphore(6)
    results = await asyncio.gather(*[one(svc, p, out_dir, sem) for p in mp4s])

    ok = sum(1 for _, b in results if b)
    print(f"[run-tribe] done: {ok}/{len(mp4s)} succeeded")


if __name__ == "__main__":
    asyncio.run(main_async())
