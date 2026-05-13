"""One-off debug function: list contents of /bmd-videos inside a Modal container."""

from __future__ import annotations

import os

import modal

from audience_vectors.modal_app.app import app, env_secrets
from audience_vectors.modal_app.image_factory import base_image

bmd_videos_volume = modal.Volume.from_name("bmd-videos-v1", create_if_missing=False)


@app.function(
    image=base_image,
    volumes={"/bmd-videos": bmd_videos_volume},
    secrets=env_secrets,
    timeout=60,
)
def inspect_bmd_volume() -> dict[str, object]:
    """Walk /bmd-videos and report what's actually visible inside a container."""
    result: dict[str, object] = {}
    print(f"[debug] /bmd-videos exists: {os.path.exists('/bmd-videos')}")
    if not os.path.exists("/bmd-videos"):
        result["error"] = "/bmd-videos path does not exist"
        print(f"[debug] result: {result}")
        return result

    # Force volume refresh
    try:
        bmd_videos_volume.reload()
        print("[debug] volume.reload() succeeded")
    except Exception as e:  # noqa: BLE001
        print(f"[debug] volume.reload() failed: {e}")

    top = sorted(os.listdir("/bmd-videos"))[:20]
    print(f"[debug] /bmd-videos top-level: {top}")
    result["top_level"] = top

    if os.path.exists("/bmd-videos/videos"):
        files = sorted(os.listdir("/bmd-videos/videos"))
        print(f"[debug] /bmd-videos/videos count={len(files)} first={files[:3]} last={files[-3:]}")
        result["videos_count"] = len(files)
        result["videos_first_3"] = files[:3]
        result["videos_last_3"] = files[-3:]
        # Try reading a known file
        for test_path in ["/bmd-videos/videos/vid_idx0001.mp4", "/bmd-videos/videos/vid_idx0250.mp4"]:
            exists = os.path.exists(test_path)
            size = os.path.getsize(test_path) if exists else 0
            print(f"[debug] {test_path}: exists={exists} size={size}")
            result[test_path] = {"exists": exists, "size": size}
    else:
        print("[debug] /bmd-videos/videos NOT found")
        result["note"] = "/bmd-videos/videos NOT found"
    return result
