"""Extract V-JEPA features for selector-study generated videos via Modal.

By default, this sends each local video as a bytes payload to avoid stale Modal
volume views during generated-video sweeps. Volume transport is still available
for BMD-style workflows. Outputs are local `.npz` feature files keyed by video
label.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.services.vjepa_service import VjepaService


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def unique_label_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for row in manifest["rows"]:
        for policy, label in row["labels"].items():
            if label is None:
                continue
            raw_path = row["video_paths"].get(policy)
            if raw_path is None:
                continue
            out[str(label)] = Path(str(raw_path))
    return dict(sorted(out.items()))


def upload_missing(
    *,
    label_paths: dict[str, Path],
    remote_dir: str,
    force_upload: bool,
) -> dict[str, str]:
    import modal  # noqa: PLC0415

    volume = modal.Volume.from_name("bmd-videos-v1", create_if_missing=True)
    remote_paths = {
        label: f"{remote_dir.rstrip('/')}/{label}.mp4" for label in label_paths
    }
    with volume.batch_upload(force=force_upload) as batch:
        for label, local_path in label_paths.items():
            if not local_path.exists():
                raise FileNotFoundError(
                    f"missing local video for {label}: {local_path}"
                )
            batch.put_file(local_path, remote_paths[label])
    return {label: f"/bmd-videos{path}" for label, path in remote_paths.items()}


async def extract_one(
    *,
    service: VjepaService,
    sem: asyncio.Semaphore,
    label: str,
    local_path: Path,
    modal_path: str,
    output_dir: Path,
    transport: str,
) -> Path | None:
    out = output_dir / f"{label}.npz"
    if out.exists() and out.stat().st_size > 0:
        return out
    async with sem:
        if transport == "bytes":
            result = await service.predict_video_bytes(local_path.read_bytes(), ".mp4")
        else:
            result = await service.predict_video(modal_path)
    if result is None:
        return None

    if hasattr(result, "embedding"):
        embedding = np.asarray(result.embedding, dtype=np.float32)
        duration = float(result.duration_seconds)
        n_frames = int(result.n_frames)
    else:
        embedding = np.asarray(result["embedding"], dtype=np.float32)
        duration = float(result["duration_seconds"])
        n_frames = int(result["n_frames"])

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embedding=embedding,
        duration_seconds=np.array(duration, dtype=np.float32),
        n_frames=np.array(n_frames, dtype=np.int32),
        label=np.array(label),
        modal_path=np.array(modal_path),
        transport=np.array(transport),
    )
    print(f"[vjepa] wrote {out} dim={embedding.shape[0]}", flush=True)
    return out


async def run(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    label_paths = unique_label_paths(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.limit is not None:
        label_paths = dict(list(label_paths.items())[: args.limit])

    cached = [
        label
        for label in label_paths
        if (args.output_dir / f"{label}.npz").exists()
        and (args.output_dir / f"{label}.npz").stat().st_size > 0
    ]
    to_upload = {
        label: path for label, path in label_paths.items() if label not in set(cached)
    }
    print(
        f"[plan] labels={len(label_paths)} cached={len(cached)} "
        f"to_extract={len(to_upload)} output={args.output_dir}",
        flush=True,
    )
    if not to_upload:
        return 0

    if args.transport == "volume":
        print(f"[upload] uploading {len(to_upload)} videos to Modal volume", flush=True)
        modal_paths = upload_missing(
            label_paths=to_upload,
            remote_dir=args.remote_dir,
            force_upload=args.force_upload,
        )
    else:
        modal_paths = {
            label: f"/bytes/{path.name}" for label, path in to_upload.items()
        }

    service = VjepaService(app_name=args.app_name)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        extract_one(
            service=service,
            sem=sem,
            label=label,
            local_path=to_upload[label],
            modal_path=modal_path,
            output_dir=args.output_dir,
            transport=args.transport,
        )
        for label, modal_path in modal_paths.items()
    ]
    results = await asyncio.gather(*tasks)
    written = [path for path in results if path is not None]
    print(f"[done] extracted {len(written)}/{len(to_upload)} missing embeddings")
    return 0 if len(written) == len(to_upload) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features/vjepa_wan22_selector_pref_weighted_r16_s300"),
    )
    parser.add_argument(
        "--remote-dir", default="/wan22_selector_pref_weighted_r16_s300"
    )
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument(
        "--transport",
        choices=("bytes", "volume"),
        default="bytes",
        help="Use RPC bytes payloads by default; volume transport is kept for BMD-style workflows.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
