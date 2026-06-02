"""Run a small Wan2.2 best-of-N replication matrix.

The one-off `wan22_best_of_n.py` script is useful for a single seed. This
wrapper makes the next research step reproducible: several BMD image seeds,
several Wan samples per seed, one checkpointed manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_APP = "audience-vectors-dev"
DEFAULT_SEED_IDS = (
    "1053",
    "1070",
    "1050",
    "1029",
    "1101",
    "1075",
    "1067",
    "1057",
)


@dataclass(frozen=True)
class MatrixSeed:
    seed_id: str
    bmd_name: str
    source_image: Path
    local_image: Path
    prompt: str
    memorability_score: float
    dataset_split: str


def _load_annotations(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _find_middle_frame(frames_dir: Path, seed_id: str) -> Path:
    matches = sorted(frames_dir.glob(f"{seed_id}_*.jpg"))
    if not matches:
        raise FileNotFoundError(f"no middle frame found for BMD seed {seed_id}")
    return matches[0]


def _prompt_from_annotation(row: dict[str, Any]) -> str:
    descriptions = row.get("text_descriptions") or []
    base = str(descriptions[0]).strip() if descriptions else ""
    if not base:
        actions = ", ".join(str(x) for x in row.get("actions", [])[:3])
        scenes = ", ".join(str(x) for x in row.get("scenes", [])[:2])
        base = f"A realistic short video involving {actions} in {scenes}."
    if not base.endswith("."):
        base += "."
    return (
        f"{base} Natural realistic short video, clear central subject, "
        "continuous motion, stable composition, no text, no watermark."
    )


def _prepare_seed_image(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with Image.open(source) as img:
        img.convert("RGB").save(dest)


def _build_matrix_seeds(
    *,
    seed_ids: list[str],
    annotations: dict[str, Any],
    frames_dir: Path,
    out_dir: Path,
) -> list[MatrixSeed]:
    seeds = []
    for raw_seed_id in seed_ids:
        seed_id = raw_seed_id.removeprefix("vid_idx").zfill(4)
        if seed_id not in annotations:
            raise KeyError(f"BMD seed {seed_id} not found in annotations")
        row = annotations[seed_id]
        source_image = _find_middle_frame(frames_dir, seed_id)
        local_image = out_dir / "seeds" / f"vid_idx{seed_id}_seed.png"
        _prepare_seed_image(source_image, local_image)
        seeds.append(
            MatrixSeed(
                seed_id=seed_id,
                bmd_name=f"vid_idx{seed_id}",
                source_image=source_image,
                local_image=local_image,
                prompt=_prompt_from_annotation(row),
                memorability_score=float(row["memorability_score"]),
                dataset_split=str(row.get("set", "unknown")),
            )
        )
    return seeds


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (str(r.get("bmd_name", "")), int(r["idx"])))
    path.write_text(json.dumps(rows, indent=2))


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-name", default=DEFAULT_APP)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/generated/wan22_replication_matrix_2026-05-20"),
    )
    parser.add_argument("--seed-ids", nargs="+", default=list(DEFAULT_SEED_IDS))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/bold_moments/annotations.json"),
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/raw/bold_moments/stimuli/stimulus_set/frames_middle"),
    )
    parser.add_argument(
        "--task", default="ti2v-5B", choices=["ti2v-5B", "i2v-A14B", "t2v-A14B"]
    )
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=520000)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--spawn-retries", type=int, default=3)
    parser.add_argument("--frame-num", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--sample-guide-scale", type=float)
    parser.add_argument("--sample-shift", type=float)
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate clips even when the local mp4 already exists.",
    )
    args = parser.parse_args()

    if args.max_in_flight < 1:
        raise ValueError("--max-in-flight must be at least 1")
    if args.n < 1:
        raise ValueError("--n must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    annotations = _load_annotations(args.annotations)
    matrix_seeds = _build_matrix_seeds(
        seed_ids=args.seed_ids,
        annotations=annotations,
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
    )

    import modal  # noqa: PLC0415

    Generator = modal.Cls.from_name(args.app_name, "Wan22Generator")
    gen = Generator()

    manifest = _read_manifest(manifest_path)
    manifest_by_label = {str(row.get("label")): row for row in manifest}
    jobs: list[tuple[MatrixSeed, int, int, str, Path]] = []
    for matrix_seed in matrix_seeds:
        for idx in range(args.n):
            generation_seed = args.seed_base + int(matrix_seed.seed_id) * 100 + idx
            label = (
                f"{matrix_seed.bmd_name}_{args.task.replace('-', '_')}_"
                f"rep_n{idx:02d}"
            )
            local_path = args.out_dir / f"{label}.mp4"
            if local_path.exists() and not args.overwrite:
                row = manifest_by_label.get(label, {})
                manifest_by_label[label] = {
                    **row,
                    "idx": idx,
                    "seed": generation_seed,
                    "label": label,
                    "bmd_name": matrix_seed.bmd_name,
                    "bmd_seed_id": matrix_seed.seed_id,
                    "bmd_memorability_score": matrix_seed.memorability_score,
                    "bmd_split": matrix_seed.dataset_split,
                    "prompt": matrix_seed.prompt,
                    "source_image": str(matrix_seed.source_image),
                    "seed_image": str(matrix_seed.local_image),
                    "local_path": str(local_path),
                    "skipped_existing": True,
                }
                continue
            jobs.append((matrix_seed, idx, generation_seed, label, local_path))

    pending: list[tuple[MatrixSeed, int, int, str, Path, Any]] = []
    next_job = 0
    completed_this_run = 0

    def spawn_one(job: tuple[MatrixSeed, int, int, str, Path]) -> None:
        matrix_seed, idx, generation_seed, label, local_path = job
        image_bytes = matrix_seed.local_image.read_bytes()
        fc = None
        last_exc: Exception | None = None
        for attempt in range(args.spawn_retries + 1):
            try:
                fc = gen.generate.spawn(
                    matrix_seed.prompt,
                    image_bytes=image_bytes,
                    task=args.task,
                    size=args.size,
                    frame_num=args.frame_num,
                    sample_steps=args.sample_steps,
                    sample_guide_scale=args.sample_guide_scale,
                    sample_shift=args.sample_shift,
                    seed=generation_seed,
                    offload_model=args.offload_model,
                    output_label=label,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= args.spawn_retries:
                    raise
                delay = 5.0 * (attempt + 1)
                print(
                    f"  spawn retry {attempt + 1}/{args.spawn_retries} "
                    f"for {label}: {exc!r}; sleeping {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
        if fc is None:
            raise RuntimeError(f"failed to spawn {label}") from last_exc
        pending.append((matrix_seed, idx, generation_seed, label, local_path, fc))

    while next_job < len(jobs) or pending:
        while next_job < len(jobs) and len(pending) < args.max_in_flight:
            spawn_one(jobs[next_job])
            next_job += 1
        print(
            "[wan22-matrix] "
            f"in-flight={len(pending)} completed-now={completed_this_run} "
            f"skipped-existing={len(manifest_by_label) - completed_this_run} "
            f"remaining-to-spawn={len(jobs) - next_job}",
            flush=True,
        )

        matrix_seed, idx, generation_seed, label, local_path, fc = pending.pop(0)
        row: dict[str, Any] = {
            "idx": idx,
            "seed": generation_seed,
            "label": label,
            "bmd_name": matrix_seed.bmd_name,
            "bmd_seed_id": matrix_seed.seed_id,
            "bmd_memorability_score": matrix_seed.memorability_score,
            "bmd_split": matrix_seed.dataset_split,
            "prompt": matrix_seed.prompt,
            "source_image": str(matrix_seed.source_image),
            "seed_image": str(matrix_seed.local_image),
            "local_path": str(local_path),
        }
        try:
            result = fc.get(timeout=4 * 60 * 60)
            video_bytes = result.pop("video_bytes")
            local_path.write_bytes(video_bytes)
            row.update(result)
            row["bytes"] = len(video_bytes)
            row["error"] = None
            completed_this_run += 1
            print(f"  ok {label}: {len(video_bytes) / 1024 / 1024:.2f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
            print(f"  x {label}: {exc!r}", flush=True)

        manifest_by_label[label] = row
        _write_manifest(manifest_path, list(manifest_by_label.values()))

    _write_manifest(manifest_path, list(manifest_by_label.values()))
    print(f"[wan22-matrix] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
