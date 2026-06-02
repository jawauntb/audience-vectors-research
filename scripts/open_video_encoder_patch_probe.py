"""Open video-encoder memorability patching probe.

This is intentionally separate from the AlexNet and generation-model probes.
It uses an open CLIP image encoder as a frame-level video encoder: sample a few
frames per BOLD Moments clip, average frame representations per video, learn a
human-memorability direction at an intermediate transformer block, then patch
that direction during the forward pass with a module hook.

Default execution is local-files-only so the probe never silently starts a
large model download. Pass --allow-downloads if you explicitly want Hugging Face
to fetch missing model files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_JSON = Path("data/reports/open_video_encoder_patch_probe.json")
DEFAULT_MD = Path("data/reports/open_video_encoder_patch_probe.md")


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    annotation_key: str
    path: Path
    score: float


@dataclass(frozen=True)
class ProbeArrays:
    records: list[VideoRecord]
    hidden: np.ndarray
    image_embeds: np.ndarray


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ra = _rankdata(np.asarray(a, dtype=np.float64))
    rb = _rankdata(np.asarray(b, dtype=np.float64))
    denom = float(np.std(ra) * np.std(rb))
    if denom <= 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _unit(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("near-zero direction norm")
    return np.asarray(x / norm, dtype=np.float32)


def _train_direction(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    top_frac: float,
    min_tail: int,
) -> np.ndarray:
    if len(features) != len(scores):
        raise ValueError("feature and score lengths differ")
    n_each = max(min_tail, int(round(len(scores) * top_frac)))
    if 2 * n_each > len(scores):
        raise ValueError(
            f"not enough examples for tails: n={len(scores)}, n_each={n_each}"
        )
    order = np.argsort(scores)
    low = features[order[:n_each]].mean(axis=0)
    high = features[order[-n_each:]].mean(axis=0)
    return _unit(high - low)


def _load_records(data_root: Path) -> list[VideoRecord]:
    annotations = data_root / "raw" / "bold_moments" / "annotations.json"
    video_dir = data_root / "raw" / "bold_moments" / "videos"
    if not annotations.exists():
        raise FileNotFoundError(f"missing annotations: {annotations}")
    if not video_dir.exists():
        raise FileNotFoundError(f"missing BOLD Moments videos: {video_dir}")

    payload = json.loads(annotations.read_text())
    records: list[VideoRecord] = []
    for key, row in payload.items():
        if "memorability_score" not in row:
            continue
        video_path = video_dir / f"vid_idx{key}.mp4"
        if not video_path.exists():
            continue
        records.append(
            VideoRecord(
                video_id=f"bmd_vid_idx{key}",
                annotation_key=str(key),
                path=video_path,
                score=float(row["memorability_score"]),
            )
        )
    if not records:
        raise ValueError(f"no scored local videos under {video_dir}")
    return records


def _select_records(
    records: Sequence[VideoRecord],
    *,
    max_videos: int,
) -> list[VideoRecord]:
    ordered = sorted(records, key=lambda r: r.score)
    if max_videos <= 0 or max_videos >= len(ordered):
        return ordered
    # Spread samples across the full memorability range instead of accidentally
    # taking only one score band from lexicographic filenames.
    indices = np.linspace(0, len(ordered) - 1, num=max_videos, dtype=int)
    return [ordered[int(i)] for i in indices]


def _split_indices(
    n: int, *, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n < 8:
        raise ValueError("need at least 8 videos for a train/test patch probe")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(4, int(round(n * test_frac)))
    n_test = min(n_test, n - 4)
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test


def _load_optional_modules() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    modules: list[Any] = []
    for name in ("torch", "transformers", "imageio"):
        try:
            modules.append(importlib.import_module(name))
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise ModuleNotFoundError(
            "missing local dependencies: "
            + ", ".join(missing)
            + ". Install the repo's ml extras or run with the existing project .venv."
        )

    transformers = modules[1]
    return (
        modules[0],
        transformers.CLIPImageProcessor,
        transformers.CLIPVisionModelWithProjection,
        modules[2],
    )


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_frames(imageio: Any, path: Path, n_frames: int) -> list[Image.Image]:
    reader = imageio.get_reader(str(path))
    try:
        try:
            n_total = int(reader.count_frames())
        except Exception:  # noqa: BLE001
            meta = reader.get_meta_data()
            n_total = int(meta.get("fps", 24) * meta.get("duration", 3.0))
        if n_total <= 0:
            raise ValueError(f"could not count frames in {path}")
        indices = np.linspace(0, n_total - 1, num=n_frames, dtype=int)
        return [
            Image.fromarray(np.asarray(reader.get_data(int(i)), dtype=np.uint8))
            for i in indices
        ]
    finally:
        reader.close()


def _to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in inputs.items()}


def _make_patch_hook(
    torch: Any,
    *,
    mode: str,
    direction: np.ndarray | None,
    center: np.ndarray | None,
    add_scale: float,
) -> Callable[[Any, tuple[Any, ...], Any], Any]:
    if mode == "baseline":
        direction_t = None
        center_t = None
    else:
        if direction is None or center is None:
            raise ValueError("patch mode requires direction and center")
        direction_t = torch.as_tensor(direction, dtype=torch.float32)
        center_t = torch.as_tensor(center, dtype=torch.float32)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if mode == "baseline":
            patched = hidden
        else:
            assert direction_t is not None
            assert center_t is not None
            direction_local = direction_t.to(device=hidden.device, dtype=hidden.dtype)
            center_local = center_t.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden.clone()
            cls = patched[:, 0, :]
            if mode == "remove":
                projection = ((cls - center_local) * direction_local).sum(dim=-1)
                patched[:, 0, :] = cls - projection[:, None] * direction_local
            elif mode == "add":
                patched[:, 0, :] = cls + add_scale * direction_local
            elif mode == "subtract":
                patched[:, 0, :] = cls - add_scale * direction_local
            else:
                raise ValueError(f"unknown patch mode: {mode}")

        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


def _forward_records(
    *,
    torch: Any,
    imageio_v3: Any,
    processor: Any,
    model: Any,
    device: Any,
    records: Sequence[VideoRecord],
    layer_index: int,
    frames_per_video: int,
    batch_videos: int,
    mode: str,
    direction: np.ndarray | None = None,
    center: np.ndarray | None = None,
    add_scale: float = 1.0,
) -> ProbeArrays:
    all_hidden: list[np.ndarray] = []
    all_embeds: list[np.ndarray] = []
    hook = _make_patch_hook(
        torch,
        mode=mode,
        direction=direction,
        center=center,
        add_scale=add_scale,
    )
    handle = model.vision_model.encoder.layers[layer_index].register_forward_hook(hook)
    capture_handle = None
    captured: list[Any] = []

    def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, 0, :].detach().cpu())

    capture_handle = model.vision_model.encoder.layers[
        layer_index
    ].register_forward_hook(capture)
    try:
        for start in range(0, len(records), batch_videos):
            batch = list(records[start : start + batch_videos])
            frames: list[Image.Image] = []
            for record in batch:
                frames.extend(_sample_frames(imageio_v3, record.path, frames_per_video))
            inputs = _to_device(processor(images=frames, return_tensors="pt"), device)
            with torch.no_grad():
                outputs = model(**inputs)

            image_embeds = outputs.image_embeds.detach().cpu().float().numpy()
            hidden = captured[-1].float().numpy()
            n_batch = len(batch)
            all_embeds.append(
                image_embeds.reshape(n_batch, frames_per_video, -1).mean(axis=1)
            )
            all_hidden.append(
                hidden.reshape(n_batch, frames_per_video, -1).mean(axis=1)
            )
    finally:
        handle.remove()
        if capture_handle is not None:
            capture_handle.remove()

    return ProbeArrays(
        records=list(records),
        hidden=np.concatenate(all_hidden, axis=0),
        image_embeds=np.concatenate(all_embeds, axis=0),
    )


def _summarize_shift(
    baseline: np.ndarray,
    patched: np.ndarray,
    final_direction: np.ndarray,
) -> dict[str, float]:
    baseline_proj = baseline @ final_direction
    patched_proj = patched @ final_direction
    delta = patched_proj - baseline_proj
    l2 = np.linalg.norm(patched - baseline, axis=1)
    return {
        "mean_projection": float(patched_proj.mean()),
        "mean_delta_vs_baseline": float(delta.mean()),
        "median_delta_vs_baseline": float(np.median(delta)),
        "projection_shift_sign_rate_positive": float(np.mean(delta > 0)),
        "mean_image_embedding_l2_delta": float(l2.mean()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    patch = payload["forward_patching"]
    intermediate = payload["intermediate_probe"]
    conclusion = payload["conclusion"]
    return "\n".join(
        [
            "# Open Video Encoder Patching Probe",
            "",
            f"- Status: **{payload['status']}**.",
            f"- Encoder: `{payload['model_id']}` as a frame-level open video encoder.",
            f"- Forward access: transformer block `{payload['layer_index']}` hook on CLS state.",
            f"- Data: {payload['n_selected']} local BOLD Moments videos "
            f"({payload['n_train']} train / {payload['n_test']} test), "
            f"{payload['frames_per_video']} frames per video.",
            f"- Intermediate memorability rho on held-out videos: "
            f"**{intermediate['test_spearman_rho']:+.3f}**.",
            f"- Final-space baseline rho on held-out videos: "
            f"**{patch['baseline_final_spearman_rho']:+.3f}**.",
            f"- Removing the intermediate direction shifted final projection by "
            f"**{patch['remove']['mean_delta_vs_baseline']:+.4f}** on average.",
            f"- Adding the intermediate direction shifted final projection by "
            f"**{patch['add']['mean_delta_vs_baseline']:+.4f}** on average.",
            f"- Subtracting the intermediate direction shifted final projection by "
            f"**{patch['subtract']['mean_delta_vs_baseline']:+.4f}** on average.",
            "",
            f"Conclusion: {conclusion}",
            "",
            "Interpretation: this is a causal forward-pass patch in an open encoder, "
            "but it is a CLIP frame-encoder approximation rather than a temporal "
            "V-JEPA hidden-state patch. V-JEPA exists in the repo through Modal and "
            "cached pooled embeddings, but the local checkout does not include a "
            "local V-JEPA checkpoint/decode stack for block-level hooks.",
            "",
        ]
    )


def _write_reports(payload: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text(_markdown(payload))


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch, image_processor_cls, model_cls, imageio_v3 = _load_optional_modules()
    records = _select_records(_load_records(args.data_root), max_videos=args.max_videos)
    train_i, test_i = _split_indices(
        len(records), test_frac=args.test_frac, seed=args.seed
    )
    local_files_only = not args.allow_downloads

    device = _resolve_device(torch, args.device)
    processor = image_processor_cls.from_pretrained(
        args.model_id,
        local_files_only=local_files_only,
    )
    model = model_cls.from_pretrained(
        args.model_id,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    layer_index = args.layer_index
    if layer_index < 0:
        layer_index = len(model.vision_model.encoder.layers) + layer_index
    if layer_index < 0 or layer_index >= len(model.vision_model.encoder.layers):
        raise ValueError(
            f"layer index {args.layer_index} out of range for "
            f"{len(model.vision_model.encoder.layers)} layers"
        )

    baseline = _forward_records(
        torch=torch,
        imageio_v3=imageio_v3,
        processor=processor,
        model=model,
        device=device,
        records=records,
        layer_index=layer_index,
        frames_per_video=args.frames_per_video,
        batch_videos=args.batch_videos,
        mode="baseline",
    )

    scores = np.asarray([r.score for r in records], dtype=np.float32)
    hidden_direction = _train_direction(
        baseline.hidden[train_i],
        scores[train_i],
        top_frac=args.top_frac,
        min_tail=args.min_tail,
    )
    final_direction = _train_direction(
        baseline.image_embeds[train_i],
        scores[train_i],
        top_frac=args.top_frac,
        min_tail=args.min_tail,
    )

    hidden_center = baseline.hidden[train_i].mean(axis=0).astype(np.float32)
    train_hidden_projection = (
        baseline.hidden[train_i] - hidden_center
    ) @ hidden_direction
    add_scale = float(
        args.patch_scale * max(float(train_hidden_projection.std()), 1e-6)
    )

    test_records = [records[int(i)] for i in test_i]
    remove = _forward_records(
        torch=torch,
        imageio_v3=imageio_v3,
        processor=processor,
        model=model,
        device=device,
        records=test_records,
        layer_index=layer_index,
        frames_per_video=args.frames_per_video,
        batch_videos=args.batch_videos,
        mode="remove",
        direction=hidden_direction,
        center=hidden_center,
        add_scale=add_scale,
    )
    add = _forward_records(
        torch=torch,
        imageio_v3=imageio_v3,
        processor=processor,
        model=model,
        device=device,
        records=test_records,
        layer_index=layer_index,
        frames_per_video=args.frames_per_video,
        batch_videos=args.batch_videos,
        mode="add",
        direction=hidden_direction,
        center=hidden_center,
        add_scale=add_scale,
    )
    subtract = _forward_records(
        torch=torch,
        imageio_v3=imageio_v3,
        processor=processor,
        model=model,
        device=device,
        records=test_records,
        layer_index=layer_index,
        frames_per_video=args.frames_per_video,
        batch_videos=args.batch_videos,
        mode="subtract",
        direction=hidden_direction,
        center=hidden_center,
        add_scale=add_scale,
    )

    test_scores = scores[test_i]
    test_hidden_projection = (
        baseline.hidden[test_i] - hidden_center
    ) @ hidden_direction
    test_final_projection = baseline.image_embeds[test_i] @ final_direction

    baseline_test_embeds = baseline.image_embeds[test_i]
    payload: dict[str, Any] = {
        "status": "ok",
        "probe_kind": "open_clip_frame_encoder_forward_patch",
        "model_id": args.model_id,
        "local_files_only": local_files_only,
        "data_root": str(args.data_root),
        "device": str(device),
        "layer_index": int(layer_index),
        "frames_per_video": int(args.frames_per_video),
        "n_available_local_scored_videos": int(len(_load_records(args.data_root))),
        "n_selected": int(len(records)),
        "n_train": int(len(train_i)),
        "n_test": int(len(test_i)),
        "seed": int(args.seed),
        "top_frac": float(args.top_frac),
        "test_frac": float(args.test_frac),
        "patch_scale_in_train_projection_std": float(args.patch_scale),
        "add_scale_raw_hidden_units": add_scale,
        "intermediate_probe": {
            "train_projection_std": float(train_hidden_projection.std()),
            "test_spearman_rho": _spearman(test_hidden_projection, test_scores),
            "all_selected_spearman_rho": _spearman(
                (baseline.hidden - hidden_center) @ hidden_direction,
                scores,
            ),
        },
        "forward_patching": {
            "baseline_final_spearman_rho": _spearman(
                test_final_projection,
                test_scores,
            ),
            "baseline_mean_projection": float(test_final_projection.mean()),
            "remove": _summarize_shift(
                baseline_test_embeds,
                remove.image_embeds,
                final_direction,
            ),
            "add": _summarize_shift(
                baseline_test_embeds,
                add.image_embeds,
                final_direction,
            ),
            "subtract": _summarize_shift(
                baseline_test_embeds,
                subtract.image_embeds,
                final_direction,
            ),
        },
        "test_video_ids": [records[int(i)].video_id for i in test_i],
    }

    remove_delta = payload["forward_patching"]["remove"]["mean_delta_vs_baseline"]
    add_delta = payload["forward_patching"]["add"]["mean_delta_vs_baseline"]
    subtract_delta = payload["forward_patching"]["subtract"]["mean_delta_vs_baseline"]
    if add_delta > 0 and subtract_delta < 0:
        if remove_delta < 0:
            conclusion = (
                "Feasible and directionally causal: adding the learned intermediate "
                "memorability direction increases the final memorability projection, "
                "while removing or subtracting it decreases that projection."
            )
        else:
            conclusion = (
                "Feasible and directionally causal for signed steering: adding the "
                "learned intermediate memorability direction increases the final "
                "memorability projection and subtracting it decreases that projection. "
                "Centered removal did not suppress the projection in this local sample."
            )
    elif not math.isclose(add_delta, 0.0, abs_tol=1e-6) or not math.isclose(
        remove_delta, 0.0, abs_tol=1e-6
    ):
        conclusion = (
            "Feasible as a forward patch, but the final projection response is "
            "not cleanly monotonic in this small local smoke run."
        )
    else:
        conclusion = (
            "Forward hook executed, but this run produced a near-zero final-space "
            "effect; increase --max-videos or patch a later layer."
        )
    payload["conclusion"] = conclusion
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--max-videos", type=int, default=64)
    parser.add_argument("--frames-per-video", type=int, default=2)
    parser.add_argument("--batch-videos", type=int, default=4)
    parser.add_argument("--layer-index", type=int, default=18)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--min-tail", type=int, default=4)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-scale", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    payload = run(args)
    _write_reports(payload, args.out_json, args.out_md)
    print(json.dumps(payload, indent=2))
    print(f"[done] wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
