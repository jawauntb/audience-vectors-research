"""Score Wan2.2 candidates with memorability plus CLIP preservation metrics.

The TRIBE/BMD projection is still the reward signal. CLIP seed-image and
prompt alignment are guardrails that help catch proxy wins caused by semantic
drift, like inserting a salient new subject that was not in the seed scene.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class Candidate:
    label: str
    seed_key: str
    prompt: str
    seed_image: Path
    video: Path
    manifest: dict[str, Any]


def seed_key_from_label(label: str) -> str:
    if label.endswith("_base") or label.endswith("_lora"):
        return label.rsplit("_", 1)[0]
    match = re.search(r"vid_idx\d{4}", label)
    return match.group(0) if match else label


def resolve_path(raw: str | None, *, roots: list[Path]) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    return None


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def frames_from_mp4(path: Path, n_frames: int) -> list[Image.Image]:
    frames = iio.imiter(path)
    sampled: list[np.ndarray] = []
    # These clips are short. Reading all frames keeps sampling deterministic
    # across local ffmpeg/imageio backends.
    all_frames = [np.asarray(frame) for frame in frames]
    if not all_frames:
        raise ValueError(f"no frames found in {path}")
    indices = np.linspace(0, len(all_frames) - 1, num=n_frames, dtype=int)
    for idx in indices:
        sampled.append(all_frames[int(idx)])
    return [Image.fromarray(frame).convert("RGB") for frame in sampled]


def normalized_np(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class ClipScorer:
    def __init__(self, model_id: str) -> None:
        self.device = choose_device()
        print(f"[clip-preserve] loading {model_id} on {self.device}", flush=True)
        self.processor: Any = CLIPProcessor.from_pretrained(model_id)
        model: Any = CLIPModel.from_pretrained(model_id)
        self.model: Any = model.to(self.device).eval()
        self.image_cache: dict[Path, np.ndarray] = {}
        self.video_cache: dict[Path, dict[str, Any]] = {}
        self.text_cache: dict[str, np.ndarray] = {}

    def image_embedding(self, path: Path) -> np.ndarray:
        if path in self.image_cache:
            return self.image_cache[path]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        out = normalized_np(emb)[0]
        self.image_cache[path] = out
        return out

    def video_embedding(self, path: Path, *, n_frames: int) -> dict[str, Any]:
        if path in self.video_cache:
            return self.video_cache[path]
        frames = frames_from_mp4(path, n_frames=n_frames)
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            frame_embs = self.model.get_image_features(**inputs)
        frame_np = normalized_np(frame_embs)
        mean_emb = frame_np.mean(axis=0)
        mean_emb = mean_emb / max(float(np.linalg.norm(mean_emb)), 1e-12)
        payload = {"mean": mean_emb, "frames": frame_np, "n_frames": len(frames)}
        self.video_cache[path] = payload
        return payload

    def text_embedding(self, text: str) -> np.ndarray:
        if text in self.text_cache:
            return self.text_cache[text]
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        out = normalized_np(emb)[0]
        self.text_cache[text] = out
        return out


def load_candidates(
    *,
    generated_dir: Path,
    tribe_report: dict[str, Any],
    seed_root: Path | None,
    project_root: Path,
) -> list[Candidate]:
    manifest_by_label = {
        str(row.get("label")): row
        for row in tribe_report.get("manifest", [])
        if isinstance(row, dict) and row.get("label")
    }
    score_labels = [str(row["label"]) for row in tribe_report["scores"]]
    roots = [generated_dir, project_root]
    if seed_root is not None:
        roots.insert(0, seed_root)

    candidates: list[Candidate] = []
    for label in score_labels:
        row = dict(manifest_by_label.get(label, {}))
        prompt = str(row.get("prompt", ""))
        seed_key = str(row.get("bmd_name") or seed_key_from_label(label))
        seed_image = resolve_path(str(row.get("seed_image", "")), roots=roots)
        video = generated_dir / f"{label}.mp4"
        if not video.exists():
            video = resolve_path(str(row.get("local_path") or row.get("video")), roots=roots) or video
        if seed_image is None or not seed_image.exists():
            raise FileNotFoundError(f"seed image missing for {label}: {row.get('seed_image')}")
        if not video.exists():
            raise FileNotFoundError(f"video missing for {label}: {video}")
        candidates.append(
            Candidate(
                label=label,
                seed_key=seed_key,
                prompt=prompt,
                seed_image=seed_image,
                video=video,
                manifest=row,
            )
        )
    return sorted(candidates, key=lambda item: item.label)


def summarize_by_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for row in rows:
        by_seed.setdefault(row["seed_key"], []).append(row)

    out: dict[str, Any] = {}
    for seed_key, seed_rows in sorted(by_seed.items()):
        mem_best = max(seed_rows, key=lambda row: row["v_mem_projection"])
        composite_best = max(seed_rows, key=lambda row: row["composite_score"])
        gated = [row for row in seed_rows if row["passes_preservation_gate"]]
        gated_best = max(gated, key=lambda row: row["v_mem_projection"]) if gated else None
        payload: dict[str, Any] = {
            "n": len(seed_rows),
            "v_mem_best_label": mem_best["label"],
            "v_mem_best_projection": mem_best["v_mem_projection"],
            "composite_best_label": composite_best["label"],
            "composite_best_score": composite_best["composite_score"],
            "winner_changed_by_composite": mem_best["label"] != composite_best["label"],
            "n_passing_gate": len(gated),
        }
        if gated_best is not None:
            payload.update(
                {
                    "gated_best_label": gated_best["label"],
                    "gated_best_projection": gated_best["v_mem_projection"],
                    "winner_changed_by_gate": mem_best["label"] != gated_best["label"],
                }
            )
        else:
            payload.update(
                {
                    "gated_best_label": None,
                    "gated_best_projection": None,
                    "winner_changed_by_gate": True,
                }
            )
        if len(seed_rows) == 2:
            by_variant = {
                str(row.get("variant") or row["label"].rsplit("_", 1)[-1]): row
                for row in seed_rows
            }
            base = by_variant.get("base")
            lora = by_variant.get("lora")
            if base and lora:
                payload["pair_delta"] = {
                    "v_mem_projection": lora["v_mem_projection"] - base["v_mem_projection"],
                    "seed_image_cosine": lora["seed_image_cosine"] - base["seed_image_cosine"],
                    "prompt_clip_cosine": lora["prompt_clip_cosine"] - base["prompt_clip_cosine"],
                    "composite_score": lora["composite_score"] - base["composite_score"],
                    "lora_passes_gate": bool(lora["passes_preservation_gate"]),
                }
        out[seed_key] = payload
    return out


def write_markdown(report: dict[str, Any], path: Path, *, top_n: int) -> None:
    rows = report["rows"]
    summary = report["summary"]
    md = [
        "# Wan2.2 Composite Preservation Score",
        "",
        "Composite score = z(TRIBE/BMD projection) + image_weight * z(CLIP seed-image cosine) + prompt_weight * z(CLIP prompt cosine).",
        "",
        "## Summary",
        "",
        f"- Candidates: **{summary['n_candidates']}**",
        f"- Seeds: **{summary['n_seeds']}**",
        f"- Composite changed winner for **{summary['n_composite_winner_changes']}** seeds",
        f"- Preservation gate changed winner for **{summary['n_gate_winner_changes']}** seeds",
        f"- Gate pass rate: **{summary['gate_pass_rate']:.3f}**",
        f"- Max seed-image cosine drop from seed best: **{summary['max_seed_cosine_drop_from_best']}**",
        "",
        "## Top Composite Candidates",
        "",
        "| rank | label | seed | v_mem | seed_cos | prompt_cos | composite | gate |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(
        sorted(rows, key=lambda item: item["composite_score"], reverse=True)[:top_n],
        start=1,
    ):
        gate = "yes" if row["passes_preservation_gate"] else "no"
        md.append(
            f"| {idx} | `{row['label']}` | `{row['seed_key']}` | "
            f"{row['v_mem_projection']:.4f} | {row['seed_image_cosine']:.4f} | "
            f"{row['prompt_clip_cosine']:.4f} | {row['composite_score']:.4f} | {gate} |"
        )

    changed = [
        (seed, payload)
        for seed, payload in report["by_seed"].items()
        if payload["winner_changed_by_composite"] or payload["winner_changed_by_gate"]
    ]
    if changed:
        md += [
            "",
            "## Changed Winners",
            "",
            "| seed | v_mem best | composite best | gated best |",
            "|---|---|---|---|",
        ]
        for seed, payload in changed[:top_n]:
            md.append(
                f"| `{seed}` | `{payload['v_mem_best_label']}` | "
                f"`{payload['composite_best_label']}` | `{payload['gated_best_label']}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--tribe-report", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-weight", type=float, default=0.75)
    parser.add_argument("--prompt-weight", type=float, default=0.25)
    parser.add_argument("--min-seed-cosine", type=float, default=None)
    parser.add_argument("--min-prompt-cosine", type=float, default=None)
    parser.add_argument("--max-seed-cosine-drop-from-best", type=float, default=None)
    parser.add_argument("--max-prompt-cosine-drop-from-best", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    project_root = Path.cwd()
    tribe_report = json.loads(args.tribe_report.read_text())
    score_by_label = {
        str(row["label"]): float(row["v_mem_projection"])
        for row in tribe_report["scores"]
    }
    candidates = load_candidates(
        generated_dir=args.generated_dir,
        tribe_report=tribe_report,
        seed_root=args.seed_root,
        project_root=project_root,
    )

    scorer = ClipScorer(args.model_id)
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[clip-preserve] {idx}/{len(candidates)} {candidate.label}", flush=True)
        seed_emb = scorer.image_embedding(candidate.seed_image)
        video_payload = scorer.video_embedding(candidate.video, n_frames=args.frames)
        video_emb = np.asarray(video_payload["mean"], dtype=np.float32)
        frame_embs = np.asarray(video_payload["frames"], dtype=np.float32)
        text_emb = scorer.text_embedding(candidate.prompt) if candidate.prompt else None
        seed_frame_cosines = frame_embs @ seed_emb
        prompt_cosine = float(video_emb @ text_emb) if text_emb is not None else 0.0
        variant = candidate.manifest.get("variant")
        if variant is None and candidate.label.endswith(("_base", "_lora")):
            variant = candidate.label.rsplit("_", 1)[-1]
        rows.append(
            {
                "label": candidate.label,
                "seed_key": candidate.seed_key,
                "variant": variant,
                "video": str(candidate.video),
                "seed_image": str(candidate.seed_image),
                "prompt": candidate.prompt,
                "v_mem_projection": score_by_label[candidate.label],
                "seed_image_cosine": float(video_emb @ seed_emb),
                "seed_image_frame_max_cosine": float(seed_frame_cosines.max()),
                "seed_image_frame_min_cosine": float(seed_frame_cosines.min()),
                "prompt_clip_cosine": prompt_cosine,
                "n_clip_frames": int(video_payload["n_frames"]),
            }
        )

    mem_z = zscore(np.asarray([row["v_mem_projection"] for row in rows], dtype=np.float32))
    image_z = zscore(np.asarray([row["seed_image_cosine"] for row in rows], dtype=np.float32))
    prompt_z = zscore(np.asarray([row["prompt_clip_cosine"] for row in rows], dtype=np.float32))
    seed_bests: dict[str, dict[str, float]] = {}
    for row in rows:
        bests = seed_bests.setdefault(
            row["seed_key"],
            {"seed_image_cosine": -1.0, "prompt_clip_cosine": -1.0},
        )
        bests["seed_image_cosine"] = max(
            bests["seed_image_cosine"], float(row["seed_image_cosine"])
        )
        bests["prompt_clip_cosine"] = max(
            bests["prompt_clip_cosine"], float(row["prompt_clip_cosine"])
        )
    for row, mz, iz, pz in zip(rows, mem_z, image_z, prompt_z, strict=True):
        seed_best = seed_bests[row["seed_key"]]
        seed_drop = seed_best["seed_image_cosine"] - float(row["seed_image_cosine"])
        prompt_drop = seed_best["prompt_clip_cosine"] - float(row["prompt_clip_cosine"])
        passes_seed = (
            True
            if args.min_seed_cosine is None
            else row["seed_image_cosine"] >= args.min_seed_cosine
        )
        passes_prompt = (
            True
            if args.min_prompt_cosine is None
            else row["prompt_clip_cosine"] >= args.min_prompt_cosine
        )
        passes_seed_drop = (
            True
            if args.max_seed_cosine_drop_from_best is None
            else seed_drop <= args.max_seed_cosine_drop_from_best
        )
        passes_prompt_drop = (
            True
            if args.max_prompt_cosine_drop_from_best is None
            else prompt_drop <= args.max_prompt_cosine_drop_from_best
        )
        row["seed_image_cosine_drop_from_seed_best"] = float(seed_drop)
        row["prompt_clip_cosine_drop_from_seed_best"] = float(prompt_drop)
        row["v_mem_z"] = float(mz)
        row["seed_image_cosine_z"] = float(iz)
        row["prompt_clip_cosine_z"] = float(pz)
        row["composite_score"] = float(mz + args.image_weight * iz + args.prompt_weight * pz)
        row["passes_seed_gate"] = bool(passes_seed)
        row["passes_prompt_gate"] = bool(passes_prompt)
        row["passes_seed_drop_gate"] = bool(passes_seed_drop)
        row["passes_prompt_drop_gate"] = bool(passes_prompt_drop)
        row["passes_preservation_gate"] = bool(
            passes_seed and passes_prompt and passes_seed_drop and passes_prompt_drop
        )

    by_seed = summarize_by_seed(rows)
    summary = {
        "n_candidates": len(rows),
        "n_seeds": len(by_seed),
        "model_id": args.model_id,
        "frames": args.frames,
        "image_weight": args.image_weight,
        "prompt_weight": args.prompt_weight,
        "min_seed_cosine": args.min_seed_cosine,
        "min_prompt_cosine": args.min_prompt_cosine,
        "max_seed_cosine_drop_from_best": args.max_seed_cosine_drop_from_best,
        "max_prompt_cosine_drop_from_best": args.max_prompt_cosine_drop_from_best,
        "gate_pass_rate": float(np.mean([row["passes_preservation_gate"] for row in rows])),
        "n_composite_winner_changes": int(
            sum(payload["winner_changed_by_composite"] for payload in by_seed.values())
        ),
        "n_gate_winner_changes": int(
            sum(payload["winner_changed_by_gate"] for payload in by_seed.values())
        ),
        "seed_image_cosine_mean": float(np.mean([row["seed_image_cosine"] for row in rows])),
        "prompt_clip_cosine_mean": float(np.mean([row["prompt_clip_cosine"] for row in rows])),
    }
    report = {
        "summary": summary,
        "rows": sorted(rows, key=lambda row: row["composite_score"], reverse=True),
        "by_seed": by_seed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"[clip-preserve] wrote {args.out_json}", flush=True)
    if args.out_md is not None:
        write_markdown(report, args.out_md, top_n=args.top_n)
        print(f"[clip-preserve] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
