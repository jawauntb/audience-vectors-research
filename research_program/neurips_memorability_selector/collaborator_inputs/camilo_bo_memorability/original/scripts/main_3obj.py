"""
BO-Memorability — 3-objective pipeline for local GPU (RTX 5080 / CUDA 12.8+).

Extension of main_gpu.py: adds VideoQualityScorer (R3D-18) as a 3rd objective.

Three objectives maximized simultaneously via qLogNoisyExpectedHypervolumeImprovement:
  1. TRIBE score    — cortical memorability (TRIBE v2)
  2. CLIP score     — visual fidelity to the prompt (CLIP ViT-H/14)
  3. Quality score  — R3D-18 similarity to the no-steering baseline (alpha=0)

Search space (identical to main_gpu.py):
  alpha       in [-10, +10]  — memorability steering strength
  guidance    in [1,   10]   — classifier-free guidance of SVD-XT
  seed_idx    in {0..15}     — index of the seed image

Usage:
  python scripts/main_3obj.py                     # recommended defaults
  python scripts/main_3obj.py --steps 10          # quick test (10 steps)
  python scripts/main_3obj.py --n-initial 12      # initial Sobol points
  python scripts/main_3obj.py --n-iterations 10   # BO iterations
  python scripts/main_3obj.py --batch-size 2      # candidates per round
  python scripts/main_3obj.py --no-resume         # start from scratch
  python scripts/main_3obj.py --cpu-offload       # for GPUs < 12 GB VRAM
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from itertools import cycle, islice
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"   # safe encoding on Windows

import matplotlib
matplotlib.use("Agg")

# Project root = parent folder of scripts/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from bo_mem.acquisition.qehvi import compute_hypervolume, get_pareto_front, optimize_qehvi
from bo_mem.generator.svd_generator import SVDGenerator
from bo_mem.objectives.clip_scorer import CLIPScorer
from bo_mem.objectives.fvd_scorer import VideoQualityScorer
from bo_mem.surrogate.gp_model import build_model_list, fit_model

# ── Blackwell compatibility (RTX 5080, sm_120) ────────────────────────────────
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

# ── Paths ─────────────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS_DIR      = PROJECT_ROOT / "seeds"
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
OUTPUT_DIR     = PROJECT_ROOT / "outputs" / "gpu_run_3obj"   # separate folder from the 2D run
VIDEOS_DIR     = OUTPUT_DIR / "videos"
STATE_FILE     = OUTPUT_DIR / "bo_state.pt"
REF_VIDEOS_DIR = OUTPUT_DIR / "reference_videos"
REF_STATS_FILE = OUTPUT_DIR / "quality_reference.npz"

# ── Search space ──────────────────────────────────────────────────────────────
FULL_BOUNDS = torch.tensor(
    [[-10.0, 1.0,  0.0],
     [ 10.0, 10.0, 15.0]],
    dtype=torch.double,
)
CAT_DIMS    = [2]
N_SEED_POOL = 16
N_REFERENCE = 8      # neutral videos (alpha=0) to calibrate the quality scorer


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    import re
    clean = re.sub(r"\[/?[^\[\]]*\]", "", str(msg))
    sys.stderr.write(clean + "\n")
    sys.stderr.flush()


def rule(title: str = "") -> None:
    w = 72
    if title:
        log(f"── {title} " + "─" * max(0, w - len(title) - 4))
    else:
        log("─" * w)


# ── Backend TRIBE v2 ──────────────────────────────────────────────────────────

class TribeBackend:
    """Wrapper around tribev2.TribeModel with automatic download and Windows patches."""

    HF_REPO     = "facebook/tribev2"
    HF_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model  = None
        self._cache  = OUTPUT_DIR / "tribe_weights" / "tribev2_weights"

    @staticmethod
    def _patch_yaml() -> None:
        import pathlib, yaml

        def _ctor(loader, node):
            try:
                parts = loader.construct_sequence(node, deep=True)
                return pathlib.Path(*parts) if parts else pathlib.Path(".")
            except Exception:
                return pathlib.Path(".")

        for tag in [
            "!!python/object/apply:pathlib.PosixPath",
            "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
            "!!python/object:pathlib.PosixPath",
        ]:
            try:
                yaml.add_constructor(tag, _ctor, Loader=yaml.UnsafeLoader)
            except Exception:
                pass

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from tribev2 import TribeModel  # type: ignore

        if self._model is not None:
            return

        self._patch_yaml()
        self._cache.mkdir(parents=True, exist_ok=True)

        log("Downloading / verifying TRIBE v2 weights...")
        try:
            path = snapshot_download(
                self.HF_REPO,
                revision=self.HF_REVISION,
                local_dir=str(self._cache),
            )
        except Exception as e:
            if any(c in str(e) for c in ("401", "403", "gated")):
                log(
                    "\nWARNING: facebook/tribev2 requires HuggingFace authentication.\n"
                    "  1. Accept the terms at https://huggingface.co/facebook/tribev2\n"
                    "  2. Generate a token at https://huggingface.co/settings/tokens\n"
                    "  3. Run: huggingface-cli login\n"
                )
                raise SystemExit(1) from e
            raise

        self._model = TribeModel.from_pretrained(
            path,
            device=self._device,
            config_update={"data.num_workers": 0},
        )
        log("TRIBE v2 loaded.")

    def predict(self, video_path: Path) -> np.ndarray:
        if self._model is None:
            self.load()
        events = self._model.get_events_dataframe(video_path=str(video_path))
        preds, _ = self._model.predict(events, verbose=False)
        return np.array(preds, dtype="float32").mean(axis=0)   # (20484,)


# ── Seeds ─────────────────────────────────────────────────────────────────────

def load_seeds(n_pool: int = N_SEED_POOL) -> list[dict]:
    with open(SEEDS_DIR / "prompts.json", encoding="utf-8") as f:
        all_prompts = json.load(f)

    available = []
    for e in all_prompts:
        si = e.get("seed_image")
        if not si:
            continue
        p = PROJECT_ROOT / si
        if p.exists():
            available.append({
                "idx": e["idx"],
                "bmd_name": e["bmd_name"],
                "prompt": e["prompt"],
                "image_path": p,
            })

    if not available:
        raise FileNotFoundError(
            f"No seed image found in {SEEDS_DIR}.\n"
            "Make sure the PNG files are in seeds/."
        )

    pool = list(islice(cycle(available), n_pool))
    log(f"{len(available)} seeds available → pool of {n_pool}")
    return pool


# ── Quality reference ─────────────────────────────────────────────────────────

def build_quality_reference(
    generator: SVDGenerator,
    seeds_pool: list[dict],
    quality_scorer: VideoQualityScorer,
    num_inference_steps: int = 25,
) -> None:
    """
    Generates N_REFERENCE videos with alpha=0 (no steering) and calibrates the quality scorer.
    From the second run onward, loads the stats from disk — does not regenerate.
    """
    REF_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if REF_STATS_FILE.exists():
        quality_scorer.load(REF_STATS_FILE)
        log(f"Quality reference loaded: {REF_STATS_FILE.name}")
        return

    rule("Generating reference videos (alpha=0, only on the 1st run)")
    ref_paths: list[Path] = []
    for i in range(N_REFERENCE):
        seed_entry = seeds_pool[i % len(seeds_pool)]
        img = Image.open(seed_entry["image_path"]).convert("RGB")
        out = REF_VIDEOS_DIR / f"ref_{i:02d}.mp4"
        t0 = time.time()
        generator.generate(
            conditioning_image  = img,
            alphas              = torch.tensor([0.0]),
            guidance_scale      = 3.0,
            seed                = 9000 + i,
            num_inference_steps = num_inference_steps,
            output_path         = out,
        )
        ref_paths.append(out)
        log(f"  ref_{i:02d}.mp4  ({time.time()-t0:.1f}s)")

    log("Calibrating reference distribution...")
    quality_scorer.fit_reference(ref_paths)
    quality_scorer.save(REF_STATS_FILE)
    log(f"Reference saved: {REF_STATS_FILE.name}")


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_video(
    video_path: Path,
    prompt: str,
    tribe: TribeBackend,
    clip: CLIPScorer,
    v_mem: np.ndarray,
    quality: VideoQualityScorer,
) -> tuple[float, float, float]:
    """Returns (tribe_score, clip_score, quality_score)."""
    activation    = tribe.predict(video_path)
    tribe_score   = float(np.dot(activation, v_mem))
    clip_score    = clip.score(video_path, prompt)
    quality_score = quality.score(video_path)
    return tribe_score, clip_score, quality_score


def evaluate_batch(
    tasks: list[dict],
    generator: SVDGenerator,
    tribe: TribeBackend,
    clip: CLIPScorer,
    seeds_pool: list[dict],
    v_mem: np.ndarray,
    quality: VideoQualityScorer,
    iteration: int,
    num_inference_steps: int = 25,
) -> tuple[list[list[float]], list[dict]]:
    """
    Generates and scores each task.
    Returns (scores [[tribe, clip, quality], ...], list of metadata).
    """
    scores_list: list[list[float]] = []
    meta_list:   list[dict]        = []

    for i, task in enumerate(tasks):
        seed_idx   = int(task["seed_idx"]) % N_SEED_POOL
        seed_entry = seeds_pool[seed_idx]
        img        = Image.open(seed_entry["image_path"]).convert("RGB")
        out_path   = VIDEOS_DIR / f"iter{iteration:03d}_t{i:02d}_{task['task_id']}.mp4"

        log(
            f"  ▶ {task['task_id']}  "
            f"alpha={task['alpha']:+.3f}  guidance={task['guidance']:.2f}  "
            f"seed={seed_idx} ({seed_entry['bmd_name']})"
        )

        t0 = time.time()
        generator.generate(
            conditioning_image  = img,
            alphas              = torch.tensor([float(task["alpha"])]),
            guidance_scale      = float(task["guidance"]),
            seed                = int(task.get("noise_seed", i + iteration * 100)),
            num_inference_steps = num_inference_steps,
            output_path         = out_path,
        )
        log(f"    generated in {time.time()-t0:.1f}s → {out_path.name}")

        ts, cs, qs = score_video(out_path, seed_entry["prompt"], tribe, clip, v_mem, quality)
        log(f"    tribe={ts:.4f}  clip={cs:.4f}  quality={qs:.4f}")

        scores_list.append([ts, cs, qs])
        meta_list.append({
            **task,
            "filename":      out_path.name,
            "prompt":        seed_entry["prompt"],
            "tribe_score":   ts,
            "clip_score":    cs,
            "quality_score": qs,
        })

    return scores_list, meta_list


# ── BO step ───────────────────────────────────────────────────────────────────

def bo_step(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    batch_size: int,
    iteration: int,
) -> list[dict]:
    """Fits the GP surrogate (3 GPs) and proposes the next candidates via qLogNEHVI."""
    log("Fitting GP (BoTorch, 3 objectives)...")
    model = build_model_list(train_x, train_y, FULL_BOUNDS, cat_dims=CAT_DIMS)
    fit_model(model)

    log("Optimizing qLogNEHVI (3D)...")
    new_x = optimize_qehvi(
        model      = model,
        train_x    = train_x,
        train_y    = train_y,
        bounds     = FULL_BOUNDS,
        batch_size = batch_size,
        cat_dims   = CAT_DIMS,
        n_seeds    = N_SEED_POOL,
    )
    del model
    gc.collect()

    tasks = []
    for j, x in enumerate(new_x):
        tasks.append({
            "task_id":    f"bo{iteration:02d}_cand{j:02d}",
            "alpha":      float(x[0].item()),
            "guidance":   float(x[1].item()),
            "seed_idx":   int(x[2].item()) % N_SEED_POOL,
            "noise_seed": iteration * 100 + j,
        })
        log(
            f"  Candidate {j}: alpha={tasks[-1]['alpha']:+.3f}  "
            f"guidance={tasks[-1]['guidance']:.2f}  seed={tasks[-1]['seed_idx']}"
        )
    return tasks


# ── State ─────────────────────────────────────────────────────────────────────

def save_state(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    all_meta: list[dict],
    hv_history: list[float],
    iteration: int,
) -> None:
    torch.save(
        {"train_x": train_x, "train_y": train_y,
         "all_meta": all_meta, "hv_history": hv_history,
         "iteration": iteration},
        STATE_FILE,
    )
    log(f"  State saved ({len(train_x)} obs, iter={iteration})")


def load_state(resume: bool) -> tuple[torch.Tensor, torch.Tensor, list, list, int]:
    if resume and STATE_FILE.exists():
        s = torch.load(STATE_FILE, weights_only=False)
        log(f"State loaded: {len(s['train_x'])} obs, iter={s['iteration']}")
        return (
            s["train_x"].double(),
            s["train_y"].double(),
            s.get("all_meta", []),
            s.get("hv_history", []),
            s.get("iteration", 0),
        )
    log("Starting from scratch.")
    return (
        torch.empty(0, 3, dtype=torch.double),   # train_x: (alpha, guidance, seed_idx)
        torch.empty(0, 3, dtype=torch.double),   # train_y: (tribe, clip, quality)
        [], [], 0,
    )


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_results(train_y: torch.Tensor, hv_history: list[float], n_init: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BO-Memorability — 3 objectives (TRIBE · CLIP · Quality)", fontsize=13, fontweight="bold")

    t    = train_y[:, 0].numpy()
    c    = train_y[:, 1].numpy()
    q    = train_y[:, 2].numpy()
    n_bo = max(0, len(train_y) - n_init)

    kw_s = dict(color="#6C8EBF", alpha=0.75, s=55, label=f"Sobol (n={n_init})")
    kw_b = dict(color="#9B7B6A", s=70, marker="D", alpha=0.75, label=f"BO (n={n_bo})")

    # Panel 1: TRIBE × CLIP
    ax = axes[0]
    ax.scatter(t[:n_init], c[:n_init], **kw_s)
    if n_bo > 0:
        ax.scatter(t[n_init:], c[n_init:], **kw_b)
    pf = get_pareto_front(train_y)
    si = np.argsort(pf[:, 0].numpy())
    ax.plot(pf[si, 0].numpy(), pf[si, 1].numpy(), "o--",
            color="#4A7A4A", lw=1.8, ms=7, label=f"Pareto (n={len(pf)})")
    ax.set_xlabel("TRIBE score (memorability)")
    ax.set_ylabel("CLIP score (fidelity)")
    ax.set_title("Memorability × Fidelity")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: TRIBE × Quality
    ax = axes[1]
    ax.scatter(t[:n_init], q[:n_init], **kw_s)
    if n_bo > 0:
        ax.scatter(t[n_init:], q[n_init:], **kw_b)
    ax.set_xlabel("TRIBE score (memorability)")
    ax.set_ylabel("Quality score R3D-18")
    ax.set_title("Memorability × Quality")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: HV
    ax = axes[2]
    if hv_history:
        iters = list(range(1, len(hv_history) + 1))
        ax.plot(iters, hv_history, "o-", color="#4A7A4A", lw=2, ms=7)
        ax.fill_between(iters, hv_history, alpha=0.12, color="#4A7A4A")
        if hv_history[0] > 0:
            gain = hv_history[-1] / hv_history[0]
            ax.set_title(f"Hypervolume (×{gain:.0f})")
        else:
            ax.set_title("Hypervolume convergence")
        ax.set_xlabel("BO round"); ax.set_ylabel("Dominated hypervolume")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "HV available\nafter 2+ rounds",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out = OUTPUT_DIR / "bo_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Figure saved: {out}")


def print_summary(train_y: torch.Tensor, hv_history: list[float]) -> None:
    pf = get_pareto_front(train_y)
    rule("Final Summary")
    log(f"  Total evaluations      : {len(train_y)}")
    log(f"  Pareto points (3D)     : {len(pf)}")
    if hv_history:
        log(f"  Initial HV             : {hv_history[0]:.6f}")
        log(f"  Final HV               : {hv_history[-1]:.6f}")
        if hv_history[0] > 0:
            log(f"  HV gain                : ×{hv_history[-1]/hv_history[0]:.1f}")
    log(f"  Best TRIBE score       : {float(train_y[:, 0].max()):.4f}")
    log(f"  Best CLIP score        : {float(train_y[:, 1].max()):.4f}")
    log(f"  Best quality score     : {float(train_y[:, 2].max()):.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-initial",    type=int,  default=12,     help="Initial Sobol points (default: 12)")
    p.add_argument("--n-iterations", type=int,  default=10,     help="BO iterations (default: 10)")
    p.add_argument("--batch-size",   type=int,  default=2,      help="Candidates per round (default: 2)")
    p.add_argument("--steps",        type=int,  default=25,     help="SVD denoising steps (default: 25)")
    p.add_argument("--no-resume",    action="store_true",       help="Ignore checkpoint and start from scratch")
    p.add_argument("--cpu-offload",  action="store_true",       help="Enable cpu_offload on SVD-XT (GPUs < 12 GB VRAM)")
    p.add_argument("--device",       type=str,  default="cuda", help="PyTorch device (default: cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    rule("BO-Memorability 3 objectives (TRIBE · CLIP · Quality)")
    if torch.cuda.is_available():
        log(f"  GPU    : {torch.cuda.get_device_name(0)}")
        log(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"  Device : {args.device}  |  cpu_offload : {args.cpu_offload}")
    log(f"  n_init : {args.n_initial}  |  n_iter : {args.n_iterations}  |  batch : {args.batch_size}")
    log(f"  steps  : {args.steps}  |  output : {OUTPUT_DIR}")
    rule()

    # ── Seeds and artifacts ───────────────────────────────────────────────────
    seeds_pool = load_seeds()
    v_raw = np.load(ARTIFACTS_DIR / "v_mem.npz")["direction"].astype("float32")
    v_mem = v_raw / np.linalg.norm(v_raw)
    log(f"v_mem: shape={v_mem.shape}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    train_x, train_y, all_meta, hv_history, start_iter = load_state(not args.no_resume)
    scored_ids = {m["task_id"] for m in all_meta}

    # ── Models ────────────────────────────────────────────────────────────────
    log("Loading CLIP ViT-H/14...")
    clip_scorer = CLIPScorer(device=args.device)
    log("CLIP loaded.")

    tribe_backend = TribeBackend(device=args.device)
    tribe_backend.load()

    # RTX 5080 (16 GB VRAM): cpu_offload=False by default → faster
    # GPUs < 12 GB: pass --cpu-offload
    log(f"Loading SVD-XT (cpu_offload={args.cpu_offload})...")
    generator = SVDGenerator(
        device      = args.device,
        dtype       = torch.float16,
        cpu_offload = args.cpu_offload,
    )
    generator._load_pipeline()
    log("SVD-XT loaded.")

    log("Initializing VideoQualityScorer (R3D-18)...")
    quality_scorer = VideoQualityScorer(device=args.device)
    build_quality_reference(generator, seeds_pool, quality_scorer, num_inference_steps=args.steps)
    log("VideoQualityScorer ready.")

    # ── Phase 1: Sobol ────────────────────────────────────────────────────────
    if start_iter == 0 and len(train_x) == 0:
        rule("Phase 1 — Sobol initialization")
        from torch.quasirandom import SobolEngine

        eng = SobolEngine(dimension=3, scramble=True, seed=42)
        s   = eng.draw(args.n_initial)
        sobol_tasks = [
            {
                "task_id":    f"sobol_{i:03d}",
                "alpha":      float(s[i, 0] * 20.0 - 10.0),
                "guidance":   float(s[i, 1] *  9.0 +  1.0),
                "seed_idx":   int(s[i, 2] * 15),
                "noise_seed": i,
            }
            for i in range(args.n_initial)
        ]

        new_tasks = [t for t in sobol_tasks if t["task_id"] not in scored_ids]
        log(f"{len(new_tasks)}/{len(sobol_tasks)} Sobol points to evaluate")

        if new_tasks:
            scores, meta = evaluate_batch(
                tasks               = new_tasks,
                generator           = generator,
                tribe               = tribe_backend,
                clip                = clip_scorer,
                seeds_pool          = seeds_pool,
                v_mem               = v_mem,
                quality             = quality_scorer,
                iteration           = 0,
                num_inference_steps = args.steps,
            )
            new_x = torch.tensor(
                [[t["alpha"], t["guidance"], float(t["seed_idx"])] for t in new_tasks],
                dtype=torch.double,
            )
            new_y = torch.tensor(scores, dtype=torch.double)
            train_x = torch.cat([train_x, new_x]) if train_x.numel() else new_x
            train_y = torch.cat([train_y, new_y]) if train_y.numel() else new_y
            all_meta.extend(meta)
            scored_ids.update(t["task_id"] for t in new_tasks)

        hv = compute_hypervolume(train_y)
        hv_history.append(hv)
        start_iter = 1
        log(f"Sobol complete | HV={hv:.6f}")
        save_state(train_x, train_y, all_meta, hv_history, start_iter)

    # ── Phase 2: BO ───────────────────────────────────────────────────────────
    rule("Phase 2 — BO loop (qLogNEHVI, 3 objectives)")
    assert len(train_x) >= 2, "At least 2 observations are required to fit the GP."

    for it in range(start_iter, start_iter + args.n_iterations):
        rule(f"BO iteration {it} / {start_iter + args.n_iterations - 1}  ({len(train_x)} obs)")

        candidates = bo_step(train_x, train_y, args.batch_size, it)

        scores, meta = evaluate_batch(
            tasks               = candidates,
            generator           = generator,
            tribe               = tribe_backend,
            clip                = clip_scorer,
            seeds_pool          = seeds_pool,
            v_mem               = v_mem,
            quality             = quality_scorer,
            iteration           = it,
            num_inference_steps = args.steps,
        )

        new_x   = torch.tensor(
            [[t["alpha"], t["guidance"], float(t["seed_idx"])] for t in candidates],
            dtype=torch.double,
        )
        new_y   = torch.tensor(scores, dtype=torch.double)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        all_meta.extend(meta)

        hv      = compute_hypervolume(train_y)
        pf_size = len(get_pareto_front(train_y))
        hv_history.append(hv)
        log(f"Iteration {it} | HV={hv:.6f} | Pareto={pf_size} points")

        save_state(train_x, train_y, all_meta, hv_history, it + 1)

    # ── Results ───────────────────────────────────────────────────────────────
    print_summary(train_y, hv_history)
    plot_results(train_y, hv_history, n_init=args.n_initial)

    results_path = OUTPUT_DIR / "all_results.json"
    results_path.write_text(
        json.dumps(
            {"n_initial": args.n_initial, "n_iterations": args.n_iterations,
             "all_meta": all_meta, "hv_history": hv_history},
            indent=2,
        )
    )
    log(f"Results saved: {results_path}")
    rule("Done!")


if __name__ == "__main__":
    main()
