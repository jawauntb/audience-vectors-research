"""BO Memorability -- Unified script for the RTX 5080 (local GPU).

Runs the entire pipeline on the local GPU:
  1. Video generation with SVD-XT (cpu_offload to save VRAM)
  2. TRIBE v2 + CLIP scoring on the GPU
  3. Multi-objective BO (qNEHVI) on the CPU with BoTorch

Search space:
  x = (alpha, guidance_scale, seed_idx) -- 3D, same as the notebooks
  alpha       in [-10, 10]  -- memorability steering coefficient
  guidance    in [1, 10]    -- classifier-free guidance
  seed_idx    in {0..15}    -- pool of 5 seeds cycled up to 16

Usage:
  python main_gpu.py                     # run with defaults
  python main_gpu.py --n-initial 6       # fewer Sobol points (faster)
  python main_gpu.py --n-iterations 5    # fewer BO iterations
  python main_gpu.py --batch-size 2      # candidates per BO round
  python main_gpu.py --steps 10          # SVD inference steps (fast)
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

# Force UTF-8 in Python to avoid encoding errors on Windows
os.environ["PYTHONUTF8"] = "1"

import matplotlib
matplotlib.use("Agg")  # display-less backend for Windows

# -- Add src/ to the Python path ----------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from bo_mem.acquisition.qehvi import (
    compute_hypervolume,
    get_pareto_front,
    optimize_qehvi,
)
from bo_mem.generator.svd_generator import SVDGenerator
from bo_mem.objectives.clip_scorer import CLIPScorer
from bo_mem.surrogate.gp_model import build_model_list, fit_model


def _err(msg: str, **kwargs) -> None:  # noqa: ANN003  (accepts flush=True without using it)
    """Writes to stderr with immediate flush (always captured by 2>&1)."""
    import sys as _sys
    import re
    clean = re.sub(r"\[/?[^\[\]]*\]", "", str(msg))
    _sys.stderr.write(clean + "\n")
    _sys.stderr.flush()


def log(msg: str) -> None:
    _err(msg)


def rule(title: str = "") -> None:
    w = 80
    if title:
        pad = max(0, w - len(title) - 4)
        _err(f"-- {title} " + "-" * pad)
    else:
        _err("-" * w)

# -- Constants -----------------------------------------------------------------
torch.backends.cudnn.benchmark = False   # avoids CUDNN_STATUS_EXECUTION_FAILED on sm_120
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS_DIR     = PROJECT_ROOT / "seeds"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "gpu_run"
VIDEOS_DIR    = OUTPUT_DIR / "videos"
STATE_FILE    = OUTPUT_DIR / "bo_state.pt"

FULL_BOUNDS = torch.tensor(
    [[-10.0, 1.0,  0.0],
     [ 10.0, 10.0, 15.0]],
    dtype=torch.double,
)
CAT_DIMS = [2]   # seed_idx is categorical


# -- TRIBE Backend (wraps tribev2.TribeModel) ----------------------------------

class LiveTribeBackend:
    """TribeBackend that uses tribev2.TribeModel directly on the GPU."""

    HF_REPO    = "facebook/tribev2"
    HF_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"

    def __init__(self, device: str = "cuda", weights_cache: Path | None = None) -> None:
        self._device = device
        self._cache  = weights_cache or (OUTPUT_DIR / "tribe_weights")
        self._model  = None

    @staticmethod
    def _patch_yaml_for_windows() -> None:
        """Patches yaml.UnsafeLoader to treat PosixPath as Path on Windows.
        TRIBE's config.yaml was saved on Unix with PosixPath, which does not exist on Windows.
        """
        import pathlib
        import yaml

        def _posixpath_constructor(loader, node):
            parts = loader.construct_sequence(node) if hasattr(loader.construct_sequence(node), '__iter__') else []
            return pathlib.Path(*parts)

        # More robust version: builds any sequence of args for PosixPath as Path
        def _posixpath_constructor2(loader, node):
            try:
                parts = loader.construct_sequence(node, deep=True)
                return pathlib.Path(*parts) if parts else pathlib.Path(".")
            except Exception:
                return pathlib.Path(".")

        for tag in ["!!python/object/apply:pathlib.PosixPath",
                    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
                    "!!python/object:pathlib.PosixPath"]:
            try:
                yaml.add_constructor(tag, _posixpath_constructor2, Loader=yaml.UnsafeLoader)
            except Exception:
                pass

    def load(self) -> None:
        """Downloads (first time) and loads TribeModel into memory."""
        from huggingface_hub import snapshot_download
        from tribev2 import TribeModel  # type: ignore

        if self._model is not None:
            return

        # Patch YAML for Windows (PosixPath in TRIBE's Unix configs)
        self._patch_yaml_for_windows()

        log("Downloading/verifying TRIBE v2 weights (HuggingFace)...")
        tribe_dir = self._cache / "tribev2_weights"
        tribe_dir.mkdir(parents=True, exist_ok=True)
        try:
            tribe_path = snapshot_download(
                self.HF_REPO,
                revision=self.HF_REVISION,
                local_dir=str(tribe_dir),   # direct download without symlinks (Windows)
            )
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                _err(
                    "\n[bold red]WARNING: Model facebook/tribev2 requires authentication.[/bold red]\n"
                    "Follow these steps:\n"
                    "  1. Create a free account at https://huggingface.co\n"
                    "  2. Accept the terms at https://huggingface.co/facebook/tribev2\n"
                    "  3. Generate a token at https://huggingface.co/settings/tokens\n"
                    "  4. Run: [bold]huggingface-cli login[/bold]\n"
                    "  5. Run this script again.\n"
                )
                raise SystemExit(1) from e
            raise

        log(f"[cyan]Loading TribeModel on device '{self._device}'...")
        self._model = TribeModel.from_pretrained(
            tribe_path,
            device=self._device,
            config_update={"data.num_workers": 0},
        )
        log("[green]TRIBE v2 loaded OK")

    def unload(self) -> None:
        """Frees TribeModel from VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()

    def predict(self, video_path: Path) -> np.ndarray:
        """Returns cortical activation (20484,) for a video."""
        if self._model is None:
            self.load()
        events = self._model.get_events_dataframe(video_path=str(video_path))
        preds, _ = self._model.predict(events, verbose=False)
        activation = np.array(preds, dtype="float32").mean(axis=0)   # (20484,)
        return activation


# -- Seed loading --------------------------------------------------------------

def load_seeds(n_pool: int = 16) -> list[dict]:
    """Loads available seeds from prompts.json and cycles them into a pool of n_pool."""
    import sys as _sys
    _sys.stderr.write("load_seeds: start\n"); _sys.stderr.flush()
    prompts_file = SEEDS_DIR / "prompts.json"
    _sys.stderr.write(f"load_seeds: opening {prompts_file}\n"); _sys.stderr.flush()
    with open(prompts_file, encoding="utf-8") as f:
        _sys.stderr.write("load_seeds: reading json\n"); _sys.stderr.flush()
        all_prompts = json.load(f)
    _sys.stderr.write(f"load_seeds: got {len(all_prompts)} prompts\n"); _sys.stderr.flush()

    available = []
    for e in all_prompts:
        si = e.get("seed_image")
        if not si:
            continue
        p = PROJECT_ROOT / si
        _sys.stderr.write(f"  checking: {si}\n"); _sys.stderr.flush()
        if p.exists():
            available.append({
                "idx": e["idx"], "bmd_name": e["bmd_name"],
                "prompt": e["prompt"], "image_path": p,
            })
    _sys.stderr.write(f"load_seeds: available={len(available)}\n"); _sys.stderr.flush()

    if not available:
        raise FileNotFoundError(
            f"No seed image found in {SEEDS_DIR}. "
            "Make sure the PNG images are in seeds/."
        )

    seeds_pool = list(islice(cycle(available), n_pool))
    log(f"[green]{len(available)} seeds available -> pool of {n_pool}")
    return seeds_pool


# -- Scoring of a single video -------------------------------------------------

def score_video(
    video_path: Path,
    prompt: str,
    tribe_backend: LiveTribeBackend,
    clip_scorer: CLIPScorer,
    v_mem: np.ndarray,
) -> tuple[float, float]:
    """Returns (tribe_score, clip_score) for a video."""
    activation   = tribe_backend.predict(video_path)
    tribe_score  = float(np.dot(activation, v_mem))
    clip_score   = clip_scorer.score(video_path, prompt)
    return tribe_score, clip_score


# -- Main generation + scoring loop -------------------------------------------

def evaluate_batch(
    tasks: list[dict],
    generator: SVDGenerator,
    tribe_backend: LiveTribeBackend,
    clip_scorer: CLIPScorer,
    seeds_pool: list[dict],
    v_mem: np.ndarray,
    videos_dir: Path,
    iteration: int,
    num_inference_steps: int = 25,
) -> tuple[list[list[float]], list[dict]]:
    """
    For each task in `tasks`:
      1. Generate the video (SVD-XT on the GPU via cpu_offload)
      2. Score with TRIBE v2 + CLIP
    Returns (scores [[tribe, clip], ...], metadata [dict, ...]).
    """
    scores_list: list[list[float]] = []
    meta_list:   list[dict]        = []

    for i, task in enumerate(tasks):
        seed_idx = int(task["seed_idx"]) % 16
        seed_entry = seeds_pool[seed_idx]
        img  = Image.open(seed_entry["image_path"]).convert("RGB")
        out_path = videos_dir / f"iter{iteration:03d}_t{i:02d}_{task['task_id']}.mp4"

        log(
            f"  [yellow]> Generating[/yellow] {task['task_id']} | "
            f"alpha={task['alpha']:+.3f} guidance={task['guidance']:.2f} "
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
        gen_time = time.time() - t0
        log(f"    generated in {gen_time:.1f}s -> {out_path.name}")

        ts, cs = score_video(out_path, seed_entry["prompt"], tribe_backend, clip_scorer, v_mem)
        log(f"    tribe={ts:.4f}  clip={cs:.4f}")

        scores_list.append([ts, cs])
        meta_list.append({
            **task,
            "filename":   out_path.name,
            "prompt":     seed_entry["prompt"],
            "tribe_score": ts,
            "clip_score":  cs,
        })

    return scores_list, meta_list


# -- BO step: fit GP + qNEHVI -------------------------------------------------

def bo_step(
    train_x:    torch.Tensor,
    train_y:    torch.Tensor,
    batch_size: int,
    iteration:  int,
) -> list[dict]:
    """Fits the GP and returns the next candidates as a list of task dicts."""
    log("[cyan]Fitting GP surrogate (BoTorch)...")
    model = build_model_list(train_x, train_y, FULL_BOUNDS, cat_dims=CAT_DIMS)
    fit_model(model)

    log("[cyan]Optimizing qNEHVI...")
    new_x = optimize_qehvi(
        model      = model,
        train_x    = train_x,
        train_y    = train_y,
        bounds     = FULL_BOUNDS,
        batch_size = batch_size,
        cat_dims   = CAT_DIMS,
        n_seeds    = 16,
    )
    del model
    gc.collect()

    tasks = []
    for j, x in enumerate(new_x):
        tasks.append({
            "task_id":    f"bo{iteration:02d}_cand{j:02d}",
            "alpha":      float(x[0].item()),
            "guidance":   float(x[1].item()),
            "seed_idx":   int(x[2].item()) % 16,
            "noise_seed": iteration * 100 + j,
        })
        log(
            f"  Candidate {j}: alpha={tasks[-1]['alpha']:+.3f} "
            f"guidance={tasks[-1]['guidance']:.2f} seed={tasks[-1]['seed_idx']}"
        )
    return tasks


# -- Save / load state --------------------------------------------------------

def save_state(
    train_x:    torch.Tensor,
    train_y:    torch.Tensor,
    all_meta:   list[dict],
    hv_history: list[float],
    iteration:  int,
) -> None:
    torch.save(
        {
            "train_x":    train_x,
            "train_y":    train_y,
            "all_meta":   all_meta,
            "hv_history": hv_history,
            "iteration":  iteration,
        },
        STATE_FILE,
    )


def load_state() -> tuple[torch.Tensor, torch.Tensor, list, list, int]:
    if STATE_FILE.exists():
        s = torch.load(STATE_FILE, weights_only=False)
        log(f"[green]Existing state loaded: {len(s['train_x'])} observations")
        return (
            s["train_x"].double(),
            s["train_y"].double(),
            s.get("all_meta", []),
            s.get("hv_history", []),
            s.get("iteration", 0),
        )
    return (
        torch.empty(0, 3, dtype=torch.double),
        torch.empty(0, 2, dtype=torch.double),
        [],
        [],
        0,
    )


# -- Visualization -------------------------------------------------------------

def plot_results(train_y: torch.Tensor, hv_history: list[float], n_init: int) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BO Memorability -- Results (GPU)", fontsize=14, fontweight="bold")

    t  = train_y[:, 0].numpy()
    c  = train_y[:, 1].numpy()
    n_bo = max(0, len(train_y) - n_init)

    ax1.scatter(t[:n_init], c[:n_init], c="steelblue", alpha=0.7, s=50, label=f"Sobol (n={n_init})")
    if n_bo > 0:
        ax1.scatter(t[n_init:], c[n_init:], c="darkorange", s=70, marker="D", label=f"BO (n={n_bo})")

    pf = get_pareto_front(train_y)
    _si = np.argsort(pf[:, 0].numpy())
    ax1.plot(pf[_si, 0].numpy(), pf[_si, 1].numpy(), "r--o", lw=2, label="Pareto front")
    ax1.set_xlabel("TRIBE memorability score")
    ax1.set_ylabel("CLIP fidelity score")
    ax1.set_title("Objective space")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if hv_history:
        iters = list(range(1, len(hv_history) + 1))
        ax2.plot(iters, hv_history, "go-", lw=2, ms=8)
        ax2.fill_between(iters, hv_history, alpha=0.15, color="green")
        ax2.set_xlabel("BO round")
        ax2.set_ylabel("Dominated hypervolume")
        ax2.set_title("Hypervolume convergence")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "HV available after\n2+ BO rounds",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "bo_pareto_hv.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"[green]Figure saved: {fig_path}")


# -- Final summary --------------------------------------------------------------

def print_summary(train_y: torch.Tensor, hv_history: list[float]) -> None:
    pf = get_pareto_front(train_y)
    rule("Final BO Summary")
    _err(f"  Total evaluations           : {len(train_y)}", flush=True)
    _err(f"  Points on the Pareto front  : {len(pf)}", flush=True)
    if hv_history:
        _err(f"  Initial HV                  : {hv_history[0]:.6f}", flush=True)
        _err(f"  Final HV                    : {hv_history[-1]:.6f}", flush=True)
        _err(f"  HV gain                     : {hv_history[-1] - hv_history[0]:.6f}", flush=True)
    print(f"  Best tribe_score            : {float(train_y[:, 0].max()):.4f}", flush=True)
    print(f"  Best clip_score             : {float(train_y[:, 1].max()):.4f}", flush=True)


# -- Entry point ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-initial",    type=int, default=12,  help="Initial Sobol points (default: 12)")
    p.add_argument("--n-iterations", type=int, default=10,  help="BO iterations (default: 10)")
    p.add_argument("--batch-size",   type=int, default=2,   help="Candidates per BO round (default: 2)")
    p.add_argument("--steps",        type=int, default=25,  help="SVD inference steps (default: 25; use 10 for a quick test)")
    p.add_argument("--no-resume",    action="store_true",   help="Ignore saved state and start from scratch")
    p.add_argument("--tribe-device", type=str, default="cuda", help="TRIBE device (default: cuda)")
    p.add_argument("--clip-device",  type=str, default="cuda", help="CLIP device (default: cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Debug log independent of stdout (to diagnose output problems)
    import atexit
    _dbg_path = OUTPUT_DIR / "debug_run.txt"
    _dbg = open(_dbg_path, "w", encoding="ascii", errors="replace")
    def _dbg_write(msg: str) -> None:
        _dbg.write(msg + "\n")
        _dbg.flush()
    atexit.register(_dbg.close)
    _dbg_write("START main()")

    # -- Banner ----------------------------------------------------------------
    rule("BO Memorability -- Local GPU")
    _dbg_write("AFTER banner rule")
    _err(f"  GPU      : {torch.cuda.get_device_name(0)}", flush=True)
    _err(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    _err(f"  n_initial: {args.n_initial}")
    _err(f"  n_iters  : {args.n_iterations}")
    _err(f"  batch    : {args.batch_size}")
    _err(f"  steps    : {args.steps}")
    _err(f"  output   : {OUTPUT_DIR}")
    rule()

    _dbg_write("BEFORE load_seeds")
    # -- Seeds and v_mem -------------------------------------------------------
    seeds_pool = load_seeds(n_pool=16)
    _dbg_write(f"AFTER load_seeds: {len(seeds_pool)} seeds")
    v_raw  = np.load(ARTIFACTS_DIR / "v_mem.npz")["direction"].astype("float32")
    v_mem  = v_raw / np.linalg.norm(v_raw)
    log(f"[green]v_mem loaded: shape={v_mem.shape}")

    # -- Load previous state (if it exists) ------------------------------------
    if args.no_resume:
        train_x, train_y, all_meta, hv_history, start_iter = (
            torch.empty(0, 3, dtype=torch.double),
            torch.empty(0, 2, dtype=torch.double),
            [], [], 0,
        )
        log("[yellow]--no-resume: starting from scratch")
    else:
        train_x, train_y, all_meta, hv_history, start_iter = load_state()

    scored_ids = {m["task_id"] for m in all_meta}

    _dbg_write("BEFORE CLIPScorer")
    # -- Initialize models -----------------------------------------------------
    log(f"[cyan]Loading CLIP ViT-H/14 on {args.clip_device}...")
    clip_scorer = CLIPScorer(device=args.clip_device)
    _dbg_write("AFTER CLIPScorer")
    log("[green]CLIP loaded OK")

    tribe_backend = LiveTribeBackend(device=args.tribe_device)
    tribe_backend.load()   # downloads HF weights the first time

    log("[cyan]Loading SVD-XT (cpu_offload=True to save VRAM)...")
    generator = SVDGenerator(
        device      = DEVICE,
        dtype       = torch.float16,
        cpu_offload = True,          # peak ~3 GB VRAM vs ~9 GB without offload
    )
    generator._load_pipeline()       # load now (not lazily) to show progress
    log("[green]SVD-XT loaded OK")

    # -- Phase 1: Sobol initialization (if not yet done) ----------------------
    if start_iter == 0 and len(train_x) == 0:
        rule("[bold]Phase 1 -- Sobol initialization")
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
        log(f"[yellow]{len(new_tasks)}/{len(sobol_tasks)} Sobol points to evaluate")

        if new_tasks:
            scores, meta = evaluate_batch(
                tasks               = new_tasks,
                generator           = generator,
                tribe_backend       = tribe_backend,
                clip_scorer         = clip_scorer,
                seeds_pool          = seeds_pool,
                v_mem               = v_mem,
                videos_dir          = VIDEOS_DIR,
                iteration           = 0,
                num_inference_steps = args.steps,
            )
            new_x = torch.tensor([[t["alpha"], t["guidance"], float(t["seed_idx"])] for t in new_tasks], dtype=torch.double)
            new_y = torch.tensor(scores, dtype=torch.double)
            train_x   = torch.cat([train_x, new_x])   if train_x.numel()   else new_x
            train_y   = torch.cat([train_y, new_y])   if train_y.numel()   else new_y
            all_meta.extend(meta)
            scored_ids.update(t["task_id"] for t in new_tasks)

        hv = compute_hypervolume(train_y)
        hv_history.append(hv)
        start_iter = 1
        log(f"[green]Sobol complete | initial HV = {hv:.6f}")
        save_state(train_x, train_y, all_meta, hv_history, start_iter)

    # -- Phase 2: BO iterations ------------------------------------------------
    rule("[bold]Phase 2 -- BO iterations (qNEHVI)")

    for it in range(start_iter, start_iter + args.n_iterations):
        rule(f"[bold]BO iteration {it} / {start_iter + args.n_iterations - 1}")
        assert len(train_x) >= 2, "At least 2 observations are required to fit the GP."

        candidates = bo_step(train_x, train_y, args.batch_size, it)

        scores, meta = evaluate_batch(
            tasks               = candidates,
            generator           = generator,
            tribe_backend       = tribe_backend,
            clip_scorer         = clip_scorer,
            seeds_pool          = seeds_pool,
            v_mem               = v_mem,
            videos_dir          = VIDEOS_DIR,
            iteration           = it,
            num_inference_steps = args.steps,
        )

        new_x = torch.tensor([[t["alpha"], t["guidance"], float(t["seed_idx"])] for t in candidates], dtype=torch.double)
        new_y = torch.tensor(scores, dtype=torch.double)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        all_meta.extend(meta)

        hv = compute_hypervolume(train_y)
        hv_history.append(hv)
        pf_size = len(get_pareto_front(train_y))
        log(
            f"[green]Iteration {it} complete | "
            f"HV = {hv:.6f} | Pareto = {pf_size} points"
        )

        save_state(train_x, train_y, all_meta, hv_history, it + 1)

    # -- Final results ---------------------------------------------------------
    rule("[bold green]Final Results")
    print_summary(train_y, hv_history)
    plot_results(train_y, hv_history, n_init=args.n_initial)

    results_path = OUTPUT_DIR / "all_results.json"
    results_path.write_text(json.dumps(
        {"n_initial": args.n_initial, "n_iterations": args.n_iterations,
         "all_meta": all_meta, "hv_history": hv_history},
        indent=2,
    ))
    log(f"[green]Results saved to: {results_path}")
    rule("[bold green]Done!")


if __name__ == "__main__":
    main()
