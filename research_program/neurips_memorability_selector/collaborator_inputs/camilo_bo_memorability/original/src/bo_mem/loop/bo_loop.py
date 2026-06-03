"""Multi-objective Bayesian Optimization closed loop (F3 of the proposal).

Orchestrates the four stages per iteration:
  1. Select next batch via qEHVI
  2. Generate videos (SVDGenerator)
  3. Evaluate objectives (TribeScorer, CLIPScorer) + constraint (FVDScorer)
  4. Update surrogate GP

Tracks hypervolume over iterations and logs to Weights & Biases.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bo_mem.acquisition.qehvi import (
    compute_hypervolume,
    get_pareto_front,
    get_reference_point,
    optimize_qehvi,
)
from bo_mem.config import BOConfig, Config
from bo_mem.generator.svd_generator import SVDGenerator
from bo_mem.objectives.clip_scorer import CLIPScorer
from bo_mem.objectives.fvd_scorer import FVDScorer
from bo_mem.objectives.tribe_scorer import TribeScorer
from bo_mem.search_space import SearchSpace
from bo_mem.surrogate.gp_model import build_model_list, fit_model

console = Console()


@dataclass
class BOState:
    """Mutable state of the BO loop — all observations so far."""
    train_x: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.double))
    train_y: Tensor = field(default_factory=lambda: torch.empty(0, 2, dtype=torch.double))
    hypervolume_history: list[float] = field(default_factory=list)
    iteration: int = 0
    generated_paths: list[Path] = field(default_factory=list)

    def append(self, x_new: Tensor, y_new: Tensor, paths: list[Path]) -> None:
        self.train_x = torch.cat([self.train_x, x_new], dim=0) if self.train_x.numel() else x_new
        self.train_y = torch.cat([self.train_y, y_new], dim=0) if self.train_y.numel() else y_new
        self.generated_paths.extend(paths)
        self.iteration += 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train_x": self.train_x.tolist(),
            "train_y": self.train_y.tolist(),
            "hv_history": self.hypervolume_history,
            "iteration": self.iteration,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class BOLoop:
    """Closed-loop multi-objective BO for memorable video synthesis."""

    def __init__(
        self,
        cfg: Config,
        conditioning_image_path: Path,
        prompt: str,
        reference_video_paths: list[Path],
        output_dir: Path = Path("outputs/bo_run"),
        use_wandb: bool = True,
    ) -> None:
        self.cfg = cfg
        self.search_space = SearchSpace(cfg.bo)
        self.prompt = prompt
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluators
        self.tribe_scorer = TribeScorer(vmem_path=cfg.paths.vmem)
        self.clip_scorer = CLIPScorer(device=cfg.device)
        self.fvd_scorer = FVDScorer(
            reference_paths=reference_video_paths,
            device=cfg.device,
            threshold=cfg.bo.fvd_threshold,
        )

        # Generator (lazy-loaded on first call)
        self.generator = SVDGenerator(
            model_id=cfg.svd.model_id,
            device=cfg.device,
            num_frames=cfg.svd.num_frames,
        )

        from PIL import Image
        self._conditioning_image = Image.open(conditioning_image_path).convert("RGB")

        self.state = BOState()

        if use_wandb:
            wandb.init(project=cfg.wandb_project, config={
                "k_directions": cfg.bo.k_directions,
                "batch_size": cfg.bo.batch_size,
                "n_initial": cfg.bo.n_initial,
                "n_iterations": cfg.bo.n_iterations,
            })
        self._use_wandb = use_wandb

    def _evaluate_batch(
        self,
        x_batch: Tensor,
        fidelity: str = "med",
    ) -> tuple[Tensor, list[Path]]:
        """Evaluate a batch of x vectors → (N, 2) objective tensor + video paths."""
        alphas_batch, guidance_batch, seed_batch = self.search_space.split(x_batch)
        steps = {"low": self.cfg.svd.steps_low, "med": self.cfg.svd.steps_med, "high": self.cfg.svd.steps_high}[fidelity]

        paths: list[Path] = []
        scores: list[list[float]] = []

        for i in range(len(x_batch)):
            seed = int(seed_batch[i].item())
            guidance = float(guidance_batch[i].item())
            alphas = alphas_batch[i]

            out_path = self.output_dir / f"iter{self.state.iteration:03d}_i{i:02d}_s{seed}.mp4"

            t0 = time.time()
            video_path = self.generator.generate(
                conditioning_image=self._conditioning_image,
                alphas=alphas,
                guidance_scale=guidance,
                seed=seed,
                num_inference_steps=steps,
                output_path=out_path,
            )
            gen_time = time.time() - t0

            mem_score = self.tribe_scorer.score(video_path)
            clip_score = self.clip_scorer.score(video_path, self.prompt)
            paths.append(video_path)
            scores.append([mem_score, clip_score])
            console.log(f"  x[{i}]: mem={mem_score:.3f} clip={clip_score:.3f} t={gen_time:.1f}s")

        y = torch.tensor(scores, dtype=torch.double)
        return y, paths

    def _build_and_fit_model(self) -> tuple:
        """Fit a fresh ModelListGP on all observations so far."""
        bounds = self.search_space.bounds_continuous
        cat_dims = [self.search_space.dim_continuous]  # seed_id column
        model = build_model_list(
            self.state.train_x,
            self.state.train_y,
            bounds,
            cat_dims=cat_dims,
        )
        fit_model(model)
        return model

    def initialize(self) -> None:
        """Sample and evaluate the initial quasi-random Sobol points."""
        console.rule("[bold]Initialization (Sobol)")
        x_init = self.search_space.sobol_initial_points(self.cfg.bo.n_initial)
        y_init, paths = self._evaluate_batch(x_init)
        self.state.append(x_init, y_init, paths)

        hv = compute_hypervolume(self.state.train_y)
        self.state.hypervolume_history.append(hv)
        console.log(f"Initial HV: {hv:.4f}")
        if self._use_wandb:
            wandb.log({"hypervolume": hv, "iteration": 0})

    def run(self) -> BOState:
        """Run the full BO loop for n_iterations iterations."""
        self.initialize()

        for t in range(self.cfg.bo.n_iterations):
            console.rule(f"[bold]Iteration {t+1}/{self.cfg.bo.n_iterations}")

            # Fit surrogate
            model = self._build_and_fit_model()

            # Optimize acquisition
            bounds = self.search_space.bounds_continuous
            candidates = optimize_qehvi(
                model=model,
                train_x=self.state.train_x,
                train_y=self.state.train_y,
                bounds=bounds,
                batch_size=self.cfg.bo.batch_size,
                cat_dims=[self.search_space.dim_continuous],
                n_seeds=self.search_space.n_seeds,
            )

            # Evaluate candidates
            y_new, paths = self._evaluate_batch(candidates)
            self.state.append(candidates, y_new, paths)

            # Track hypervolume
            hv = compute_hypervolume(self.state.train_y)
            self.state.hypervolume_history.append(hv)
            console.log(f"HV after iter {t+1}: {hv:.4f}")

            if self._use_wandb:
                wandb.log({"hypervolume": hv, "iteration": t + 1,
                           "n_pareto": int(get_pareto_front(self.state.train_y).shape[0])})

            self.state.save(self.output_dir / "state.json")

        if self._use_wandb:
            wandb.finish()

        self._print_summary()
        return self.state

    def _print_summary(self) -> None:
        table = Table(title="BO Run Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total evaluations", str(len(self.state.train_x)))
        table.add_row("Final HV", f"{self.state.hypervolume_history[-1]:.4f}")
        table.add_row("HV gain", f"{self.state.hypervolume_history[-1] - self.state.hypervolume_history[0]:.4f}")
        pareto_y = get_pareto_front(self.state.train_y)
        table.add_row("Pareto front size", str(len(pareto_y)))
        best_mem = float(self.state.train_y[:, 0].max().item())
        table.add_row("Best memorability", f"{best_mem:.4f}")
        console.print(table)
