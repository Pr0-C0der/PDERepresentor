from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pde import Domain1D
from pde.dataset import ParameterRange, generate_dataset
from pde.json_loader import build_problem_from_dict
from pde.problem import SecondOrderPDEProblem
from pde.plotting import plot_1d_combined, plot_xt_heatmap


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def _build_time_grid(time_cfg: Dict[str, Any]) -> tuple[tuple[float, float], Optional[np.ndarray], Dict[str, Any]]:
    t0 = float(time_cfg.get("t0", 0.0))
    t1 = float(time_cfg.get("t1", 1.0))

    if "t_eval" in time_cfg:
        t_eval = np.asarray(time_cfg["t_eval"], dtype=float)
    else:
        num_points = int(time_cfg.get("num_points", 101))
        t_eval = np.linspace(t0, t1, num_points)

    method = time_cfg.get("method", "RK45")
    rtol = float(time_cfg.get("rtol", 1e-6))
    atol = float(time_cfg.get("atol", 1e-8))

    solve_kwargs: Dict[str, Any] = {"method": method, "rtol": rtol, "atol": atol}
    return (t0, t1), t_eval, solve_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a JSON-defined PDE problem via Method of Lines.")
    parser.add_argument("config", type=str, help="Path to JSON configuration file.")
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Suppress detailed output; only exit code indicates success.",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Enable dataset generation mode (overrides JSON dataset.enabled).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples for dataset generation (overrides JSON dataset.num_samples).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save path for dataset .pt file (overrides JSON dataset.savepath).",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)

    # Check for dataset generation mode
    dataset_cfg = config.get("dataset", {})
    dataset_enabled = args.dataset or bool(dataset_cfg.get("enabled", False))

    if dataset_enabled:
        # Dataset generation mode
        if not args.no_output:
            print("Dataset generation mode enabled.")

        # Parse parameter ranges
        param_dict = dataset_cfg.get("parameters", {})
        if not param_dict:
            raise ValueError("Dataset generation requires 'dataset.parameters' in JSON config.")

        param_ranges = [
            ParameterRange(name=name, low=float(bounds[0]), high=float(bounds[1]))
            for name, bounds in param_dict.items()
        ]

        num_samples = args.samples if args.samples is not None else dataset_cfg.get("num_samples", 50)
        savepath = args.save if args.save is not None else dataset_cfg.get("savepath", "dataset.pt")
        seed = dataset_cfg.get("seed", None)

        # Extract time configuration
        time_cfg = config.get("time", {})
        t_span = (time_cfg.get("t0", 0.0), time_cfg.get("t1", 1.0))
        t_eval = None
        if "num_points" in time_cfg:
            t_eval = np.linspace(t_span[0], t_span[1], time_cfg["num_points"])

        solver_method = time_cfg.get("method", "RK45")
        solver_kwargs = {}
        if "rtol" in time_cfg:
            solver_kwargs["rtol"] = time_cfg["rtol"]
        if "atol" in time_cfg:
            solver_kwargs["atol"] = time_cfg["atol"]

        # Extract problem name from config or savepath
        problem_name = None
        if isinstance(config_path, Path):
            problem_name = config_path.stem
            # Remove _parameterized suffix if present
            if problem_name.endswith("_parameterized"):
                problem_name = problem_name[:-14]

        # Check if heatmaps should be saved (from dataset config or default to True)
        save_heatmaps = dataset_cfg.get("save_heatmaps", True)
        overwrite = dataset_cfg.get("overwrite", False)

        # Check if dataset already exists
        savepath_path = Path(savepath)
        if savepath_path.exists() and not overwrite:
            if not args.no_output:
                print(f"Dataset already exists at: {savepath}")
                print("Loading existing dataset...")
            import torch
            dataset = torch.load(savepath_path)
            if not args.no_output:
                print(f"Loaded existing dataset:")
                print(f"  Parameters shape: {dataset['params'].shape}")
                print(f"  Solution shape: {dataset['u'].shape}")
                print(f"  Time points: {dataset['t'].shape[0]}")
            return

        # Generate dataset
        if not args.no_output:
            print(f"Generating {num_samples} samples...")
            print(f"Parameters: {[pr.name for pr in param_ranges]}")
            print(f"Saving to: {savepath}")
            if save_heatmaps:
                print(f"Saving heatmaps to: dataset_generate_plots/{problem_name}/")

        dataset = generate_dataset(
            json_template=config,
            param_ranges=param_ranges,
            num_samples=num_samples,
            savepath=savepath,
            seed=seed,
            t_span=t_span,
            t_eval=t_eval,
            solver_method=solver_method,
            solver_kwargs=solver_kwargs,
            save_heatmaps=save_heatmaps,
            problem_name=problem_name,
            overwrite=overwrite,
        )

        if not args.no_output:
            print(f"Dataset generated successfully!")
            print(f"  Parameters shape: {dataset['params'].shape}")
            print(f"  Solution shape: {dataset['u'].shape}")
            print(f"  Time points: {dataset['t'].shape[0]}")
            print(f"  Spatial points: {dataset['x'].shape[0] if len(dataset['x'].shape) == 1 else dataset['x'].shape}")

        return

    # Normal single solve mode
    # Build PDEProblem from config dictionary
    problem = build_problem_from_dict(config)

    time_cfg = config.get("time", {})
    t_span, t_eval, solve_kwargs = _build_time_grid(time_cfg)

    # Optional visualization block
    vis_cfg = config.get("visualization", {})
    vis_enable = bool(vis_cfg.get("enable", False))
    vis_type = vis_cfg.get("type", "1d")

    # All plots from JSON-driven runs go under "plots/<save_dir>/"
    save_subdir = vis_cfg.get("save_dir") or "default"
    base_plots_dir = Path("plots") / save_subdir

    result = problem.solve(
        t_span,
        t_eval=t_eval,
        plot=vis_enable,
        plot_dir=str(base_plots_dir),
        **solve_kwargs,
    )

    # Optional combined 1D time-series plot for convenience
    if vis_enable and isinstance(problem.domain, Domain1D) and vis_type == "1d":
        x = problem.domain.x
        
        # Handle second-order problems differently (they have result.u attribute)
        if isinstance(problem, SecondOrderPDEProblem):
            # For second-order problems, use the u component
            if problem.domain.periodic:
                full_solutions = result.u
            else:
                # Reconstruct full u from interior
                full_states = []
                for k, t in enumerate(result.t):
                    interior_k = result.y[:, k]
                    u_full, _ = problem._reconstruct_full_from_interior(interior_k, t)
                    full_states.append(u_full)
                full_solutions = np.stack(full_states, axis=1)
        else:
            # First-order problems: standard handling
            if problem.domain.periodic:
                full_solutions = result.y
            else:
                full_states = []
                for k, t in enumerate(result.t):
                    interior_k = result.y[:, k]
                    full_k = problem._reconstruct_full_from_interior(  # type: ignore[attr-defined]
                        interior_k, t
                    )
                    full_states.append(full_k)
                full_solutions = np.stack(full_states, axis=1)

        combined_path = base_plots_dir / "solution1d_combined.png"
        plot_1d_combined(
            x,
            full_solutions,
            result.t,
            title="Combined 1D time series",
            savepath=combined_path,
            max_curves=8,
        )

        # u(x,t) heatmap (t on x-axis, x on y-axis)
        xt_path = base_plots_dir / "solution1d_xt_heatmap.png"
        plot_xt_heatmap(
            x,
            result.t,
            full_solutions,
            title="u(x,t) heatmap",
            savepath=xt_path,
        )

    if not result.success:
        raise SystemExit("Time integration failed.")

    if not args.no_output:
        import math

        # Report simple diagnostics: final time, L2 norm of solution, etc.
        # For second-order problems, use result.u; otherwise use result.y
        if isinstance(problem, SecondOrderPDEProblem):
            u_final = result.u[:, -1]
        else:
            u_final = result.y[:, -1]
        t_final = result.t[-1]

        l2 = math.sqrt(float(np.sum(u_final**2)) / u_final.size)
        print(f"Solved problem from t={t_span[0]} to t={t_final}")
        print(f"Grid size: {u_final.size}")
        print(f"L2 norm of solution at final time: {l2:.6e}")


if __name__ == "__main__":
    main()


