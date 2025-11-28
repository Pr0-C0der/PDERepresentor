from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pde import Domain1D
from pde.json_loader import build_problem_from_dict
from pde.plotting import plot_1d_combined


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

    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)

    # Build PDEProblem from config dictionary
    problem = build_problem_from_dict(config)

    time_cfg = config.get("time", {})
    t_span, t_eval, solve_kwargs = _build_time_grid(time_cfg)

    # Optional visualization block
    vis_cfg = config.get("visualization", {})
    vis_enable = bool(vis_cfg.get("enable", False))
    vis_type = vis_cfg.get("type", "1d")
    save_dir = vis_cfg.get("save_dir") or "test_plots"

    result = problem.solve(
        t_span,
        t_eval=t_eval,
        plot=vis_enable,
        plot_dir=save_dir,
        **solve_kwargs,
    )

    # Optional combined 1D time-series plot for convenience
    if vis_enable and isinstance(problem.domain, Domain1D) and vis_type == "1d":
        x = problem.domain.x
        # Reconstruct full solutions for non-periodic 1D problems
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

        combined_path = Path(save_dir) / "solution1d_combined.png"
        plot_1d_combined(
            x,
            full_solutions,
            result.t,
            title="Combined 1D time series",
            savepath=combined_path,
            max_curves=8,
        )

    if not result.success:
        raise SystemExit("Time integration failed.")

    if not args.no_output:
        import math

        # Report simple diagnostics: final time, L2 norm of solution, etc.
        u_final = result.y[:, -1]
        t_final = result.t[-1]

        l2 = math.sqrt(float(np.sum(u_final**2)) / u_final.size)
        print(f"Solved problem from t={t_span[0]} to t={t_final}")
        print(f"Grid size: {u_final.size}")
        print(f"L2 norm of solution at final time: {l2:.6e}")


if __name__ == "__main__":
    main()


