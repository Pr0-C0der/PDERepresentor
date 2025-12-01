from __future__ import annotations

"""
Dataset generation for parameterized PDE problems.

This module provides utilities for sampling PDE parameters, solving the PDE
for each parameter set, and collecting results into PyTorch tensor files.
"""

import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .domain import Domain1D
from .json_loader import build_problem_from_dict, load_from_json
from .plotting import plot_xt_heatmap
from .problem import PDEProblem, SecondOrderPDEProblem


@dataclass
class ParameterRange:
    """
    Represents a parameter range for sampling.

    Parameters
    ----------
    name:
        Parameter name (e.g., "H", "A0").
    low:
        Lower bound of the parameter range.
    high:
        Upper bound of the parameter range.
    """

    name: str
    low: float
    high: float

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"ParameterRange {self.name}: low ({self.low}) must be < high ({self.high})")


class ParameterSampler:
    """
    Samples parameter values from specified ranges.

    Parameters
    ----------
    param_ranges:
        List of ParameterRange objects defining the parameter space.
    seed:
        Optional random seed for reproducible sampling.
    """

    def __init__(self, param_ranges: List[ParameterRange], seed: Optional[int] = None):
        self.param_ranges = param_ranges
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def sample(self) -> Dict[str, float]:
        """
        Sample one set of parameter values.

        Returns
        -------
        dict
            Dictionary mapping parameter names to sampled values.
            Example: {"H": 2.1831, "A0": 0.4421}
        """
        return {pr.name: self.rng.uniform(pr.low, pr.high) for pr in self.param_ranges}

    def sample_n(self, n: int) -> List[Dict[str, float]]:
        """
        Sample n sets of parameter values.

        Parameters
        ----------
        n:
            Number of samples to generate.

        Returns
        -------
        list of dict
            List of parameter dictionaries.
        """
        return [self.sample() for _ in range(n)]


def _substitute_parameters(obj: Any, params: Dict[str, float]) -> Any:
    """
    Recursively substitute parameter placeholders in a JSON-like structure.

    This function replaces string parameter names (e.g., "H") with their
    numeric values from the params dictionary.

    Parameters
    ----------
    obj:
        JSON-like object (dict, list, str, number, etc.)
    params:
        Dictionary mapping parameter names to values.

    Returns
    -------
    object
        Object with parameter placeholders replaced by numeric values.
    """
    if isinstance(obj, dict):
        return {key: _substitute_parameters(value, params) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_parameters(item, params) for item in obj]
    elif isinstance(obj, str):
        # Check if the string is a parameter name
        if obj in params:
            return params[obj]
        # Check if it's a numeric string (for backward compatibility)
        try:
            return float(obj)
        except ValueError:
            # It's a regular string (e.g., expression)
            # Substitute parameter names in the expression string
            # Use word boundaries to avoid partial matches (e.g., "A0" in "A0_value")
            result = obj
            for param_name, param_value in params.items():
                # Replace parameter name as a whole word (using word boundaries)
                # Pattern: \b ensures word boundaries, but we also need to handle cases
                # where the parameter might be followed by operators, spaces, etc.
                # More robust: replace parameter name when it's not part of another identifier
                pattern = r'\b' + re.escape(param_name) + r'\b'
                result = re.sub(pattern, str(param_value), result)
            return result
    else:
        # Number, bool, None, etc. - return as-is
        return obj


def generate_dataset(
    json_template: Union[str, Path, Dict[str, Any]],
    param_ranges: List[ParameterRange],
    num_samples: int,
    savepath: Union[str, Path] = "dataset.pt",
    seed: Optional[int] = None,
    t_span: Optional[tuple[float, float]] = None,
    t_eval: Optional[Sequence[float]] = None,
    solver_method: str = "RK45",
    solver_kwargs: Optional[Dict[str, Any]] = None,
    save_full_time: bool = True,
    save_heatmaps: bool = False,
    problem_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Generate a dataset by sampling PDE parameters and solving for each sample.

    Parameters
    ----------
    json_template:
        Path to JSON template file or a dictionary containing the template.
        The template may contain parameter placeholders (e.g., "H" instead of 2.0).
    param_ranges:
        List of ParameterRange objects defining the parameter space.
    num_samples:
        Number of parameter samples to generate and solve.
    savepath:
        Path where the .pt file will be saved.
    seed:
        Random seed for reproducible parameter sampling.
    t_span:
        Time span (t0, tf) for solving. If None, uses values from JSON "time" block.
    t_eval:
        Time points at which to evaluate the solution. If None, uses values from JSON.
    solver_method:
        ODE solver method (e.g., "RK45", "BDF").
    solver_kwargs:
        Additional keyword arguments for solve_ivp.
    save_full_time:
        If True, save full time sequence. If False, save only final snapshot.
    save_heatmaps:
        If True, save heatmaps for each sample in dataset_generate_plots/{problem_name}/.
    problem_name:
        Name for the problem (used in plot directory). If None, extracted from savepath.
    overwrite:
        If False and the dataset file already exists, load and return it instead of regenerating.
        If True, always regenerate the dataset.

    Returns
    -------
    dict
        Dictionary containing the dataset with keys:
        - "params": tensor of shape (num_samples, num_params)
        - "x": tensor of shape (nx,)
        - "t": tensor of shape (nt,)
        - "u": tensor of shape (num_samples, nt, nx)
        - "param_names": list of parameter names in order
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for dataset generation. Install with: pip install torch")

    # Check if dataset already exists
    savepath = Path(savepath)
    if savepath.exists() and not overwrite:
        # Load existing dataset
        dataset = torch.load(savepath)
        return dataset

    # Determine problem name for plot directory
    if problem_name is None:
        savepath_str = str(savepath)
        # Extract name from savepath (e.g., "data/heat1d_dataset.pt" -> "heat1d_dataset")
        problem_name = Path(savepath_str).stem
        # Remove common suffixes
        if problem_name.endswith("_dataset"):
            problem_name = problem_name[:-8]

    # Set up plot directory if saving heatmaps
    plot_dir = None
    if save_heatmaps:
        plot_dir = Path("dataset_generate_plots") / problem_name
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON template (make a deep copy to avoid modifying original)
    if isinstance(json_template, (str, Path)):
        with open(json_template, "r", encoding="utf8") as f:
            original_template = json.load(f)
    else:
        original_template = copy.deepcopy(json_template)

    # Extract time configuration if not provided
    time_cfg = original_template.get("time", {})
    if t_span is None:
        t_span = (time_cfg.get("t0", 0.0), time_cfg.get("t1", 1.0))
    if t_eval is None and "num_points" in time_cfg:
        t_eval = np.linspace(t_span[0], t_span[1], time_cfg["num_points"])
    if solver_kwargs is None:
        solver_kwargs = {}
        if "rtol" in time_cfg:
            solver_kwargs["rtol"] = time_cfg["rtol"]
        if "atol" in time_cfg:
            solver_kwargs["atol"] = time_cfg["atol"]

    # Initialize sampler
    sampler = ParameterSampler(param_ranges, seed=seed)

    # Collect parameter names in order
    param_names = [pr.name for pr in param_ranges]

    # Storage for results
    all_params = []
    all_solutions = []
    domain = None
    time_vector = None

    # Generate samples with progress bar
    for i in tqdm(range(num_samples), desc="Generating dataset", unit="sample"):
        # Sample parameters
        params = sampler.sample()
        all_params.append([params[name] for name in param_names])

        # Print parameter values for this sample
        param_str = ", ".join([f"{name}={params[name]:.6f}" for name in param_names])
        tqdm.write(f"Sample {i+1}/{num_samples}: {param_str}")

        # Substitute parameters into template (use fresh copy each time)
        config = _substitute_parameters(copy.deepcopy(original_template), params)

        # Build and solve problem
        problem = build_problem_from_dict(config)
        
        # Store domain and time vector from first sample (assuming they're the same)
        if domain is None:
            domain = problem.domain
            if t_eval is None:
                # Use default time evaluation
                t_eval = np.linspace(t_span[0], t_span[1], 100)

        # Solve
        sol = problem.solve(
            t_span=t_span,
            t_eval=t_eval,
            method=solver_method,
            plot=False,  # Disable plotting during dataset generation
            **solver_kwargs
        )

        # Extract solution
        # Handle second-order problems differently
        if isinstance(problem, SecondOrderPDEProblem):
            # For second-order problems, use sol.u (u component)
            if domain.periodic:
                u_full = sol.u  # shape (nx, nt)
            else:
                # Reconstruct full u from interior
                u_full = np.zeros((domain.nx, sol.t.size))
                for j, t in enumerate(sol.t):
                    interior_k = sol.y[:, j]
                    u_full_vec, _ = problem._reconstruct_full_from_interior(interior_k, t)
                    u_full[:, j] = u_full_vec
            # Transpose to (nt, nx)
            u_solution = u_full.T
        else:
            # First-order problem: standard handling
            # 1D: sol.y has shape (nx, nt) for periodic or (nx-2, nt) for non-periodic
            if domain.periodic:
                u_full = sol.y  # shape (nx, nt)
            else:
                # Reconstruct full solution from interior
                u_full = np.zeros((domain.nx, sol.t.size))
                for j, t in enumerate(sol.t):
                    u_interior = sol.y[:, j]
                    u_full[1:-1, j] = u_interior
                    # Apply BCs to get boundary values
                    if problem.bc_left:
                        problem.bc_left.apply_to_full(u_full[:, j], t, domain)
                    if problem.bc_right:
                        problem.bc_right.apply_to_full(u_full[:, j], t, domain)
            # Transpose to (nt, nx)
            u_solution = u_full.T

        if time_vector is None:
            time_vector = sol.t

        if save_full_time:
            all_solutions.append(u_solution)
        else:
            # Save only final snapshot
            all_solutions.append(u_solution[-1:])

        # Save heatmap for this sample if requested
        if save_heatmaps and plot_dir is not None:
            # 1D: save u(x,t) heatmap
            heatmap_path = plot_dir / f"sample_{i+1:04d}_heatmap.png"
            param_str = "_".join([f"{name}_{params[name]:.4f}" for name in param_names])
            title = f"Sample {i+1}: {param_str}"
            # plot_xt_heatmap expects solutions of shape (nx, nt)
            # u_solution is (nt, nx), so we transpose
            plot_xt_heatmap(
                x=domain.x,
                times=time_vector,
                solutions=u_solution.T,  # Shape (nx, nt)
                title=title,
                savepath=heatmap_path,
            )

    # Convert to PyTorch tensors
    params_tensor = torch.tensor(all_params, dtype=torch.float32)  # (num_samples, num_params)

    x_tensor = torch.tensor(domain.x, dtype=torch.float32)  # (nx,)
    if save_full_time:
        u_tensor = torch.tensor(np.array(all_solutions), dtype=torch.float32)  # (num_samples, nt, nx)
    else:
        u_tensor = torch.tensor(np.array(all_solutions), dtype=torch.float32)  # (num_samples, 1, nx)

    t_tensor = torch.tensor(time_vector, dtype=torch.float32)  # (nt,)

    # Create dataset dictionary
    dataset = {
        "params": params_tensor,
        "param_names": param_names,
        "t": t_tensor,
        "u": u_tensor,
    }

    if isinstance(domain, Domain1D):
        dataset["x"] = x_tensor
    else:
        dataset["x"] = x_tensor
        dataset["y"] = y_tensor

    # Save to file
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, savepath)

    return dataset

