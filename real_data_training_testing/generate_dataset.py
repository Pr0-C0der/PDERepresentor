"""
Minimal dataset generation for PDE solutions.

Generates datasets by sampling parameters and solving PDEs, saving results as .pt files.
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Callable
import importlib
from tqdm import tqdm

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from solvers import heat1d, moisture1d, burgers1d, custom_pde, robin_general, moisture_1d_sagar


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def sample_parameters(param_ranges: Dict[str, Dict[str, float]], seed: int = None) -> Dict[str, float]:
    """Sample parameter values from their ranges."""
    if seed is not None:
        np.random.seed(seed)
    params = {}
    for name, range_dict in param_ranges.items():
        low = range_dict["low"]
        high = range_dict["high"]
        params[name] = np.random.uniform(low, high)
    return params


def solve_pde(pde_type: str, params: Dict[str, float], config: Dict[str, Any]) -> tuple:
    """
    Solve PDE with given parameters.

    Returns
    -------
    x : np.ndarray
        Spatial grid
    t : np.ndarray
        Time grid
    u : np.ndarray
        Solution array (nx, nt) or (nz, nt)
    """
    domain = config["domain"]
    
    if pde_type == "heat1d":
        x0, x1 = domain["x"]
        nx = domain["nx"]
        t0, t1 = domain["t"]
        nt = domain["nt"]
        u0_expr = config["initial_condition"]["expr"]
        u0_func = lambda x: eval(u0_expr, {"np": np, "x": x})
        x, t, u = heat1d.solve_heat1d(params["nu"], x0, x1, nx, t0, t1, nt, u0_func)
        return x, t, u
    
    elif pde_type == "moisture1d":
        z0, z1 = domain["z"]
        nz = domain["nz"]
        t0, t1 = domain["t"]
        nt = domain["nt"]
        z, t, u = moisture1d.solve_moisture1d(
            params["D"], params["h"], params["X_env"],
            z0, z1, nz, t0, t1, nt, params["X0"]
        )
        return z, t, u
    
    elif pde_type == "moisture_1d_sagar":
        z0, z1 = domain["z"]
        nz = domain["nz"]
        t0, t1 = domain["t"]
        nt = domain["nt"]
        # Get constant initial condition
        X0 = config["initial_condition"]["value"]
        z, t, u = moisture_1d_sagar.solve_moisture_1d_sagar(
            params["D"], params["h"], params["X_env"],
            z0, z1, nz, t0, t1, nt, X0
        )
        return z, t, u
    
    elif pde_type == "burgers1d":
        x0, x1 = domain["x"]
        nx = domain["nx"]
        t0, t1 = domain["t"]
        nt = domain["nt"]
        u0_expr = config["initial_condition"]["expr"]
        u0_func = lambda x: eval(u0_expr, {"np": np, "x": x})
        x, t, u = burgers1d.solve_burgers1d(params["nu"], x0, x1, nx, t0, t1, nt, u0_func)
        return x, t, u
    
    elif pde_type == "custom_pde":
        x0, x1 = domain["x"]
        nx = domain["nx"]
        t0, t1 = domain["t"]
        nt = domain["nt"]
        u0_expr = config["initial_condition"]["expr"]
        u0_func = lambda x: eval(u0_expr, {"np": np, "x": x})
        x, t, u = custom_pde.solve_custom_pde(
            params["a"], params["nu"], params["lam"],
            x0, x1, nx, t0, t1, nt, u0_func
        )
        return x, t, u
    
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


def generate_dataset(config_path: str, output_path: str = None, overwrite: bool = False):
    """
    Generate dataset from config file.

    Parameters
    ----------
    config_path : str
        Path to JSON config file
    output_path : str, optional
        Output .pt file path. If None, uses config name + "_dataset.pt" in data/ folder
    overwrite : bool
        If False and file exists, skip generation
    """
    config = load_config(config_path)
    pde_type = config["pde_type"]
    param_ranges = config["parameters"]
    dataset_cfg = config.get("dataset", {})
    num_samples = dataset_cfg.get("num_samples", 100)
    seed = dataset_cfg.get("seed", None)

    # Determine output path
    if output_path is None:
        config_name = Path(config_path).stem
        output_path = f"data/{config_name}_dataset.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if exists
    if output_path.exists() and not overwrite:
        print(f"Dataset already exists at {output_path}. Skipping generation.")
        return

    # Get parameter names
    param_names = list(param_ranges.keys())
    
    # Storage
    all_params = []
    all_solutions = []
    x_grid = None
    t_grid = None

    # Generate samples with progress bar
    print(f"Generating {num_samples} samples for {pde_type}...")
    for i in tqdm(range(num_samples), desc="Generating dataset", unit="sample"):
        # Sample parameters
        sample_seed = seed + i if seed is not None else None
        params = sample_parameters(param_ranges, seed=sample_seed)
        all_params.append([params[name] for name in param_names])

        # Solve PDE
        x, t, u = solve_pde(pde_type, params, config)
        
        # Store first grid (all should be same)
        if x_grid is None:
            x_grid = x
            t_grid = t

        # Store solution (nx, nt) -> (nt, nx) for consistency
        all_solutions.append(u.T)

    # Convert to tensors
    params_tensor = torch.tensor(all_params, dtype=torch.float32)  # (num_samples, n_params)
    u_tensor = torch.tensor(np.array(all_solutions), dtype=torch.float32)  # (num_samples, nt, nx)
    x_tensor = torch.tensor(x_grid, dtype=torch.float32)  # (nx,)
    t_tensor = torch.tensor(t_grid, dtype=torch.float32)  # (nt,)

    # Save dataset
    dataset = {
        "params": params_tensor,
        "u": u_tensor,
        "x": x_tensor,
        "t": t_tensor,
        "param_names": param_names,
        "pde_type": pde_type,
    }
    torch.save(dataset, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"  Parameters shape: {params_tensor.shape}")
    print(f"  Solution shape: {u_tensor.shape}")
    print(f"  Spatial grid size: {x_tensor.shape[0]}")
    print(f"  Time grid size: {t_tensor.shape[0]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_dataset.py <config.json> [output.pt]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    generate_dataset(config_path, output_path)

