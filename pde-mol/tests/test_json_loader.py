import json
from pathlib import Path

import numpy as np

from pde.json_loader import load_from_json


def _project_root() -> Path:
    # tests/ -> project root is one directory up thanks to conftest sys.path
    return Path(__file__).resolve().parent.parent


def test_load_heat1d_json_and_compare_analytic():
    """
    Load the heat1d example JSON, build the PDEProblem, and compare against
    the known analytic solution.
    """
    root = _project_root()
    cfg_path = root / "examples" / "heat1d.json"

    # Load the configuration dictionary to get time settings
    with cfg_path.open("r", encoding="utf8") as f:
        cfg = json.load(f)

    problem = load_from_json(str(cfg_path))

    time_cfg = cfg.get("time", {})
    t0 = float(time_cfg.get("t0", 0.0))
    t1 = float(time_cfg.get("t1", 1.0))
    num_points = int(time_cfg.get("num_points", 6))

    t_eval = np.linspace(t0, t1, num_points)

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=float(time_cfg.get("rtol", 1e-8)),
        atol=float(time_cfg.get("atol", 1e-10)),
    )
    assert sol.success

    from pde import Domain1D  # for type clarity

    dom: Domain1D = problem.domain  # type: ignore[assignment]
    x = dom.x
    nu = cfg["operators"][0]["nu"]

    u_num = sol.y[:, -1]
    u_exact = np.exp(-nu * t1) * np.sin(x)

    error = np.max(np.abs(u_num - u_exact))
    assert error < 1e-2


def test_load_burgers1d_json_runs():
    """
    Basic smoke test: load Burgers example from JSON and ensure the
    integration runs without error.
    """
    root = _project_root()
    cfg_path = root / "examples" / "burgers1d.json"

    with cfg_path.open("r", encoding="utf8") as f:
        cfg = json.load(f)

    problem = load_from_json(str(cfg_path))

    time_cfg = cfg.get("time", {})
    t0 = float(time_cfg.get("t0", 0.0))
    t1 = float(time_cfg.get("t1", 0.1))
    num_points = int(time_cfg.get("num_points", 4))

    t_eval = np.linspace(t0, t1, num_points)

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=float(time_cfg.get("rtol", 1e-6)),
        atol=float(time_cfg.get("atol", 1e-8)),
    )

    assert sol.success
    # Ensure the solution changed from initial condition (non-trivial dynamics).
    u0 = sol.y[:, 0]
    u1 = sol.y[:, -1]
    assert np.max(np.abs(u1 - u0)) > 0.0


def test_load_burgers1d_json_rhs_matches_mol_small_dt():
    """
    Load the Burgers example from JSON and check that, for a very small
    time step, the numerical time derivative from solve() matches the
    MOL RHS implied by the loaded operators.
    """
    root = _project_root()
    cfg_path = root / "examples" / "burgers1d.json"

    with cfg_path.open("r", encoding="utf8") as f:
        cfg = json.load(f)

    problem = load_from_json(str(cfg_path))

    # Override time settings to use a very small dt for this consistency test.
    t0 = 0.0
    dt = 5e-4
    t1 = t0 + dt
    t_eval = [t0, t1]

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    assert sol.success

    # For periodic 1D, the state is the full field.
    u0 = sol.y[:, 0]
    u1 = sol.y[:, -1]
    du_dt_num = (u1 - u0) / dt

    # Use the combined operator inside the problem to get the MOL RHS.
    du_dt_op = problem._op.apply(u0, problem.domain, t0)  # type: ignore[attr-defined]

    error = np.max(np.abs(du_dt_num - du_dt_op))
    assert error < 1e-2


