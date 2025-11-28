import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pde import Domain1D, Domain2D, InitialCondition, Diffusion, PDEProblem
from pde.json_loader import load_from_json
from pde.plotting import plot_1d, plot_2d


def test_plot_1d_creates_png(tmp_path: Path):
    dom = Domain1D(0.0, 1.0, 51)
    x = dom.x
    u = np.sin(np.pi * x)

    savepath = tmp_path / "test_plot1d.png"
    plot_1d(x, u, title="1D test", savepath=savepath)

    assert savepath.is_file()
    assert savepath.stat().st_size > 0


def test_plot_2d_creates_png(tmp_path: Path):
    dom = Domain2D(0.0, 1.0, 41, 0.0, 1.0, 41)
    X, Y = dom.X, dom.Y
    U = np.exp(-50.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    savepath = tmp_path / "test_plot2d.png"
    plot_2d(X, Y, U, title="2D test", savepath=savepath)

    assert savepath.is_file()
    assert savepath.stat().st_size > 0


def test_pdeproblem_solve_with_plot_1d(tmp_path: Path):
    dom = Domain1D(0.0, 2 * np.pi, 101, periodic=True)
    ic = InitialCondition.from_expression("np.sin(x)")
    op = Diffusion(0.1)
    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0, t1 = 0.0, 0.2
    t_eval = np.linspace(t0, t1, 4)

    sol = problem.solve((t0, t1), t_eval=t_eval, plot=True, plot_dir=str(tmp_path))
    assert sol.success

    # Initial and final snapshots
    assert (tmp_path / "solution1d_initial.png").is_file()
    assert (tmp_path / "solution1d_final.png").is_file()

    # Time series frames
    pngs = list(tmp_path.glob("solution1d_t*.png"))
    assert len(pngs) == len(t_eval)


def test_json_visualization_like_flow_1d(tmp_path: Path):
    """
    Mimic a JSON configuration with visualization enabled and ensure
    PDEProblem.solve(plot=True, plot_dir=...) writes PNGs.
    """
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "examples" / "heat1d.json"

    with cfg_path.open("r", encoding="utf8") as f:
        cfg = json.load(f)

    problem = load_from_json(str(cfg_path))

    time_cfg = cfg.get("time", {})
    t0 = float(time_cfg.get("t0", 0.0))
    t1 = float(time_cfg.get("t1", 1.0))
    num_points = int(time_cfg.get("num_points", 6))
    t_eval = np.linspace(t0, t1, num_points)

    sol = problem.solve((t0, t1), t_eval=t_eval, plot=True, plot_dir=str(tmp_path))
    assert sol.success

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) >= 1


