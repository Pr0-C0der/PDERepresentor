from __future__ import annotations

"""
Plotting utilities for 1D and 2D PDE solutions.

All functions in this module are optional helpers – solving works without
matplotlib. The module forces the Agg backend so it is safe in headless
environments (e.g. CI servers).
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

# Use non-interactive backend suitable for headless environments.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_1d(
    x: np.ndarray,
    u: np.ndarray,
    title: Optional[str] = None,
    savepath: Optional[str | Path] = None,
) -> None:
    """
    Plot a 1D solution u(x) as a simple line plot.

    Parameters
    ----------
    x:
        1D NumPy array of grid coordinates.
    u:
        1D NumPy array of solution values at the same grid points as ``x``.
    title:
        Optional plot title.
    savepath:
        Optional filesystem path (string or Path). If provided, the plot is
        saved as a PNG at this location using dpi=150.

    Output
    ------
    None. The figure is saved to disk if ``savepath`` is given and then closed.
    """
    x = np.asarray(x)
    u = np.asarray(u)

    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if savepath is not None:
        savepath = str(savepath)
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_1d_time_series(
    x: np.ndarray,
    solutions: np.ndarray,
    times: Sequence[float],
    prefix: str = "solution1d",
    out_dir: Optional[str | Path] = None,
) -> List[Path]:
    """
    Plot a time series of 1D solutions and save each as a PNG.

    Parameters
    ----------
    x:
        1D NumPy array of grid coordinates.
    solutions:
        2D array of shape (nx, nt), where each column is u(x, t_k).
    times:
        Sequence of times corresponding to the columns of ``solutions``.
    prefix:
        Filename prefix (without extension) for the generated PNG files.
    out_dir:
        Optional directory in which to save the PNG files. Defaults to ``"."``.

    Returns
    -------
    paths:
        List of Path objects for the generated PNG files.
    """
    x = np.asarray(x)
    sol = np.asarray(solutions)
    times = list(times)

    if sol.ndim != 2:
        raise ValueError("solutions must be a 2D array of shape (nx, nt).")
    if sol.shape[1] != len(times):
        raise ValueError("solutions second dimension must match len(times).")

    if out_dir is None:
        out_dir = "."
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for k, t in enumerate(times):
        u = sol[:, k]
        fname = f"{prefix}_t{k:04d}.png"
        fpath = out_dir / fname
        plot_1d(
            x,
            u,
            title=f"{prefix} at t={t:.3g}",
            savepath=fpath,
        )
        paths.append(fpath)

    return paths


def plot_2d(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    title: Optional[str] = None,
    savepath: Optional[str | Path] = None,
) -> None:
    """
    Plot a 2D scalar field U(X, Y) using pcolormesh with a colorbar.

    Parameters
    ----------
    X, Y:
        2D meshgrid arrays defining the coordinates (as from numpy.meshgrid).
    U:
        2D NumPy array of the same shape as X and Y representing the field.
    title:
        Optional plot title.
    savepath:
        Optional filesystem path for saving the figure as a PNG (dpi=150).

    Output
    ------
    None. The figure is saved to disk if ``savepath`` is provided and then
    closed.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    U = np.asarray(U)

    if X.shape != Y.shape or X.shape != U.shape:
        raise ValueError("X, Y, and U must all have the same 2D shape.")

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, U, shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.colorbar(mesh, ax=ax)

    if savepath is not None:
        savepath = str(savepath)
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_2d_time_series(
    X: np.ndarray,
    Y: np.ndarray,
    solutions: Iterable[np.ndarray],
    times: Sequence[float],
    prefix: str = "solution2d",
    out_dir: Optional[str | Path] = None,
) -> List[Path]:
    """
    Plot a time series of 2D scalar fields and save each as a PNG.

    Parameters
    ----------
    X, Y:
        2D meshgrid arrays.
    solutions:
        Iterable of 2D arrays U_k of the same shape as X and Y.
    times:
        Sequence of times corresponding to each element of ``solutions``.
    prefix:
        Filename prefix (without extension) for the generated PNG files.
    out_dir:
        Optional directory in which to save the PNG files. Defaults to ``"."``.

    Returns
    -------
    paths:
        List of Path objects for the generated PNG files.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    sols = list(solutions)
    times = list(times)

    if len(sols) != len(times):
        raise ValueError("Number of solutions must match number of times.")

    if out_dir is None:
        out_dir = "."
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for k, (U, t) in enumerate(zip(sols, times)):
        U_arr = np.asarray(U)
        if U_arr.shape != X.shape:
            raise ValueError("Each solution must have the same shape as X and Y.")
        fname = f"{prefix}_t{k:04d}.png"
        fpath = out_dir / fname
        plot_2d(
            X,
            Y,
            U_arr,
            title=f"{prefix} at t={t:.3g}",
            savepath=fpath,
        )
        paths.append(fpath)

    return paths


