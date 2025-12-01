from __future__ import annotations

"""
Plotting utilities for 1D PDE solutions.

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

try:  # optional, used for progress bars when available
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, *args, **kwargs):
        return iterable


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
    iterator = tqdm(
        enumerate(times),
        total=len(times),
        desc="Plotting 1D time series",
        leave=False,
    )
    for k, t in iterator:
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


def plot_1d_combined(
    x: np.ndarray,
    solutions: np.ndarray,
    times: Sequence[float],
    title: Optional[str] = None,
    savepath: Optional[str | Path] = None,
    max_curves: Optional[int] = None,
) -> None:
    """
    Plot multiple 1D solution curves at different times on a single figure.

    Parameters
    ----------
    x:
        1D NumPy array of grid coordinates.
    solutions:
        2D array of shape (nx, nt), where each column is u(x, t_k).
    times:
        Sequence of times corresponding to the columns of ``solutions``.
    title:
        Optional plot title.
    savepath:
        Optional filesystem path. If provided, the combined figure is saved as
        a PNG (dpi=150).
    max_curves:
        Optional maximum number of time curves to plot. If provided and
        ``len(times) > max_curves``, a subset of evenly spaced times is chosen.

    Output
    ------
    None. The figure is saved and closed if ``savepath`` is provided.
    """
    x = np.asarray(x)
    sol = np.asarray(solutions)
    times = list(times)

    if sol.ndim != 2:
        raise ValueError("solutions must be a 2D array of shape (nx, nt).")
    if sol.shape[1] != len(times):
        raise ValueError("solutions second dimension must match len(times).")

    nt = sol.shape[1]
    indices = list(range(nt))
    if max_curves is not None and nt > max_curves:
        # Choose a subset of indices, including first and last.
        indices = np.linspace(0, nt - 1, max_curves, dtype=int).tolist()

    fig, ax = plt.subplots()
    for k in indices:
        u = sol[:, k]
        t = times[k]
        ax.plot(x, u, label=f"t={t:.3g}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")

    if savepath is not None:
        savepath = str(savepath)
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_xt_heatmap(
    x: np.ndarray,
    times: Sequence[float],
    solutions: np.ndarray,
    title: Optional[str] = None,
    savepath: Optional[str | Path] = None,
) -> None:
    """
    Plot u(x, t) as a 2D heatmap with x on the x-axis and t on the y-axis.

    Parameters
    ----------
    x:
        1D NumPy array of spatial grid points (length nx).
    times:
        Sequence of time points (length nt).
    solutions:
        2D array of shape (nx, nt) with u(x_i, t_j).
    title:
        Optional plot title.
    savepath:
        Optional filesystem path. If provided, the heatmap is saved as a PNG.

    Output
    ------
    None. The figure is saved and closed if ``savepath`` is provided.
    """
    x = np.asarray(x)
    sol = np.asarray(solutions)
    times = np.asarray(times)

    if sol.ndim != 2:
        raise ValueError("solutions must be a 2D array of shape (nx, nt).")
    if sol.shape[0] != x.size:
        raise ValueError("solutions first dimension must match len(x).")
    if sol.shape[1] != times.size:
        raise ValueError("solutions second dimension must match len(times).")

    # Create grids with x along the horizontal axis and t along the vertical.
    Xg, Tg = np.meshgrid(x, times, indexing="xy")  # shapes (nt, nx)
    Z = sol.T  # transpose to shape (nt, nx) to match (Tg, Xg)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(Xg, Tg, Z, shading="auto", cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title:
        ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="u(x,t)")

    if savepath is not None:
        savepath = str(savepath)
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.close(fig)

