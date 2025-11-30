"""
Visualization utilities for comparing real and predicted PDE solutions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_heatmap_comparison(
    x: np.ndarray,
    t: np.ndarray,
    u_real: np.ndarray,
    u_pred: np.ndarray,
    title: str = "Solution Comparison",
    savepath: Optional[str] = None,
):
    """
    Plot side-by-side heatmaps comparing real and predicted solutions.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid (nx,)
    t : np.ndarray
        Time grid (nt,)
    u_real : np.ndarray
        Real solution (nx, nt) or (nt, nx)
    u_pred : np.ndarray
        Predicted solution (nx, nt) or (nt, nx)
    title : str
        Plot title
    savepath : str, optional
        Path to save figure
    """
    # Ensure u is (nt, nx) for plotting
    if u_real.shape[0] == len(x) and u_real.shape[1] == len(t):
        u_real = u_real.T  # (nx, nt) -> (nt, nx)
    if u_pred.shape[0] == len(x) and u_pred.shape[1] == len(t):
        u_pred = u_pred.T  # (nx, nt) -> (nt, nx)

    # Create meshgrid
    Xg, Tg = np.meshgrid(x, t, indexing="xy")  # (nt, nx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Real solution
    mesh1 = ax1.pcolormesh(Xg, Tg, u_real, shading="auto", cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("Real Solution")
    fig.colorbar(mesh1, ax=ax1, label="u(x,t)")

    # Predicted solution
    mesh2 = ax2.pcolormesh(Xg, Tg, u_pred, shading="auto", cmap="viridis")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("Predicted Solution")
    fig.colorbar(mesh2, ax=ax2, label="u(x,t)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {savepath}")

    plt.close(fig)


def plot_std_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    std: np.ndarray,
    title: str = "Prediction Uncertainty (Std Dev)",
    savepath: Optional[str] = None,
):
    """
    Plot heatmap of standard deviation (uncertainty).

    Parameters
    ----------
    x : np.ndarray
        Spatial grid (nx,)
    t : np.ndarray
        Time grid (nt,)
    std : np.ndarray
        Standard deviation (nx, nt) or (nt, nx)
    title : str
        Plot title
    savepath : str, optional
        Path to save figure
    """
    # Ensure std is (nt, nx) for plotting
    if std.shape[0] == len(x) and std.shape[1] == len(t):
        std = std.T  # (nx, nt) -> (nt, nx)

    # Create meshgrid
    Xg, Tg = np.meshgrid(x, t, indexing="xy")  # (nt, nx)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(Xg, Tg, std, shading="auto", cmap="hot")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Std Dev")

    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved std heatmap to {savepath}")

    plt.close(fig)

