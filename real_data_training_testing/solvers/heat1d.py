"""
1D Heat Equation Solver: du/dt = nu * d^2u/dx^2

Uses Crank-Nicolson method with periodic boundary conditions.
Optimized with matrix factorization.
"""
import numpy as np
import scipy.linalg


def solve_heat1d(nu, x0, x1, nx, t0, t1, nt, u0_func):
    """
    Solve 1D heat equation: du/dt = nu * d^2u/dx^2 with periodic BCs.

    Parameters
    ----------
    nu : float
        Diffusion coefficient
    x0, x1 : float
        Spatial domain boundaries
    nx : int
        Number of spatial grid points
    t0, t1 : float
        Time domain boundaries
    nt : int
        Number of time steps
    u0_func : callable
        Initial condition function u0(x) -> float

    Returns
    -------
    x : np.ndarray
        Spatial grid points (shape: (nx,))
    t : np.ndarray
        Time points (shape: (nt,))
    u : np.ndarray
        Solution array (shape: (nx, nt))
    """
    dx = (x1 - x0) / nx
    x = np.linspace(x0, x1 - dx, nx)  # periodic grid (exclude endpoint)
    t = np.linspace(t0, t1, nt)
    dt = t[1] - t[0]

    # Initialize solution
    u = np.zeros((nx, nt))
    u[:, 0] = u0_func(x)

    # Build periodic Laplacian matrix (only depends on nx, not on nu)
    L = np.zeros((nx, nx))
    for i in range(nx):
        L[i, i] = -2.0
        L[i, (i - 1) % nx] = 1.0
        L[i, (i + 1) % nx] = 1.0

    # Crank-Nicolson matrices
    alpha = nu * dt / dx**2
    I = np.eye(nx)
    A = I - 0.5 * alpha * L
    B = I + 0.5 * alpha * L

    # Factorize A once (LU decomposition) for efficiency
    lu, piv = scipy.linalg.lu_factor(A)

    # Time-stepping (use factorized solve)
    for n in range(nt - 1):
        rhs = B.dot(u[:, n])
        u[:, n + 1] = scipy.linalg.lu_solve((lu, piv), rhs)

    return x, t, u

