"""
1D Burgers Equation Solver: du/dt = -u * du/dx + nu * d^2u/dx^2

Uses explicit Euler with periodic boundary conditions.
"""
import numpy as np


def solve_burgers1d(nu, x0, x1, nx, t0, t1, nt, u0_func):
    """
    Solve 1D Burgers equation: du/dt = -u * du/dx + nu * d^2u/dx^2 with periodic BCs.

    Parameters
    ----------
    nu : float
        Viscosity coefficient
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
    x = np.linspace(x0, x1 - dx, nx)  # periodic grid
    t = np.linspace(t0, t1, nt)
    dt = t[1] - t[0]

    # Initialize
    u = np.zeros((nx, nt))
    u[:, 0] = u0_func(x)

    # Time-stepping (explicit Euler)
    for n in range(nt - 1):
        u_n = u[:, n]

        # Periodic neighbors
        u_ip1 = np.roll(u_n, -1)  # i+1
        u_im1 = np.roll(u_n, 1)   # i-1

        # Derivatives
        ux = (u_ip1 - u_im1) / (2.0 * dx)
        uxx = (u_ip1 - 2.0 * u_n + u_im1) / (dx * dx)

        # RHS
        adv_term = -u_n * ux
        diff_term = nu * uxx
        rhs = adv_term + diff_term

        # Advance
        u[:, n + 1] = u_n + dt * rhs

    return x, t, u

