"""
Custom 1D PDE Solver: du/dt = -a * du/dx + nu * d^2u/dx^2 + lambda * sin(t) * u

Uses implicit Euler with Dirichlet boundary conditions (u=0 at boundaries).
"""
import numpy as np


def thomas_solve(sub, main, sup, rhs):
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(rhs)
    cprime = np.zeros(n)
    dprime = np.zeros(n)
    xsol = np.zeros(n)

    denom = main[0]
    if abs(denom) < 1e-14:
        denom = 1e-14
    cprime[0] = sup[0] / denom
    dprime[0] = rhs[0] / denom

    for i in range(1, n):
        denom = main[i] - sub[i] * cprime[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cprime[i] = sup[i] / denom if i < n - 1 else 0.0
        dprime[i] = (rhs[i] - sub[i] * dprime[i - 1]) / denom

    xsol[-1] = dprime[-1]
    for i in range(n - 2, -1, -1):
        xsol[i] = dprime[i] - cprime[i] * xsol[i + 1]

    return xsol


def build_tridiagonal_for_time(tnext, dt, adv_coef, diff_coef, lam, M):
    """Build tridiagonal system for backward Euler at time tnext."""
    s = np.sin(tnext)
    sub = -dt * (adv_coef + diff_coef)
    main = 1.0 + dt * (adv_coef + 2.0 * diff_coef - lam * s)
    sup = -dt * diff_coef

    sub_arr = np.full(M, sub)
    main_arr = np.full(M, main)
    sup_arr = np.full(M, sup)

    return sub_arr, main_arr, sup_arr


def solve_custom_pde(a, nu, lam, x0, x1, nx, t0, t1, nt, u0_func):
    """
    Solve custom 1D PDE: du/dt = -a * du/dx + nu * d^2u/dx^2 + lambda * sin(t) * u.

    Parameters
    ----------
    a : float
        Advection speed
    nu : float
        Diffusion coefficient
    lam : float
        Reaction coefficient multiplier
    x0, x1 : float
        Spatial domain boundaries
    nx : int
        Number of spatial grid points (including boundaries)
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
    x = np.linspace(x0, x1, nx)
    dx = x[1] - x[0]
    t = np.linspace(t0, t1, nt)
    dt = t[1] - t[0]

    # Initialize
    u = np.zeros((nx, nt))
    u[:, 0] = u0_func(x)
    u[0, :] = 0.0  # Dirichlet BCs
    u[-1, :] = 0.0

    # Interior nodes
    M = nx - 2
    idx = np.arange(1, nx - 1)

    adv_coef = a / dx
    diff_coef = nu / dx**2

    # Time-stepping (implicit Euler)
    for n in range(nt - 1):
        tnext = t[n + 1]
        sub_arr, main_arr, sup_arr = build_tridiagonal_for_time(
            tnext, dt, adv_coef, diff_coef, lam, M
        )

        rhs = u[idx, n].copy()
        u_interior_new = thomas_solve(sub_arr, main_arr, sup_arr, rhs)

        u[1:-1, n + 1] = u_interior_new
        u[0, n + 1] = 0.0
        u[-1, n + 1] = 0.0

    return x, t, u

