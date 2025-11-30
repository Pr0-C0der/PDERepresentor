"""
General Robin Boundary Condition 1D PDE Solver: du/dt = nu * d^2u/dx^2

Uses implicit Euler with general Robin BCs:
- Left: aL*u(0) + bL*u_x(0) = cL (constant)
- Right: aR*u(1) + bR*u_x(1) = cR(t) (time-dependent)
"""
import numpy as np


def thomas_solve(a, b, c, d):
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def build_tridiag(nx, alpha, dt, nu, dx, aL, bL, cL, aR, bR, cR_func, tnext):
    """Build tridiagonal system for backward Euler with general Robin BCs."""
    a = np.zeros(nx)
    b = np.zeros(nx)
    c = np.zeros(nx)
    rhs_const = np.zeros(nx)

    # Interior rows
    for i in range(1, nx - 1):
        a[i] = -alpha
        b[i] = 1.0 + 2.0 * alpha
        c[i] = -alpha

    # Left boundary: aL*u0 + bL*u_x(0) = cL
    b[0] = 1.0 + dt * nu * (2.0 / dx**2 - 2.0 * aL / (bL * dx))
    c[0] = -dt * nu * (2.0 / dx**2)
    a[0] = 0.0
    rhs_const[0] = -dt * nu * (2.0 * cL / (bL * dx))

    # Right boundary: aR*uN + bR*u_x(1) = cR(t)
    a[nx - 1] = -dt * nu * (2.0 / dx**2)
    b[nx - 1] = 1.0 + dt * nu * (2.0 / dx**2 + 2.0 * aR / (bR * dx))
    c[nx - 1] = 0.0
    rhs_const[nx - 1] = +dt * nu * (2.0 * cR_func(tnext) / (bR * dx))

    return a, b, c, rhs_const


def solve_robin_general(nu, x0, x1, nx, t0, t1, nt, u0_func, aL, bL, cL, aR, bR, cR_func):
    """
    Solve 1D diffusion with general Robin BCs.

    Parameters
    ----------
    nu : float
        Diffusion coefficient
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
    aL, bL, cL : float
        Left BC: aL*u(0) + bL*u_x(0) = cL
    aR, bR : float
        Right BC coefficients: aR*u(1) + bR*u_x(1) = cR(t)
    cR_func : callable
        Right BC time-dependent function cR(t) -> float

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

    alpha = nu * dt / dx**2

    # Time-stepping
    for n in range(nt - 1):
        tnext = t[n + 1]
        a, b, c, rhs_const = build_tridiag(
            nx, alpha, dt, nu, dx, aL, bL, cL, aR, bR, cR_func, tnext
        )
        rhs = u[:, n].copy() + rhs_const
        u[:, n + 1] = thomas_solve(a, b, c, rhs)

    return x, t, u

