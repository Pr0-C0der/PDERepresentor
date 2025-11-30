"""
1D Moisture Diffusion Solver: dX/dt = D * d^2X/dz^2

Uses implicit Euler with Robin boundary conditions:
dX/dz = -h * (X - X_env) at both boundaries.
Optimized with pre-built tridiagonal system.
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


def build_tridiagonal(nz, alpha, dt, D, dz, h, X_env):
    """Build tridiagonal system for implicit Euler with Robin BCs."""
    a = np.zeros(nz)  # lower diag
    b = np.zeros(nz)  # main diag
    c = np.zeros(nz)  # upper diag
    rhs_const = np.zeros(nz)

    # Interior nodes
    for i in range(1, nz - 1):
        a[i] = -alpha
        b[i] = 1.0 + 2.0 * alpha
        c[i] = -alpha
        rhs_const[i] = 0.0

    # Left boundary
    b[0] = 1.0 + dt * D * (2.0 / dz**2 - 2.0 * h / dz)
    c[0] = -dt * D * (2.0 / dz**2)
    rhs_const[0] = -dt * D * (2.0 * h / dz) * X_env

    # Right boundary
    a[nz - 1] = -dt * D * (2.0 / dz**2)
    b[nz - 1] = 1.0 + dt * D * (2.0 / dz**2 - 2.0 * h / dz)
    rhs_const[nz - 1] = -dt * D * (2.0 * h / dz) * X_env

    return a, b, c, rhs_const


def solve_moisture1d(D, h, X_env, z0, z1, nz, t0, t1, nt, X0):
    """
    Solve 1D moisture diffusion: dX/dt = D * d^2X/dz^2 with Robin BCs.

    Parameters
    ----------
    D : float
        Diffusion coefficient
    h : float
        Robin boundary condition coefficient
    X_env : float
        Environmental moisture concentration
    z0, z1 : float
        Spatial domain boundaries
    nz : int
        Number of spatial grid points
    t0, t1 : float
        Time domain boundaries
    nt : int
        Number of time steps
    X0 : float
        Initial moisture concentration (constant)

    Returns
    -------
    z : np.ndarray
        Spatial grid points (shape: (nz,))
    t : np.ndarray
        Time points (shape: (nt,))
    u : np.ndarray
        Solution array (shape: (nz, nt))
    """
    z = np.linspace(z0, z1, nz)
    dz = z[1] - z[0]
    t = np.linspace(t0, t1, nt)
    dt = t[1] - t[0]

    # Initialize
    u = np.zeros((nz, nt))
    u[:, 0] = X0
    alpha = D * dt / dz**2

    # Build tridiagonal system
    a, b, c, rhs_const = build_tridiagonal(nz, alpha, dt, D, dz, h, X_env)

    # Time-stepping
    for n in range(nt - 1):
        rhs = u[:, n].copy() + rhs_const
        u_new = thomas_solve(a, b, c, rhs)
        u[:, n + 1] = u_new

    return z, t, u

