"""
1D Moisture Diffusion Solver (Sagar's version): dX/dt = D * d^2X/dz^2

Uses solve_ivp with BDF method and Robin boundary conditions:
dX/dz = -h * (X - X_env) at both boundaries.
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve_moisture_1d_sagar(D, h, X_env, z0, z1, nz, t0, t1, nt, X0):
    """
    Solve 1D moisture diffusion: dX/dt = D * d^2X/dz^2 with Robin BCs.
    
    Uses solve_ivp with BDF method for adaptive time stepping.

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
        Number of time evaluation points
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
    L = z1 - z0
    dz = L / (nz - 1)
    
    # Build spatial discretization matrix
    A = np.zeros((nz, nz))
    
    # Interior nodes: standard second-order finite difference
    for i in range(1, nz - 1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
    
    # Left boundary: Robin BC dX/dz = -h * (X - X_env)
    # Using ghost point method: (X[-1] - X[ghost])/(2*dz) = -h * (X[0] - X_env)
    # This gives: X[ghost] = X[-1] + 2*dz*h*(X[0] - X_env)
    # Substituting into the FD scheme at i=0:
    A[0, 0] = -2 - 2*dz*h
    A[0, 1] = 2
    
    # Right boundary: Robin BC dX/dz = -h * (X - X_env)
    A[-1, -2] = 2
    A[-1, -1] = -2 - 2*dz*h
    
    # Scale by diffusion coefficient
    A = (D / dz**2) * A
    
    # Build constant term for boundary conditions
    b = np.zeros(nz)
    b[0] = 2*dz*h*X_env*(D/dz**2)
    b[-1] = 2*dz*h*X_env*(D/dz**2)
    
    # ODE system: dX/dt = A@X + b
    def ode_system(t, y):
        return A @ y + b
    
    # Initial condition
    y0 = np.full(nz, X0)
    
    # Time evaluation points
    t_eval = np.linspace(t0, t1, nt)
    
    # Solve with BDF method
    sol = solve_ivp(
        ode_system, 
        [t0, t1], 
        y0=y0, 
        t_eval=t_eval, 
        method='BDF', 
        rtol=1e-6, 
        atol=1e-8
    )
    
    # Extract solution
    z = np.linspace(z0, z1, nz)
    t = sol.t
    u = sol.y  # (nz, nt)
    
    return z, t, u

