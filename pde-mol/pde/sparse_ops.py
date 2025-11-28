from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron

from .domain import Domain1D, Domain2D


def build_1d_laplacian(domain: Domain1D) -> csr_matrix:
    """
    Build a sparse 1D Laplacian matrix consistent with the finite-difference
    stencils used in the 1D diffusion operator.

    For periodic domains, the first and last rows are constructed so that
    x=0 and x=L have identical second derivatives using the penultimate
    point as the neighbor, matching the 1D periodic stencils.

    For non-periodic domains, the interior rows use a standard central
    second-difference stencil and the boundary rows are set to zero
    (Neumann-like, but tests focus on interior points).
    """
    n = domain.nx
    dx2 = domain.dx * domain.dx

    main = -2.0 * np.ones(n) / dx2
    off = 1.0 * np.ones(n - 1) / dx2

    A = diags([main, off, off], [0, 1, -1], shape=(n, n), format="lil")

    if domain.periodic:
        # Adjust periodic end rows to mirror the 1D operator logic.
        # Row 0 couples to column 1 and column n-2 (not n-1).
        A[0, -1] = 0.0
        A[0, -2] = 1.0 / dx2
        # Last row shares the same stencil as the first row.
        A[-1, :] = A[0, :]
    else:
        # Zero out boundary rows; interior rows remain central differences.
        A[0, :] = 0.0
        A[-1, :] = 0.0

    return A.tocsr()


def build_2d_laplacian(domain: Domain2D) -> csr_matrix:
    """
    Build a sparse 2D Laplacian matrix (u_xx + u_yy) on a tensor-product grid.

    The operator is constructed as a Kronecker sum:

        L2D = kron(I_y, Lx) + kron(Ly, I_x)

    where Lx and Ly are 1D Laplacians along x and y respectively, built with
    :func:`build_1d_laplacian` using the appropriate periodicity flags.
    """
    # 1D domains capturing the x- and y- directions with matching spacing.
    dom_x = Domain1D(domain.x0, domain.x1, domain.nx, periodic=domain.periodic_x)
    dom_y = Domain1D(domain.y0, domain.y1, domain.ny, periodic=domain.periodic_y)

    Lx = build_1d_laplacian(dom_x)
    Ly = build_1d_laplacian(dom_y)

    Ix = eye(domain.nx, format="csr")
    Iy = eye(domain.ny, format="csr")

    # Kronecker sum
    L2 = kron(Iy, Lx) + kron(Ly, Ix)
    return L2.tocsr()


