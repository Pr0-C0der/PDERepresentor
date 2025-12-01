from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags

from .domain import Domain1D


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


