import numpy as np

from pde import Domain1D, Domain2D
from pde.operators import Diffusion, Diffusion2D
from pde.sparse_ops import build_1d_laplacian, build_2d_laplacian


def test_sparse_1d_laplacian_matches_diffusion_periodic():
    """
    For a periodic 1D domain, the sparse Laplacian should match the
    Diffusion operator applied to a smooth mode.
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    A = build_1d_laplacian(dom)
    lap_sparse = A @ u

    op = Diffusion(1.0)
    lap_fd = op.apply(u, dom, t=0.0)

    error = np.max(np.abs(lap_sparse - lap_fd))
    assert error < 1e-10


def test_sparse_2d_laplacian_matches_diffusion2d_periodic():
    """
    For a periodic 2D domain, the sparse Laplacian constructed via Kronecker
    products should match the Diffusion2D operator.
    """
    dom = Domain2D(
        0.0,
        2 * np.pi,
        81,
        0.0,
        2 * np.pi,
        81,
        periodic_x=True,
        periodic_y=True,
    )
    X, Y = dom.X, dom.Y
    U = np.sin(X) * np.sin(Y)
    u_flat = dom.flatten(U)

    A2 = build_2d_laplacian(dom)
    lap_sparse_flat = A2 @ u_flat

    op = Diffusion2D(1.0)
    lap_fd_flat = op.apply(u_flat, dom, t=0.0)

    error = np.max(np.abs(lap_sparse_flat - lap_fd_flat))
    assert error < 1e-8


