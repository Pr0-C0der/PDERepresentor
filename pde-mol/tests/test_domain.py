import numpy as np

from pde import Domain1D, Domain2D


def test_domain1d_basic_grid():
    dom = Domain1D(0.0, 1.0, 11)

    assert dom.nx == 11
    assert np.isclose(dom.dx, 0.1)
    assert np.isclose(dom.x[0], 0.0)
    assert np.isclose(dom.x[-1], 1.0)
    assert dom.x.shape == (11,)


def test_domain1d_to_from_dict_roundtrip():
    dom = Domain1D(0.0, 2.0, 21, periodic=True)
    d = dom.to_dict()

    dom2 = Domain1D.from_dict(d)

    assert dom2.x0 == 0.0
    assert dom2.x1 == 2.0
    assert dom2.nx == 21
    assert dom2.periodic is True
    assert np.allclose(dom.x, dom2.x)
    assert np.isclose(dom.dx, dom2.dx)


def test_domain2d_basic_grid_and_mesh():
    dom = Domain2D(0.0, 1.0, 11, 0.0, 2.0, 21, periodic_x=True, periodic_y=False)

    assert dom.nx == 11
    assert dom.ny == 21
    assert np.isclose(dom.dx, 0.1)
    assert np.isclose(dom.dy, 2.0 / 20.0)

    assert dom.shape == (21, 11)
    assert dom.size == 21 * 11

    X, Y = dom.X, dom.Y
    assert X.shape == (21, 11)
    assert Y.shape == (21, 11)

    # Check endpoints
    assert np.isclose(dom.x[0], 0.0)
    assert np.isclose(dom.x[-1], 1.0)
    assert np.isclose(dom.y[0], 0.0)
    assert np.isclose(dom.y[-1], 2.0)


def test_domain2d_flatten_unflatten_roundtrip():
    dom = Domain2D(0.0, 1.0, 5, 0.0, 1.0, 4)
    X, Y = dom.X, dom.Y

    u2d = np.sin(np.pi * X) * np.cos(np.pi * Y)
    flat = dom.flatten(u2d)
    assert flat.shape == (dom.size,)

    u2d_back = dom.unflatten(flat)
    assert u2d_back.shape == dom.shape
    assert np.allclose(u2d_back, u2d)

