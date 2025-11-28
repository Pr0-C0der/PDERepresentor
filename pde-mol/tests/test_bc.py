import numpy as np

from pde import (
    Domain1D,
    DirichletLeft,
    DirichletRight,
    Periodic,
    NeumannLeft,
    NeumannRight,
)


def test_dirichlet_left_constant():
    dom = Domain1D(0.0, 1.0, 5)
    u = np.zeros(dom.nx)

    bc = DirichletLeft(value=2.0)
    bc.apply_to_full(u, t=0.0, domain=dom)

    assert u[0] == 2.0


def test_dirichlet_right_expression_in_time():
    dom = Domain1D(0.0, 1.0, 5)
    u = np.zeros(dom.nx)

    bc = DirichletRight(expr="np.sin(t)")
    t = 0.5
    bc.apply_to_full(u, t=t, domain=dom)

    assert np.isclose(u[-1], np.sin(t))


def test_periodic_is_noop():
    dom = Domain1D(0.0, 1.0, 5)
    u = np.arange(dom.nx, dtype=float)
    before = u.copy()

    bc = Periodic()
    bc.apply_to_full(u, t=1.23, domain=dom)

    # Periodic BC currently does not touch u_full
    assert np.allclose(u, before)


def test_neumann_placeholders_exist_and_do_not_error():
    dom = Domain1D(0.0, 1.0, 5)
    u = np.zeros(dom.nx)

    left = NeumannLeft(derivative_value=0.0)
    right = NeumannRight(expr="0.0")

    # Placeholders should run without raising
    left.apply_to_full(u, t=0.0, domain=dom)
    right.apply_to_full(u, t=0.0, domain=dom)

