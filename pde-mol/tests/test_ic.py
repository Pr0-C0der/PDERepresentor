import numpy as np

from pde import Domain1D, InitialCondition


def test_ic_from_expression_matches_numpy():
    dom = Domain1D(0.0, 1.0, 101)
    ic = InitialCondition.from_expression("np.sin(np.pi * x)")

    u0 = ic.evaluate(dom)
    x = dom.x
    expected = np.sin(np.pi * x)

    assert u0.shape == x.shape
    assert np.allclose(u0, expected)


def test_ic_from_values_uses_given_array():
    dom = Domain1D(0.0, 1.0, 5)
    values = np.linspace(0.0, 1.0, 5)
    ic = InitialCondition.from_values(values)

    u0 = ic.evaluate(dom)
    assert np.allclose(u0, values)


def test_ic_from_callable():
    dom = Domain1D(0.0, 1.0, 11)

    def f(x):
        return x**2

    ic = InitialCondition.from_callable(f)
    u0 = ic.evaluate(dom)

    assert np.allclose(u0, dom.x**2)

