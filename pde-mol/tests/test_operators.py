import numpy as np

from pde import Domain1D, Domain2D
from pde.operators import (
    Diffusion,
    Advection,
    ExpressionOperator,
    Diffusion2D,
    ExpressionOperator2D,
    sum_operators,
)


def test_diffusion_on_sin():
    # Periodic domain for clean second derivative of sin(x)
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    nu = 0.5
    op = Diffusion(nu)
    result = op.apply(u, dom, t=0.0)

    expected = nu * (-np.sin(x))  # u_xx = -sin(x)
    error = np.max(np.abs(result - expected))

    # Second-order accuracy: with dx ~ 0.03, we expect errors ~ O(dx^2)
    assert error < 5e-3


def test_advection_on_sin():
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    a = 1.0
    op = Advection(a)
    result = op.apply(u, dom, t=0.0)

    expected = -a * np.cos(x)  # -a * u_x
    error = np.max(np.abs(result - expected))
    assert error < 5e-3


def test_expression_operator_burgers_like():
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    nu = 0.1
    expr = "-u*ux + nu*uxx"
    op = ExpressionOperator(expr, params={"nu": nu})

    result = op.apply(u, dom, t=0.0)

    # Analytic values: ux = cos(x), uxx = -sin(x)
    expected = -u * np.cos(x) + nu * (-np.sin(x))

    error = np.max(np.abs(result - expected))
    assert error < 1e-2


def test_sum_operators_matches_manual_sum():
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    diff = Diffusion(0.2)
    adv = Advection(1.0)
    total = sum_operators([diff, adv])

    res_sum = total.apply(u, dom, t=0.0)
    res_manual = diff.apply(u, dom, t=0.0) + adv.apply(u, dom, t=0.0)

    assert np.allclose(res_sum, res_manual)


def test_diffusion2d_on_separable_mode():
    """
    2D diffusion on u(x,y) = sin(x) * sin(y) with periodic domain and nu=1.

    Analytic Laplacian: u_xx + u_yy = -2 * sin(x) * sin(y).
    """
    dom = Domain2D(0.0, 2 * np.pi, 81, 0.0, 2 * np.pi, 81, periodic_x=True, periodic_y=True)
    X, Y = dom.X, dom.Y
    U = np.sin(X) * np.sin(Y)

    u_flat = dom.flatten(U)

    op = Diffusion2D(1.0)
    result_flat = op.apply(u_flat, dom, t=0.0)
    result = dom.unflatten(result_flat)

    expected = -2.0 * np.sin(X) * np.sin(Y)
    error = np.max(np.abs(result - expected))
    assert error < 5e-3


def test_expression_operator2d_reaction_diffusion_mode():
    """
    2D reaction-diffusion for a single mode:
        u_t = nu*(u_xx + u_yy) + lam*u
    on u(x,y) = sin(x) sin(y), which has Laplacian -2u.
    """
    dom = Domain2D(0.0, 2 * np.pi, 81, 0.0, 2 * np.pi, 81, periodic_x=True, periodic_y=True)
    X, Y = dom.X, dom.Y
    U = np.sin(X) * np.sin(Y)
    u_flat = dom.flatten(U)

    nu = 0.1
    lam = -0.05
    expr = "nu_param * (uxx + uyy) + lam * u"
    op = ExpressionOperator2D(expr, params={"nu_param": nu, "lam": lam})

    result_flat = op.apply(u_flat, dom, t=0.0)
    result = dom.unflatten(result_flat)

    expected = nu * (-2.0 * np.sin(X) * np.sin(Y)) + lam * np.sin(X) * np.sin(Y)
    error = np.max(np.abs(result - expected))
    assert error < 5e-3


def test_diffusion_convergence_polynomial_nonperiodic():
    """
    Check that the diffusion operator shows ~second-order convergence
    on a smooth polynomial in a non-periodic domain.
    """
    ns = [51, 101, 201]
    errors = []

    for nx in ns:
        dom = Domain1D(0.0, 1.0, nx, periodic=False)
        x = dom.x
        u = np.sin(np.pi * x)

        op = Diffusion(1.0)
        result = op.apply(u, dom, t=0.0)
        expected = -(np.pi**2) * np.sin(np.pi * x)

        # Ignore a few points near the boundaries where one-sided stencils
        # dominate and focus on the interior where accuracy is highest.
        interior = slice(5, -5)
        error = np.max(np.abs(result[interior] - expected[interior]))
        errors.append(error)

    # Expect roughly O(dx^2) behaviour: when nx roughly doubles, error should
    # drop by about 4×. Be slightly lenient in the assertion.
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]


def test_advection_variable_coefficient_string():
    """
    Advection with a spatially varying coefficient given as an expression.
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    a_expr = "1.0 + 0.5 * np.cos(x)"
    op = Advection(a_expr)
    result = op.apply(u, dom, t=0.0)

    a_vals = 1.0 + 0.5 * np.cos(x)
    expected = -a_vals * np.cos(x)

    error = np.max(np.abs(result - expected))
    assert error < 1e-2


def test_advection_variable_coefficient_callable_with_time():
    """
    Advection with a coefficient that depends on both x and t via a callable.
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    def a_func(x, t):
        return 0.5 + 0.2 * t + 0.1 * x

    t = 0.7
    op = Advection(a_func)
    result = op.apply(u, dom, t=t)

    a_vals = a_func(x, t)
    expected = -a_vals * np.cos(x)

    error = np.max(np.abs(result - expected))
    assert error < 1e-2


def test_expression_operator_linear_combo_with_time_and_params():
    """
    ExpressionOperator with multiple parameters and explicit time dependence.
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    x = dom.x
    u = np.sin(x)

    alpha = 2.0
    beta = 0.3
    gamma = 0.1
    expr = "alpha*u + beta*uxx + gamma*sin(x)*t"

    op = ExpressionOperator(expr, params={"alpha": alpha, "beta": beta, "gamma": gamma})

    t = 0.5
    result = op.apply(u, dom, t=t)

    # Analytic values: ux = cos(x), uxx = -sin(x)
    expected = alpha * u + beta * (-np.sin(x)) + gamma * np.sin(x) * t

    error = np.max(np.abs(result - expected))
    assert error < 1e-2

