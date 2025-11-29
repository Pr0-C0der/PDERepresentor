"""
Tests for Robin boundary conditions (general form and backward compatibility).
"""

import numpy as np

from pde import Domain1D, InitialCondition, Diffusion, PDEProblem
from pde.bc import RobinLeft, RobinRight


def test_robin_backward_compatibility():
    """
    Test that the old form (h, u_env) still works and produces the same result
    as the equivalent general form.
    """
    dom = Domain1D(0.0, 1.0, 101, periodic=False)
    x = dom.x

    # Old form: u_x = -h * (u - u_env) with h=1.0, u_env=0.2
    bc_left_old = RobinLeft(h=1.0, u_env=0.2)
    bc_right_old = RobinRight(h=1.0, u_env=0.2)

    # Equivalent general form: a*u + b*u_x = c
    # u_x = -h*(u - u_env) = -h*u + h*u_env
    # So: h*u + 1*u_x = h*u_env
    # Therefore: a = h = 1.0, b = 1.0, c = h*u_env = 0.2
    bc_left_new = RobinLeft(a=1.0, b=1.0, c=0.2)
    bc_right_new = RobinRight(a=1.0, b=1.0, c=0.2)

    # Test that both produce the same boundary values
    u_test = np.ones(dom.nx) * 0.5
    t = 0.0

    u_old = u_test.copy()
    u_new = u_test.copy()

    bc_left_old.apply_to_full(u_old, t, dom)
    bc_left_new.apply_to_full(u_new, t, dom)

    assert np.isclose(u_old[0], u_new[0]), f"Left BC mismatch: {u_old[0]} vs {u_new[0]}"

    u_old = u_test.copy()
    u_new = u_test.copy()

    bc_right_old.apply_to_full(u_old, t, dom)
    bc_right_new.apply_to_full(u_new, t, dom)

    assert np.isclose(u_old[-1], u_new[-1]), f"Right BC mismatch: {u_old[-1]} vs {u_new[-1]}"


def test_robin_general_form_constant():
    """
    Test general form with constant coefficients: 2*u + 3*u_x = 5
    """
    dom = Domain1D(0.0, 1.0, 101, periodic=False)

    # BC: 2*u + 3*u_x = 5
    # Rearranging: u_x = (5 - 2*u) / 3
    bc_left = RobinLeft(a=2.0, b=3.0, c=5.0)
    bc_right = RobinRight(a=2.0, b=3.0, c=5.0)

    u = np.ones(dom.nx) * 1.0
    t = 0.0

    bc_left.apply_to_full(u, t, dom)
    bc_right.apply_to_full(u, t, dom)

    # Verify the BC is approximately satisfied
    # For left: (u[1] - u[0]) / dx ≈ u_x[0] = (5 - 2*u[0]) / 3
    ux_left_approx = (u[1] - u[0]) / dom.dx
    ux_left_expected = (5.0 - 2.0 * u[0]) / 3.0
    assert np.isclose(ux_left_approx, ux_left_expected, rtol=1e-3)

    # For right: (u[-1] - u[-2]) / dx ≈ u_x[-1] = (5 - 2*u[-1]) / 3
    ux_right_approx = (u[-1] - u[-2]) / dom.dx
    ux_right_expected = (5.0 - 2.0 * u[-1]) / 3.0
    assert np.isclose(ux_right_approx, ux_right_expected, rtol=1e-3)


def test_robin_time_dependent():
    """
    Test general form with time-dependent right-hand side: u + u_x = sin(t)
    """
    dom = Domain1D(0.0, 1.0, 101, periodic=False)

    bc_left = RobinLeft(a=1.0, b=1.0, c_expr="np.sin(t)")
    bc_right = RobinRight(a=1.0, b=1.0, c_expr="np.sin(t)")

    u = np.ones(dom.nx) * 0.5

    # Test at different times
    for t_val in [0.0, 0.5, 1.0, np.pi / 2]:
        u_test = u.copy()
        bc_left.apply_to_full(u_test, t_val, dom)
        bc_right.apply_to_full(u_test, t_val, dom)

        # Verify BC is approximately satisfied
        c_val = np.sin(t_val)
        ux_left_approx = (u_test[1] - u_test[0]) / dom.dx
        ux_left_expected = c_val - u_test[0]  # From: u + u_x = c, so u_x = c - u
        assert np.isclose(ux_left_approx, ux_left_expected, rtol=1e-3)

        ux_right_approx = (u_test[-1] - u_test[-2]) / dom.dx
        ux_right_expected = c_val - u_test[-1]
        assert np.isclose(ux_right_approx, ux_right_expected, rtol=1e-3)


def test_robin_moisture_diffusion_backward_compat():
    """
    Test that moisture diffusion problem still works with backward-compatible form.
    """
    dom = Domain1D(0.0, 1.0, 101, periodic=False)
    x = dom.x

    ic = InitialCondition.from_expression("np.ones_like(x)")
    op = Diffusion(0.1)

    # Use backward-compatible form
    bc_left = RobinLeft(h=10.0, u_env=0.2)
    bc_right = RobinRight(h=10.0, u_env=0.2)

    problem = PDEProblem(
        domain=dom,
        operators=[op],
        ic=ic,
        bc_left=bc_left,
        bc_right=bc_right,
    )

    # Test that RHS can be evaluated
    interior = ic.evaluate(dom)[1:-1]
    rhs_val = problem.rhs(t=0.0, y=interior)

    assert rhs_val.shape == interior.shape
    assert np.all(np.isfinite(rhs_val))


def test_robin_general_form_moisture_equivalent():
    """
    Test that general form can represent the moisture diffusion BC equivalently.
    """
    dom = Domain1D(0.0, 1.0, 101, periodic=False)
    x = dom.x

    ic = InitialCondition.from_expression("np.ones_like(x)")
    op = Diffusion(0.1)

    # Old form: u_x = -h*(u - u_env) with h=10.0, u_env=0.2
    # This is: u_x = -10*u + 2.0
    # In general form: a*u + b*u_x = c
    # We need: u_x = (c - a*u) / b = -10*u + 2.0
    # So: -a/b = -10, c/b = 2.0
    # If b = 1: a = 10, c = 2.0
    bc_left = RobinLeft(a=10.0, b=1.0, c=2.0)
    bc_right = RobinRight(a=10.0, b=1.0, c=2.0)

    problem = PDEProblem(
        domain=dom,
        operators=[op],
        ic=ic,
        bc_left=bc_left,
        bc_right=bc_right,
    )

    # Test that RHS can be evaluated
    interior = ic.evaluate(dom)[1:-1]
    rhs_val = problem.rhs(t=0.0, y=interior)

    assert rhs_val.shape == interior.shape
    assert np.all(np.isfinite(rhs_val))


def test_robin_error_cases():
    """
    Test error handling for invalid Robin BC configurations.
    """
    # Cannot specify both old and new forms
    try:
        RobinLeft(h=1.0, u_env=0.2, a=2.0, b=1.0, c=3.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Must specify either c or c_expr, not both
    try:
        RobinLeft(a=1.0, b=1.0, c=1.0, c_expr="1.0")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Must specify a (or h for backward compat)
    try:
        RobinLeft(b=1.0, c=1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # b cannot be zero
    try:
        RobinLeft(a=1.0, b=0.0, c=1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

