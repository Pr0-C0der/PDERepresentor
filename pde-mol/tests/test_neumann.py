import numpy as np

from pde import Domain1D, InitialCondition, Diffusion, PDEProblem
from pde.bc import NeumannLeft, NeumannRight, apply_neumann_ghosts


def test_neumann_linear_steady_state_diffusion():
    """
    For u(x) = x with Neumann BCs u_x(0)=1, u_x(1)=1, the Laplacian should
    be approximately zero everywhere (steady linear profile).
    """
    dom = Domain1D(0.0, 1.0, 101, periodic=False)
    x = dom.x

    ic = InitialCondition.from_values(x)  # linear profile
    op = Diffusion(1.0)

    bc_left = NeumannLeft(derivative_value=1.0)
    bc_right = NeumannRight(derivative_value=1.0)

    problem = PDEProblem(
        domain=dom,
        operators=[op],
        ic=ic,
        bc_left=bc_left,
        bc_right=bc_right,
    )

    interior = ic.evaluate(dom)[1:-1]
    rhs_val = problem.rhs(t=0.0, y=interior)

    # Expect near-zero diffusion on the interior for a linear function.
    max_abs = float(np.max(np.abs(rhs_val)))
    assert max_abs < 5e-3


def test_neumann_zero_flux_preserves_integral_short_time():
    """
    Heat equation with zero-flux Neumann BCs should approximately conserve
    the integral of u over the domain for short times.
    """
    dom = Domain1D(0.0, 1.0, 201, periodic=False)
    x = dom.x

    # Initial condition: smooth bump inside the domain
    ic = InitialCondition.from_expression("np.exp(-100*(x-0.5)**2)")
    op = Diffusion(0.1)

    bc_left = NeumannLeft(derivative_value=0.0)
    bc_right = NeumannRight(derivative_value=0.0)

    problem = PDEProblem(
        domain=dom,
        operators=[op],
        ic=ic,
        bc_left=bc_left,
        bc_right=bc_right,
    )

    t0 = 0.0
    t1 = 0.01
    t_eval = [t0, t1]

    sol = problem.solve((t0, t1), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    assert sol.success

    # Reconstruct full solutions from interior states
    u0_full = np.zeros(dom.nx)
    u1_full = np.zeros(dom.nx)
    u0_full[1:-1] = sol.y[:, 0]
    u1_full[1:-1] = sol.y[:, -1]

    # Approximate integral via trapezoidal rule. Use numpy.trapz for
    # compatibility with the installed NumPy version.
    integral0 = float(np.trapz(u0_full, x))
    integral1 = float(np.trapz(u1_full, x))

    assert abs(integral1 - integral0) < 1e-3


def test_neumann_quadratic_diffusion_value():
    """
    For u(x) = x^2 with Neumann BCs u_x(0)=0, u_x(1)=2, we expect u_xx = 2.
    Check that the diffusion operator produces approximately 2 in the interior.
    """
    dom = Domain1D(0.0, 1.0, 201, periodic=False)
    x = dom.x

    u_vals = x**2
    ic = InitialCondition.from_values(u_vals)
    op = Diffusion(1.0)

    bc_left = NeumannLeft(derivative_value=0.0)
    bc_right = NeumannRight(derivative_value=2.0)

    problem = PDEProblem(
        domain=dom,
        operators=[op],
        ic=ic,
        bc_left=bc_left,
        bc_right=bc_right,
    )

    interior = ic.evaluate(dom)[1:-1]
    rhs_val = problem.rhs(t=0.0, y=interior)

    # Expected diffusion term is u_xx = 2 everywhere.
    interior_slice = slice(10, -10)
    expected = 2.0
    error = float(np.max(np.abs(rhs_val[interior_slice] - expected)))
    assert error < 5e-2


def test_apply_neumann_ghosts_with_time_dependent_flux():
    """
    Unit test for apply_neumann_ghosts using time-dependent Neumann fluxes.
    """
    dom = Domain1D(0.0, 1.0, 5, periodic=False)
    x = dom.x
    dx = dom.dx

    u_full = np.zeros(dom.nx)
    u_full[1] = 1.0
    u_full[-2] = 2.0

    bc_left = NeumannLeft(expr="1.0 + 0.5*t")
    bc_right = NeumannRight(expr="0.5 * t")

    t = 0.4
    q_left = 1.0 + 0.5 * t
    q_right = 0.5 * t

    apply_neumann_ghosts(u_full, bc_left, bc_right, t, dom)

    # Check the ghost-point-inspired updates
    assert np.isclose(u_full[0], u_full[1] - dx * q_left, rtol=0, atol=1e-12)
    assert np.isclose(u_full[-1], u_full[-2] + dx * q_right, rtol=0, atol=1e-12)
