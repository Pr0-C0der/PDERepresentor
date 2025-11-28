import numpy as np

from pde import (
    Domain1D,
    Domain2D,
    InitialCondition,
    Diffusion,
    Advection,
    ExpressionOperator,
    Diffusion2D,
    ExpressionOperator2D,
    PDEProblem,
)


def test_heat_equation_analytic_solution_periodic():
    """
    u_t = nu * u_xx with u(x, 0) = sin(x) on [0, 2π], periodic.

    Analytic solution:
        u(x, t) = exp(-nu * t) * sin(x)
    because u_xx = -sin(x).
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    nu = 0.5

    ic = InitialCondition.from_expression("np.sin(x)")
    op = Diffusion(nu)

    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0, t1 = 0.0, 1.0
    t_eval = np.linspace(t0, t1, 6)

    sol = problem.solve((t0, t1), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    assert sol.success

    x = dom.x
    u_num = sol.y[:, -1]
    u_exact = np.exp(-nu * t1) * np.sin(x)

    error = np.max(np.abs(u_num - u_exact))

    # Allow modest spatial + temporal discretisation error.
    assert error < 1e-2


def test_burgers_sanity_small_time_step():
    """
    Sanity check: Burgers-like PDE integrated for a very small time step
    should agree with the ExpressionOperator to first order in dt.

    u_t = -u*u_x + nu*u_xx, u(x,0) = sin(x), periodic on [0, 2π].
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    nu = 0.1

    ic = InitialCondition.from_expression("np.sin(x)")
    expr = "-u*ux + nu*uxx"
    op = ExpressionOperator(expr, params={"nu": nu})

    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0 = 0.0
    dt = 1e-3
    t1 = t0 + dt
    t_eval = [t0, t1]

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    assert sol.success

    x = dom.x
    u0 = ic.evaluate(dom)
    u1 = sol.y[:, -1]

    du_dt_num = (u1 - u0) / dt
    du_dt_op = op.apply(u0, dom, t0)

    error = np.max(np.abs(du_dt_num - du_dt_op))

    # First-order in time with a small dt and tight tolerances.
    assert error < 1e-2


def test_heat_equation_dirichlet_nonperiodic():
    """
    u_t = nu * u_xx with u(x, 0) = sin(pi x) on [0, 1], u(0,t)=u(1,t)=0.

    Analytic solution:
        u(x, t) = exp(-nu * pi^2 * t) * sin(pi x)
    """
    from pde import DirichletLeft, DirichletRight

    dom = Domain1D(0.0, 1.0, 201, periodic=False)
    nu = 0.2

    ic = InitialCondition.from_expression("np.sin(np.pi * x)")
    op = Diffusion(nu)

    problem = PDEProblem(
        domain=dom,
        operators=[op],
        ic=ic,
        bc_left=DirichletLeft(0.0),
        bc_right=DirichletRight(0.0),
    )

    t0, t1 = 0.0, 0.5
    t_eval = np.linspace(t0, t1, 5)

    sol = problem.solve((t0, t1), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    assert sol.success

    x = dom.x
    # Reconstruct full solution from interior state
    u_full = np.zeros(dom.nx)
    u_full[1:-1] = sol.y[:, -1]

    u_exact = np.exp(-nu * (np.pi**2) * t1) * np.sin(np.pi * x)

    # Focus on interior where the scheme is most accurate.
    interior = slice(5, -5)
    error = np.max(np.abs(u_full[interior] - u_exact[interior]))
    assert error < 2e-2


def test_advection_analytic_shift_periodic():
    """
    Pure advection u_t + a u_x = 0 with u(x,0)=sin(x) on [0, 2π], periodic.

    Analytic solution (with our sign convention u_t = -a u_x):
        u(x, t) = sin(x - a t)
    """
    a = 0.5
    dom = Domain1D(0.0, 2 * np.pi, 401, periodic=True)

    ic = InitialCondition.from_expression("np.sin(x)")
    op = Advection(a)

    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0, t1 = 0.0, 0.5
    t_eval = np.linspace(t0, t1, 6)

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    assert sol.success

    x = dom.x
    u_num = sol.y[:, -1]
    u_exact = np.sin(x - a * t1)

    error = np.max(np.abs(u_num - u_exact))
    # Centered advection is dispersive; keep tolerance moderate.
    assert error < 3e-2


def test_problem_rhs_matches_operator_periodic():
    """
    Directly compare PDEProblem.rhs against the underlying operator
    for a periodic heat equation at a single time.
    """
    dom = Domain1D(0.0, 2 * np.pi, 201, periodic=True)
    nu = 0.3

    ic = InitialCondition.from_expression("np.cos(2 * x)")
    op = Diffusion(nu)
    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t = 0.4
    u = ic.evaluate(dom)

    rhs_val = problem.rhs(t, u)
    op_val = op.apply(u, dom, t)

    error = np.max(np.abs(rhs_val - op_val))
    assert error < 1e-12


def test_heat_equation_2d_separable_mode_periodic():
    """
    2D heat equation on u(x,y,0) = sin(x) sin(y) with periodic BCs.

    With u_t = nu (u_xx + u_yy), Laplacian is -2u, so:
        u(x,y,t) = exp(-2*nu*t) * sin(x) sin(y)
    """
    nu = 0.1
    dom = Domain2D(0.0, 2 * np.pi, 81, 0.0, 2 * np.pi, 81, periodic_x=True, periodic_y=True)

    ic = InitialCondition.from_expression("np.sin(x) * np.sin(y)")
    op = Diffusion2D(nu)
    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0, t1 = 0.0, 0.5
    t_eval = np.linspace(t0, t1, 4)

    sol = problem.solve((t0, t1), t_eval=t_eval, rtol=1e-7, atol=1e-9)
    assert sol.success

    U0 = ic.evaluate(dom)
    U1 = dom.unflatten(sol.y[:, -1])

    factor = np.exp(-2.0 * nu * t1)
    X, Y = dom.X, dom.Y
    U_exact = factor * np.sin(X) * np.sin(Y)

    error = np.max(np.abs(U1 - U_exact))
    assert error < 3e-2


def test_expression_operator2d_burgers_like_small_time_step():
    """
    2D Burgers-like PDE:
        u_t = -u*ux - u*uy + nu*(uxx + uyy)
    with u(x,y,0) = sin(x) sin(y), periodic.

    For a very small time step, the numerical time derivative should
    match the ExpressionOperator2D evaluation.
    """
    nu = 0.05
    dom = Domain2D(0.0, 2 * np.pi, 81, 0.0, 2 * np.pi, 81, periodic_x=True, periodic_y=True)

    ic = InitialCondition.from_expression("np.sin(x) * np.sin(y)")

    expr = "-u*ux - u*uy + nu_param * (uxx + uyy)"
    op = ExpressionOperator2D(expr, params={"nu_param": nu})
    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0 = 0.0
    dt = 5e-4
    t1 = t0 + dt
    t_eval = [t0, t1]

    sol = problem.solve((t0, t1), t_eval=t_eval, rtol=1e-7, atol=1e-9)
    assert sol.success

    U0 = ic.evaluate(dom)
    U1 = dom.unflatten(sol.y[:, -1])

    du_dt_num = (U1 - U0) / dt

    u_flat0 = dom.flatten(U0)
    du_dt_op_flat = op.apply(u_flat0, dom, t0)
    du_dt_op = dom.unflatten(du_dt_op_flat)

    error = np.max(np.abs(du_dt_num - du_dt_op))
    assert error < 5e-2


def test_advection_diffusion_fourier_mode_periodic():
    """
    u_t = -a u_x + nu u_xx with u(x,0) = sin(x) on [0, 2π], periodic.

    Analytic solution:
        u(x, t) = exp(-nu * t) * sin(x - a t)
    """
    a = 0.7
    nu = 0.2
    dom = Domain1D(0.0, 2 * np.pi, 401, periodic=True)

    ic = InitialCondition.from_expression("np.sin(x)")
    ops = [Advection(a), Diffusion(nu)]
    problem = PDEProblem(domain=dom, operators=ops, ic=ic)

    t0, t1 = 0.0, 0.5
    t_eval = np.linspace(t0, t1, 6)

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    assert sol.success

    x = dom.x
    u_num = sol.y[:, -1]
    u_exact = np.exp(-nu * t1) * np.sin(x - a * t1)

    error = np.max(np.abs(u_num - u_exact))
    assert error < 3e-2


def test_reaction_diffusion_multimode_expression_operator():
    """
    Reaction-diffusion PDE:
        u_t = nu * u_xx + lam * u
    with initial condition u(x,0) = sin(x) + 0.5 sin(2x) on [0, 2π], periodic.

    Analytic solution:
        Each mode sin(kx) evolves as exp((lam - nu*k^2) * t) * sin(kx).
    """
    dom = Domain1D(0.0, 2 * np.pi, 401, periodic=True)
    x = dom.x

    nu = 0.1
    lam = -0.05

    ic = InitialCondition.from_expression("np.sin(x) + 0.5*np.sin(2*x)")

    expr = "nu_param*uxx + lam*u"
    op = ExpressionOperator(expr, params={"nu_param": nu, "lam": lam})
    problem = PDEProblem(domain=dom, operators=[op], ic=ic)

    t0, t1 = 0.0, 0.8
    t_eval = np.linspace(t0, t1, 5)

    sol = problem.solve(
        (t0, t1),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    assert sol.success

    u_num = sol.y[:, -1]

    # Analytic evolution of each Fourier mode
    mode1_factor = np.exp((lam - nu * 1.0**2) * t1)
    mode2_factor = np.exp((lam - nu * 4.0) * t1)
    u_exact = mode1_factor * np.sin(x) + 0.5 * mode2_factor * np.sin(2 * x)

    error = np.max(np.abs(u_num - u_exact))
    assert error < 3e-2

