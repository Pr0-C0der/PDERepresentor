## pde-mol — JSON-driven Method-of-Lines PDE Solver

`pde-mol` is a small, modular framework for solving time-dependent PDEs using
the **Method of Lines (MOL)**, driven by **JSON configuration files**.

In MOL we:

- **discretise space** → obtain a semi-discrete ODE system,
- **integrate in time** → use standard ODE solvers (`scipy.integrate.solve_ivp`).

The design emphasises:

- **Modularity** – domain, grid, IC, BC, operators, and problem wrapper are
  independent components with clear interfaces.
- **Simplicity** – straightforward finite-difference defaults, no hidden magic.
- **Extensibility** – 1D and 2D structured grids, multiple operator types,
  expression-based operators, and sparse matrix back-ends.
- **Safety & performance** – restricted expression evaluation, SymPy-based
  precompilation, and optional sparse operators/Jacobians.

---

### Project layout

- `pde/`
  - `domain.py` – `Domain1D`, `Domain2D` (structured grids, flatten/unflatten).
  - `ic.py` – `InitialCondition` (expression / callable / array ICs).
  - `bc.py` – Dirichlet, Neumann, Periodic BCs + Neumann ghost helpers.
  - `operators.py` – 1D/2D diffusion, advection, expression operators, sums.
  - `plotting.py` – 1D/2D plotting helpers and time-series PNG generation.
  - `sparse_ops.py` – sparse 1D/2D Laplacian builders (CSR matrices).
  - `problem.py` – `PDEProblem` MOL wrapper and RHS/solve logic.
  - `json_loader.py` – JSON → `PDEProblem` builder.
  - `run_problem.py` – CLI entry point for running JSON-defined problems.
- `examples/`
  - `heat1d.json` – 1D periodic heat equation.
  - `burgers1d.json` – 1D Burgers-like equation.
- `tests/`
  - Unit tests for domains, IC, BC, operators (1D/2D), Neumann BCs,
    `PDEProblem`, JSON loader, and sparse operators.

All public classes and functions include docstrings describing **what they do**,
their **inputs** (types/shapes) and **outputs**.

---

### Module reference (files in `pde/`)

Below is a brief description of each class/function in the `pde` package and
what it does. For detailed signatures and parameter types, see the docstrings.

#### `domain.py`

- **`Domain1D(x0, x1, nx, periodic=False)`**  
  Structured 1D domain on \([x_0, x_1]\) with `nx` grid points.  
  - **Inputs**: floats `x0, x1`; int `nx`; bool `periodic`.  
  - **Properties**: `x` (1D NumPy array of length `nx`), `dx` (float spacing).  
  - **Methods**: `to_dict() -> dict`, `from_dict(d: dict) -> Domain1D`.

- **`Domain2D(x0, x1, nx, y0, y1, ny, periodic_x=False, periodic_y=False)`**  
  Structured 2D rectangular domain with tensor-product grid.  
  - **Inputs**: bounds and counts in x/y; booleans for periodicity along each axis.  
  - **Properties**:  
    - `x`, `y` (1D arrays), `X`, `Y` (meshgrids, shape `(ny, nx)`),  
    - `dx`, `dy` (spacings), `shape` `(ny, nx)`, `size` (`nx*ny`), `periodic` (bool, true if both axes periodic).  
  - **Methods**:  
    - `flatten(u2d: array) -> array` (ravel `(ny, nx)` to `(ny*nx,)`),  
    - `unflatten(u_flat: array) -> array` (reshape back to `(ny, nx)`),  
    - `to_dict()`, `from_dict(d: dict) -> Domain2D`.

#### `ic.py`

- **`InitialCondition`** (dataclass with `expr`, `values`, `func`)  
  Represents an initial condition as:
  - string expression in `x` (1D) or `x, y` (2D) with `np` (e.g. `"np.sin(x)"`, `"np.sin(x)*np.sin(y)"`), or  
  - array-like of values matching the grid, or  
  - callable `f(x)` (1D) or `f(X)` (2D) returning an array.  
  - **Key methods**:  
    - `from_expression(expr: str) -> InitialCondition`  
    - `from_values(values: array_like) -> InitialCondition`  
    - `from_callable(func: callable) -> InitialCondition`  
    - `evaluate(domain: Domain1D | Domain2D) -> np.ndarray` – returns 1D or 2D NumPy array on the domain grid.

#### `bc.py`

- **`BoundaryCondition` (base class)**  
  Abstract interface with `apply_to_full(u_full, t, domain)` for in-place BC enforcement.

- **`DirichletLeft(value=None, expr=None)` / `DirichletRight(value=None, expr=None)`**  
  Fixed value at left/right boundaries.  
  - `value`: constant float, or  
  - `expr`: string in `t` and `np` (e.g. `"np.sin(t)"`).  
  - `apply_to_full(u_full, t, domain)` sets `u_full[0]` or `u_full[-1]`.

- **`Periodic()`**  
  Marker BC for periodic domains. `apply_to_full` is a no-op; periodicity is handled by operators.

- **`NeumannLeft(derivative_value=None, expr=None)` / `NeumannRight(...)`**  
  Neumann conditions specifying normal derivative \(u_x\) at left/right boundary.  
  - `derivative_value`: constant flux, or  
  - `expr`: string in `t` and `np`.  
  - `_evaluate_flux(t) -> float` returns \(q(t)\).  
  - `apply_to_full` does not modify `u_full` directly (enforced via helper).

- **`apply_neumann_ghosts(u_full, bc_left, bc_right, t, domain)`**  
  Adjusts boundary values of `u_full` to approximate Neumann fluxes using a
  one-sided first-order stencil:  
  - left: `u[0] = u[1] - dx * q_left(t)`  
  - right: `u[-1] = u[-2] + dx * q_right(t)`  
  - **Inputs**: solution array `u_full`, optional left/right BCs, time `t`, `Domain1D`.  
  - **Output**: modifies `u_full` in-place; returns `None`.

#### `operators.py`

- **`Operator` (base class)**  
  - Method: `apply(u_full: array, domain, t: float) -> array` – spatial RHS contribution.

- **1D helper functions**:  
  - `_central_first_derivative(u, dx, periodic) -> array` – 1D central first derivative.  
  - `_central_second_derivative(u, dx, periodic) -> array` – 1D central second derivative.  
  - `_eval_coeff(coeff, x, t) -> array` – evaluate scalar/callable/string coefficient.

- **`Diffusion(nu)`**  
  1D diffusion operator: returns `nu(x,t) * u_xx`.  
  - `nu`: scalar, callable `nu(x,t)`, or string expression in `x,t`.  
  - `apply(u_full, domain: Domain1D, t) -> array` of same shape as `u_full`.

- **`Advection(a)`**  
  1D advection operator: returns `-a(x,t) * u_x`.  
  - `a`: scalar, callable `a(x,t)`, or string expression in `x,t`.  
  - `apply(u_full, domain: Domain1D, t) -> array`.

- **`ExpressionOperator(expr_string, params=None)`**  
  1D expression operator in terms of `(u, ux, uxx, x, t, params)`.  
  - `expr_string`: SymPy expression string (e.g. `"-u*ux + nu*uxx"`).  
  - `params`: dict of parameter names → values.  
  - Internally: parses with `sympy.sympify`, compiles with `sympy.lambdify`.  
  - `apply(u_full, domain: Domain1D, t) -> array`.

- **`sum_operators(ops: Iterable[Operator]) -> Operator`**  
  Returns an `Operator` whose `apply` gives the sum of `apply` for all `ops`.

- **`Diffusion2D(nu)`**  
  2D diffusion: returns `nu * (u_xx + u_yy)` on `Domain2D`.  
  - `apply(u_full_flat, domain: Domain2D, t) -> flat array` (same size as input).  
  - Internally unflattens to `(ny, nx)`, applies 2D Laplacian, then flattens.

- **`ExpressionOperator2D(expr_string, params=None)`**  
  2D expression operator in `(u, ux, uy, uxx, uyy, x, y, t, params)`.  
  - `expr_string`: e.g. `"nu*(uxx+uyy) - u*ux"`.  
  - `params`: dict of parameter names → values.  
  - Uses SymPy to parse and lambdify; computes 2D derivatives via FD stencils.  
  - `apply(u_full_flat, domain: Domain2D, t) -> flat array`.

#### `sparse_ops.py`

- **`build_1d_laplacian(domain: Domain1D) -> csr_matrix`**  
  Builds a sparse CSR matrix representing the 1D Laplacian consistent with the
  `Diffusion` operator’s stencil (including periodic endpoints).

- **`build_2d_laplacian(domain: Domain2D) -> csr_matrix`**  
  Builds a sparse CSR matrix for `u_xx + u_yy` on a tensor-product grid using a
  Kronecker-sum: `kron(I_y, Lx) + kron(Ly, I_x)`, where `Lx`, `Ly` are 1D Laplacians.

#### `problem.py`

- **`PDEProblem(domain, operators, ic, bc_left=None, bc_right=None)`**  
  High-level MOL wrapper that assembles domain, operators, IC, and BCs.  
  - **Inputs**:  
    - `domain`: `Domain1D` or `Domain2D`,  
    - `operators`: sequence of `Operator` instances (1D or 2D as appropriate),  
    - `ic`: `InitialCondition`,  
    - `bc_left`, `bc_right`: optional `BoundaryCondition` (1D only).  
  - **Key methods**:  
    - `initial_full(t0=0.0) -> array` – full state; flattened for 2D.  
    - `initial_interior(t0=0.0) -> array` – 1D interior (non-periodic).  
    - `rhs(t, y) -> array` – unified RHS for testing (dispatches 1D/2D/periodic).  
    - `solve(t_span, t_eval=None, method="RK45", **kwargs) -> OdeResult` – wraps `solve_ivp`.  
  - Internally uses `_rhs_periodic`, `_rhs_nonperiodic`, `_rhs_2d` and `_reconstruct_full_from_interior` for 1D non-periodic Neumann/Dirichlet handling.

#### `json_loader.py`

- **`build_problem_from_dict(config: dict) -> PDEProblem`**  
  Parses a JSON-like dict containing `domain`, `initial_condition`,
  `boundary_conditions` (optional), and `operators`, and returns a configured
  `PDEProblem`.

- **`load_from_json(path: str | Path) -> PDEProblem`**  
  Reads a JSON file from disk, parses it into a dict, and calls
  `build_problem_from_dict`.  
  - **Input**: file path string or `Path`.  
  - **Output**: ready-to-solve `PDEProblem` instance.

#### `__init__.py`

Exports the main public API:

- `Domain1D`, `Domain2D`  
- `InitialCondition`  
- `BoundaryCondition`, `DirichletLeft`, `DirichletRight`, `Periodic`, `NeumannLeft`, `NeumannRight`  
- `Operator`, `Diffusion`, `Advection`, `ExpressionOperator`, `sum_operators`, `Diffusion2D`, `ExpressionOperator2D`  
- `PDEProblem`

#### `plotting.py`

- **`plot_1d(x, u, title=None, savepath=None)`**  
  Creates a simple 1D line plot of `u(x)` with labeled axes and optional title;
  saves a PNG if `savepath` is provided.

- **`plot_1d_time_series(x, solutions, times, prefix="solution1d", out_dir=None)`**  
  Saves a sequence of 1D line plots (one per time) as PNGs named with the
  given prefix and time index.

- **`plot_2d(X, Y, U, title=None, savepath=None)`**  
  Uses `pcolormesh` to visualise a 2D scalar field `U(X,Y)` with labeled axes
  and a colorbar; saves a PNG if `savepath` is provided.

- **`plot_2d_time_series(X, Y, solutions, times, prefix="solution2d", out_dir=None)`**  
  Saves a sequence of 2D colormaps (one per time) as PNGs, suitable for
  combining into GIFs or videos externally.


---

### Installation & development

From the `pde-mol` directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run tests
pytest
```

You can also install the package in editable mode once a `pyproject.toml` or
`setup.cfg` is added:

```bash
pip install -e .
```

---

### JSON schema

A problem is described by a JSON file with the following top-level keys:

- **`domain`** (required)
- **`initial_condition`** (required)
- **`boundary_conditions`** (optional)
- **`operators`** (required)
- **`time`** (optional for CLI/tests)

#### `domain`

1D:

```json
"domain": {
  "type": "1d",
  "x0": 0.0,
  "x1": 6.283185307179586,
  "nx": 201,
  "periodic": true
}
```

2D:

```json
"domain": {
  "type": "2d",
  "x0": 0.0,
  "x1": 6.283185307179586,
  "nx": 81,
  "y0": 0.0,
  "y1": 6.283185307179586,
  "ny": 81,
  "periodic_x": true,
  "periodic_y": true
}
```

#### `initial_condition`

```json
"initial_condition": {
  "type": "expression",      // "expression" or "values"
  "expr": "np.sin(x)"        // for 2D: e.g. "np.sin(x)*np.sin(y)"
}
```

For `"values"`, `values` must be an array (1D or 2D) matching the grid.

#### `boundary_conditions` (1D)

```json
"boundary_conditions": {
  "left":  { "type": "dirichlet", "value": 0.0 },
  "right": { "type": "dirichlet", "expr": "0.0" }
}
```

- Types: `"dirichlet"`, `"neumann"`, `"periodic"`.
- Neumann BCs support `derivative_value` or `expr` for the flux.

#### `operators`

```json
"operators": [
  { "type": "diffusion", "nu": 0.5 },
  { "type": "advection", "a": "1.0 + 0.5 * np.cos(x)" },
  {
    "type": "expression",
    "expr": "-u*ux + nu*uxx",
    "params": { "nu": 0.1 }
  }
]
```

Supported operator entries:

- `"diffusion"` (1D): `{"type": "diffusion", "nu": ...}`
- `"advection"` (1D): `{"type": "advection", "a": ...}`
- `"expression"` / `"expression_operator"` (1D):
  - `expr` in variables `u, ux, uxx, x, t` plus optional `params`.
- 2D expression operator (via code, not JSON in examples):
  - `ExpressionOperator2D` uses `u, ux, uy, uxx, uyy, x, y, t`.

#### `time`

Used by the CLI and tests to define the time grid:

```json
"time": {
  "t0": 0.0,
  "t1": 1.0,
  "num_points": 6,
  "method": "RK45",
  "rtol": 1e-8,
  "atol": 1e-10
}
```

If `t_eval` is given explicitly it overrides `num_points`.

---

### CLI usage

You can run a JSON-defined problem using the small CLI:

```bash
cd pde-mol
python run_problem.py examples/heat1d.json
python run_problem.py examples/burgers1d.json
```

This will:

- Load the JSON,
- Build a `PDEProblem` via `json_loader.build_problem_from_dict`,
- Integrate in time using the `"time"` block, and
- Print simple diagnostics (final time and L2 norm).

Use `--no-output` to suppress printing and rely on the exit code only.

---

### Python API quickstart

**1D heat equation** \(u_t = \nu u_{xx}\) on \([0, 2\pi]\):

```python
import numpy as np
from pde import Domain1D, InitialCondition, Diffusion, PDEProblem

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic = InitialCondition.from_expression("np.sin(x)")
op = Diffusion(0.5)

problem = PDEProblem(domain=dom, operators=[op], ic=ic)
sol = problem.solve((0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 6))
u_final = sol.y[:, -1]
```

**2D heat equation** \(u_t = \nu (u_{xx} + u_{yy})\) on \([0, 2\pi]^2\):

```python
from pde import Domain2D, InitialCondition, PDEProblem
from pde.operators import Diffusion2D

dom2d = Domain2D(0.0, 2*np.pi, 81, 0.0, 2*np.pi, 81, periodic_x=True, periodic_y=True)
ic2d = InitialCondition.from_expression("np.sin(x) * np.sin(y)")
op2d = Diffusion2D(0.1)

problem2d = PDEProblem(domain=dom2d, operators=[op2d], ic=ic2d)
sol2d = problem2d.solve((0.0, 0.5), t_eval=[0.0, 0.5])
U_final = dom2d.unflatten(sol2d.y[:, -1])
```

---

### Testing & CI

Run the full test suite from the `pde-mol` directory:

```bash
pytest
```

The tests cover:

- `Domain1D` / `Domain2D` creation, spacing, flatten/unflatten.
- `InitialCondition` evaluation for expressions, arrays, callables (1D/2D).
- Boundary conditions (Dirichlet, Periodic, Neumann with ghost enforcement).
- 1D/2D diffusion, advection, and expression operators.
- `PDEProblem` RHS and time integration for multiple analytic test PDEs.
- JSON loading and CLI-style problem execution.
- Sparse Laplacian builders vs FD operators.

You can wire this into CI (e.g. GitHub Actions) by running `pytest` in a job
after installing `requirements.txt`. Each module and function is documented
with docstrings so the codebase remains understandable and easy to extend.
