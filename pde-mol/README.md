## pde-mol — JSON-driven Method-of-Lines PDE Solver

`pde-mol` is a small, modular framework for solving time-dependent PDEs using
the **Method of Lines (MOL)**, driven by **JSON configuration files**.

**📖 For detailed implementation and architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md)**

In MOL we:

- **discretise space** → obtain a semi-discrete ODE system,
- **integrate in time** → use standard ODE solvers (`scipy.integrate.solve_ivp`).

The design emphasises:

- **Modularity** – domain, grid, IC, BC, operators, and problem wrapper are
  independent components with clear interfaces.
- **Simplicity** – straightforward finite-difference defaults, no hidden magic.
- **Extensibility** – 1D structured grids, multiple operator types,
  expression-based operators, and sparse matrix back-ends.
- **Safety & performance** – restricted expression evaluation, SymPy-based
  precompilation, and optional sparse operators/Jacobians.

---

### Project layout

- `pde/`
  - `domain.py` – `Domain1D` (structured grids).
  - `ic.py` – `InitialCondition` (expression / callable / array ICs).
  - `bc.py` – Dirichlet, Neumann, Periodic, and Robin BCs + Neumann ghost helpers.
  - `operators.py` – 1D diffusion, advection, expression operators, sums.
  - `plotting.py` – 1D plotting helpers and time-series PNG generation.
  - `sparse_ops.py` – sparse 1D Laplacian builders (CSR matrices).
  - `problem.py` – `PDEProblem` MOL wrapper and RHS/solve logic.
  - `json_loader.py` – JSON → `PDEProblem` builder.
  - `run_problem.py` – CLI entry point for running JSON-defined problems.
- `examples/`
  - `heat1d.json` – 1D periodic heat equation.
  - `burgers1d.json` – 1D Burgers-like equation.
  - `moisture1d.json` – 1D moisture diffusion with Robin BCs.
  - `moisture1d_dirichlet.json` – 1D moisture diffusion with Dirichlet BCs.
  - `custom1d_pde.json` – Custom 1D PDE with advection-diffusion-reaction.
  - `robin_general_example.json` – Example of generalized Robin BCs.
- `tests/`
  - Unit tests for domains, IC, BC, operators (1D), Neumann BCs,
    `PDEProblem`, JSON loader, and sparse operators.

All public classes and functions include docstrings describing **what they do**,
their **inputs** (types/shapes) and **outputs**.

---

### Module reference (files in `pde/`)

Below is a brief description of each class/function in the `pde` package and
what it does. For detailed signatures and parameter types, see the docstrings.

#### `domain.py`

- **`Domain1D(x0, x1, nx, periodic=False)`**  
  Structured 1D domain on $[x_0, x_1]$ with `nx` grid points.  
  - **Inputs**: floats `x0, x1`; int `nx`; bool `periodic`.  
  - **Properties**: `x` (1D NumPy array of length `nx`), `dx` (float spacing).  
  - **Methods**: `to_dict() -> dict`, `from_dict(d: dict) -> Domain1D`.

#### `ic.py`

- **`InitialCondition`** (dataclass with `expr`, `values`, `func`)  
  Represents an initial condition as:
  - string expression in `x` with `np` (e.g. `"np.sin(x)"`), or  
  - array-like of values matching the grid, or  
  - callable `f(x)` returning an array.  
  - **Key methods**:  
    - `from_expression(expr: str) -> InitialCondition`  
    - `from_values(values: array_like) -> InitialCondition`  
    - `from_callable(func: callable) -> InitialCondition`  
    - `evaluate(domain: Domain1D) -> np.ndarray` – returns 1D NumPy array on the domain grid.

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
  Neumann conditions specifying normal derivative $u_x$ at left/right boundary.  
  - `derivative_value`: constant flux, or  
  - `expr`: string in `t` and `np`.  
  - `_evaluate_flux(t) -> float` returns $q(t)$.  
  - `apply_to_full` does not modify `u_full` directly (enforced via helper).

- **`RobinLeft(a=None, b=None, c=None, c_expr=None, h=None, u_env=None)` / `RobinRight(...)`**  
  General Robin boundary condition of the form $a \cdot u + b \cdot u_x = c(t)$ at left/right boundary.  
  - **General form**: `a`, `b` (coefficients), `c` (constant) or `c_expr` (time-dependent expression in `t` and `np`).  
  - **Backward compatibility**: `h` (heat/mass transfer coefficient) and `u_env` (environmental value)  
    converts to $h \cdot u + u_x = h \cdot u_{env}$ (equivalent to $u_x = -h \cdot (u - u_{env})$).  
  - Uses one-sided first-order finite differences to enforce the condition.  
  - `apply_to_full(u_full, t, domain)` modifies `u_full[0]` or `u_full[-1]` in-place.

- **`apply_neumann_ghosts(u_full, bc_left, bc_right, t, domain)`**  
  Adjusts boundary values of `u_full` to approximate Neumann fluxes using a
  one-sided first-order stencil:  
  - left: `u[0] = u[1] - dx * q_left(t)`  
  - right: `u[-1] = u[-2] + dx * q_right(t)`  
  - **Inputs**: solution array `u_full`, optional left/right BCs, time `t`, `Domain1D`.  
  - **Output**: modifies `u_full` in-place; returns `None`.

#### `spatial_discretization.py`

- **`DiscretizationScheme` (enum)**  
  Enumeration of available discretization schemes:
  - `CENTRAL`: Second-order central differences (default, good for diffusion)
  - `UPWIND_FIRST`: First-order upwind (stable for advection, less accurate)
  - `UPWIND_SECOND`: Second-order upwind (stable and more accurate for advection)
  - `BACKWARD`: First-order backward differences
  - `FORWARD`: First-order forward differences

- **`first_derivative(u, dx, domain, scheme, velocity=None) -> array`**  
  Compute first derivative using specified scheme.  
  - For upwind schemes, `velocity` array is required to determine upwind direction.
  - Returns derivative approximation with same shape as `u`.

- **`second_derivative(u, dx, domain, scheme) -> array`**  
  Compute second derivative (currently only `CENTRAL` is supported).

- **`third_derivative(u, dx, domain, scheme) -> array`**  
  Compute third derivative (currently only `CENTRAL` is supported).

**Key Feature**: Different discretization schemes can be applied to different terms in the same PDE for optimal accuracy and stability. For example, use upwind schemes for advection terms and central differences for diffusion terms.

#### `operators.py`

- **`Operator` (base class)**  
  - Method: `apply(u_full: array, domain, t: float) -> array` – spatial RHS contribution.

- **`Diffusion(nu, scheme="central")`**  
  1D diffusion operator: returns `nu(x,t) * u_xx`.  
  - `nu`: scalar, callable `nu(x,t)`, or string expression in `x,t`.  
  - `scheme`: discretization scheme (default: `"central"`).  
  - `apply(u_full, domain: Domain1D, t) -> array` of same shape as `u_full`.

- **`Advection(a, scheme="upwind_first")`**  
  1D advection operator: returns `-a(x,t) * u_x`.  
  - `a`: scalar, callable `a(x,t)`, or string expression in `x,t`.  
  - `scheme`: discretization scheme. Options:
    - `"upwind_first"` (default): First-order upwind, stable for advection
    - `"upwind_second"`: Second-order upwind, more accurate while maintaining stability
    - `"central"`: Second-order central (may cause oscillations for sharp gradients)
  - `apply(u_full, domain: Domain1D, t) -> array`.

- **`ExpressionOperator(expr_string, params=None, schemes=None)`**  
  1D expression operator in terms of `(u, ux, uxx, uxxx, x, t, params)`.  
  - `expr_string`: SymPy expression string (e.g. `"-u*ux + nu*uxx"`).  
  - `params`: dict of parameter names → values.  
  - `schemes`: dict mapping derivative names to schemes, e.g. `{"ux": "upwind_second", "uxx": "central"}`.  
    - Keys: `"ux"`, `"uxx"`, `"uxxx"`  
    - Values: scheme names (default: `"central"` for all)  
  - Internally: parses with `sympy.sympify`, compiles with `sympy.lambdify`.  
  - `apply(u_full, domain: Domain1D, t) -> array`.

- **`sum_operators(ops: Iterable[Operator]) -> Operator`**  
  Returns an `Operator` whose `apply` gives the sum of `apply` for all `ops`.

**Note**: `ExpressionOperator` now supports third-order spatial derivatives via the `uxxx` symbol and configurable discretization schemes per derivative type.

#### `sparse_ops.py`

- **`build_1d_laplacian(domain: Domain1D) -> csr_matrix`**  
  Builds a sparse CSR matrix representing the 1D Laplacian consistent with the
  `Diffusion` operator's stencil (including periodic endpoints).

#### `problem.py`

- **`PDEProblem(domain, operators, ic, bc_left=None, bc_right=None)`**  
  High-level MOL wrapper that assembles domain, operators, IC, and BCs.  
  - **Inputs**:  
    - `domain`: `Domain1D`,  
    - `operators`: sequence of `Operator` instances,  
    - `ic`: `InitialCondition`,  
    - `bc_left`, `bc_right`: optional `BoundaryCondition`.  
  - **Key methods**:  
    - `initial_full(t0=0.0) -> array` – full state.  
    - `initial_interior(t0=0.0) -> array` – interior (non-periodic).  
    - `rhs(t, y) -> array` – unified RHS for testing (dispatches periodic/non-periodic).  
    - `solve(t_span, t_eval=None, method="RK45", plot=False, plot_dir=None, **kwargs) -> OdeResult` – wraps `solve_ivp`.  
      - `plot`: if `True`, generates plots during/after solving (initial, final, time series, combined plots, heatmaps).  
      - `plot_dir`: directory name for saving plots (plots are saved to `plots/<plot_dir>/`).  
  - Internally uses `_rhs_periodic`, `_rhs_nonperiodic` and `_reconstruct_full_from_interior` for non-periodic Neumann/Dirichlet/Robin handling.

- **`SecondOrderPDEProblem(domain, spatial_operator, ic_u, ic_ut, u_t_operator=None, bc_left=None, bc_right=None)`**  
  Handles second-order time derivatives (u_tt) by converting to first-order system.  
  - **Inputs**:  
    - `domain`: `Domain1D`,  
    - `spatial_operator`: `Operator` L such that u_tt = L[u] + M[u_t],  
    - `ic_u`: `InitialCondition` for u(x, 0),  
    - `ic_ut`: `InitialCondition` for u_t(x, 0),  
    - `u_t_operator`: optional `Operator` M applied to u_t (for damping terms),  
    - `bc_left`, `bc_right`: optional `BoundaryCondition` (applied to u).  
  - **Key methods**:  
    - `initial_full(t0=0.0) -> array` – extended state [u, v] where v = u_t.  
    - `rhs(t, y) -> array` – returns [u_t, v_t] = [v, L[u] + M[v]].  
    - `solve(...) -> OdeResult` – same interface as `PDEProblem.solve()`.  
      - Result object has additional attributes: `result.u` (u component) and `result.v` (v = u_t component).  
  - **Usage**: For equations like `u_tt = c² * u_xx` (wave equation) or `u_tt = c² * u_xx - γ * u_t` (damped wave).

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

- `Domain1D`  
- `InitialCondition`  
- `BoundaryCondition`, `DirichletLeft`, `DirichletRight`, `Periodic`, `NeumannLeft`, `NeumannRight`, `RobinLeft`, `RobinRight`  
- `Operator`, `Diffusion`, `Advection`, `ExpressionOperator`, `sum_operators`  
- `PDEProblem`, `SecondOrderPDEProblem`  
- `DiscretizationScheme`, `first_derivative`, `second_derivative`, `third_derivative`

#### `plotting.py`

- **`plot_1d(x, u, title=None, savepath=None)`**  
  Creates a simple 1D line plot of `u(x)` with labeled axes and optional title;
  saves a PNG if `savepath` is provided.

- **`plot_1d_time_series(x, solutions, times, prefix="solution1d", out_dir=None)`**  
  Saves a sequence of 1D line plots (one per time) as PNGs named with the
  given prefix and time index. Uses `tqdm` for progress bars when available.

- **`plot_1d_combined(x, solutions, times, title=None, savepath=None, max_curves=None)`**  
  Plots multiple 1D solution curves at different times on a single figure with
  a legend. Useful for visualizing time evolution. If `max_curves` is provided,
  only a subset of evenly spaced times is plotted.

- **`plot_xt_heatmap(x, times, solutions, title=None, savepath=None)`**  
  Creates a 2D heatmap of $u(x,t)$ with `x` on the x-axis and `t` on the y-axis.
  Useful for visualizing the full spatiotemporal evolution of 1D solutions.
  `solutions` must be a 2D array of shape `(nx, nt)`.


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

#### `initial_condition`

For first-order problems (u_t = L[u]):

```json
"initial_condition": {
  "type": "expression",      // "expression" or "values"
  "expr": "np.sin(x)"
}
```

For second-order problems (u_tt = L[u] + M[u_t]), also include:

```json
"initial_condition_ut": {
  "type": "expression",
  "expr": "0.0"              // u_t(x, 0)
}
```

For `"values"`, `values` must be an array matching the grid.

#### `boundary_conditions` (1D)

```json
"boundary_conditions": {
  "left":  { "type": "dirichlet", "value": 0.0 },
  "right": { "type": "dirichlet", "expr": "0.0" }
}
```

- Types: `"dirichlet"`, `"neumann"`, `"periodic"`, `"robin"`.
- **Dirichlet**: `value` (constant) or `expr` (time-dependent expression in `t` and `np`).
- **Neumann**: `derivative_value` (constant flux) or `expr` (time-dependent flux expression).
- **Robin** (general form): `a`, `b` (coefficients), `c` (constant) or `c_expr` (time-dependent expression):
  ```json
  "left": {
    "type": "robin",
    "a": 2.0,
    "b": 3.0,
    "c": 5.0
  }
  ```
- **Robin** (backward-compatible form): `h` (heat/mass transfer coefficient) and `u_env` (environmental value):
  ```json
  "left": {
    "type": "robin",
    "h": 1.0,
    "u_env": 0.2
  }
  ```
  This is equivalent to $h \cdot u + u_x = h \cdot u_{env}$ (or $u_x = -h \cdot (u - u_{env})$).

#### `operators`

For first-order problems (u_t = L[u]):

```json
"operators": [
  { 
    "type": "diffusion", 
    "nu": 0.5,
    "scheme": "central"
  },
  { 
    "type": "advection", 
    "a": "1.0 + 0.5 * np.cos(x)",
    "scheme": "upwind_second"
  },
  {
    "type": "expression",
    "expr": "-u*ux + nu*uxx - alpha*uxxx",
    "params": { "nu": 0.1, "alpha": 0.01 },
    "schemes": {
      "ux": "upwind_second",
      "uxx": "central",
      "uxxx": "central"
    }
  }
]
```

For second-order problems (u_tt = L[u] + M[u_t]):

```json
"spatial_operators": [
  { "type": "diffusion", "nu": 1.0 }    // L[u] = c² * u_xx
],
"u_t_operators": [                       // Optional: M[u_t] for damping
  { "type": "advection", "a": 0.1 }     // M[u_t] = -γ * u_t
]
```

Supported operator entries:

- `"diffusion"`: `{"type": "diffusion", "nu": ..., "scheme": "central"}` (optional `scheme`, default: `"central"`)
- `"advection"`: `{"type": "advection", "a": ..., "scheme": "upwind_first"}` (optional `scheme`, default: `"upwind_first"`)
  - Scheme options: `"upwind_first"`, `"upwind_second"`, `"central"`, `"backward"`, `"forward"`
- `"expression"` / `"expression_operator"`:
  - `expr` in variables `u, ux, uxx, uxxx, x, t` plus optional `params`.
  - Optional `schemes` dict: `{"ux": "upwind_second", "uxx": "central", "uxxx": "central"}`
  - Example with third-order derivative: `"expr": "-u*ux - alpha*uxxx"` (KdV-type)

**Example: Mixed-order equation in JSON** $u_t + u_{tt} = u_x + u_{xx} + u + 5$:

```json
{
  "domain": {
    "type": "1d",
    "x0": 0.0,
    "x1": 10.0,
    "nx": 201,
    "periodic": false
  },
  "initial_condition": {
    "type": "expression",
    "expr": "np.sin(x)"
  },
  "initial_condition_ut": {
    "type": "expression",
    "expr": "0.0"
  },
  "spatial_operators": [
    {
      "type": "expression",
      "expr": "ux + uxx + u + 5",
      "params": {}
    }
  ],
  "u_t_operators": [
    {
      "type": "expression",
      "expr": "-u",
      "params": {}
    }
  ],
  "time": {
    "t0": 0.0,
    "t1": 1.0,
    "num_points": 100,
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-8
  }
}
```

**Note**: In `u_t_operators`, the symbol `u` in the expression represents `u_t` (the input to the operator), so `"-u"` gives `-u_t`.

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

#### `visualization` (optional)

Controls automatic plotting during/after solving:

```json
"visualization": {
  "enable": true,
  "type": "1d",
  "save_dir": "my_problem",
  "plot_initial": true,
  "plot_final": true,
  "plot_time_series": true
}
```

- `enable`: if `true`, enables plotting (equivalent to `plot=True` in `solve()`).
- `save_dir`: directory name for saving plots (plots saved to `plots/<save_dir>/`).
- `plot_initial`: if `true`, saves initial condition plot.
- `plot_final`: if `true`, saves final solution plot.
- `plot_time_series`: if `true`, saves time series plots (individual frames, plus combined plot and heatmap).

When enabled, automatically generates:
- Initial and final solution plots
- Time series plots (individual frames)
- Combined plot showing multiple time slices
- Heatmap of $u(x,t)$ with `x` on x-axis and `t` on y-axis

---

### CLI usage

You can run a JSON-defined problem using the small CLI:

```bash
cd pde-mol
python run_problem.py examples/heat1d.json
python run_problem.py examples/burgers1d.json
python run_problem.py examples/wave1d.json          # Wave equation (u_tt = c²*u_xx)
python run_problem.py examples/damped_wave1d.json   # Damped wave equation
python run_problem.py examples/kdv1d.json           # KdV equation (with u_xxx)
```

This will:

- Load the JSON,
- Build a `PDEProblem` via `json_loader.build_problem_from_dict`,
- Integrate in time using the `"time"` block, and
- Print simple diagnostics (final time and L2 norm).

Use `--no-output` to suppress printing and rely on the exit code only.

---

### Python API quickstart

**1D heat equation** $u_t = \nu u_{xx}$ on $[0, 2\pi]$:

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

**Wave equation** $u_{tt} = c^2 u_{xx}$ on $[0, 2\pi]$:

```python
from pde import SecondOrderPDEProblem

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic_u = InitialCondition.from_expression("np.sin(x)")      # u(x, 0)
ic_ut = InitialCondition.from_expression("0.0")            # u_t(x, 0)
spatial_op = Diffusion(1.0)  # c² * u_xx

wave_problem = SecondOrderPDEProblem(
    domain=dom,
    spatial_operator=spatial_op,
    ic_u=ic_u,
    ic_ut=ic_ut
)
sol = wave_problem.solve((0.0, 2.0), t_eval=np.linspace(0.0, 2.0, 100))
u_final = sol.u[:, -1]  # Extract u component
```

**KdV equation** $u_t = -u u_x - \alpha u_{xxx}$ (with third-order derivative):

```python
from pde.operators import ExpressionOperator

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic = InitialCondition.from_expression("np.sin(x)")
op = ExpressionOperator(
    expr_string="-u*ux - alpha*uxxx",
    params={"alpha": 0.01}
)

problem = PDEProblem(domain=dom, operators=[op], ic=ic)
sol = problem.solve((0.0, 0.5), t_eval=np.linspace(0.0, 0.5, 100))
u_final = sol.y[:, -1]
```

**Mixed-order equation** $u_t + u_{tt} = u_x + u_{xx} + u + 5$:

This equation contains both first-order and second-order time derivatives. Rearranging: $u_{tt} = -u_t + u_x + u_{xx} + u + 5$

```python
from pde import SecondOrderPDEProblem, ExpressionOperator

dom = Domain1D(0.0, 10.0, 201, periodic=False)
ic_u = InitialCondition.from_expression("np.sin(x)")
ic_ut = InitialCondition.from_expression("0.0")

# Spatial operator: u_x + u_xx + u + 5
spatial_op = ExpressionOperator(
    expr_string="ux + uxx + u + 5",
    params={}
)

# u_t operator: -u_t (when applied to u_t, 'u' represents u_t)
u_t_op = ExpressionOperator(
    expr_string="-u",
    params={}
)

problem = SecondOrderPDEProblem(
    domain=dom,
    spatial_operator=spatial_op,
    ic_u=ic_u,
    ic_ut=ic_ut,
    u_t_operator=u_t_op
)
sol = problem.solve((0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 100), method="BDF")
u_final = sol.u[:, -1]
```

**Advection-diffusion-reaction equation** $u_t = -a u_x + \nu u_{xx} + r u (1 - u)$:

```python
from pde.operators import Advection, Diffusion, ExpressionOperator, sum_operators

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic = InitialCondition.from_expression("np.sin(x)")

# Combine multiple operators with optimal schemes
advection = Advection(a=1.0, scheme="upwind_second")  # Upwind for stability
diffusion = Diffusion(nu=0.1, scheme="central")       # Central for accuracy
reaction = ExpressionOperator(
    expr_string="r*u*(1 - u)",
    params={"r": 0.5}
)

# Sum all operators
op = sum_operators([advection, diffusion, reaction])

problem = PDEProblem(domain=dom, operators=[op], ic=ic)
sol = problem.solve((0.0, 2.0), t_eval=np.linspace(0.0, 2.0, 100))
u_final = sol.y[:, -1]
```

**Using ExpressionOperator with mixed schemes** (upwind for advection, central for diffusion):

```python
from pde.operators import ExpressionOperator

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic = InitialCondition.from_expression("np.sin(x)")

# Single expression with different schemes for different terms
op = ExpressionOperator(
    expr_string="-a*u*ux + nu*uxx + r*u*(1 - u)",
    params={"a": 1.0, "nu": 0.1, "r": 0.5},
    schemes={
        "ux": "upwind_second",  # Upwind for advection term
        "uxx": "central"        # Central for diffusion term
    }
)

problem = PDEProblem(domain=dom, operators=[op], ic=ic)
sol = problem.solve((0.0, 2.0), t_eval=np.linspace(0.0, 2.0, 100))
u_final = sol.y[:, -1]
```

**Damped wave with source term** $u_{tt} = c^2 u_{xx} - \gamma u_t + f(x,t)$:

```python
from pde import SecondOrderPDEProblem, ExpressionOperator

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic_u = InitialCondition.from_expression("np.sin(x)")
ic_ut = InitialCondition.from_expression("0.0")

# Spatial operator: c²*u_xx + source term f(x,t) = sin(x)*exp(-t)
spatial_op = ExpressionOperator(
    expr_string="c2*uxx + sin(x)*exp(-t)",
    params={"c2": 1.0}
)

# Damping operator: -γ*u_t
u_t_op = ExpressionOperator(
    expr_string="-gamma*u",
    params={"gamma": 0.1}
)

problem = SecondOrderPDEProblem(
    domain=dom,
    spatial_operator=spatial_op,
    ic_u=ic_u,
    ic_ut=ic_ut,
    u_t_operator=u_t_op
)
sol = problem.solve((0.0, 2.0), t_eval=np.linspace(0.0, 2.0, 100), method="BDF")
u_final = sol.u[:, -1]
```

**Complex expression with multiple spatial derivatives** $u_t = u u_x + \nu u_{xx} - \alpha u_{xxx} + \beta u$:

```python
from pde.operators import ExpressionOperator

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic = InitialCondition.from_expression("np.sin(x)")

op = ExpressionOperator(
    expr_string="u*ux + nu*uxx - alpha*uxxx + beta*u",
    params={"nu": 0.1, "alpha": 0.01, "beta": 0.5}
)

problem = PDEProblem(domain=dom, operators=[op], ic=ic)
sol = problem.solve((0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 100))
u_final = sol.y[:, -1]
```

For more details on second-order time derivatives and third-order spatial derivatives, see [SECOND_ORDER_AND_THIRD_ORDER.md](SECOND_ORDER_AND_THIRD_ORDER.md).

---

### Spatial Discretization Schemes

The library supports different discretization schemes optimized for different types of PDE terms. This allows you to use the most appropriate scheme for each term, improving both accuracy and stability.

#### Available Schemes

- **`CENTRAL`**: Second-order central differences
  - Best for: Diffusion terms, smooth solutions
  - Accuracy: $O(\Delta x^2)$
  - May cause oscillations for advection-dominated problems with sharp gradients

- **`UPWIND_FIRST`**: First-order upwind scheme
  - Best for: Advection terms, stability-critical problems
  - Accuracy: $O(\Delta x)$
  - Very stable, prevents oscillations by using information from the upwind direction

- **`UPWIND_SECOND`**: Second-order upwind scheme (ENO-like)
  - Best for: Advection terms where both accuracy and stability are important
  - Accuracy: $O(\Delta x^2)$
  - More accurate than first-order upwind while maintaining stability

- **`BACKWARD`**: First-order backward differences
- **`FORWARD`**: First-order forward differences

#### Why Use Different Schemes?

Different PDE terms benefit from different discretization methods:

1. **Advection terms** ($u_x$): Use upwind schemes to respect the direction of information flow and prevent oscillations
2. **Diffusion terms** ($u_{xx}$): Use central differences for optimal accuracy
3. **Reaction terms**: No spatial discretization needed

#### Usage Examples

**Python API:**

```python
from pde import Advection, Diffusion, ExpressionOperator
from pde.spatial_discretization import DiscretizationScheme

# Advection with second-order upwind (recommended)
advection = Advection(a=1.0, scheme=DiscretizationScheme.UPWIND_SECOND)

# Or use string
advection = Advection(a=1.0, scheme="upwind_second")

# Expression operator with mixed schemes
op = ExpressionOperator(
    expr_string="-u*ux + nu*uxx",
    params={"nu": 0.1},
    schemes={
        "ux": "upwind_second",  # Upwind for advection
        "uxx": "central"        # Central for diffusion
    }
)
```

**JSON Configuration:**

```json
{
  "operators": [
    {
      "type": "advection",
      "a": 1.0,
      "scheme": "upwind_second"
    },
    {
      "type": "expression",
      "expr": "-u*ux + nu*uxx",
      "params": {"nu": 0.1},
      "schemes": {
        "ux": "upwind_second",
        "uxx": "central"
      }
    }
  ]
}
```

**Default Schemes:**
- `Advection`: `"upwind_first"` (stable default)
- `Diffusion`: `"central"` (optimal for diffusion)
- `ExpressionOperator`: `"central"` for all derivatives (can be overridden with `schemes` dict)

---

### Testing & CI

Run the full test suite from the `pde-mol` directory:

```bash
pytest
```

The tests cover:

- `Domain1D` creation, spacing.
- `InitialCondition` evaluation for expressions, arrays, callables.
- Boundary conditions (Dirichlet, Periodic, Neumann with ghost enforcement, Robin with general and backward-compatible forms).
- 1D diffusion, advection, and expression operators.
- `PDEProblem` RHS and time integration for multiple analytic test PDEs.
- JSON loading and CLI-style problem execution.
- Sparse Laplacian builders vs FD operators.
- Plotting utilities for 1D solutions, time series, combined plots, and heatmaps.

You can wire this into CI (e.g. GitHub Actions) by running `pytest` in a job
after installing `requirements.txt`. Each module and function is documented
with docstrings so the codebase remains understandable and easy to extend.
