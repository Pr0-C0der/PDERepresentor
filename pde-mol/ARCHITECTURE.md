# Architecture and Implementation Guide

This document provides a comprehensive explanation of the code flow, mathematical foundations, and implementation details of the `pde-mol` library. It covers how JSON configurations are parsed, how initial and boundary conditions are applied, how PDEs are solved, and how results are visualized.

> **Note:** This document uses LaTeX math notation. On GitHub, math is rendered using `$$...$$` for display equations and `$...$` for inline math. If viewing in a plain text editor, LaTeX syntax will be visible but the math will still be readable.

---

## Table of Contents

1. [Overview](#overview)
2. [JSON Parsing Flow](#json-parsing-flow)
3. [Domain and Grid Construction](#domain-and-grid-construction)
4. [Initial Condition Application](#initial-condition-application)
5. [Boundary Condition Application](#boundary-condition-application)
6. [Spatial Operators and Finite Differences](#spatial-operators-and-finite-differences)
7. [PDE Problem Solving](#pde-problem-solving)
8. [Plotting and Visualization](#plotting-and-visualization)
9. [Design Principles in Practice](#design-principles-in-practice)

---

## Overview

The `pde-mol` library implements the **Method of Lines (MOL)** for solving partial differential equations. The core idea is:

1. **Discretize space** using finite differences on a structured grid
2. **Convert the PDE to a system of ODEs** where each ODE represents the time evolution at one spatial point
3. **Solve the ODE system** using standard time integrators (e.g., `scipy.integrate.solve_ivp`)

### Mathematical Foundation

Given a PDE of the form:
$$\frac{\partial u}{\partial t} = L[u]$$
$$where $L[u]$ is a spatial operator (e.g., $$L[u] = \nu \frac{\partial^2 u}{\partial x^2}$$ for diffusion), we discretize space:
$$x_i = x_0 + i \cdot \Delta x, \quad i = 0, 1, \ldots, N-1$$
$$where $\Delta x = \frac{x_1 - x_0}{N-1}$$ is the grid spacing.

The solution becomes a vector:
$$\mathbf{u}(t) = [u(x_0, t), u(x_1, t), \ldots, u(x_{N-1}, t)]^T$$
The PDE becomes a system of ODEs:
$$\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}, t)$$ where $\mathbf{F}$$ applies the spatial operator $$L$$ at each grid point using finite differences.

---

## JSON Parsing Flow

### Entry Point: `load_from_json()`

The parsing process begins when a JSON file is loaded:

```python
# pde/json_loader.py
def load_from_json(path: str | Path):
    p = Path(path)
    with p.open("r", encoding="utf8") as f:
        config = json.load(f)
    return build_problem_from_dict(config)
```

**Flow:**
1. Read JSON file from disk
2. Parse into Python dictionary
3. Pass to `build_problem_from_dict()` for construction

### Core Parser: `build_problem_from_dict()`

This function orchestrates the parsing of all components:

```python
def build_problem_from_dict(config: JsonDict):
    # 1. Parse domain
    domain_cfg = config["domain"]
    domain = _parse_domain(domain_cfg)
    
    # 2. Parse boundary conditions
    bc_cfg = config.get("boundary_conditions", {})
    bc_left, bc_right = _parse_boundary_conditions(bc_cfg)
    
    # 3. Detect problem type (first-order vs second-order)
    if "initial_condition_ut" in config:
        # Second-order problem: u_tt = L[u] + M[u_t]
        # ... parse second-order components
    else:
        # First-order problem: u_t = L[u]
        # ... parse first-order components
```

**Key Design Decision:** The parser automatically detects second-order problems by checking for `"initial_condition_ut"` in the JSON. This maintains backward compatibility while supporting new features.

### Component Parsers

#### Domain Parser: `_parse_domain()`

```python
def _parse_domain(cfg: JsonDict):
    domain_type = cfg.get("type", "1d").lower()
    if domain_type == "1d":
        x0 = cfg["x0"]
        x1 = cfg["x1"]
        nx = cfg["nx"]
        periodic = bool(cfg.get("periodic", False))
        return Domain1D(x0=x0, x1=x1, nx=nx, periodic=periodic)
```

**Mathematical Result:** Creates a structured grid:
$$x_i = x_0 + i \cdot \frac{x_1 - x_0}{N-1}, \quad i = 0, 1, \ldots, N-1$$

#### Initial Condition Parser: `_parse_initial_condition()`

```python
def _parse_initial_condition(cfg: JsonDict) -> InitialCondition:
    ic_type = cfg.get("type", "expression")
    if ic_type == "expression":
        expr = cfg["expr"]
        return InitialCondition.from_expression(expr)
    elif ic_type == "values":
        values = cfg["values"]
        return InitialCondition.from_values(values)
```

**Supported Formats:**
- **Expression**: String like `"np.sin(x)"` evaluated on the grid
- **Values**: Direct array of values matching grid size
- **Callable**: Python function (programmatic API only)

#### Boundary Condition Parser: `_parse_bc_one()`

Handles all boundary condition types:

```python
def _parse_bc_one(side_cfg: Optional[JsonDict], side: str):
    bc_type = side_cfg.get("type", "").lower()
    
    if bc_type == "dirichlet":
        # Parse Dirichlet: u = value or u = expr(t)
    elif bc_type == "neumann":
        # Parse Neumann: u_x = value or u_x = expr(t)
    elif bc_type == "robin":
        # Parse Robin: a*u + b*u_x = c(t)
    elif bc_type == "periodic":
        # Parse Periodic: u(0) = u(L)
```

**Modularity:** Each BC type is a separate class, making it easy to add new types.

#### Operator Parser: `_parse_operator()`

```python
def _parse_operator(op_cfg: JsonDict) -> Operator:
    op_type = op_cfg["type"].lower()
    
    if op_type == "diffusion":
        return Diffusion(nu=op_cfg["nu"])
    elif op_type == "advection":
        return Advection(a=op_cfg["a"])
    elif op_type in ("expression", "expression_operator"):
        return ExpressionOperator(
            expr_string=op_cfg["expr"],
            params=op_cfg.get("params", {})
        )
```

**Extensibility:** New operator types can be added by extending this function and creating new `Operator` subclasses.

---

## Domain and Grid Construction

### Domain1D Class

The `Domain1D` class encapsulates the spatial discretization:

```python
@dataclass
class Domain1D:
    x0: float      # Left boundary
    x1: float      # Right boundary
    nx: int        # Number of grid points
    periodic: bool # Whether domain is periodic
```

**Grid Construction (in `__post_init__`):**

```python
self._x = np.linspace(self.x0, self.x1, self.nx)
self._dx = float(self._x[1] - self._x[0])
```

**Mathematical Details:**

For a uniform grid:
$$x_i = x_0 + i \cdot \Delta x, \quad \Delta x = \frac{x_1 - x_0}{N-1}$$

The grid spacing $\Delta x$ is constant, ensuring second-order accuracy for central differences.

**Periodic Domains:**

When `periodic=True`, the domain satisfies:
$$u(x_0, t) = u(x_1, t)$$
This is enforced in finite-difference stencils by wrapping indices (e.g., `u[-1]` wraps to `u[0]`).

---

## Initial Condition Application

### InitialCondition Class

The `InitialCondition` class supports three modes:

1. **Expression**: String evaluated on the grid
2. **Values**: Direct array
3. **Callable**: Python function

### Evaluation Process

```python
def evaluate(self, domain: Domain1D) -> np.ndarray:
    x = domain.x  # Grid points: [x_0, x_1, ..., x_{N-1}]
    
    if self.expr is not None:
        # Safe evaluation with minimal namespace
        env = {"np": np, "x": x}
        result = eval(self.expr, {"__builtins__": {}}, env)
        return np.asarray(result, dtype=float)
    
    if self.func is not None:
        result = self.func(x)
        return np.asarray(result, dtype=float)
    
    # Direct values (already validated for shape)
    return np.asarray(self.values, dtype=float)
```

### Mathematical Interpretation

Given an initial condition $u(x, 0) = u_0(x)$, we evaluate it at each grid point:
$$\mathbf{u}(0) = [u_0(x_0), u_0(x_1), \ldots, u_0(x_{N-1})]^T$$
**Example:** For $$u_0(x) = \sin(x)$$ on grid points $$x_i$$:
$$u_i(0) = \sin(x_i), \quad i = 0, 1, \ldots, N-1$$
### Safety: Expression Evaluation

The expression evaluator uses a **minimal namespace** to prevent code injection:

```python
env = {"np": np, "x": x}  # Only numpy and grid points
result = eval(self.expr, {"__builtins__": {}}, env)  # No builtins
```

This allows mathematical expressions like `"np.sin(x)"` while preventing dangerous code execution.

### Integration with Problem Classes

**First-Order Problems (`PDEProblem`):**

```python
def initial_full(self, t0: float = 0.0) -> Array:
    u0 = self.ic.evaluate(self.domain)
    # Optionally apply boundary conditions
    if not self.domain.periodic:
        if self.bc_left is not None:
            self.bc_left.apply_to_full(u0, t0, self.domain)
        if self.bc_right is not None:
            self.bc_right.apply_to_full(u0, t0, self.domain)
    return u0
```

**Second-Order Problems (`SecondOrderPDEProblem`):**

```python
def initial_full(self, t0: float = 0.0) -> Array:
    u0 = self.ic_u.evaluate(self.domain)      # u(x, 0)
    v0 = self.ic_ut.evaluate(self.domain)     # u_t(x, 0)
    
    # Apply BCs to u0 (not v0)
    # ... (same as first-order)
    
    # Return extended state [u, v] where v = u_t
    return np.concatenate([u0, v0])
```

**Mathematical Result:** For second-order problems, the initial state vector is doubled:
$$\mathbf{y}(0) = \begin{bmatrix} \mathbf{u}(0) \\ \mathbf{v}(0) \end{bmatrix}$$
where $$\mathbf{v}(0) = \frac{\partial \mathbf{u}}{\partial t}(0)$$

---

## Boundary Condition Application

Boundary conditions are applied at each time step to enforce constraints on the solution. The implementation follows a **modular design** where each BC type is a separate class.

### Dirichlet Boundary Conditions

**Mathematical Form:**
$$u(x_0, t) = g_0(t) \quad \text{(left)}, \quad u(x_1, t) = g_1(t) \quad \text{(right)}$$
**Implementation:**

```python
@dataclass
class DirichletLeft(BoundaryCondition):
    value: Optional[float] = None      # Constant: g_0(t) = constant
    expr: Optional[str] = None         # Time-dependent: g_0(t) = expr(t)
    
    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D):
        if self.value is not None:
            u_full[0] = float(self.value)
        else:
            env = {"np": np, "t": t}
            u_full[0] = float(eval(self.expr, {"__builtins__": {}}, env))
```

**Code Flow:**
1. Evaluate boundary value (constant or time-dependent expression)
2. Directly assign to boundary grid point: `u_full[0] = g_0(t)`

**Mathematical Result:**
$$u_0(t) = g_0(t), \quad u_{N-1}(t) = g_1(t)$$
### Neumann Boundary Conditions

**Mathematical Form:**
$$\frac{\partial u}{\partial x}(x_0, t) = q_0(t) \quad \text{(left)}, \quad \frac{\partial u}{\partial x}(x_1, t) = q_1(t) \quad \text{(right)}$$
**Implementation Strategy:**

Neumann conditions are enforced using **ghost points** or **one-sided differences**. The library uses a helper function `apply_neumann_ghosts()`:

```python
def apply_neumann_ghosts(u_full, bc_left, bc_right, t, domain):
    dx = domain.dx
    
    if isinstance(bc_left, NeumannLeft):
        q_left = bc_left._evaluate_flux(t)
        u_full[0] = u_full[1] - dx * q_left
    
    if isinstance(bc_right, NeumannRight):
        q_right = bc_right._evaluate_flux(t)
        u_full[-1] = u_full[-2] + dx * q_right
```

**Mathematical Derivation:**

Using a first-order forward difference at the left boundary:
$$\frac{u_1 - u_0}{\Delta x} \approx \frac{\partial u}{\partial x}(x_0, t) = q_0(t)$$
Rearranging:
$$u_0 = u_1 - \Delta x \cdot q_0(t)$$
Similarly, at the right boundary using a backward difference:
$$\frac{u_{N-1} - u_{N-2}}{\Delta x} \approx \frac{\partial u}{\partial x}(x_1, t) = q_1(t)$$
Rearranging:
$$u_{N-1} = u_{N-2} + \Delta x \cdot q_1(t)$$
**Accuracy:** This is first-order accurate. Higher-order approximations are possible but require more interior points.

### Robin Boundary Conditions

**Mathematical Form:**
$$a \cdot u(x_0, t) + b \cdot \frac{\partial u}{\partial x}(x_0, t) = c(t) \quad \text{(left)}$$
This is a **linear combination** of Dirichlet and Neumann conditions.

**Implementation:**

```python
@dataclass
class RobinLeft(BoundaryCondition):
    a: Optional[float] = None      # Coefficient of u
    b: Optional[float] = None      # Coefficient of u_x (defaults to 1.0)
    c: Optional[float] = None      # Right-hand side (constant)
    c_expr: Optional[str] = None   # Right-hand side (time-dependent)
    
    def apply_to_full(self, u_full, t, domain):
        dx = domain.dx
        u1 = float(u_full[1])
        c_val = self._evaluate_c(t)
        
        # From: a * u[0] + b * (u[1] - u[0]) / dx = c
        # Rearranging: (a - b/dx) * u[0] = c - (b/dx) * u[1]
        b_over_dx = self.b / dx
        denom = self.a - b_over_dx
        u_full[0] = (c_val - b_over_dx * u1) / denom
```

**Mathematical Derivation:**

Starting from the Robin condition:
$$a \cdot u_0 + b \cdot \frac{\partial u}{\partial x}(x_0, t) = c(t)$$
Approximating the derivative with a forward difference:
$$\frac{\partial u}{\partial x}(x_0, t) \approx \frac{u_1 - u_0}{\Delta x}$$
Substituting:
$$a \cdot u_0 + b \cdot \frac{u_1 - u_0}{\Delta x} = c(t)$$
Rearranging:
$$\left(a - \frac{b}{\Delta x}\right) u_0 = c(t) - \frac{b}{\Delta x} u_1$$
Solving for $u_0$:
$$u_0 = \frac{c(t) - \frac{b}{\Delta x} u_1}{a - \frac{b}{\Delta x}}$$
**Special Cases:**
- **Dirichlet** ($b = 0$): $u_0 = \frac{c(t)}{a}$
- **Neumann** ($a = 0$): $u_0 = u_1 - \Delta x \cdot \frac{c(t)}{b}$

**Right Boundary:**

For the right boundary, we use a backward difference:
$$\frac{\partial u}{\partial x}(x_1, t) \approx \frac{u_{N-1} - u_{N-2}}{\Delta x}$$
The Robin condition becomes:
$$a \cdot u_{N-1} + b \cdot \frac{u_{N-1} - u_{N-2}}{\Delta x} = c(t)$$
Rearranging:
$$\left(a + \frac{b}{\Delta x}\right) u_{N-1} = c(t) + \frac{b}{\Delta x} u_{N-2}$$
Solving:
$$u_{N-1} = \frac{c(t) + \frac{b}{\Delta x} u_{N-2}}{a + \frac{b}{\Delta x}}$$
### Periodic Boundary Conditions

**Mathematical Form:**
$$u(x_0, t) = u(x_1, t), \quad \frac{\partial u}{\partial x}(x_0, t) = \frac{\partial u}{\partial x}(x_1, t)$$
**Implementation:**

```python
class Periodic(BoundaryCondition):
    def apply_to_full(self, u_full, t, domain):
        # No-op: periodicity is handled in finite-difference stencils
        return None
```

**Enforcement:** Periodicity is enforced **in the finite-difference stencils**, not by direct assignment. For example, when computing derivatives:

```python
# In _central_first_derivative for periodic domains:
du[0] = (u[1] - u[-2]) / (2.0 * dx)  # Wrap to penultimate point
du[-1] = du[0]  # Same as left boundary
```

**Mathematical Result:** The grid points satisfy $u_0 = u_{N-1}$ implicitly through the stencil construction.

### Boundary Condition Application Order

For non-periodic domains, boundary conditions are applied in this order:

1. **Dirichlet/Robin**: Applied first to set boundary values
2. **Neumann**: Applied via `apply_neumann_ghosts()` after Dirichlet/Robin

This ensures Neumann conditions can override Dirichlet values if both are specified (though typically only one type is used per boundary).

---

## Spatial Operators and Finite Differences

Spatial operators compute derivatives using **finite difference methods**. The library implements second-order accurate stencils.

### First-Order Derivative: `_central_first_derivative()`

**Mathematical Formula (Central Difference):**
$$\frac{\partial u}{\partial x}(x_i) \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}$$
**Accuracy:** $$O(\Delta x^2)$$ (second-order)

**Implementation:**

```python
def _central_first_derivative(u: Array, dx: float, periodic: bool) -> Array:
    if periodic:
        # Interior points
        du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        # Periodic boundaries: wrap indices
        du[0] = (u[1] - u[-2]) / (2.0 * dx)
        du[-1] = du[0]
    else:
        # Use np.gradient for non-periodic (handles boundaries automatically)
        return np.gradient(u, dx, edge_order=2)
```

**Non-Periodic Boundaries:** `np.gradient` with `edge_order=2` uses:
- **Interior**: Central difference (second-order)
- **Boundaries**: One-sided differences (second-order)

### Second-Order Derivative: `_central_second_derivative()`

**Mathematical Formula (Central Difference):**
$$\frac{\partial^2 u}{\partial x^2}(x_i) \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$
**Accuracy:** $O(\Delta x^2)$ (second-order)

**Implementation:**

```python
def _central_second_derivative(u: Array, dx: float, periodic: bool) -> Array:
    if periodic:
        # Interior
        d2[1:-1] = (u[2:] - 2.0*u[1:-1] + u[:-2]) / (dx * dx)
        # Periodic boundaries
        d2[0] = (u[1] - 2.0*u[0] + u[-2]) / (dx * dx)
        d2[-1] = d2[0]
    else:
        # Apply gradient twice
        return np.gradient(np.gradient(u, dx, edge_order=2), dx, edge_order=2)
```

**Derivation:** The second derivative is the derivative of the first derivative:
$$\frac{\partial^2 u}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial u}{\partial x}\right)$$
### Third-Order Derivative: `_central_third_derivative()`

**Mathematical Formula (Central Difference):**
$$\frac{\partial^3 u}{\partial x^3}(x_i) \approx \frac{u_{i+2} - 2u_{i+1} + 2u_{i-1} - u_{i-2}}{2\Delta x^3}$$
**Accuracy:** $O(\Delta x^2)$ (second-order)

**Implementation:**

```python
def _central_third_derivative(u: Array, dx: float, periodic: bool) -> Array:
    if periodic:
        # Interior: central stencil
        d3[2:-2] = (u[4:] - 2.0*u[3:-1] + 2.0*u[1:-3] - u[:-4]) / (2.0 * dx**3)
        # Periodic boundaries: wrap indices
        d3[0] = (u[2] - 2.0*u[1] + 2.0*u[-2] - u[-3]) / (2.0 * dx**3)
        # ... (similar for other boundary points)
    else:
        # Non-periodic: one-sided differences at boundaries
        # Forward difference at left boundary
        d3[0] = (-u[3] + 3.0*u[2] - 3.0*u[1] + u[0]) / (dx**3)
        # Backward difference at right boundary
        d3[-1] = (u[-1] - 3.0*u[-2] + 3.0*u[-3] - u[-4]) / (dx**3)
```

**Derivation:** The third derivative stencil is derived using Taylor series expansions to cancel lower-order terms.

### Operator Classes

#### Diffusion Operator

**Mathematical Form:**
$$L[u] = \nu \frac{\partial^2 u}{\partial x^2}$$
**Implementation:**

```python
@dataclass
class Diffusion(Operator):
    nu: float | str | Callable  # Diffusivity coefficient
    
    def apply(self, u_full, domain, t):
        u_xx = _central_second_derivative(u_full, domain.dx, domain.periodic)
        nu_vals = _eval_coeff(self.nu, domain.x, t)  # Can be space/time-dependent
        return nu_vals * u_xx
```

#### Advection Operator

**Mathematical Form:**
$$L[u] = -a \frac{\partial u}{\partial x}$$
**Implementation:**

```python
@dataclass
class Advection(Operator):
    a: float | str | Callable  # Advection speed
    
    def apply(self, u_full, domain, t):
        u_x = _central_first_derivative(u_full, domain.dx, domain.periodic)
        a_vals = _eval_coeff(self.a, domain.x, t)
        return -a_vals * u_x
```

#### ExpressionOperator

**Mathematical Form:**
$$L[u] = f(u, u_x, u_{xx}, u_{xxx}, x, t, \mathbf{p})$$
where $$\mathbf{p}$$ are parameters.

**Implementation:**

```python
@dataclass
class ExpressionOperator(Operator):
    expr_string: str
    params: Optional[dict] = None
    
    def __post_init__(self):
        # Parse expression using SymPy
        u, ux, uxx, uxxx, x, t = sp.symbols("u ux uxx uxxx x t")
        self._base_symbols = {"u": u, "ux": ux, "uxx": uxx, "uxxx": uxxx, "x": x, "t": t}
        
        # Compile to fast NumPy function
        self._func = sp.lambdify(args, self._expr, modules=["numpy"])
    
    def apply(self, u_full, domain, t):
        # Compute derivatives
        u = u_full
        ux = _central_first_derivative(u, domain.dx, domain.periodic)
        uxx = _central_second_derivative(u, domain.dx, domain.periodic)
        uxxx = _central_third_derivative(u, domain.dx, domain.periodic)
        
        # Evaluate expression
        result = self._func(u, ux, uxx, uxxx, domain.x, t, *param_vals)
        return result
```

**Safety:** SymPy parsing prevents code injection, and `lambdify` compiles to fast NumPy code.

---

## PDE Problem Solving

The library supports two types of PDE problems:

1. **First-order in time**: $u_t = L[u]$
2. **Second-order in time**: $u_{tt} = L[u] + M[u_t]$

### First-Order Problems: `PDEProblem`

**Mathematical Form:**
$$\frac{\partial u}{\partial t} = L[u]$$
After spatial discretization:
$$\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}, t)$$ where $\mathbf{F}$ applies the spatial operator $L$ at each grid point.

**State Vector:**

For **periodic** domains:
$$\mathbf{u} = [u_0, u_1, \ldots, u_{N-1}]^T$$
For **non-periodic** domains (interior only):
$$\mathbf{u} = [u_1, u_2, \ldots, u_{N-2}]^T$$
Boundary values are reconstructed when needed.

**Right-Hand Side Function:**

```python
def _rhs_periodic(self, t: float, u_full: Array) -> Array:
    # Apply spatial operator directly to full state
    return self._op.apply(u_full, self.domain, t)

def _rhs_nonperiodic(self, t: float, interior: Array) -> Array:
    # Reconstruct full state
    u_full = self._reconstruct_full_from_interior(interior, t)
    # Apply operator
    du_full = self._op.apply(u_full, self.domain, t)
    # Return interior part only
    return du_full[1:-1]
```

**Reconstruction Process (Non-Periodic):**

```python
def _reconstruct_full_from_interior(self, interior: Array, t: float) -> Array:
    u_full = np.zeros(nx, dtype=float)
    u_full[1:-1] = interior  # Copy interior values
    
    # Apply boundary conditions
    if self.bc_left is not None:
        self.bc_left.apply_to_full(u_full, t, self.domain)
    if self.bc_right is not None:
        self.bc_right.apply_to_full(u_full, t, self.domain)
    
    # Adjust for Neumann BCs (ghost point method)
    apply_neumann_ghosts(u_full, self.bc_left, self.bc_right, t, self.domain)
    
    return u_full
```

**Time Integration:**

```python
def solve(self, t_span, t_eval=None, method="RK45", **kwargs):
    y0 = self.initial_full(t_span[0])  # Initial state
    fun = self._rhs_periodic if self.domain.periodic else self._rhs_nonperiodic
    
    result = solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        **kwargs
    )
    return result
```

**Mathematical Result:** `solve_ivp` integrates the ODE system:
$$\mathbf{u}(t) = \mathbf{u}(0) + \int_0^t \mathbf{F}(\mathbf{u}(\tau), \tau) \, d\tau$$
### Second-Order Problems: `SecondOrderPDEProblem`

**Mathematical Form:**
$$\frac{\partial^2 u}{\partial t^2} = L[u] + M\left[\frac{\partial u}{\partial t}\right]$$
**Conversion to First-Order System:**

Introduce $v = \frac{\partial u}{\partial t}$:
$$\frac{\partial u}{\partial t} = v$$
$$\frac{\partial v}{\partial t} = L[u] + M[v]$$

**State Vector:**

$$\mathbf{y} = \begin{bmatrix} \mathbf{u} \\ \mathbf{v} \end{bmatrix}$$
For periodic domains:
$$\mathbf{y} = [u_0, u_1, \ldots, u_{N-1}, v_0, v_1, \ldots, v_{N-1}]^T$$
**Right-Hand Side Function:**

```python
def _rhs_periodic(self, t: float, y: Array) -> Array:
    nx = self.domain.nx
    u = y[:nx]      # Extract u component
    v = y[nx:]      # Extract v = u_t component
    
    # Apply spatial operator to u
    L_u = self.spatial_operator.apply(u, self.domain, t)
    
    # Apply operator to v (u_t) if provided
    if self.u_t_operator is not None:
        M_v = self.u_t_operator.apply(v, self.domain, t)
    else:
        M_v = np.zeros_like(v)
    
    # Return [u_t, v_t] = [v, L[u] + M[v]]
    return np.concatenate([v, L_u + M_v])
```

**Mathematical Result:**

$$\frac{d\mathbf{y}}{dt} = \begin{bmatrix} \mathbf{v} \\ L[\mathbf{u}] + M[\mathbf{v}] \end{bmatrix}$$
**Initial Conditions:**

```python
def initial_full(self, t0: float = 0.0) -> Array:
    u0 = self.ic_u.evaluate(self.domain)   # u(x, 0)
    v0 = self.ic_ut.evaluate(self.domain)  # u_t(x, 0)
    return np.concatenate([u0, v0])
```

**Mathematical Result:**
$$\mathbf{y}(0) = \begin{bmatrix} \mathbf{u}(0) \\ \mathbf{v}(0) \end{bmatrix} = \begin{bmatrix} \mathbf{u}_0 \\ \mathbf{v}_0 \end{bmatrix}$$
### Higher-Order Spatial Derivatives

The library supports up to **third-order spatial derivatives** ($u_{xxx}$) through the `ExpressionOperator`.

**Usage Example:**

For the KdV equation: $u_t = -u u_x - \alpha u_{xxx}$

```python
op = ExpressionOperator(
    expr_string="-u*ux - alpha*uxxx",
    params={"alpha": 0.01}
)
```

**Mathematical Implementation:**

The third derivative is computed using the central difference stencil:
$$\frac{\partial^3 u}{\partial x^3}(x_i) \approx \frac{u_{i+2} - 2u_{i+1} + 2u_{i-1} - u_{i-2}}{2\Delta x^3}$$
This is second-order accurate: $O(\Delta x^2)$.

**Extensibility:** To add fourth-order or higher derivatives, extend `_central_third_derivative()` with additional stencils and update `ExpressionOperator` to include new symbols.

### Time Integration Methods

The library uses `scipy.integrate.solve_ivp`, which supports multiple methods:

- **RK45**: Runge-Kutta 4(5) - explicit, adaptive, good for non-stiff problems
- **RK23**: Runge-Kutta 2(3) - explicit, adaptive, lower order
- **BDF**: Backward Differentiation Formula - implicit, adaptive, good for stiff problems
- **Radau**: Implicit Runge-Kutta - very stable for stiff problems

**Recommendations:**
- **Non-stiff problems**: Use `RK45` (default)
- **Stiff problems** (e.g., with damping): Use `BDF`
- **High accuracy**: Use `Radau` with tight tolerances

**Adaptive Time Stepping:**

The solver automatically adjusts the time step to maintain error below:
$$\text{error} < \text{rtol} \cdot |\mathbf{u}| + \text{atol}$$
where `rtol` (relative tolerance) and `atol` (absolute tolerance) are user-specified.

---

## Plotting and Visualization

The plotting module provides utilities for visualizing 1D PDE solutions. It uses matplotlib with the Agg backend (suitable for headless environments).

### Plot Types

#### 1. Single Snapshot: `plot_1d()`

**Purpose:** Plot $u(x)$ at a single time.

**Implementation:**

```python
def plot_1d(x: np.ndarray, u: np.ndarray, title: Optional[str] = None, savepath: Optional[str] = None):
    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Mathematical Representation:** Plots the function $u(x, t_k)$ for a fixed time $t_k$.

#### 2. Time Series: `plot_1d_time_series()`

**Purpose:** Generate multiple plots showing $u(x, t)$ at different times.

**Implementation:**

```python
def plot_1d_time_series(x, solutions, times, prefix="solution1d", out_dir=None):
    # solutions: shape (nx, nt) - each column is u(x, t_k)
    # times: sequence of times t_k
    
    paths = []
    for k, t in enumerate(times):
        u_k = solutions[:, k]
        filename = f"{prefix}_t{k:04d}.png"
        path = Path(out_dir) / filename
        plot_1d(x, u_k, title=f"t = {t:.3f}", savepath=path)
        paths.append(path)
    return paths
```

**Mathematical Representation:** Generates a sequence of plots:
$$\{u(x, t_0), u(x, t_1), \ldots, u(x, t_{N_t-1})\}$$
#### 3. Combined Plot: `plot_1d_combined()`

**Purpose:** Show multiple time slices on a single plot.

**Implementation:**

```python
def plot_1d_combined(x, solutions, times, title="Combined 1D time series", savepath=None, max_curves=8):
    fig, ax = plt.subplots()
    
    # Select subset of times to plot
    n_plot = min(max_curves, len(times))
    indices = np.linspace(0, len(times) - 1, n_plot, dtype=int)
    
    for idx in indices:
        t = times[idx]
        u = solutions[:, idx]
        ax.plot(x, u, label=f"t={t:.3g}")
    
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Mathematical Representation:** Overlays multiple time slices:
$$\{u(x, t_{i_0}), u(x, t_{i_1}), \ldots, u(x, t_{i_{n-1}})\}$$
on a single axes.

#### 4. Heatmap: `plot_xt_heatmap()`

**Purpose:** Visualize $u(x, t)$ as a 2D color map.

**Implementation:**

```python
def plot_xt_heatmap(x, t, solutions, title="u(x,t) heatmap", savepath=None):
    # solutions: shape (nx, nt)
    fig, ax = plt.subplots()
    
    # Create meshgrid for pcolormesh
    T, X = np.meshgrid(t, x)
    
    # Plot heatmap
    im = ax.pcolormesh(T, X, solutions, shading='auto', cmap='viridis')
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="u(x, t)")
    
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Mathematical Representation:** Creates a 2D visualization of the function $u(x, t)$ where:
- **x-axis**: Time $t$
- **y-axis**: Space $x$
- **Color**: Value $u(x, t)$

### Integration with Problem Classes

**Automatic Plotting in `solve()`:**

```python
def solve(self, t_span, t_eval=None, method="RK45", plot=False, plot_dir=None, **kwargs):
    # ... solve PDE ...
    
    if plot:
        base_dir = plot_dir or "test_plots"
        os.makedirs(base_dir, exist_ok=True)
        
        # Reconstruct full solutions for plotting
        if self.domain.periodic:
            full_solutions = result.y
        else:
            full_states = []
            for k, t in enumerate(result.t):
                interior_k = result.y[:, k]
                full_k = self._reconstruct_full_from_interior(interior_k, t)
                full_states.append(full_k)
            full_solutions = np.stack(full_states, axis=1)
        
        # Generate plots
        plot_1d(x, full_solutions[:, 0], title="Initial solution", savepath=...)
        plot_1d(x, full_solutions[:, -1], title="Final solution", savepath=...)
        plot_1d_time_series(x, full_solutions, result.t, prefix="solution1d", out_dir=base_dir)
```

**For Second-Order Problems:**

The plotting extracts the `u` component (first half of state vector):

```python
# In SecondOrderPDEProblem.solve()
result.u = result.y[:nx, :]  # Extract u component
result.v = result.y[nx:, :]  # Extract v = u_t component

# Plotting uses result.u
plot_1d(x, result.u[:, 0], title="Initial solution u(x,0)", savepath=...)
```

---

## Design Principles in Practice

### 1. Modularity

**Domain, Grid, BC, IC, Operators, Time Integrator, Problem wrapper, JSON loader are all independent components.**

**Evidence:**
- Each component is in its own module/class
- `Domain1D` can be used independently
- `InitialCondition` can be evaluated without a problem
- `BoundaryCondition` classes are independent
- Operators can be tested in isolation

**Example:**

```python
# Test domain independently
domain = Domain1D(0, 1, 101, periodic=False)
assert domain.dx == 0.01

# Test initial condition independently
ic = InitialCondition.from_expression("np.sin(x)")
u0 = ic.evaluate(domain)
assert len(u0) == 101

# Test operator independently
op = Diffusion(0.1)
du_dt = op.apply(u0, domain, 0.0)
assert len(du_dt) == 101
```

### 2. Simplicity and Understandability

**Clear, small functions. Straightforward finite-difference defaults. No hidden magic.**

**Evidence:**
- Finite difference stencils are explicit and well-documented
- Boundary condition application is transparent
- No complex abstractions or hidden state

**Example:**

```python
# Clear, explicit finite difference
def _central_first_derivative(u, dx, periodic):
    if periodic:
        du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)  # Obvious central difference
    else:
        return np.gradient(u, dx, edge_order=2)  # Standard NumPy function
```

### 3. Extensibility

**Start with 1D, then extend to 2D. Support structured and nonuniform grids initially; later support mapped and unstructured FEM domains. Add multiple operator types, including custom expression operators.**

**Evidence:**
- `ExpressionOperator` allows custom operators without code changes
- New boundary condition types can be added by subclassing `BoundaryCondition`
- New operator types can be added by subclassing `Operator`
- Architecture supports extension to 2D (though currently only 1D is implemented)

**Example:**

```python
# Easy to add new operator type
@dataclass
class ReactionOperator(Operator):
    rate: float
    
    def apply(self, u_full, domain, t):
        return self.rate * u_full * (1 - u_full)  # Logistic growth
```

### 4. Safety and Performance

**Avoid raw eval. Precompile expressions with sympy (or safe AST) for speed. Allow optional sparse matrices and Jacobians for stiff PDEs. Add well-defined tests.**

**Evidence:**
- Expression evaluation uses minimal namespace (no `__builtins__`)
- SymPy compilation (`lambdify`) provides fast execution
- Sparse matrix support exists (`sparse_ops.py`) for future use
- Comprehensive test suite

**Example:**

```python
# Safe expression evaluation
env = {"np": np, "x": x}  # Only numpy and grid
result = eval(self.expr, {"__builtins__": {}}, env)  # No dangerous builtins

# Fast compiled expressions
self._func = sp.lambdify(args, self._expr, modules=["numpy"])  # Compiled to NumPy
result = self._func(u, ux, uxx, uxxx, x, t, *param_vals)  # Fast execution
```

---

## Summary

The `pde-mol` library implements a **modular, extensible, and safe** framework for solving PDEs using the Method of Lines. Key features:

1. **JSON-driven configuration** for easy problem specification
2. **Modular components** (domain, IC, BC, operators) that can be tested independently
3. **Clear finite-difference implementations** with second-order accuracy
4. **Support for first- and second-order time derivatives**
5. **Support for up to third-order spatial derivatives**
6. **Comprehensive boundary condition support** (Dirichlet, Neumann, Robin, Periodic)
7. **Safe expression evaluation** using SymPy
8. **Flexible plotting utilities** for visualization

The architecture follows the stated design principles, making it easy to understand, extend, and maintain.

