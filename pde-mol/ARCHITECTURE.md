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
   - [Discretization Schemes](#discretization-schemes)
7. [PDE Problem Solving](#pde-problem-solving)
8. [Plotting and Visualization](#plotting-and-visualization)
9. [Dataset Generation](#dataset-generation)
10. [Command-Line Interface](#command-line-interface)
11. [Design Principles in Practice](#design-principles-in-practice)
12. [Summary](#summary)

---

## Overview

The `pde-mol` library implements the **Method of Lines (MOL)** for solving partial differential equations. The core idea is:

1. **Discretize space** using finite differences on a structured grid
2. **Convert the PDE to a system of ODEs** where each ODE represents the time evolution at one spatial point
3. **Solve the ODE system** using standard time integrators (e.g., `scipy.integrate.solve_ivp`)

### Mathematical Foundation

Given a PDE of the form:
$$\frac{\partial u}{\partial t} = L[u]$$
where $L[u]$ is a spatial operator (e.g., $L[u] = \nu \frac{\partial^2 u}{\partial x^2}$ for diffusion), we discretize space:
$$x_i = x_0 + i \cdot \Delta x, \quad i = 0, 1, \ldots, N-1$$
where $\Delta x = \frac{x_1 - x_0}{N-1}$ is the grid spacing.

The solution becomes a vector:
$$\mathbf{u}(t) = [u(x_0, t), u(x_1, t), \ldots, u(x_{N-1}, t)]^T$$
The PDE becomes a system of ODEs:
$$\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}, t)$$
where $\mathbf{F}$ applies the spatial operator $L$ at each grid point using finite differences.

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

**Complete Flow Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          JSON Configuration File                         │
│  {                                                                       │
│    "domain": {...},                                                      │
│    "initial_condition": {...},                                           │
│    "boundary_conditions": {...},                                         │
│    "operators": [...]                                                    │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │ load_from_json()    │
                    │ - Read file         │
                    │ - UTF-8 encoding    │
                    └─────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │ json.load()         │
                    │ → Python dict       │
                    └─────────────────────┘
                              ↓
            ┌─────────────────────────────────────┐
            │ build_problem_from_dict(config)     │
            └─────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ↓                                           ↓
┌───────────────────────┐              ┌──────────────────────────┐
│ _parse_domain()       │              │ _parse_boundary_         │
│                       │              │   conditions()           │
│ Input:                │              │                          │
│   cfg["domain"]       │              │ Input:                   │
│   - type: "1d"        │              │   cfg.get("boundary_     │
│   - x0, x1, nx        │              │     conditions", {})     │
│   - periodic: bool    │              │                          │
│                       │              │ Process:                 │
│ Process:              │              │   ├─→ Extract "left"     │
│   ├─→ Extract fields  │              │   ├─→ Extract "right"    │
│   ├─→ Validate type   │              │   └─→ For each side:     │
│   └─→ Create Domain1D │              │       _parse_bc_one()    │
│                       │              │                          │
│ Output:               │              │ _parse_bc_one() details: │
│   Domain1D(           │              │   ├─→ Extract "type"     │
│     x0, x1, nx,       │              │   ├─→ Dispatch by type:  │
│     periodic          │              │   │   - "dirichlet"      │
│   )                   │              │   │     → DirichletLeft/ │
│                       │              │   │       Right(value/   │
│                       │              │   │        expr)         │
│                       │              │   │   - "neumann"        │
│                       │              │   │     → NeumannLeft/   │
│                       │              │   │       Right(deriv_   │
│                       │              │   │        value/expr)   │
│                       │              │   │   - "robin"          │
│                       │              │   │     → RobinLeft/     │
│                       │              │   │       Right(a,b,c/   │
│                       │              │   │        h,u_env)      │
│                       │              │   │   - "periodic"       │
│                       │              │   │     → Periodic()     │
│                       │              │   └─→ Validate fields    │
│                       │              │                          │
│                       │              │ Output:                  │
│                       │              │   (bc_left, bc_right)    │
│                       │              │   or (None, None)        │
└───────────────────────┘              └──────────────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ↓
                    ┌─────────────────────────┐
                    │ Problem Type Detection  │
                    │                         │
                    │ Check:                  │
                    │   "initial_condition_   │
                    │    ut" in config?       │
                    └─────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ↓                                           ↓
┌───────────────────────────────┐    ┌──────────────────────────────┐
│ SECOND-ORDER PROBLEM          │    │ FIRST-ORDER PROBLEM          │
│ u_tt = L[u] + M[u_t]          │    │ u_t = L[u]                   │
│                               │    │                              │
│ ┌─────────────────────────┐   │    │ ┌─────────────────────────┐ │
│ │ Parse Initial Conditions│   │    │ │ Parse Initial Condition │ │
│ │                         │   │    │ │                         │ │
│ │ ic_u_cfg = config[      │   │    │ │ ic_cfg = config[        │ │
│ │   "initial_condition"   │   │    │ │   "initial_condition"   │ │
│ │ ]                       │   │    │ │ ]                       │ │
│ │                         │   │    │ │                         │ │
│ │ _parse_initial_         │   │    │ │ _parse_initial_         │ │
│ │   condition(ic_u_cfg)   │   │    │ │   condition(ic_cfg)     │ │
│ │   ├─→ Extract "type"    │   │    │ │   ├─→ Extract "type"    │ │
│ │   ├─→ Dispatch:         │   │    │ │   ├─→ Dispatch:         │ │
│ │   │   - "expression"    │   │    │ │   │   - "expression"    │ │
│ │   │     → IC.from_      │   │    │ │   │     → IC.from_      │ │
│ │   │        expression() │   │    │ │   │        expression() │ │
│ │   │   - "values"        │   │    │ │   │   - "values"        │ │
│ │   │     → IC.from_      │   │    │ │   │     → IC.from_      │ │
│ │   │        values()     │   │    │ │   │        values()     │ │
│ │   └─→ Return IC         │   │    │ │   └─→ Return IC         │ │
│ │                         │   │    │ │                         │ │
│ │ Output: ic_u            │   │    │ │ Output: ic              │ │
│ └─────────────────────────┘   │    │ └─────────────────────────┘ │
│         │                     │    │         │                   │
│         ↓                     │    │         ↓                   │
│ ┌─────────────────────────┐   │    │ ┌─────────────────────────┐ │
│ │ Parse u_t Initial Cond. │   │    │ │ Parse Operators         │ │
│ │                         │   │    │ │                         │ │
│ │ ic_ut_cfg = config[     │   │    │ │ ops_cfg = config[       │ │
│ │   "initial_condition_   │   │    │ │   "operators"           │ │
│ │    ut"                  │   │    │ │ ]                       │ │
│ │ ]                       │   │    │ │                         │ │
│ │                         │   │    │ │ _parse_operators()      │ │
│ │ _parse_initial_         │   │    │ │   └─→ For each op_cfg:  │ │
│ │   condition(ic_ut_cfg)  │   │    │ │       _parse_operator() │ │
│ │   (same as above)       │   │    │ │                         │ │
│ │                         │   │    │ │ _parse_operator()       │ │
│ │ Output: ic_ut           │   │    │ │   ├─→ Extract "type"    │ │
│ └─────────────────────────┘   │    │ │   ├─→ Dispatch:         │ │
│         │                     │    │ │   │   - "diffusion"     │ │
│         ↓                     │    │ │   │     → Diffusion(    │ │
│ ┌─────────────────────────┐   │    │ │   │        nu, scheme) │ │
│ │ Parse Spatial Operators │   │    │ │   │   - "advection"    │ │
│ │                         │   │    │ │   │     → Advection(   │ │
│ │ spatial_ops_cfg =       │   │    │ │   │        a, scheme)  │ │
│ │   config.get(           │   │    │ │   │   - "expression"   │ │
│ │     "spatial_operators",│   │    │ │   │     → Expression   │ │
│ │     config.get(         │   │    │ │   │        Operator(   │ │
│ │       "operators", []   │   │    │ │   │        expr,       │ │
│ │     )                   │   │    │ │   │        params,     │ │
│ │   )                     │   │    │ │   │        schemes)    │ │
│ │                         │   │    │ │   └─→ Validate fields  │ │
│ │ Validate:               │   │    │ │                         │ │
│ │   if not spatial_ops_   │   │    │ │ Output: List[Operator] │ │
│ │     cfg: raise Error    │   │    │ └─────────────────────────┘ │
│ │                         │   │    │         │                   │
│ │ For each op_cfg:        │   │    │         ↓                   │
│ │   _parse_operator()     │   │    │ ┌─────────────────────────┐ │
│ │   (see right column)    │   │    │ │ Combine Operators       │ │
│ │                         │   │    │ │                         │ │
│ │ Output:                 │   │    │ │ sum_operators(ops)      │ │
│ │   List[Operator]        │   │    │ │   ├─→ If len(ops) > 1:  │ │
│ └─────────────────────────┘   │    │ │   │   → _SumOperator    │ │
│         │                     │    │ │   │     (combines all)  │ │
│         ↓                     │    │ │   └─→ Else: return ops[0]│ │
│ ┌─────────────────────────┐   │    │ │                         │ │
│ │ Combine Spatial Ops     │   │    │ │ Output: Operator        │ │
│ │                         │   │    │ └─────────────────────────┘ │
│ │ sum_operators(          │   │    │         │                   │
│ │   spatial_operators     │   │    │         ↓                   │
│ │ )                       │   │    │ ┌─────────────────────────┐ │
│ │   ├─→ If len > 1:       │   │    │ │ Construct PDEProblem    │ │
│ │   │   → _SumOperator    │   │    │ │                         │ │
│ │   └─→ Else: return [0]  │   │    │ │ PDEProblem(             │ │
│ │                         │   │    │ │   domain=domain,        │ │
│ │ Output: spatial_op      │   │    │ │   operators=operators,  │ │
│ └─────────────────────────┘   │    │ │   ic=ic,                │ │
│         │                     │    │ │   bc_left=bc_left,      │ │
│         ↓                     │    │ │   bc_right=bc_right     │ │
│ ┌─────────────────────────┐   │    │ │ )                       │ │
│ │ Parse u_t Operators     │   │    │ │                         │ │
│ │ (Optional)              │   │    │ │ __post_init__():        │ │
│ │                         │   │    │ │   ├─→ Normalize ops     │ │
│ │ if "u_t_operators" in   │   │    │ │   │   to list           │ │
│ │   config:               │   │    │ │   ├─→ Validate: len > 0 │ │
│ │   u_t_ops_cfg = config[ │   │    │ │   └─→ sum_operators()   │ │
│ │     "u_t_operators"     │   │    │ │      → self._op         │ │
│ │   ]                     │   │    │ │                         │ │
│ │   For each op_cfg:      │   │    │ │ Output: PDEProblem      │ │
│ │     _parse_operator()   │   │    │ └─────────────────────────┘ │
│ │   u_t_operator =        │   │    │                             │
│ │     sum_operators(...)  │   │    │                             │
│ │ else:                   │   │    │                             │
│ │   u_t_operator = None   │   │    │                             │
│ │                         │   │    │                             │
│ │ Output: u_t_operator    │   │    │                             │
│ └─────────────────────────┘   │    │                             │
│         │                     │    │                             │
│         ↓                     │    │                             │
│ ┌─────────────────────────┐   │    │                             │
│ │ Construct SecondOrder   │   │    │                             │
│ │   PDEProblem            │   │    │                             │
│ │                         │   │    │                             │
│ │ SecondOrderPDEProblem(  │   │    │                             │
│ │   domain=domain,        │   │    │                             │
│ │   spatial_operator=     │   │    │                             │
│ │     spatial_op,         │   │    │                             │
│ │   ic_u=ic_u,            │   │    │                             │
│ │   ic_ut=ic_ut,          │   │    │                             │
│ │   u_t_operator=         │   │    │                             │
│ │     u_t_operator,       │   │    │                             │
│ │   bc_left=bc_left,      │   │    │                             │
│ │   bc_right=bc_right     │   │    │                             │
│ │ )                       │   │    │                             │
│ │                         │   │    │                             │
│ │ __post_init__():        │   │    │                             │
│ │   └─→ Validate domain   │   │    │                             │
│ │      is Domain1D        │   │    │                             │
│ │                         │   │    │                             │
│ │ Output:                 │   │    │                             │
│ │   SecondOrderPDEProblem │   │    │                             │
│ └─────────────────────────┘   │    │                             │
└───────────────────────────────┴────┴─────────────────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │ Return Problem      │
                    │ (PDEProblem or      │
                    │  SecondOrderPDE     │
                    │  Problem)           │
                    └─────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │ Ready for solving!  │
                    │ problem.solve(...)  │
                    └─────────────────────┘
```

**Key Validation Points:**
- Domain type must be "1d"
- Boundary condition types must be recognized
- Initial condition types must be recognized
- Operator types must be recognized
- Required fields must be present (e.g., `nu` for diffusion, `a` for advection)
- Second-order problems require `spatial_operators` or `operators`
- At least one operator required for `PDEProblem`

**Error Handling:**
- All parsers raise `ValueError` with descriptive messages
- Field validation occurs at parse time
- Type validation occurs during object construction

**Error Handling:** The parser validates:
- Required fields are present
- Field types are correct
- Component-specific constraints (e.g., exactly one of `value` or `expr` for Dirichlet BCs)
- Raises descriptive `ValueError` messages for debugging

### Component Parsers

The JSON parser uses a modular approach where each component type has its own parsing function. This design makes it easy to extend the parser with new component types.

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
    else:
        raise ValueError(f"Unsupported domain type {domain_type!r}.")
```

**Flow:**
1. Extract domain type (defaults to "1d")
2. Extract spatial bounds `x0`, `x1` and number of grid points `nx`
3. Extract periodic flag (defaults to `False`)
4. Create and return `Domain1D` instance

**Mathematical Result:** Creates a structured grid:
$$x_i = x_0 + i \cdot \frac{x_1 - x_0}{N-1}, \quad i = 0, 1, \ldots, N-1$$

**Error Handling:** Raises `ValueError` if domain type is not "1d" (extensibility point for future 2D support).

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
    else:
        raise ValueError(f"Unknown initial condition type {ic_type!r}.")
```

**Flow:**
1. Extract type (defaults to "expression")
2. Based on type:
   - **"expression"**: Extract `expr` string and create `InitialCondition.from_expression(expr)`
   - **"values"**: Extract `values` array and create `InitialCondition.from_values(values)`
3. Return `InitialCondition` instance

**Supported Formats:**
- **Expression**: String like `"np.sin(x)"` evaluated on the grid using safe `eval()` with minimal namespace
- **Values**: Direct array of values matching grid size (validated during evaluation)
- **Callable**: Python function (programmatic API only, not supported in JSON)

**Error Handling:** Raises `ValueError` if type is not recognized or if required fields are missing.

#### Boundary Condition Parser: `_parse_bc_one()`

Handles all boundary condition types for a single boundary (left or right):

```python
def _parse_bc_one(side_cfg: Optional[JsonDict], side: str) -> Optional[BoundaryCondition]:
    if side_cfg is None:
        return None
    
    bc_type = side_cfg.get("type", "").lower()
    
    if bc_type == "dirichlet":
        value = side_cfg.get("value")
        expr = side_cfg.get("expr")
        if side == "left":
            return DirichletLeft(value=value, expr=expr)
        elif side == "right":
            return DirichletRight(value=value, expr=expr)
    
    elif bc_type == "neumann":
        derivative_value = side_cfg.get("derivative_value")
        expr = side_cfg.get("expr")
        if side == "left":
            return NeumannLeft(derivative_value=derivative_value, expr=expr)
        elif side == "right":
            return NeumannRight(derivative_value=derivative_value, expr=expr)
    
    elif bc_type == "robin":
        # Supports both general form (a, b, c/c_expr) and backward-compatible (h, u_env)
        if "a" in side_cfg or "b" in side_cfg or "c" in side_cfg or "c_expr" in side_cfg:
            # General form: a*u + b*u_x = c(t)
            a = side_cfg.get("a")
            b = side_cfg.get("b")
            c = side_cfg.get("c")
            c_expr = side_cfg.get("c_expr")
            if side == "left":
                return RobinLeft(a=a, b=b, c=c, c_expr=c_expr)
            elif side == "right":
                return RobinRight(a=a, b=b, c=c, c_expr=c_expr)
        else:
            # Backward-compatible form: h*u + u_x = h*u_env
            h = side_cfg.get("h")
            u_env = side_cfg.get("u_env")
            if side == "left":
                return RobinLeft(h=h, u_env=u_env)
            elif side == "right":
                return RobinRight(h=h, u_env=u_env)
    
    elif bc_type == "periodic":
        return Periodic()
    
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
```

**Flow:**
1. If `side_cfg` is `None`, return `None` (no boundary condition)
2. Extract `type` field and convert to lowercase
3. Based on type, extract relevant fields and create appropriate BC class
4. Return `BoundaryCondition` instance (or `None`)

**Modularity:** Each BC type is a separate class, making it easy to add new types. The parser dispatches to the appropriate constructor based on the type string.

**Error Handling:** Raises `ValueError` for unknown BC types or invalid side specifications.

#### Operator Parser: `_parse_operator()`

```python
def _parse_operator(op_cfg: JsonDict) -> Operator:
    op_type = op_cfg["type"].lower()
    
    if op_type == "diffusion":
        if "nu" not in op_cfg:
            raise ValueError("Diffusion operator requires 'nu' coefficient.")
        nu = op_cfg["nu"]
        scheme = op_cfg.get("scheme", "central")
        return Diffusion(nu=nu, scheme=scheme)
    
    elif op_type == "advection":
        if "a" not in op_cfg:
            raise ValueError("Advection operator requires 'a' coefficient.")
        a = op_cfg["a"]
        scheme = op_cfg.get("scheme", "upwind_first")
        return Advection(a=a, scheme=scheme)
    
    elif op_type in ("expression", "expression_operator"):
        expr = op_cfg["expr"]
        params = op_cfg.get("params", {})
        schemes = op_cfg.get("schemes", {})
        return ExpressionOperator(
            expr_string=expr,
            params=params,
            schemes=schemes
        )
    
    else:
        raise ValueError(f"Unknown operator type {op_type!r}.")
```

**Flow:**
1. Extract `type` field and convert to lowercase
2. Based on type:
   - **"diffusion"**: Extract `nu` (required), `scheme` (optional, default: "central"), create `Diffusion`
   - **"advection"**: Extract `a` (required), `scheme` (optional, default: "upwind_first"), create `Advection`
   - **"expression"** or **"expression_operator"**: Extract `expr` (required), `params` (optional), `schemes` (optional), create `ExpressionOperator`
3. Return `Operator` instance

**Extensibility:** New operator types can be added by:
1. Creating a new `Operator` subclass
2. Adding a new `elif` branch in `_parse_operator()`
3. Specifying required and optional fields

**Error Handling:** 
- Raises `ValueError` if required fields are missing (e.g., `nu` for diffusion, `a` for advection)
- Raises `ValueError` for unknown operator types

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
        env = _build_safe_eval_env()  # Returns {"np": np}
        env["x"] = x  # Inject grid points
        result = eval(self.expr, {"__builtins__": {}}, env)
        return np.asarray(result, dtype=float)
    
    if self.func is not None:
        result = self.func(x)
        return np.asarray(result, dtype=float)
    
    # Direct values (validate shape matches grid)
    arr = np.asarray(self.values, dtype=float)
    if arr.shape != x.shape:
        raise ValueError(
            f"Initial condition values have shape {arr.shape}, "
            f"but domain grid has shape {x.shape}."
        )
    return arr
```

**Step-by-Step Process:**

1. **Extract grid points**: `x = domain.x` (1D array of length `nx`)

2. **Expression mode** (`self.expr is not None`):
   - Build safe evaluation environment: `{"np": np, "x": x}`
   - No `__builtins__` to prevent code injection
   - Evaluate expression string using `eval()`
   - Convert result to NumPy array with `dtype=float`

3. **Callable mode** (`self.func is not None`):
   - Call function with grid points: `result = self.func(x)`
   - Convert result to NumPy array

4. **Values mode** (`self.values is not None`):
   - Convert to NumPy array
   - **Validate shape**: Must match grid shape `x.shape`
   - Raise `ValueError` if shapes don't match

**Safety Features:**
- **Minimal namespace**: Only `np` and `x` are available in expression evaluation
- **No builtins**: `{"__builtins__": {}}` prevents access to dangerous functions
- **Shape validation**: Ensures values array matches grid size
- **Type coercion**: All results converted to `float` dtype for numerical consistency

### Mathematical Interpretation

Given an initial condition $u(x, 0) = u_0(x)$, we evaluate it at each grid point:
$$\mathbf{u}(0) = [u_0(x_0), u_0(x_1), \ldots, u_0(x_{N-1})]^T$$
**Example:** For $u_0(x) = \sin(x)$ on grid points $x_i$:
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
    # Step 1: Evaluate initial condition on grid
    u0 = self.ic.evaluate(self.domain)
    
    # Step 2: Apply boundary conditions for non-periodic domains
    if not self.domain.periodic:
        if self.bc_left is not None:
            self.bc_left.apply_to_full(u0, t0, self.domain)
        if self.bc_right is not None:
            self.bc_right.apply_to_full(u0, t0, self.domain)
    
    return u0  # Shape: (nx,)
```

**Flow:**
1. Evaluate IC expression/values/callable on grid → `u0` (shape: `(nx,)`)
2. For non-periodic domains, apply boundary conditions to enforce constraints at $t=0$
3. Return full initial state vector

**Second-Order Problems (`SecondOrderPDEProblem`):**

```python
def initial_full(self, t0: float = 0.0) -> Array:
    # Step 1: Evaluate both initial conditions
    u0 = self.ic_u.evaluate(self.domain)      # u(x, 0)
    v0 = self.ic_ut.evaluate(self.domain)     # u_t(x, 0) = v(x, 0)
    
    # Step 2: Ensure arrays are 1D and handle scalar ICs
    u0 = np.asarray(u0, dtype=float).flatten()
    v0 = np.asarray(v0, dtype=float).flatten()
    
    # Step 3: Handle scalar initial conditions (broadcast to array)
    if u0.size == 1:
        u0 = np.full(self.domain.nx, float(u0))
    if v0.size == 1:
        v0 = np.full(self.domain.nx, float(v0))
    
    # Step 4: Apply BCs to u0 (not v0) for non-periodic domains
    if not self.domain.periodic:
        if self.bc_left is not None:
            self.bc_left.apply_to_full(u0, t0, self.domain)
        if self.bc_right is not None:
            self.bc_right.apply_to_full(u0, t0, self.domain)
    
    # Step 5: Return extended state [u, v] where v = u_t
    if self.domain.periodic:
        return np.concatenate([u0, v0])  # Shape: (2*nx,)
    else:
        return np.concatenate([u0[1:-1], v0[1:-1]])  # Shape: (2*(nx-2),)
```

**Flow:**
1. Evaluate both ICs: `u0 = u(x, 0)`, `v0 = u_t(x, 0)`
2. Handle scalar ICs by broadcasting to full grid
3. Apply boundary conditions to `u0` only (boundary conditions apply to $u$, not $u_t$)
4. For periodic: return `[u0, v0]` (full state)
5. For non-periodic: return `[u0[1:-1], v0[1:-1]]` (interior only)

**Mathematical Result:** For second-order problems, the initial state vector is doubled:
$$\mathbf{y}(0) = \begin{bmatrix} \mathbf{u}(0) \\ \mathbf{v}(0) \end{bmatrix}$$
where $\mathbf{v}(0) = \frac{\partial \mathbf{u}}{\partial t}(0)$.

**Important:** Boundary conditions are applied to $u$ only, not to $v = u_t$. This is because physical boundary conditions typically constrain the solution $u$, not its time derivative.

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
**Accuracy:** $O(\Delta x^2)$ (second-order)

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

### Discretization Schemes

The library supports multiple discretization schemes through the `spatial_discretization` module. Different schemes are optimal for different types of PDE terms, allowing for improved accuracy and stability.

#### Available Schemes

The `DiscretizationScheme` enum provides the following options:

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

#### Why Different Schemes?

Different PDE terms benefit from different discretization methods:

1. **Advection terms** ($u_x$): Use upwind schemes to respect the direction of information flow and prevent oscillations
2. **Diffusion terms** ($u_{xx}$): Use central differences for optimal accuracy
3. **Reaction terms**: No spatial discretization needed

#### Implementation: `spatial_discretization.py`

The discretization module provides unified functions for computing derivatives:

```python
def first_derivative(
    u: Array,
    dx: float,
    domain: Domain1D,
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL,
    velocity: Optional[Array] = None,
) -> Array:
    """Compute first derivative using specified scheme."""
    # Dispatches to appropriate implementation based on scheme
```

**Key Features:**
- **Modular design**: All derivative computations go through a single interface
- **Scheme selection**: Each operator can specify its preferred scheme
- **Velocity-aware**: Upwind schemes automatically use velocity to determine upwind direction

#### Upwind Scheme Implementation

**First-Order Upwind:**

For advection terms, the upwind direction is determined by the sign of the velocity:

- If $a > 0$: Use backward difference $u_x \approx \frac{u_i - u_{i-1}}{\Delta x}$
- If $a < 0$: Use forward difference $u_x \approx \frac{u_{i+1} - u_i}{\Delta x}$

**Mathematical Form:**
$$u_x(x_i) \approx \begin{cases}
\frac{u_i - u_{i-1}}{\Delta x} & \text{if } a > 0 \\
\frac{u_{i+1} - u_i}{\Delta x} & \text{if } a < 0
\end{cases}$$

**Second-Order Upwind:**

Uses a three-point stencil biased in the upwind direction:

- If $a > 0$: $u_x \approx \frac{3u_i - 4u_{i-1} + u_{i-2}}{2\Delta x}$
- If $a < 0$: $u_x \approx \frac{-u_{i+2} + 4u_{i+1} - 3u_i}{2\Delta x}$

**Mathematical Form:**
$$u_x(x_i) \approx \begin{cases}
\frac{3u_i - 4u_{i-1} + u_{i-2}}{2\Delta x} & \text{if } a > 0 \\
\frac{-u_{i+2} + 4u_{i+1} - 3u_i}{2\Delta x} & \text{if } a < 0
\end{cases}$$

#### Integration with Operators

**Advection Operator:**

```python
@dataclass
class Advection(Operator):
    a: float | str | Callable
    scheme: DiscretizationScheme | str = DiscretizationScheme.UPWIND_FIRST
    
    def apply(self, u_full, domain, t):
        a_vals = _eval_coeff(self.a, domain.x, t)
        u_x = first_derivative(u_full, domain.dx, domain, self.scheme, velocity=a_vals)
        return -a_vals * u_x
```

**ExpressionOperator with Mixed Schemes:**

```python
@dataclass
class ExpressionOperator(Operator):
    expr_string: str
    params: Optional[dict] = None
    schemes: Optional[dict[str, DiscretizationScheme | str]] = None
    
    def apply(self, u_full, domain, t):
        # Compute derivatives with specified schemes
        ux = first_derivative(u, dx, domain, self._ux_scheme, velocity=velocity)
        uxx = second_derivative(u, dx, domain, self._uxx_scheme)
        uxxx = third_derivative(u, dx, domain, self._uxxx_scheme)
        # ... evaluate expression
```

This allows different derivatives in the same expression to use different schemes, e.g., upwind for $u_x$ and central for $u_{xx}$.

#### JSON Configuration

Schemes can be specified in JSON:

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

### Operator Classes

Operators implement the `Operator` interface with a single `apply()` method that computes the spatial contribution to the time derivative.

#### Coefficient Evaluation: `_eval_coeff()`

Before discussing specific operators, it's important to understand how coefficients are evaluated:

```python
def _eval_coeff(coeff: float | str | Callable, x: Array, t: float) -> Array:
    """Evaluate a coefficient that may be constant, space/time-dependent, or a string expression."""
    if isinstance(coeff, (int, float)):
        return float(coeff) * np.ones_like(x, dtype=float)  # Broadcast constant
    
    if callable(coeff):
        return np.asarray(coeff(x, t), dtype=float)  # Call function
    
    if isinstance(coeff, str):
        env = {"np": np, "x": x, "t": t}
        return np.asarray(eval(coeff, {"__builtins__": {}}, env), dtype=float)  # Safe eval
    
    raise TypeError(f"Unsupported coefficient type: {type(coeff)!r}")
```

**Supported Coefficient Types:**
1. **Scalar**: Constant value broadcast to all grid points
2. **Callable**: Function `f(x, t)` that returns an array of values
3. **String**: Expression in `x`, `t`, and `np` (e.g., `"0.1 + 0.05 * x"`)

**Safety:** String expressions use the same safe evaluation as initial conditions.

#### Diffusion Operator

**Mathematical Form:**
$$L[u] = \nu \frac{\partial^2 u}{\partial x^2}$$

**Implementation:**

```python
@dataclass
class Diffusion(Operator):
    nu: float | str | Callable  # Diffusivity coefficient
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL
    
    def apply(self, u_full, domain, t):
        x = domain.x
        dx = domain.dx
        
        # Compute second derivative using specified scheme
        u_xx = second_derivative(u_full, dx, domain, self.scheme)
        
        # Evaluate coefficient (may be space/time-dependent)
        nu_vals = _eval_coeff(self.nu, x, t)
        
        # Element-wise multiplication
        return nu_vals * u_xx
```

**Key Features:**
- Supports constant, space-dependent, or time-dependent diffusivity
- Uses central differences by default (optimal for diffusion)
- Element-wise multiplication allows for spatially varying coefficients

**Example:** For $\nu(x) = 0.1 + 0.05x$:
```python
diffusion = Diffusion(nu="0.1 + 0.05 * x")
```

#### Advection Operator

**Mathematical Form:**
$$L[u] = -a \frac{\partial u}{\partial x}$$

**Implementation:**

```python
@dataclass
class Advection(Operator):
    a: float | str | Callable  # Advection speed
    scheme: DiscretizationScheme | str = DiscretizationScheme.UPWIND_FIRST
    
    def apply(self, u_full, domain, t):
        x = domain.x
        dx = domain.dx
        
        # Evaluate advection speed (may be space/time-dependent)
        a_vals = _eval_coeff(self.a, x, t)
        
        # Compute first derivative using specified scheme
        # For upwind schemes, pass velocity to determine upwind direction
        if scheme in (DiscretizationScheme.UPWIND_FIRST, DiscretizationScheme.UPWIND_SECOND):
            u_x = first_derivative(u_full, dx, domain, scheme, velocity=a_vals)
        else:
            u_x = first_derivative(u_full, dx, domain, scheme)
        
        return -a_vals * u_x
```

**Key Features:**
- **Default scheme**: `UPWIND_FIRST` for stability (can be overridden)
- **Velocity-aware**: Upwind schemes use the advection speed to determine upwind direction
- Supports constant, space-dependent, or time-dependent advection speed

**Why Upwind by Default?**
- Central differences can cause oscillations for advection-dominated problems
- Upwind schemes respect the direction of information flow
- First-order upwind is very stable, though less accurate

**Example:** For $a(x,t) = 1.0 + 0.5\cos(x)$ with second-order upwind:
```python
advection = Advection(a="1.0 + 0.5 * np.cos(x)", scheme="upwind_second")
```

#### ExpressionOperator

**Mathematical Form:**
$$L[u] = f(u, u_x, u_{xx}, u_{xxx}, x, t, \mathbf{p})$$
where $\mathbf{p}$ are parameters.

**Initialization (`__post_init__`):**

```python
def __post_init__(self):
    if self.params is None:
        self.params = {}
    
    # Parse schemes (if provided)
    if self.schemes is None:
        self.schemes = {}
    
    # Convert string schemes to enums
    self._ux_scheme = DiscretizationScheme(self.schemes.get("ux", "central"))
    self._uxx_scheme = DiscretizationScheme(self.schemes.get("uxx", "central"))
    self._uxxx_scheme = DiscretizationScheme(self.schemes.get("uxxx", "central"))
    
    # Try to detect velocity for upwind schemes (heuristic)
    # ... (velocity detection logic)
    
    # Define symbols for SymPy parsing
    u, ux, uxx, uxxx, x, t = sp.symbols("u ux uxx uxxx x t")
    self._base_symbols = {"u": u, "ux": ux, "uxx": uxx, "uxxx": uxxx, "x": x, "t": t}
    
    # Parameter symbols
    self._param_symbols = {name: sp.symbols(name) for name in self.params.keys()}
    
    # Allowed functions (sin, cos, exp, log, tanh, sqrt)
    allowed_funcs = {name: getattr(sp, name) for name in ["sin", "cos", "exp", "log", "tanh", "sqrt"]}
    
    # Build local dictionary for safe parsing
    local_dict = {**self._base_symbols, **self._param_symbols, **allowed_funcs}
    
    # Parse expression safely
    self._expr = sp.sympify(self.expr_string, locals=local_dict)
    
    # Build argument list and compile to NumPy function
    args = list(self._base_symbols.values()) + list(self._param_symbols.values())
    self._param_order = list(self._param_symbols.keys())
    self._func = sp.lambdify(args, self._expr, modules=["numpy"])
```

**Application (`apply`):**

```python
def apply(self, u_full, domain, t):
    x = domain.x
    dx = domain.dx
    
    u = np.asarray(u_full, dtype=float)
    
    # Compute derivatives with specified schemes
    # For upwind, try to detect velocity from expression
    velocity = None
    if self._ux_scheme in (DiscretizationScheme.UPWIND_FIRST, DiscretizationScheme.UPWIND_SECOND):
        # Attempt to extract velocity from expression
        # (simplified heuristic - may not work for complex expressions)
        if self._velocity_func is not None:
            # ... compute velocity from expression
        else:
            velocity = np.ones_like(u)  # Fallback
    
    ux = first_derivative(u, dx, domain, self._ux_scheme, velocity=velocity)
    uxx = second_derivative(u, dx, domain, self._uxx_scheme)
    uxxx = third_derivative(u, dx, domain, self._uxxx_scheme)
    
    # Get parameter values in stable order
    param_vals = [self.params[name] for name in self._param_order]
    
    # Evaluate compiled expression
    result = self._func(u, ux, uxx, uxxx, x, t, *param_vals)
    return np.asarray(result, dtype=float)
```

**Key Features:**
- **SymPy parsing**: Safe expression parsing prevents code injection
- **Fast compilation**: `lambdify` compiles to optimized NumPy code
- **Scheme selection**: Different schemes can be used for different derivatives
- **Parameter support**: Named parameters can be passed and substituted
- **Function support**: Standard mathematical functions (sin, cos, exp, etc.)

**Safety:** 
- SymPy's `sympify` only parses valid mathematical expressions
- No access to dangerous Python builtins
- `lambdify` generates safe NumPy code

**Performance:** 
- Expression is parsed once during initialization
- Compiled function is reused for every `apply()` call
- No repeated parsing overhead

**Example:** Burgers' equation with upwind for advection:
```python
op = ExpressionOperator(
    expr_string="-u*ux + nu*uxx",
    params={"nu": 0.1},
    schemes={"ux": "upwind_second", "uxx": "central"}
)
```

#### Operator Combination: `sum_operators()`

Multiple operators can be combined into a single operator:

```python
def sum_operators(ops: Iterable[Operator]) -> Operator:
    """Combine several operators into a single Operator that returns their sum."""
    return _SumOperator(list(ops))

class _SumOperator(Operator):
    def __init__(self, operators: Sequence[Operator]):
        if not operators:
            raise ValueError("At least one operator is required.")
        self._operators: List[Operator] = list(operators)
    
    def apply(self, u_full: Array, domain, t: float) -> Array:
        result = np.zeros_like(u_full, dtype=float)
        for op in self._operators:
            result += op.apply(u_full, domain, t)
        return result
```

**Mathematical Result:** If $L_1, L_2, \ldots, L_n$ are operators, then:
$$(L_1 + L_2 + \cdots + L_n)[u] = L_1[u] + L_2[u] + \cdots + L_n[u]$$

**Usage:**
```python
advection = Advection(a=1.0)
diffusion = Diffusion(nu=0.1)
combined = sum_operators([advection, diffusion])
# combined.apply(u, domain, t) = advection.apply(u, domain, t) + diffusion.apply(u, domain, t)
```

---

## PDE Problem Solving

The library supports two types of PDE problems:

1. **First-order in time**: $u_t = L[u]$
2. **Second-order in time**: $u_{tt} = L[u] + M[u_t]$

### First-Order Problems: `PDEProblem`

**Mathematical Form:**
$$\frac{\partial u}{\partial t} = L[u]$$
After spatial discretization:
$$\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}, t)$$
where $\mathbf{F}$ applies the spatial operator $L$ at each grid point.

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

**Complete Solve Process:**

```python
def solve(self, t_span, t_eval=None, method="RK45", plot=False, plot_dir=None, **kwargs):
    t0, _ = t_span
    
    # Step 1: Determine initial state and RHS function
    if self.domain.periodic:
        y0 = self.initial_full(t0)  # Full state vector [u_0, ..., u_{N-1}]
        fun = self._rhs_periodic     # RHS function for periodic domains
    else:
        y0 = self.initial_interior(t0)  # Interior only [u_1, ..., u_{N-2}]
        fun = self._rhs_nonperiodic      # RHS function for non-periodic domains
    
    # Step 2: Call SciPy's ODE solver
    result = solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        **kwargs  # e.g., rtol, atol, max_step, etc.
    )
    
    # Step 3: Optional plotting
    if plot:
        # Reconstruct full solutions for visualization
        # ... (plotting code)
    
    return result
```

**Step-by-Step Breakdown:**

1. **Initial State Preparation:**
   - **Periodic**: `y0 = initial_full(t0)` → full state vector of length `nx`
   - **Non-periodic**: `y0 = initial_interior(t0)` → interior state of length `nx-2`

2. **RHS Function Selection:**
   - **Periodic**: `fun = _rhs_periodic` → directly applies operators to full state
   - **Non-periodic**: `fun = _rhs_nonperiodic` → reconstructs full state, applies operators, returns interior

3. **Time Integration:**
   - `solve_ivp` is called with the selected RHS function
   - Adaptive time stepping adjusts step size to maintain error tolerances
   - Solution is stored at times specified by `t_eval` (if provided)

4. **Post-Processing:**
   - If `plot=True`, full solutions are reconstructed and plots are generated
   - Result object contains `result.t` (time points) and `result.y` (solution snapshots)

**Mathematical Result:** `solve_ivp` integrates the ODE system:
$$\mathbf{u}(t) = \mathbf{u}(0) + \int_0^t \mathbf{F}(\mathbf{u}(\tau), \tau) \, d\tau$$

**Adaptive Time Stepping:**

The solver automatically adjusts the time step $\Delta t$ to maintain:
$$\text{error} < \text{rtol} \cdot |\mathbf{u}| + \text{atol}$$

where:
- `rtol`: Relative tolerance (default: 1e-6)
- `atol`: Absolute tolerance (default: 1e-8)

This ensures accuracy while maximizing efficiency.
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

The plotting module provides utilities for visualizing 1D PDE solutions. It uses matplotlib with the Agg backend (suitable for headless environments like CI servers).

**Design Principles:**
- **Headless-safe**: Uses Agg backend, no GUI required
- **File-based**: All plots saved to disk, no interactive display
- **Modular**: Each plot type is a separate function
- **Optional**: Plotting is completely optional; solving works without matplotlib

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
        
        x = self.domain.x
        
        # Reconstruct full solutions for plotting
        if self.domain.periodic:
            # Periodic: result.y already contains full state
            full_solutions = result.y
        else:
            # Non-periodic: reconstruct full state from interior at each time
            full_states = []
            for k, t in enumerate(result.t):
                interior_k = result.y[:, k]
                full_k = self._reconstruct_full_from_interior(interior_k, t)
                full_states.append(full_k)
            full_solutions = np.stack(full_states, axis=1)  # Shape: (nx, nt)
        
        # Generate plots
        plot_1d(
            x, full_solutions[:, 0], 
            title="Initial solution (1D)", 
            savepath=os.path.join(base_dir, "solution1d_initial.png")
        )
        plot_1d(
            x, full_solutions[:, -1], 
            title="Final solution (1D)", 
            savepath=os.path.join(base_dir, "solution1d_final.png")
        )
        plot_1d_time_series(
            x, full_solutions, result.t, 
            prefix="solution1d", 
            out_dir=base_dir
        )
    
    return result
```

**For Second-Order Problems:**

The plotting extracts the `u` component (first half of state vector):

```python
# In SecondOrderPDEProblem.solve()
# After solving, extract components
nx = self.domain.nx
if self.domain.periodic:
    result.u = result.y[:nx, :]  # Extract u component
    result.v = result.y[nx:, :]  # Extract v = u_t component
else:
    # For non-periodic, reconstruct full u and v
    u_states = []
    v_states = []
    for k, t in enumerate(result.t):
        interior_k = result.y[:, k]
        u_full, v_full = self._reconstruct_full_from_interior(interior_k, t)
        u_states.append(u_full)
        v_states.append(v_full)
    result.u = np.stack(u_states, axis=1)
    result.v = np.stack(v_states, axis=1)

# Plotting uses result.u
plot_1d(x, result.u[:, 0], title="Initial solution u(x,0)", savepath=...)
```

**CLI Integration (`run_problem.py`):**

The CLI automatically generates additional plots when visualization is enabled:

```python
# After solving, generate combined plot and heatmap
if vis_enable:
    plot_1d_combined(
        x, full_solutions, result.t,
        title="Combined 1D time series",
        savepath=base_plots_dir / "solution1d_combined.png",
        max_curves=8
    )
    plot_xt_heatmap(
        x, result.t, full_solutions,
        title="u(x,t) heatmap",
        savepath=base_plots_dir / "solution1d_xt_heatmap.png"
    )
```

**Plot Organization:**
- All plots saved to `plots/<save_dir>/` directory
- Initial and final snapshots: `solution1d_initial.png`, `solution1d_final.png`
- Time series frames: `solution1d_t0000.png`, `solution1d_t0001.png`, ...
- Combined plot: `solution1d_combined.png`
- Heatmap: `solution1d_xt_heatmap.png`

---

## Dataset Generation

The library includes utilities for generating parameterized datasets, useful for machine learning applications (e.g., training Physics-Informed Neural Networks).

### Module: `dataset.py`

**Purpose:** Generate multiple PDE solutions by sampling parameters from specified ranges.

**Key Classes:**

#### `ParameterRange`

```python
@dataclass
class ParameterRange:
    name: str      # Parameter name (e.g., "nu", "alpha")
    low: float     # Lower bound
    high: float    # Upper bound
```

Defines a range for uniform random sampling.

#### `ParameterSampler`

```python
class ParameterSampler:
    def __init__(self, param_ranges: List[ParameterRange], seed: Optional[int] = None):
        self.param_ranges = param_ranges
        self.rng = random.Random(seed)
    
    def sample(self) -> Dict[str, float]:
        """Sample one set of parameter values."""
        return {pr.name: self.rng.uniform(pr.low, pr.high) for pr in self.param_ranges}
```

Samples parameter values uniformly from specified ranges.

#### `generate_dataset()`

```python
def generate_dataset(
    json_template: Union[str, Path, Dict[str, Any]],
    param_ranges: List[ParameterRange],
    num_samples: int,
    savepath: Union[str, Path] = "dataset.pt",
    seed: Optional[int] = None,
    t_span: Optional[tuple[float, float]] = None,
    t_eval: Optional[Sequence[float]] = None,
    solver_method: str = "RK45",
    solver_kwargs: Optional[Dict[str, Any]] = None,
    save_heatmaps: bool = True,
    problem_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, torch.Tensor]:
```

**Process:**

1. **Load template**: JSON configuration file or dictionary
2. **Create sampler**: `ParameterSampler(param_ranges, seed)`
3. **For each sample**:
   - Sample parameter values
   - Substitute parameters into JSON template
   - Build `PDEProblem` from modified config
   - Solve PDE
   - Extract solution
4. **Collect results**: Stack all solutions into PyTorch tensors
5. **Save dataset**: Save as `.pt` file containing:
   - `params`: Parameter values (shape: `[num_samples, num_params]`)
   - `u`: Solutions (shape: `[num_samples, nx, nt]`)
   - `x`: Spatial grid (shape: `[nx]`)
   - `t`: Time points (shape: `[nt]`)

**Output Format:**

```python
dataset = {
    "params": torch.Tensor,  # Shape: (num_samples, num_params)
    "u": torch.Tensor,       # Shape: (num_samples, nx, nt)
    "x": torch.Tensor,       # Shape: (nx,)
    "t": torch.Tensor,       # Shape: (nt,)
}
```

**Usage:**

```python
from pde.dataset import ParameterRange, generate_dataset

param_ranges = [
    ParameterRange(name="nu", low=0.01, high=0.1),
    ParameterRange(name="alpha", low=0.001, high=0.01)
]

dataset = generate_dataset(
    json_template="examples/heat1d_parameterized.json",
    param_ranges=param_ranges,
    num_samples=100,
    savepath="heat1d_dataset.pt",
    seed=42
)
```

**CLI Integration:**

```bash
python run_problem.py examples/heat1d_parameterized.json --dataset --samples 100
```

---

## Command-Line Interface

The library provides a CLI (`run_problem.py`) for running JSON-defined problems without writing Python code.

### Entry Point: `main()`

**Modes of Operation:**

1. **Single Solve Mode** (default):
   - Load JSON configuration
   - Build `PDEProblem` or `SecondOrderPDEProblem`
   - Solve PDE
   - Generate plots (if enabled)
   - Print diagnostics

2. **Dataset Generation Mode** (`--dataset` flag):
   - Load JSON template
   - Sample parameters
   - Generate multiple solutions
   - Save dataset to `.pt` file

**Command-Line Arguments:**

```bash
python run_problem.py <config.json> [options]

Options:
  --no-output      Suppress detailed output (exit code only)
  --dataset        Enable dataset generation mode
  --samples N      Number of samples for dataset generation
  --save PATH      Save path for dataset file
```

**Time Configuration Parsing:**

```python
def _build_time_grid(time_cfg: Dict[str, Any]):
    t0 = float(time_cfg.get("t0", 0.0))
    t1 = float(time_cfg.get("t1", 1.0))
    
    if "t_eval" in time_cfg:
        t_eval = np.asarray(time_cfg["t_eval"], dtype=float)
    else:
        num_points = int(time_cfg.get("num_points", 101))
        t_eval = np.linspace(t0, t1, num_points)
    
    method = time_cfg.get("method", "RK45")
    rtol = float(time_cfg.get("rtol", 1e-6))
    atol = float(time_cfg.get("atol", 1e-8))
    
    solve_kwargs = {"method": method, "rtol": rtol, "atol": atol}
    return (t0, t1), t_eval, solve_kwargs
```

**Visualization Configuration:**

The CLI respects the `visualization` block in JSON:
- `enable`: Enable/disable plotting
- `save_dir`: Directory for saving plots
- `type`: Plot type (currently only "1d" supported)

**Output:**
- Prints final time and L2 norm of solution
- Saves plots to `plots/<save_dir>/` if visualization enabled
- Returns exit code 0 on success, non-zero on failure

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

The `pde-mol` library implements a **modular, extensible, and safe** framework for solving PDEs using the Method of Lines. This document has provided a comprehensive overview of:

### Core Components

1. **Domain and Grid Construction**: Structured 1D grids with periodic and non-periodic support
2. **Initial Conditions**: Expression, values, or callable-based initialization
3. **Boundary Conditions**: Dirichlet, Neumann, Robin, and Periodic with detailed mathematical formulations
4. **Spatial Discretization**: Multiple schemes (central, upwind) optimized for different PDE terms
5. **Operators**: Diffusion, Advection, and Expression operators with configurable discretization
6. **Problem Classes**: First-order (`PDEProblem`) and second-order (`SecondOrderPDEProblem`) PDE solvers
7. **Time Integration**: Integration with SciPy's `solve_ivp` with adaptive time stepping
8. **Plotting**: Comprehensive visualization utilities for 1D solutions
9. **Dataset Generation**: Parameterized problem solving for machine learning applications
10. **CLI Interface**: Command-line tool for running JSON-defined problems

### Key Features

- **JSON-driven configuration** for easy problem specification
- **Modular components** (domain, IC, BC, operators, discretization schemes) that can be tested independently
- **Clear finite-difference implementations** with second-order accuracy
- **Flexible discretization schemes** (central, upwind) optimized for different PDE terms
- **Support for first- and second-order time derivatives**
- **Support for up to third-order spatial derivatives**
- **Comprehensive boundary condition support** (Dirichlet, Neumann, Robin, Periodic)
- **Safe expression evaluation** using SymPy (no code injection)
- **Fast compiled expressions** using `lambdify` for performance
- **Flexible plotting utilities** for visualization
- **Dataset generation** for parameterized problems
- **Command-line interface** for easy execution

### Architecture Highlights

- **Separation of Concerns**: Each component (domain, IC, BC, operators, time integrator) is independent
- **Extensibility**: New operators, BCs, and schemes can be added without modifying existing code
- **Safety**: Safe expression evaluation prevents code injection
- **Performance**: Precompiled expressions and optional sparse matrices
- **Testability**: Each component can be tested in isolation

### Mathematical Foundations

The library implements the Method of Lines:
1. **Spatial discretization**: PDE → system of ODEs using finite differences
2. **Time integration**: ODE system → solution using adaptive time integrators
3. **Boundary enforcement**: Applied at each time step for non-periodic domains
4. **State management**: Different state representations for periodic vs. non-periodic domains

The architecture follows the stated design principles (modularity, simplicity, extensibility, safety, performance), making it easy to understand, extend, and maintain.

