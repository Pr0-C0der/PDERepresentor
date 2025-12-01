# Second-Order Time Derivatives and Third-Order Spatial Derivatives

This document explains how to use the new features for handling:
1. **Second-order time derivatives** (u_tt) - e.g., wave equations
2. **Third-order spatial derivatives** (u_xxx) - e.g., KdV equation
3. **Mixed terms** (d²u/dxdt) - currently requires custom implementation

## Third-Order Spatial Derivatives (u_xxx)

### Usage in ExpressionOperator

The `ExpressionOperator` now supports the `uxxx` symbol for third-order spatial derivatives:

```python
from pde import Domain1D, InitialCondition, ExpressionOperator, PDEProblem

dom = Domain1D(0.0, 2*np.pi, 201, periodic=True)
ic = InitialCondition.from_expression("np.sin(x)")

# KdV-type equation: u_t = -u*u_x - alpha*u_xxx
op = ExpressionOperator(
    expr_string="-u*ux - alpha*uxxx",
    params={"alpha": 0.01}
)

problem = PDEProblem(domain=dom, operators=[op], ic=ic)
sol = problem.solve((0.0, 0.5), t_eval=np.linspace(0.0, 0.5, 100))
```

### JSON Configuration

```json
{
  "operators": [
    {
      "type": "expression",
      "expr": "-u*ux - alpha*uxxx",
      "params": {
        "alpha": 0.01
      }
    }
  ]
}
```

### Implementation Details

- Uses second-order central finite differences for interior points
- Periodic boundaries: wrapped stencil
- Non-periodic boundaries: one-sided forward/backward differences
- Accuracy: O(dx²) for the third derivative

## Second-Order Time Derivatives (u_tt)

### Basic Wave Equation

For equations of the form `u_tt = L[u]` where L is a spatial operator:

```python
from pde import SecondOrderPDEProblem, Domain1D, InitialCondition, Diffusion

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
u_solution = sol.u  # Shape: (nx, nt)
v_solution = sol.v  # Shape: (nx, nt) where v = u_t
```

### Damped Wave Equation

For equations with u_t terms: `u_tt = L[u] - γ * u_t`:

```python
from pde.operators import Advection

# Damped wave: u_tt = c² * u_xx - γ * u_t
spatial_op = Diffusion(1.0)      # c² * u_xx
u_t_op = Advection(a=0.1)        # -γ * u_t (note: Advection returns -a*u_x, 
                                  # but here we apply it to u_t, so it becomes -γ*u_t)

damped_wave = SecondOrderPDEProblem(
    domain=dom,
    spatial_operator=spatial_op,
    ic_u=ic_u,
    ic_ut=ic_ut,
    u_t_operator=u_t_op
)

# Use BDF method for stiff systems with damping
sol = damped_wave.solve(
    (0.0, 2.0), 
    t_eval=np.linspace(0.0, 2.0, 100),
    method="BDF"  # Recommended for stiff/damped systems
)
```

### JSON Configuration

For second-order problems, use `"initial_condition_ut"` and `"spatial_operators"`:

```json
{
  "domain": {
    "type": "1d",
    "x0": 0.0,
    "x1": 6.283185307179586,
    "nx": 201,
    "periodic": true
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
      "type": "diffusion",
      "nu": 1.0
    }
  ],
  "u_t_operators": [
    {
      "type": "advection",
      "a": 0.1
    }
  ],
  "time": {
    "t0": 0.0,
    "t1": 2.0,
    "num_points": 100,
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-8
  }
}
```

### How It Works

The `SecondOrderPDEProblem` converts the second-order equation to a first-order system:

1. **Original**: `u_tt = L[u] + M[u_t]`
2. **Converted**: 
   - `u_t = v` (where v = u_t)
   - `v_t = L[u] + M[v]`

The state vector is doubled: `[u, v]` instead of just `[u]`.

### Accuracy

- **Spatial accuracy**: Same as first-order problems (O(dx²) with second-order stencils)
- **Time accuracy**: Controlled by `rtol` and `atol` (adaptive)
- **Conversion error**: None (exact mathematical transformation)

## Examples

See the example JSON files:
- `examples/wave1d.json` - Simple wave equation
- `examples/damped_wave1d.json` - Damped wave equation
- `examples/kdv1d.json` - KdV equation with third-order derivative

## Limitations

### Mixed Derivatives (d²u/dxdt)

Mixed derivatives like `d²u/dxdt` are **not yet directly supported** in the current implementation. To handle them, you would need to:

1. Store the previous time step
2. Compute `u_t` via backward difference: `u_t ≈ (u(t) - u(t-dt)) / dt`
3. Then compute `d²u/dxdt ≈ d/dx[u_t]`

This requires modifying the RHS function to track previous states, which is more complex. For most applications, this is rarely needed.

### Higher-Order Time Derivatives

Currently only second-order time derivatives (u_tt) are supported. For higher orders (u_ttt, etc.), you would need to extend the conversion approach:

- `u_ttt = L[u]` → introduce `v = u_t`, `w = u_tt`, giving:
  - `u_t = v`
  - `v_t = w`
  - `w_t = L[u]`

This would triple the state vector size.

## Design Principles

The implementation follows the core design principles:

1. **Modularity**: `SecondOrderPDEProblem` is a separate class, independent of `PDEProblem`
2. **Simplicity**: Clear conversion to first-order system, no hidden complexity
3. **Extensibility**: Easy to add support for more operator types
4. **Safety**: Uses same safe expression evaluation as first-order problems

