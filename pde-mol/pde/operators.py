from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import sympy as sp

from .domain import Domain1D


Array = np.ndarray


class Operator:
    """
    Abstract base class for spatial operators.

    An operator maps a full solution vector ``u_full`` to its spatial
    contribution ``du_dt = L[u]`` at a given time.
    """

    def apply(self, u_full: Array, domain, t: float) -> Array:
        raise NotImplementedError


def _central_first_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Second-order central first derivative in 1D.

    - For periodic domains, use a wrapped central stencil for *all* points.
    - For non-periodic domains, delegate to ``np.gradient`` with
      edge_order=2 (central in the interior, one-sided at boundaries).
    """
    if u.size < 3:
        return np.zeros_like(u, dtype=float)

    du = np.zeros_like(u, dtype=float)

    if periodic:
        # Interior points: standard central stencil
        du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        # Periodic end points: wrap to the penultimate point, not the duplicate
        # last point, so that x=0 and x=L share the same derivative.
        du[0] = (u[1] - u[-2]) / (2.0 * dx)
        du[-1] = du[0]
        return du

    # Non-periodic: central in interior, one-sided at boundaries via gradient
    return np.gradient(u, dx, edge_order=2)


def _central_second_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Second-order central second derivative in 1D.

    - For periodic domains, use a wrapped central stencil.
    - For non-periodic domains, approximate via a second application of
      ``np.gradient`` (which is central in the interior).
    """
    if u.size < 3:
        return np.zeros_like(u, dtype=float)

    if u.size < 3:
        return np.zeros_like(u, dtype=float)

    if periodic:
        d2 = np.zeros_like(u, dtype=float)
        # Interior
        d2[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        # Periodic ends: wrap to penultimate point
        d2[0] = (u[1] - 2.0 * u[0] + u[-2]) / (dx * dx)
        d2[-1] = d2[0]
        return d2

    # Non-periodic: approximate u_xx by gradient of gradient
    return np.gradient(np.gradient(u, dx, edge_order=2), dx, edge_order=2)


def _central_third_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Second-order central third derivative in 1D.
    
    Uses the stencil: u_xxx ≈ (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2]) / (2*dx^3)
    
    Parameters
    ----------
    u:
        1D array of function values.
    dx:
        Grid spacing.
    periodic:
        Whether the domain is periodic.
        
    Returns
    -------
    Array
        Third derivative approximation with same shape as u.
    """
    if u.size < 5:
        return np.zeros_like(u, dtype=float)
    
    d3 = np.zeros_like(u, dtype=float)
    
    if periodic:
        # Interior points: central difference stencil
        # u_xxx[i] ≈ (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2]) / (2*dx^3)
        d3[2:-2] = (u[4:] - 2.0*u[3:-1] + 2.0*u[1:-3] - u[:-4]) / (2.0 * dx**3)
        
        # Handle boundaries with periodic wrapping
        # Point 0: wrap to end
        d3[0] = (u[2] - 2.0*u[1] + 2.0*u[-2] - u[-3]) / (2.0 * dx**3)
        # Point 1: wrap to end
        d3[1] = (u[3] - 2.0*u[2] + 2.0*u[0] - u[-2]) / (2.0 * dx**3)
        # Point -2: wrap to beginning
        d3[-2] = (u[0] - 2.0*u[-1] + 2.0*u[-3] - u[-4]) / (2.0 * dx**3)
        # Point -1: same as point 0 (periodic)
        d3[-1] = d3[0]
        return d3
    
    # Non-periodic: use one-sided differences at boundaries
    # Interior: central difference
    d3[2:-2] = (u[4:] - 2.0*u[3:-1] + 2.0*u[1:-3] - u[:-4]) / (2.0 * dx**3)
    
    # Left boundary: forward differences
    # Third-order forward: u_xxx[0] ≈ (-u[3] + 3*u[2] - 3*u[1] + u[0]) / dx^3
    d3[0] = (-u[3] + 3.0*u[2] - 3.0*u[1] + u[0]) / (dx**3)
    d3[1] = (-u[4] + 3.0*u[3] - 3.0*u[2] + u[1]) / (dx**3)
    
    # Right boundary: backward differences
    # Third-order backward: u_xxx[-1] ≈ (u[-1] - 3*u[-2] + 3*u[-3] - u[-4]) / dx^3
    d3[-2] = (u[-1] - 3.0*u[-2] + 3.0*u[-3] - u[-4]) / (dx**3)
    d3[-1] = (u[-1] - 3.0*u[-2] + 3.0*u[-3] - u[-4]) / (dx**3)
    
    return d3


def _eval_coeff(
    coeff: float | str | Callable[[Array, float], Array],
    x: Array,
    t: float,
) -> Array:
    """
    Evaluate a scalar/space- or time-dependent coefficient.

    Allowed forms:
    - numeric scalar -> broadcast
    - callable(x, t) -> array
    - string expression in x, t using numpy as ``np`` (sandboxed eval)
    """
    if isinstance(coeff, (int, float)):
        return float(coeff) * np.ones_like(x, dtype=float)

    if callable(coeff):
        return np.asarray(coeff(x, t), dtype=float)

    if isinstance(coeff, str):
        env = {"np": np, "x": x, "t": t}
        return np.asarray(eval(coeff, {"__builtins__": {}}, env), dtype=float)

    raise TypeError(f"Unsupported coefficient type: {type(coeff)!r}")


@dataclass
class Diffusion(Operator):
    """
    Simple diffusion operator:  nu * u_xx.

    Parameters
    ----------
    nu:
        Diffusivity coefficient. May be:
        - a scalar,
        - a callable ``nu(x, t)``,
        - or a string expression in x, t (e.g. ``\"0.1 + 0.05 * x\"``).
    """

    nu: float | str | Callable[[Array, float], Array]

    def apply(self, u_full: Array, domain: Domain1D, t: float) -> Array:
        x = domain.x
        dx = domain.dx
        periodic = domain.periodic

        u_xx = _central_second_derivative(u_full, dx, periodic)
        nu_vals = _eval_coeff(self.nu, x, t)
        return nu_vals * u_xx


@dataclass
class Advection(Operator):
    """
    Linear advection operator: -a * u_x.

    Parameters
    ----------
    a:
        Advection speed. May be scalar, callable a(x, t), or string expression.
    """

    a: float | str | Callable[[Array, float], Array]

    def apply(self, u_full: Array, domain: Domain1D, t: float) -> Array:
        x = domain.x
        dx = domain.dx
        periodic = domain.periodic

        u_x = _central_first_derivative(u_full, dx, periodic)
        a_vals = _eval_coeff(self.a, x, t)
        return -a_vals * u_x


class _SumOperator(Operator):
    """Internal helper that represents the sum of multiple operators."""

    def __init__(self, operators: Sequence[Operator]) -> None:
        if not operators:
            raise ValueError("At least one operator is required.")
        self._operators: List[Operator] = list(operators)

    def apply(self, u_full: Array, domain, t: float) -> Array:
        result = np.zeros_like(u_full, dtype=float)
        for op in self._operators:
            result += op.apply(u_full, domain, t)
        return result


def sum_operators(ops: Iterable[Operator]) -> Operator:
    """
    Combine several operators into a single Operator that returns their sum.
    """
    return _SumOperator(list(ops))


@dataclass
class ExpressionOperator(Operator):
    """
    Operator defined by a symbolic expression in (u, ux, uxx, uxxx, x, t, params).

    The expression is parsed once using sympy, then compiled to a fast
    NumPy function with ``sympy.lambdify``.

    Example
    -------
    Burgers'-type operator:
        expr = "-u*ux + nu*uxx"
        params = {"nu": 0.1}
    
    KdV-type operator with third-order term:
        expr = "-u*ux + nu*uxx - alpha*uxxx"
        params = {"nu": 0.1, "alpha": 0.01}
    """

    expr_string: str
    params: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}

        # Define core symbols (including uxxx for third-order derivatives)
        u, ux, uxx, uxxx, x, t = sp.symbols("u ux uxx uxxx x t")
        self._base_symbols = {"u": u, "ux": ux, "uxx": uxx, "uxxx": uxxx, "x": x, "t": t}

        # Parameter symbols
        self._param_symbols = {
            name: sp.symbols(name) for name in self.params.keys()
        }

        # Allow a small set of standard functions
        allowed_funcs = {
            name: getattr(sp, name)
            for name in ["sin", "cos", "exp", "log", "tanh", "sqrt"]
        }

        local_dict = {
            **self._base_symbols,
            **self._param_symbols,
            **allowed_funcs,
        }

        # Parse safely with sympy
        self._expr = sp.sympify(self.expr_string, locals=local_dict)

        # Build argument list and compile
        args = list(self._base_symbols.values()) + list(
            self._param_symbols.values()
        )
        self._param_order = list(self._param_symbols.keys())

        self._func = sp.lambdify(
            args,
            self._expr,
            modules=["numpy"],
        )

    def apply(self, u_full: Array, domain: Domain1D, t: float) -> Array:
        x = domain.x
        dx = domain.dx
        periodic = domain.periodic

        u = np.asarray(u_full, dtype=float)
        ux = _central_first_derivative(u, dx, periodic)
        uxx = _central_second_derivative(u, dx, periodic)
        uxxx = _central_third_derivative(u, dx, periodic)

        # Parameter values in a stable order
        param_vals = [self.params[name] for name in self._param_order]

        result = self._func(u, ux, uxx, uxxx, x, t, *param_vals)
        return np.asarray(result, dtype=float)


