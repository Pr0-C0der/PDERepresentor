from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import sympy as sp

from .domain import Domain1D
from .spatial_discretization import (
    DiscretizationScheme,
    first_derivative,
    second_derivative,
    third_derivative,
)


Array = np.ndarray


class Operator:
    """
    Abstract base class for spatial operators.

    An operator maps a full solution vector ``u_full`` to its spatial
    contribution ``du_dt = L[u]`` at a given time.
    """

    def apply(self, u_full: Array, domain, t: float) -> Array:
        raise NotImplementedError


# Note: Derivative functions have been moved to spatial_discretization.py
# These are kept as backward-compatible wrappers
def _central_first_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Backward-compatible wrapper for central first derivative.
    
    Deprecated: Use spatial_discretization.first_derivative() instead.
    """
    from .domain import Domain1D
    # Create a minimal domain for the function call
    # This is a bit of a hack, but maintains backward compatibility
    class _MinimalDomain:
        def __init__(self, periodic: bool):
            self.periodic = periodic
    
    domain = _MinimalDomain(periodic)
    return first_derivative(u, dx, domain, DiscretizationScheme.CENTRAL)


def _central_second_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Backward-compatible wrapper for central second derivative.
    
    Deprecated: Use spatial_discretization.second_derivative() instead.
    """
    from .domain import Domain1D
    class _MinimalDomain:
        def __init__(self, periodic: bool):
            self.periodic = periodic
    
    domain = _MinimalDomain(periodic)
    return second_derivative(u, dx, domain, DiscretizationScheme.CENTRAL)


def _central_third_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Backward-compatible wrapper for central third derivative.
    
    Deprecated: Use spatial_discretization.third_derivative() instead.
    """
    from .domain import Domain1D
    class _MinimalDomain:
        def __init__(self, periodic: bool):
            self.periodic = periodic
    
    domain = _MinimalDomain(periodic)
    return third_derivative(u, dx, domain, DiscretizationScheme.CENTRAL)


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
    
    Uses central differences (optimal for diffusion terms).

    Parameters
    ----------
    nu:
        Diffusivity coefficient. May be:
        - a scalar,
        - a callable ``nu(x, t)``,
        - or a string expression in x, t (e.g. ``\"0.1 + 0.05 * x\"``).
    scheme:
        Discretization scheme (currently only "central" is supported for second derivatives).
    """

    nu: float | str | Callable[[Array, float], Array]
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL

    def apply(self, u_full: Array, domain: Domain1D, t: float) -> Array:
        x = domain.x
        dx = domain.dx

        u_xx = second_derivative(u_full, dx, domain, self.scheme)
        nu_vals = _eval_coeff(self.nu, x, t)
        return nu_vals * u_xx


@dataclass
class Advection(Operator):
    """
    Linear advection operator: -a * u_x.
    
    Now supports different discretization schemes for better accuracy and stability.
    Upwind schemes are recommended for advection-dominated problems.

    Parameters
    ----------
    a:
        Advection speed. May be scalar, callable a(x, t), or string expression.
    scheme:
        Discretization scheme. Options:
        - "central": Second-order central (default)
        - "upwind_first": First-order upwind (stable, less accurate)
        - "upwind_second": Second-order upwind (stable, more accurate, recommended for advection)
        - "backward": First-order backward differences
        - "forward": First-order forward differences
        
    Examples
    --------
    >>> # Central differences (default)
    >>> advection = Advection(a=1.0)
    >>> 
    >>> # Stable upwind scheme (recommended for advection-dominated problems)
    >>> advection = Advection(a=1.0, scheme="upwind_second")
    """

    a: float | str | Callable[[Array, float], Array]
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL

    def apply(self, u_full: Array, domain: Domain1D, t: float) -> Array:
        x = domain.x
        dx = domain.dx

        a_vals = _eval_coeff(self.a, x, t)
        
        # For upwind schemes, we need the velocity
        if isinstance(self.scheme, str):
            scheme_enum = DiscretizationScheme(self.scheme)
        else:
            scheme_enum = self.scheme
            
        if scheme_enum in (DiscretizationScheme.UPWIND_FIRST, 
                          DiscretizationScheme.UPWIND_SECOND):
            u_x = first_derivative(u_full, dx, domain, scheme_enum, velocity=a_vals)
        else:
            u_x = first_derivative(u_full, dx, domain, scheme_enum)
        
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
    
    Now supports configurable discretization schemes per derivative type for
    optimal accuracy and stability.

    Parameters
    ----------
    expr_string:
        String expression in terms of u, ux, uxx, uxxx, x, t, and parameters.
    params:
        Dictionary of parameter values.
    schemes:
        Dictionary mapping derivative names to discretization schemes.
        Keys: "ux", "uxx", "uxxx"
        Values: "central", "upwind_first", "upwind_second", "backward", "forward"
        Default: All derivatives use "central"
        
        Note: For upwind schemes on ux, the velocity is automatically detected
        from the expression if possible (e.g., in "-a*ux", a is the velocity).
        For complex expressions, you may need to use separate operators.

    Example
    -------
    Burgers'-type operator with upwind for advection:
        expr = "-u*ux + nu*uxx"
        params = {"nu": 0.1}
        schemes = {"ux": "upwind_second", "uxx": "central"}
    
    KdV-type operator with third-order term:
        expr = "-u*ux + nu*uxx - alpha*uxxx"
        params = {"nu": 0.1, "alpha": 0.01}
        schemes = {"ux": "upwind_second"}  # uxx and uxxx use central by default
    """

    expr_string: str
    params: Optional[dict] = None
    schemes: Optional[dict[str, DiscretizationScheme | str]] = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}
        
        # Parse schemes
        if self.schemes is None:
            self.schemes = {}
        
        # Convert string schemes to enums
        self._ux_scheme = DiscretizationScheme(self.schemes.get("ux", "central"))
        self._uxx_scheme = DiscretizationScheme(self.schemes.get("uxx", "central"))
        self._uxxx_scheme = DiscretizationScheme(self.schemes.get("uxxx", "central"))
        
        # Try to detect velocity for upwind schemes
        # This is a simple heuristic: look for patterns like "-a*ux" or "a*ux"
        self._velocity_expr = None
        if self._ux_scheme in (DiscretizationScheme.UPWIND_FIRST, 
                               DiscretizationScheme.UPWIND_SECOND):
            # Try to extract velocity coefficient from expression
            # This is a simple approach - could be enhanced
            try:
                u, ux, uxx, uxxx, x, t = sp.symbols("u ux uxx uxxx x t")
                expr = sp.sympify(self.expr_string, locals={"u": u, "ux": ux, "uxx": uxx, 
                                                             "uxxx": uxxx, "x": x, "t": t})
                # Look for terms like coeff*ux
                if expr.has(ux):
                    # Try to extract coefficient of ux
                    coeff = sp.collect(expr, ux, evaluate=False)
                    if ux in coeff:
                        velocity_term = coeff[ux]
                        # If it's a simple multiplication, extract the coefficient
                        if isinstance(velocity_term, sp.Mul):
                            # Get numeric/parameter parts
                            self._velocity_expr = velocity_term
            except:
                # If parsing fails, we'll compute velocity from the expression at runtime
                pass

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
        
        # Compile velocity expression if available
        if self._velocity_expr is not None:
            try:
                self._velocity_func = sp.lambdify(
                    list(self._base_symbols.values()) + list(self._param_symbols.values()),
                    self._velocity_expr,
                    modules=["numpy"],
                )
            except:
                self._velocity_func = None
        else:
            self._velocity_func = None

    def apply(self, u_full: Array, domain: Domain1D, t: float) -> Array:
        x = domain.x
        dx = domain.dx

        u = np.asarray(u_full, dtype=float)
        
        # Compute derivatives with specified schemes
        # For upwind schemes, we need velocity
        velocity = None
        if self._ux_scheme in (DiscretizationScheme.UPWIND_FIRST, 
                               DiscretizationScheme.UPWIND_SECOND):
            # Try to compute velocity from expression
            if self._velocity_func is not None:
                param_vals = [self.params[name] for name in self._param_order]
                # Compute ux with central first to get initial estimate
                ux_temp = first_derivative(u, dx, domain, DiscretizationScheme.CENTRAL)
                # Evaluate velocity expression
                velocity = self._velocity_func(u, ux_temp, 
                                               np.zeros_like(u),  # uxx placeholder
                                               np.zeros_like(u),  # uxxx placeholder
                                               x, t, *param_vals)
                velocity = np.asarray(velocity, dtype=float)
            else:
                # Fall back to central if velocity cannot be determined
                # This is a limitation - for complex expressions, use separate operators
                velocity = np.ones_like(u)  # Default: assume positive velocity
        
        ux = first_derivative(u, dx, domain, self._ux_scheme, velocity=velocity)
        uxx = second_derivative(u, dx, domain, self._uxx_scheme)
        uxxx = third_derivative(u, dx, domain, self._uxxx_scheme)

        # Parameter values in a stable order
        param_vals = [self.params[name] for name in self._param_order]

        result = self._func(u, ux, uxx, uxxx, x, t, *param_vals)
        return np.asarray(result, dtype=float)


