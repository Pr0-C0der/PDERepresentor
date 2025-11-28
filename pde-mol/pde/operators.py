from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import sympy as sp

from .domain import Domain1D, Domain2D


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
    Operator defined by a symbolic expression in (u, ux, uxx, x, t, params).

    The expression is parsed once using sympy, then compiled to a fast
    NumPy function with ``sympy.lambdify``.

    Example
    -------
    Burgers'-type operator:
        expr = "-u*ux + nu*uxx"
        params = {"nu": 0.1}
    """

    expr_string: str
    params: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}

        # Define core symbols
        u, ux, uxx, x, t = sp.symbols("u ux uxx x t")
        self._base_symbols = {"u": u, "ux": ux, "uxx": uxx, "x": x, "t": t}

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

        # Parameter values in a stable order
        param_vals = [self.params[name] for name in self._param_order]

        result = self._func(u, ux, uxx, x, t, *param_vals)
        return np.asarray(result, dtype=float)


# ---------------------------------------------------------------------------
# 2D operators
# ---------------------------------------------------------------------------


def _second_derivative_axis(u: Array, d: float, axis: int, periodic: bool) -> Array:
    """
    Second derivative along a given axis for 2D arrays.

    Uses a wrapped central stencil when ``periodic`` is True, otherwise
    approximates via ``np.gradient`` twice.
    """
    if u.ndim != 2:
        raise ValueError("Expected a 2D array for _second_derivative_axis.")

    if periodic:
        d2 = np.zeros_like(u, dtype=float)
        if axis == 1:  # x-direction (columns)
            # Interior columns
            d2[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (d * d)
            # Periodic ends: wrap to the penultimate column; last column shares derivative with first
            d2[:, 0] = (u[:, 1] - 2.0 * u[:, 0] + u[:, -2]) / (d * d)
            d2[:, -1] = d2[:, 0]
        elif axis == 0:  # y-direction (rows)
            d2[1:-1, :] = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (d * d)
            d2[0, :] = (u[1, :] - 2.0 * u[0, :] + u[-2, :]) / (d * d)
            d2[-1, :] = d2[0, :]
        else:
            raise ValueError("axis must be 0 or 1 for 2D arrays.")
        return d2

    return np.gradient(np.gradient(u, d, axis=axis, edge_order=2), d, axis=axis, edge_order=2)


def _laplacian_2d(u: Array, dx: float, dy: float, periodic_x: bool, periodic_y: bool) -> Array:
    """
    2D Laplacian u_xx + u_yy for a 2D array ``u``.
    """
    if u.ndim != 2:
        raise ValueError("Expected a 2D array for _laplacian_2d.")

    u_xx = _second_derivative_axis(u, dx, axis=1, periodic=periodic_x)
    u_yy = _second_derivative_axis(u, dy, axis=0, periodic=periodic_y)
    return u_xx + u_yy


@dataclass
class Diffusion2D(Operator):
    """
    2D diffusion operator:  nu * (u_xx + u_yy).

    Currently ``nu`` is assumed to be a scalar.
    """

    nu: float

    def apply(self, u_full: Array, domain, t: float) -> Array:
        if not isinstance(domain, Domain2D):
            raise TypeError("Diffusion2D requires a Domain2D instance.")

        U = domain.unflatten(u_full)
        lap = _laplacian_2d(U, domain.dx, domain.dy, domain.periodic_x, domain.periodic_y)
        return (float(self.nu) * lap).ravel()


@dataclass
class ExpressionOperator2D(Operator):
    """
    2D expression-based operator in terms of (u, ux, uy, uxx, uyy, x, y, t, params).

    Example:
        expr = "nu*(uxx + uyy) - u*ux"
        params = {"nu": 0.1}
    """

    expr_string: str
    params: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}

        u, ux, uy, uxx, uyy, x, y, t = sp.symbols("u ux uy uxx uyy x y t")
        self._base_symbols = {
            "u": u,
            "ux": ux,
            "uy": uy,
            "uxx": uxx,
            "uyy": uyy,
            "x": x,
            "y": y,
            "t": t,
        }

        self._param_symbols = {name: sp.symbols(name) for name in self.params.keys()}

        allowed_funcs = {
            name: getattr(sp, name)
            for name in ["sin", "cos", "exp", "log", "tanh", "sqrt"]
        }

        local_dict = {
            **self._base_symbols,
            **self._param_symbols,
            **allowed_funcs,
        }

        self._expr = sp.sympify(self.expr_string, locals=local_dict)

        args = list(self._base_symbols.values()) + list(self._param_symbols.values())
        self._param_order = list(self._param_symbols.keys())

        self._func = sp.lambdify(args, self._expr, modules=["numpy"])

    def apply(self, u_full: Array, domain, t: float) -> Array:
        if not isinstance(domain, Domain2D):
            raise TypeError("ExpressionOperator2D requires a Domain2D instance.")

        U = domain.unflatten(u_full)
        dx, dy = domain.dx, domain.dy

        # First derivatives
        if domain.periodic_x:
            ux = np.zeros_like(U, dtype=float)
            ux[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * dx)
            ux[:, 0] = (U[:, 1] - U[:, -2]) / (2.0 * dx)
            ux[:, -1] = ux[:, 0]
        else:
            ux = np.gradient(U, dx, axis=1, edge_order=2)

        if domain.periodic_y:
            uy = np.zeros_like(U, dtype=float)
            uy[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2.0 * dy)
            uy[0, :] = (U[1, :] - U[-2, :]) / (2.0 * dy)
            uy[-1, :] = uy[0, :]
        else:
            uy = np.gradient(U, dy, axis=0, edge_order=2)

        uxx = _second_derivative_axis(U, dx, axis=1, periodic=domain.periodic_x)
        uyy = _second_derivative_axis(U, dy, axis=0, periodic=domain.periodic_y)

        X = domain.X
        Y = domain.Y

        param_vals = [self.params[name] for name in self._param_order]

        result = self._func(U, ux, uy, uxx, uyy, X, Y, t, *param_vals)
        return np.asarray(result, dtype=float).ravel()


