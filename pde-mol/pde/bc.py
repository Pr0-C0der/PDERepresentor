from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .domain import Domain1D


class BoundaryCondition:
    """
    Base class for boundary conditions.

    Subclasses can implement :meth:`apply_to_full` to modify the full state
    vector ``u_full`` in-place at a given time ``t``.
    """

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        """
        Apply this boundary condition in-place to ``u_full``.

        Parameters
        ----------
        u_full:
            1D solution array including boundary points.
        t:
            Current time.
        domain:
            Associated spatial domain.
        """
        raise NotImplementedError


@dataclass
class DirichletLeft(BoundaryCondition):
    """
    Dirichlet boundary condition at the left boundary x = x0.

    The value can be given either as:
    - a constant (float),
    - or a string expression in terms of ``t`` and ``np`` (e.g. ``\"0.0\"``).
    """

    value: Optional[float] = None
    expr: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.expr is None):
            raise ValueError("DirichletLeft expects exactly one of value or expr.")

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        if self.value is not None:
            u_full[0] = float(self.value)
        else:
            env = {"np": np, "t": t}
            u_full[0] = float(eval(self.expr, {"__builtins__": {}}, env))


@dataclass
class DirichletRight(BoundaryCondition):
    """
    Dirichlet boundary condition at the right boundary x = x1.
    """

    value: Optional[float] = None
    expr: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.expr is None):
            raise ValueError("DirichletRight expects exactly one of value or expr.")

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        if self.value is not None:
            u_full[-1] = float(self.value)
        else:
            env = {"np": np, "t": t}
            u_full[-1] = float(eval(self.expr, {"__builtins__": {}}, env))


class Periodic(BoundaryCondition):
    """
    Periodic boundary condition.

    For now this is a marker class; most enforcement will happen in the
    differential operator construction. To be explicit we provide a no-op
    :meth:`apply_to_full`.
    """

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        # No-op: periodicity is typically handled by finite-difference stencil
        # construction rather than manual assignment at each step.
        return None


@dataclass
class NeumannLeft(BoundaryCondition):
    """
    Neumann condition at the left boundary x = x0 specifying u_x(x0, t) = q(t).

    This class does not directly modify ``u_full``; instead, helper functions
    (e.g. ``apply_neumann_ghosts``) use its value to construct ghost points or
    approximate boundary values for finite-difference stencils.
    """

    derivative_value: Optional[float] = None
    expr: Optional[str] = None

    def _evaluate_flux(self, t: float) -> float:
        if (self.derivative_value is None) == (self.expr is None):
            raise ValueError("NeumannLeft expects exactly one of derivative_value or expr.")
        if self.derivative_value is not None:
            return float(self.derivative_value)
        env = {"np": np, "t": t}
        return float(eval(self.expr, {"__builtins__": {}}, env))

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        # Neumann enforcement is handled via ghost-point helpers rather than
        # overwriting boundary values here.
        return None


@dataclass
class NeumannRight(BoundaryCondition):
    """
    Neumann condition at the right boundary x = x1 specifying u_x(x1, t) = q(t).
    """

    derivative_value: Optional[float] = None
    expr: Optional[str] = None

    def _evaluate_flux(self, t: float) -> float:
        if (self.derivative_value is None) == (self.expr is None):
            raise ValueError("NeumannRight expects exactly one of derivative_value or expr.")
        if self.derivative_value is not None:
            return float(self.derivative_value)
        env = {"np": np, "t": t}
        return float(eval(self.expr, {"__builtins__": {}}, env))

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        return None


@dataclass
class RobinLeft(BoundaryCondition):
    """
    General Robin boundary condition at the left boundary x = x0 of the form

        a * u(x0, t) + b * u_x(x0, t) = c(t)

    This is enforced via a one-sided first-order finite-difference relation
    using the first interior point u[1]:

        (u[1] - u[0]) / dx ≈ u_x(x0, t)

    which is rearranged to solve for u[0].

    Parameters
    ----------
    a:
        Coefficient of u in the boundary condition. Defaults to None.
    b:
        Coefficient of u_x in the boundary condition. Defaults to 1.0.
    c:
        Right-hand side value (constant). Defaults to None.
    c_expr:
        Right-hand side expression in terms of ``t`` and ``np``. Defaults to None.
    h:
        (Backward compatibility) Heat/mass transfer coefficient. If provided with
        u_env, converts to general form: a = h, b = 1, c = h * u_env.
    u_env:
        (Backward compatibility) Environmental value. Used with h for backward
        compatibility.

    Examples
    --------
    # General form: 2*u + 3*u_x = 5
    RobinLeft(a=2.0, b=3.0, c=5.0)

    # Time-dependent: u + u_x = sin(t)
    RobinLeft(a=1.0, b=1.0, c_expr="np.sin(t)")

    # Backward compatibility: u_x = -h*(u - u_env)
    RobinLeft(h=1.0, u_env=0.2)  # equivalent to a=1.0, b=1.0, c=0.2
    """

    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    c_expr: Optional[str] = None
    # Backward compatibility fields
    h: Optional[float] = None
    u_env: Optional[float] = None

    def __post_init__(self) -> None:
        # Backward compatibility: convert h, u_env to general form
        if self.h is not None and self.u_env is not None:
            if self.a is not None or self.b is not None or self.c is not None or self.c_expr is not None:
                raise ValueError("Cannot specify both (h, u_env) and (a, b, c/c_expr) in RobinLeft.")
            self.a = self.h
            self.b = 1.0
            self.c = self.h * self.u_env
            self.c_expr = None

        # Validate general form parameters
        if self.a is None:
            raise ValueError("RobinLeft requires 'a' coefficient (or 'h' for backward compatibility).")
        if self.b is None:
            self.b = 1.0  # Default b = 1
        if self.b == 0.0:
            raise ValueError("RobinLeft: b cannot be zero (would be Dirichlet, not Robin).")
        if (self.c is None) == (self.c_expr is None):
            raise ValueError("RobinLeft expects exactly one of c or c_expr.")

    def _evaluate_c(self, t: float) -> float:
        """Evaluate the right-hand side c(t)."""
        if self.c is not None:
            return float(self.c)
        env = {"np": np, "t": t}
        return float(eval(self.c_expr, {"__builtins__": {}}, env))

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        if u_full.size < 2:
            return
        dx = domain.dx
        u1 = float(u_full[1])
        c_val = self._evaluate_c(t)
        # From: a * u[0] + b * (u[1] - u[0]) / dx = c
        # Rearranging: (a - b/dx) * u[0] = c - (b/dx) * u[1]
        # So: u[0] = (c - (b/dx) * u[1]) / (a - b/dx)
        b_over_dx = self.b / dx
        denom = self.a - b_over_dx
        if abs(denom) < 1e-14:
            raise ZeroDivisionError(f"Invalid RobinLeft parameters: a - b/dx == {denom} (too close to zero).")
        u_full[0] = (c_val - b_over_dx * u1) / denom


@dataclass
class RobinRight(BoundaryCondition):
    """
    General Robin boundary condition at the right boundary x = x1 of the form

        a * u(x1, t) + b * u_x(x1, t) = c(t)

    This is enforced via a one-sided first-order finite-difference relation
    using the penultimate point u[N-2]:

        (u[N-1] - u[N-2]) / dx ≈ u_x(x1, t)

    which is rearranged to solve for u[N-1].

    Parameters
    ----------
    a:
        Coefficient of u in the boundary condition. Defaults to None.
    b:
        Coefficient of u_x in the boundary condition. Defaults to 1.0.
    c:
        Right-hand side value (constant). Defaults to None.
    c_expr:
        Right-hand side expression in terms of ``t`` and ``np``. Defaults to None.
    h:
        (Backward compatibility) Heat/mass transfer coefficient. If provided with
        u_env, converts to general form: a = h, b = 1, c = h * u_env.
    u_env:
        (Backward compatibility) Environmental value. Used with h for backward
        compatibility.

    Examples
    --------
    # General form: 2*u + 3*u_x = 5
    RobinRight(a=2.0, b=3.0, c=5.0)

    # Time-dependent: u + u_x = sin(t)
    RobinRight(a=1.0, b=1.0, c_expr="np.sin(t)")

    # Backward compatibility: u_x = -h*(u - u_env)
    RobinRight(h=1.0, u_env=0.2)  # equivalent to a=1.0, b=1.0, c=0.2
    """

    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    c_expr: Optional[str] = None
    # Backward compatibility fields
    h: Optional[float] = None
    u_env: Optional[float] = None

    def __post_init__(self) -> None:
        # Backward compatibility: convert h, u_env to general form
        if self.h is not None and self.u_env is not None:
            if self.a is not None or self.b is not None or self.c is not None or self.c_expr is not None:
                raise ValueError("Cannot specify both (h, u_env) and (a, b, c/c_expr) in RobinRight.")
            # For u_x = -h*(u - u_env) at right boundary:
            # u_x = -h*u + h*u_env
            # Rearranging: h*u + u_x = h*u_env
            # But for right boundary, we use backward difference: u_x ≈ (u[-1] - u[-2])/dx
            # So: h*u[-1] + (u[-1] - u[-2])/dx = h*u_env
            # This gives: a = h, b = 1, c = h*u_env
            self.a = self.h
            self.b = 1.0
            self.c = self.h * self.u_env
            self.c_expr = None

        # Validate general form parameters
        if self.a is None:
            raise ValueError("RobinRight requires 'a' coefficient (or 'h' for backward compatibility).")
        if self.b is None:
            self.b = 1.0  # Default b = 1
        if self.b == 0.0:
            raise ValueError("RobinRight: b cannot be zero (would be Dirichlet, not Robin).")
        if (self.c is None) == (self.c_expr is None):
            raise ValueError("RobinRight expects exactly one of c or c_expr.")

    def _evaluate_c(self, t: float) -> float:
        """Evaluate the right-hand side c(t)."""
        if self.c is not None:
            return float(self.c)
        env = {"np": np, "t": t}
        return float(eval(self.c_expr, {"__builtins__": {}}, env))

    def apply_to_full(self, u_full: np.ndarray, t: float, domain: Domain1D) -> None:
        if u_full.size < 2:
            return
        dx = domain.dx
        u_nm2 = float(u_full[-2])
        c_val = self._evaluate_c(t)
        # From: a * u[-1] + b * (u[-1] - u[-2]) / dx = c
        # Rearranging: (a + b/dx) * u[-1] = c + (b/dx) * u[-2]
        # So: u[-1] = (c + (b/dx) * u[-2]) / (a + b/dx)
        b_over_dx = self.b / dx
        denom = self.a + b_over_dx
        if abs(denom) < 1e-14:
            raise ZeroDivisionError(f"Invalid RobinRight parameters: a + b/dx == {denom} (too close to zero).")
        u_full[-1] = (c_val + b_over_dx * u_nm2) / denom


def apply_neumann_ghosts(
    u_full: np.ndarray,
    bc_left: Optional[BoundaryCondition],
    bc_right: Optional[BoundaryCondition],
    t: float,
    domain: Domain1D,
) -> None:
    """
    Approximate Neumann boundary conditions by adjusting boundary values.

    For a Neumann flux q(t) = u_x at the boundary, a one-sided first-order
    approximation is:

        (u_1 - u_0) / dx ≈ q   at the left boundary
        (u_{N-1} - u_{N-2}) / dx ≈ q   at the right boundary

    which implies:

        u_0 = u_1 - dx * q
        u_{N-1} = u_{N-2} + dx * q

    This plays the same conceptual role as ghost points while keeping the
    physical grid size unchanged.
    """
    dx = domain.dx

    if isinstance(bc_left, NeumannLeft):
        q_left = bc_left._evaluate_flux(t)
        if u_full.size >= 2:
            u_full[0] = u_full[1] - dx * q_left

    if isinstance(bc_right, NeumannRight):
        q_right = bc_right._evaluate_flux(t)
        if u_full.size >= 2:
            u_full[-1] = u_full[-2] + dx * q_right


