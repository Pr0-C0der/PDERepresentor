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


