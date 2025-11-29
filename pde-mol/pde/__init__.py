"""
Core package for a JSON-driven Method-of-Lines PDE solver.

This initial version exposes the fundamental data structures:

- Domain1D: simple 1D spatial domain and grid
- InitialCondition: expression or array-based initial conditions
- Boundary conditions: Dirichlet, Periodic, and Neumann placeholders
"""

from .domain import Domain1D, Domain2D
from .ic import InitialCondition
from .bc import (
    BoundaryCondition,
    DirichletLeft,
    DirichletRight,
    Periodic,
    NeumannLeft,
    NeumannRight,
    RobinLeft,
    RobinRight,
)
from .operators import (
    Operator,
    Diffusion,
    Advection,
    ExpressionOperator,
    Diffusion2D,
    ExpressionOperator2D,
    sum_operators,
)
from .problem import PDEProblem

__all__ = [
    "Domain1D",
    "Domain2D",
    "InitialCondition",
    "BoundaryCondition",
    "DirichletLeft",
    "DirichletRight",
    "Periodic",
    "NeumannLeft",
    "NeumannRight",
    "RobinLeft",
    "RobinRight",
    "Operator",
    "Diffusion",
    "Advection",
    "ExpressionOperator",
    "Diffusion2D",
    "ExpressionOperator2D",
    "sum_operators",
    "PDEProblem",
]

