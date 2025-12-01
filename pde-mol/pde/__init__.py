"""
Core package for a JSON-driven Method-of-Lines PDE solver.

This initial version exposes the fundamental data structures:

- Domain1D: simple 1D spatial domain and grid
- InitialCondition: expression or array-based initial conditions
- Boundary conditions: Dirichlet, Periodic, and Neumann placeholders
"""

from .domain import Domain1D
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
    sum_operators,
)
from .problem import PDEProblem, SecondOrderPDEProblem

# Dataset generation (optional, requires PyTorch)
try:
    from .dataset import ParameterRange, ParameterSampler, generate_dataset
    _DATASET_AVAILABLE = True
except ImportError:
    _DATASET_AVAILABLE = False

__all__ = [
    "Domain1D",
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
    "sum_operators",
    "PDEProblem",
    "SecondOrderPDEProblem",
]

if _DATASET_AVAILABLE:
    __all__.extend(["ParameterRange", "ParameterSampler", "generate_dataset"])

