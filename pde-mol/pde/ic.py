from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import numpy as np

from .domain import Domain1D

ArrayLike = Union[np.ndarray, Iterable[float]]


def _build_safe_eval_env() -> dict:
    """
    Very small, explicit namespace for expression evaluation.

    We expose only numpy under the name ``np``. The variable ``x`` will be
    injected at call time.
    """
    return {"np": np}


@dataclass
class InitialCondition:
    """
    Initial condition specified either as:

    - a string expression in terms of ``x`` and ``np`` (e.g. ``\"np.sin(np.pi*x)\"``), or
    - a callable ``f(x: np.ndarray) -> np.ndarray``, or
    - a concrete array-like with the same length as the domain grid.
    """

    expr: Optional[str] = None
    values: Optional[ArrayLike] = None
    func: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        modes = [self.expr is not None, self.values is not None, self.func is not None]
        if sum(modes) != 1:
            raise ValueError(
                "InitialCondition expects exactly one of expr, values, or func to be provided."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_expression(cls, expr: str) -> "InitialCondition":
        return cls(expr=expr)

    @classmethod
    def from_values(cls, values: ArrayLike) -> "InitialCondition":
        return cls(values=np.asarray(values, dtype=float))

    @classmethod
    def from_callable(cls, func: Callable[[np.ndarray], np.ndarray]) -> "InitialCondition":
        return cls(func=func)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, domain: Domain1D) -> np.ndarray:
        """
        Evaluate the initial condition on the grid defined by ``domain``.

        Expressions use the variable ``x``.
        """
        x = domain.x
        if self.expr is not None:
            env = _build_safe_eval_env()
            env["x"] = x
            result = eval(self.expr, {"__builtins__": {}}, env)
            return np.asarray(result, dtype=float)

        if self.func is not None:
            result = self.func(x)
            return np.asarray(result, dtype=float)

        arr = np.asarray(self.values, dtype=float)
        if arr.shape != x.shape:
            raise ValueError(
                f"Initial condition values have shape {arr.shape}, "
                f"but domain grid has shape {x.shape}."
            )
        return arr

