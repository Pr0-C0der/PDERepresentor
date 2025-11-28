from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bc import (
    BoundaryCondition,
    DirichletLeft,
    DirichletRight,
    NeumannLeft,
    NeumannRight,
    Periodic,
)
from .domain import Domain1D
from .ic import InitialCondition
from .operators import Advection, Diffusion, ExpressionOperator, Operator
from .problem import PDEProblem


JsonDict = Dict[str, Any]


def _parse_domain(cfg: JsonDict) -> Domain1D:
    """Parse the ``domain`` section."""
    domain_type = cfg.get("type", "1d")
    if domain_type != "1d":
        raise ValueError(f"Only '1d' domain type is supported, got {domain_type!r}.")

    x0 = cfg["x0"]
    x1 = cfg["x1"]
    nx = cfg["nx"]
    periodic = bool(cfg.get("periodic", False))
    return Domain1D(x0=x0, x1=x1, nx=nx, periodic=periodic)


def _parse_initial_condition(cfg: JsonDict) -> InitialCondition:
    """Parse the ``initial_condition`` section."""
    ic_type = cfg.get("type", "expression")
    if ic_type == "expression":
        expr = cfg["expr"]
        return InitialCondition.from_expression(expr)
    elif ic_type == "values":
        values = cfg["values"]
        return InitialCondition.from_values(values)
    else:
        raise ValueError(f"Unknown initial condition type {ic_type!r}.")


def _parse_bc_one(side_cfg: Optional[JsonDict], side: str) -> Optional[BoundaryCondition]:
    """Parse a single boundary condition specification."""
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
        else:
            raise ValueError(f"Unknown Dirichlet side {side!r}.")

    if bc_type == "neumann":
        value = side_cfg.get("derivative_value")
        expr = side_cfg.get("expr")
        if side == "left":
            return NeumannLeft(derivative_value=value, expr=expr)
        elif side == "right":
            return NeumannRight(derivative_value=value, expr=expr)
        else:
            raise ValueError(f"Unknown Neumann side {side!r}.")

    if bc_type == "periodic":
        return Periodic()

    raise ValueError(f"Unknown boundary condition type {bc_type!r} for side {side!r}.")


def _parse_boundary_conditions(cfg: JsonDict) -> (Optional[BoundaryCondition], Optional[BoundaryCondition]):
    """Parse the optional ``boundary_conditions`` section."""
    if cfg is None:
        return None, None

    left_cfg = cfg.get("left")
    right_cfg = cfg.get("right")

    bc_left = _parse_bc_one(left_cfg, "left") if left_cfg is not None else None
    bc_right = _parse_bc_one(right_cfg, "right") if right_cfg is not None else None
    return bc_left, bc_right


def _parse_operator(op_cfg: JsonDict) -> Operator:
    """Parse a single operator configuration."""
    op_type = op_cfg["type"].lower()

    if op_type == "diffusion":
        if "nu" not in op_cfg:
            raise ValueError("Diffusion operator requires 'nu' coefficient.")
        nu = op_cfg["nu"]
        return Diffusion(nu)

    if op_type == "advection":
        if "a" not in op_cfg:
            raise ValueError("Advection operator requires 'a' coefficient.")
        a = op_cfg["a"]
        return Advection(a)

    if op_type in ("expression", "expression_operator"):
        expr = op_cfg["expr"]
        params = op_cfg.get("params", {})
        return ExpressionOperator(expr_string=expr, params=params)

    raise ValueError(f"Unknown operator type {op_type!r}.")


def _parse_operators(cfg: JsonDict) -> List[Operator]:
    ops_cfg = cfg.get("operators")
    if not ops_cfg:
        raise ValueError("JSON configuration must contain a non-empty 'operators' list.")
    return [_parse_operator(op_cfg) for op_cfg in ops_cfg]


def build_problem_from_dict(config: JsonDict) -> PDEProblem:
    """
    Build a PDEProblem from an in-memory JSON-like dictionary.

    This is the core entry point; ``load_from_json`` is a small wrapper around it.
    """
    domain_cfg = config["domain"]
    ic_cfg = config["initial_condition"]
    bc_cfg = config.get("boundary_conditions", {})

    domain = _parse_domain(domain_cfg)
    ic = _parse_initial_condition(ic_cfg)
    operators = _parse_operators(config)
    bc_left, bc_right = _parse_boundary_conditions(bc_cfg)

    problem = PDEProblem(
        domain=domain,
        operators=operators,
        ic=ic,
        bc_left=bc_left,
        bc_right=bc_right,
    )
    return problem


def load_from_json(path: str | Path) -> PDEProblem:
    """
    Load a PDEProblem from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON configuration file.
    """
    p = Path(path)
    with p.open("r", encoding="utf8") as f:
        config = json.load(f)
    return build_problem_from_dict(config)

