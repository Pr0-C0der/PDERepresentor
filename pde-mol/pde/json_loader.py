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
    RobinLeft,
    RobinRight,
)
from .domain import Domain1D, Domain2D
from .ic import InitialCondition
from .operators import (
    Advection,
    Diffusion,
    Diffusion2D,
    ExpressionOperator,
    ExpressionOperator2D,
    Operator,
)
from .problem import PDEProblem


JsonDict = Dict[str, Any]


def _parse_domain(cfg: JsonDict):
    """Parse the ``domain`` section."""
    domain_type = cfg.get("type", "1d").lower()

    if domain_type == "1d":
        x0 = cfg["x0"]
        x1 = cfg["x1"]
        nx = cfg["nx"]
        periodic = bool(cfg.get("periodic", False))
        return Domain1D(x0=x0, x1=x1, nx=nx, periodic=periodic)
    elif domain_type == "2d":
        x0 = cfg["x0"]
        x1 = cfg["x1"]
        nx = cfg["nx"]
        y0 = cfg["y0"]
        y1 = cfg["y1"]
        ny = cfg["ny"]
        periodic_x = bool(cfg.get("periodic_x", False))
        periodic_y = bool(cfg.get("periodic_y", False))
        return Domain2D(
            x0=x0, x1=x1, nx=nx, y0=y0, y1=y1, ny=ny,
            periodic_x=periodic_x, periodic_y=periodic_y
        )
    else:
        raise ValueError(f"Unsupported domain type {domain_type!r}. Use '1d' or '2d'.")


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

    if bc_type == "robin":
        # Support both general form (a, b, c/c_expr) and backward-compatible form (h, u_env)
        if "a" in side_cfg or "b" in side_cfg or "c" in side_cfg or "c_expr" in side_cfg:
            # General form: a * u + b * u_x = c(t)
            a = side_cfg.get("a")
            b = side_cfg.get("b")
            c = side_cfg.get("c")
            c_expr = side_cfg.get("c_expr")
            
            if a is not None:
                a = float(a)
            if b is not None:
                b = float(b)
            if c is not None:
                c = float(c)
            
            if side == "left":
                return RobinLeft(a=a, b=b, c=c, c_expr=c_expr)
            elif side == "right":
                return RobinRight(a=a, b=b, c=c, c_expr=c_expr)
            else:
                raise ValueError(f"Unknown Robin side {side!r}.")
        else:
            # Backward-compatible form: h and u_env
            h = float(side_cfg["h"])
            # Allow several possible keys for the environmental value.
            if "u_env" in side_cfg:
                u_env = float(side_cfg["u_env"])
            elif "value_env" in side_cfg:
                u_env = float(side_cfg["value_env"])
            elif "x_env" in side_cfg:
                u_env = float(side_cfg["x_env"])
            else:
                raise ValueError("Robin boundary condition requires 'u_env' (or 'value_env'/'x_env') for backward-compatible form, or 'a', 'b', 'c'/'c_expr' for general form.")

            if side == "left":
                return RobinLeft(h=h, u_env=u_env)
            elif side == "right":
                return RobinRight(h=h, u_env=u_env)
            else:
                raise ValueError(f"Unknown Robin side {side!r}.")

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


def _parse_operator(op_cfg: JsonDict, domain_type: str = "1d") -> Operator:
    """Parse a single operator configuration."""
    op_type = op_cfg["type"].lower()

    if op_type == "diffusion":
        if "nu" not in op_cfg:
            raise ValueError("Diffusion operator requires 'nu' coefficient.")
        nu = op_cfg["nu"]
        if domain_type == "2d":
            return Diffusion2D(nu)
        else:
            return Diffusion(nu)

    if op_type == "advection":
        if "a" not in op_cfg:
            raise ValueError("Advection operator requires 'a' coefficient.")
        a = op_cfg["a"]
        # Note: Advection2D not yet implemented, would need direction
        if domain_type == "2d":
            raise ValueError("2D advection operator not yet supported. Use expression operator instead.")
        return Advection(a)

    if op_type in ("expression", "expression_operator"):
        expr = op_cfg["expr"]
        params = op_cfg.get("params", {})
        if domain_type == "2d":
            return ExpressionOperator2D(expr_string=expr, params=params)
        else:
            return ExpressionOperator(expr_string=expr, params=params)

    raise ValueError(f"Unknown operator type {op_type!r}.")


def _parse_operators(cfg: JsonDict, domain_type: str = "1d") -> List[Operator]:
    ops_cfg = cfg.get("operators")
    if not ops_cfg:
        raise ValueError("JSON configuration must contain a non-empty 'operators' list.")
    return [_parse_operator(op_cfg, domain_type) for op_cfg in ops_cfg]


def build_problem_from_dict(config: JsonDict) -> PDEProblem:
    """
    Build a PDEProblem from an in-memory JSON-like dictionary.

    This is the core entry point; ``load_from_json`` is a small wrapper around it.
    """
    domain_cfg = config["domain"]
    ic_cfg = config["initial_condition"]
    bc_cfg = config.get("boundary_conditions", {})

    domain = _parse_domain(domain_cfg)
    domain_type = domain_cfg.get("type", "1d").lower()
    ic = _parse_initial_condition(ic_cfg)
    operators = _parse_operators(config, domain_type)
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

