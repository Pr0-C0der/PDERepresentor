from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .bc import BoundaryCondition, apply_neumann_ghosts
from .domain import Domain1D, Domain2D
from .ic import InitialCondition
from .operators import Operator, sum_operators
from .plotting import plot_1d, plot_1d_time_series, plot_2d, plot_2d_time_series


Array = np.ndarray
TimeSpan = Tuple[float, float]


@dataclass
class PDEProblem:
    """
    Simple Method-of-Lines PDE problem wrapper.

    This bundles together:
    - spatial domain
    - spatial operators
    - initial condition
    - (optional) boundary conditions

    and exposes:
    - a right-hand-side function compatible with ``solve_ivp``
    - a convenience ``solve`` method to integrate in time.

    Design
    ------
    * Periodic domains:
        - The ODE state is the full vector ``u_full`` of length ``nx``.
        - Operators are applied directly to ``u_full`` and return a full
          derivative vector.

    * Non-periodic domains:
        - The ODE state contains only interior values of length ``nx-2``.
        - At each RHS evaluation, a full vector is reconstructed, boundary
          conditions are applied, spatial operators are evaluated, and the
          interior part of the derivative is returned.
    """

    domain: object  # Domain1D or Domain2D
    operators: Sequence[Operator]
    ic: InitialCondition
    bc_left: Optional[BoundaryCondition] = None
    bc_right: Optional[BoundaryCondition] = None

    # Internal combined operator (sum of all operators)
    _op: Operator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operators, Sequence) or isinstance(self.operators, (Operator,)):
            # Allow a single Operator to be passed in place of a sequence
            ops: List[Operator]
            if isinstance(self.operators, Operator):
                ops = [self.operators]
            else:
                ops = list(self.operators)  # type: ignore[arg-type]
            self.operators = ops

        if len(self.operators) == 0:
            raise ValueError("PDEProblem requires at least one spatial operator.")

        self._op = sum_operators(self.operators)

        # For periodic domains, any non-periodic BCs are ignored. A Periodic()
        # BC acts as a marker and is not required here.

    # ------------------------------------------------------------------
    # Initial data helpers
    # ------------------------------------------------------------------
    def initial_full(self, t0: float = 0.0) -> Array:
        """
        Evaluate the initial condition on the full grid.

        - For 1D domains, returns a 1D array of length nx.
        - For 2D domains, returns a flattened 1D array of length nx * ny.
        """
        u0 = self.ic.evaluate(self.domain)  # type: ignore[arg-type]

        if isinstance(self.domain, Domain2D):
            # No BC enforcement for 2D yet; assume operators handle periodicity.
            return self.domain.flatten(u0)  # type: ignore[union-attr]

        # Domain1D: optionally apply boundary conditions for non-periodic case.
        assert isinstance(self.domain, Domain1D)
        if not self.domain.periodic:
            if self.bc_left is not None:
                self.bc_left.apply_to_full(u0, t0, self.domain)
            if self.bc_right is not None:
                self.bc_right.apply_to_full(u0, t0, self.domain)
        return u0

    def initial_interior(self, t0: float = 0.0) -> Array:
        """
        Return only the interior part of the initial condition.

        This is meaningful for non-periodic domains where the ODE state
        excludes boundary values.
        """
        assert isinstance(self.domain, Domain1D), "initial_interior is only defined for 1D domains."
        u_full = self.initial_full(t0)
        return u_full[1:-1]

    # ------------------------------------------------------------------
    # Internal reconstruction helpers
    # ------------------------------------------------------------------
    def _reconstruct_full_from_interior(self, interior: Array, t: float) -> Array:
        """
        Given an interior state vector, reconstruct the full state and
        apply boundary conditions.
        """
        assert isinstance(self.domain, Domain1D)
        nx = self.domain.nx
        if interior.shape[0] != nx - 2:
            raise ValueError(
                f"Interior state has length {interior.shape[0]}, expected {nx - 2}."
            )

        u_full = np.zeros(nx, dtype=float)
        u_full[1:-1] = interior

        if self.bc_left is not None:
            self.bc_left.apply_to_full(u_full, t, self.domain)
        if self.bc_right is not None:
            self.bc_right.apply_to_full(u_full, t, self.domain)

        # Adjust boundary values to satisfy Neumann fluxes when requested.
        apply_neumann_ghosts(u_full, self.bc_left, self.bc_right, t, self.domain)

        return u_full

    # ------------------------------------------------------------------
    # Right-hand side functions
    # ------------------------------------------------------------------
    def _rhs_periodic(self, t: float, u_full: Array) -> Array:
        return self._op.apply(u_full, self.domain, t)

    def _rhs_2d(self, t: float, u_flat: Array) -> Array:
        """
        RHS for 2D problems. The state is always the full flattened field.
        """
        return self._op.apply(u_flat, self.domain, t)

    def _rhs_nonperiodic(self, t: float, interior: Array) -> Array:
        u_full = self._reconstruct_full_from_interior(interior, t)
        du_full = self._op.apply(u_full, self.domain, t)
        return du_full[1:-1]

    def rhs(self, t: float, y: Array) -> Array:
        """
        Unified RHS dispatch used primarily for testing.
        """
        if isinstance(self.domain, Domain2D):
            return self._rhs_2d(t, y)

        assert isinstance(self.domain, Domain1D)
        if self.domain.periodic:
            return self._rhs_periodic(t, y)
        return self._rhs_nonperiodic(t, y)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def solve(
        self,
        t_span: TimeSpan,
        t_eval: Optional[Array] = None,
        method: str = "RK45",
        plot: bool = False,
        plot_dir: Optional[str] = None,
        **solve_ivp_kwargs,
    ):
        """
        Solve the PDE in time using SciPy's ``solve_ivp``.

        Parameters
        ----------
        t_span:
            Tuple ``(t0, tf)`` specifying the integration interval.
        t_eval:
            Optional array of times at which to store the solution.
        method:
            Name of the time integrator (e.g. 'RK45', 'BDF', ...).
        solve_ivp_kwargs:
            Additional keyword arguments passed directly to ``solve_ivp``.

        Returns
        -------
        scipy.integrate.OdeResult
            The standard SciPy result object containing the time grid and
            solution snapshots.
        """
        t0, _ = t_span

        if isinstance(self.domain, Domain2D):
            y0 = self.initial_full(t0)
            fun = self._rhs_2d
        else:
            assert isinstance(self.domain, Domain1D)
            if self.domain.periodic:
                y0 = self.initial_full(t0)
                fun = self._rhs_periodic
            else:
                y0 = self.initial_interior(t0)
                fun = self._rhs_nonperiodic

        result = solve_ivp(
            fun=fun,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            **solve_ivp_kwargs,
        )

        # Optional plotting hooks
        if plot:
            import os

            # Default plotting directory if none is provided
            base_dir = plot_dir or "test_plots"
            os.makedirs(base_dir, exist_ok=True)

            if isinstance(self.domain, Domain1D):
                x = self.domain.x
                # Reconstruct full solutions for non-periodic 1D problems
                if self.domain.periodic:
                    full_solutions = result.y
                else:
                    full_states = []
                    for k, t in enumerate(result.t):
                        interior_k = result.y[:, k]
                        full_k = self._reconstruct_full_from_interior(interior_k, t)
                        full_states.append(full_k)
                    full_solutions = np.stack(full_states, axis=1)

                # Initial and final snapshots
                plot_1d(
                    x,
                    full_solutions[:, 0],
                    title="Initial solution (1D)",
                    savepath=os.path.join(base_dir, "solution1d_initial.png"),
                )
                plot_1d(
                    x,
                    full_solutions[:, -1],
                    title="Final solution (1D)",
                    savepath=os.path.join(base_dir, "solution1d_final.png"),
                )
                # Time series frames
                plot_1d_time_series(
                    x,
                    full_solutions,
                    result.t,
                    prefix="solution1d",
                    out_dir=base_dir,
                )

            elif isinstance(self.domain, Domain2D):
                X, Y = self.domain.X, self.domain.Y  # type: ignore[union-attr]
                U_list = [
                    self.domain.unflatten(result.y[:, k])  # type: ignore[union-attr]
                    for k in range(result.y.shape[1])
                ]

                # Initial and final snapshots
                plot_2d(
                    X,
                    Y,
                    U_list[0],
                    title="Initial solution (2D)",
                    savepath=os.path.join(base_dir, "solution2d_initial.png"),
                )
                plot_2d(
                    X,
                    Y,
                    U_list[-1],
                    title="Final solution (2D)",
                    savepath=os.path.join(base_dir, "solution2d_final.png"),
                )
                # Time series frames
                plot_2d_time_series(
                    X,
                    Y,
                    U_list,
                    result.t,
                    prefix="solution2d",
                    out_dir=base_dir,
                )

        return result

