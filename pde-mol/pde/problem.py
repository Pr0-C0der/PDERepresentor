from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .bc import BoundaryCondition, apply_neumann_ghosts
from .domain import Domain1D
from .ic import InitialCondition
from .operators import Operator, sum_operators
from .plotting import plot_1d, plot_1d_time_series


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

    domain: Domain1D
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

        Returns a 1D array of length nx.
        """
        u0 = self.ic.evaluate(self.domain)

        # Optionally apply boundary conditions for non-periodic case.
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

    def _rhs_nonperiodic(self, t: float, interior: Array) -> Array:
        u_full = self._reconstruct_full_from_interior(interior, t)
        du_full = self._op.apply(u_full, self.domain, t)
        return du_full[1:-1]

    def rhs(self, t: float, y: Array) -> Array:
        """
        Unified RHS dispatch used primarily for testing.
        """
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

        return result


@dataclass
class SecondOrderPDEProblem:
    """
    Handles second-order time derivatives by converting to first-order system.
    
    For equations of the form: u_tt = L[u] + M[u_t]
    where L and M are spatial operators, we introduce v = u_t, giving:
    - u_t = v
    - v_t = L[u] + M[v]
    
    This class maintains the same modular design as PDEProblem but handles
    the extended state vector [u, v] where v = u_t.
    
    Parameters
    ----------
    domain:
        1D spatial domain.
    spatial_operator:
        Operator L such that u_tt = L[u] + ... (applied to u).
    ic_u:
        Initial condition for u(x, 0).
    ic_ut:
        Initial condition for u_t(x, 0) = v(x, 0).
    u_t_operator:
        Optional operator M applied to u_t (for damping terms like -γ*u_t).
        If None, assumes u_tt = L[u] only.
    bc_left, bc_right:
        Optional boundary conditions (applied to u, not v).
    """
    
    domain: Domain1D
    spatial_operator: Operator
    ic_u: InitialCondition
    ic_ut: InitialCondition
    u_t_operator: Optional[Operator] = None
    bc_left: Optional[BoundaryCondition] = None
    bc_right: Optional[BoundaryCondition] = None
    
    def __post_init__(self) -> None:
        """Validate the problem setup."""
        if not isinstance(self.domain, Domain1D):
            raise TypeError("SecondOrderPDEProblem requires a Domain1D instance.")
    
    # ------------------------------------------------------------------
    # Initial data helpers
    # ------------------------------------------------------------------
    def initial_full(self, t0: float = 0.0) -> Array:
        """
        Evaluate initial conditions and return extended state [u, v].
        
        Returns
        -------
        Array
            Extended state vector of length 2*nx (or 2*(nx-2) for non-periodic)
            containing [u(x,0), v(x,0)] where v = u_t.
        """
        u0 = self.ic_u.evaluate(self.domain)
        v0 = self.ic_ut.evaluate(self.domain)
        
        # Ensure arrays are 1D
        u0 = np.asarray(u0, dtype=float).flatten()
        v0 = np.asarray(v0, dtype=float).flatten()
        
        # Handle scalar initial conditions (broadcast to array)
        if u0.size == 1:
            u0 = np.full(self.domain.nx, float(u0))
        if v0.size == 1:
            v0 = np.full(self.domain.nx, float(v0))
        
        # Apply boundary conditions to u0 (not v0)
        if not self.domain.periodic:
            if self.bc_left is not None:
                self.bc_left.apply_to_full(u0, t0, self.domain)
            if self.bc_right is not None:
                self.bc_right.apply_to_full(u0, t0, self.domain)
        
        # For non-periodic, return interior only
        if self.domain.periodic:
            return np.concatenate([u0, v0])
        else:
            return np.concatenate([u0[1:-1], v0[1:-1]])
    
    def _reconstruct_full_from_interior(self, interior: Array, t: float) -> tuple[Array, Array]:
        """
        Reconstruct full u and v vectors from interior state.
        
        Returns
        -------
        tuple[Array, Array]
            (u_full, v_full) arrays of length nx.
        """
        nx = self.domain.nx
        interior_size = interior.size // 2
        
        if not self.domain.periodic:
            if interior_size != nx - 2:
                raise ValueError(
                    f"Interior state has length {interior_size}, expected {nx - 2}."
                )
            
            u_interior = interior[:interior_size]
            v_interior = interior[interior_size:]
            
            u_full = np.zeros(nx, dtype=float)
            v_full = np.zeros(nx, dtype=float)
            u_full[1:-1] = u_interior
            v_full[1:-1] = v_interior
            
            # Apply BCs to u (not v)
            if self.bc_left is not None:
                self.bc_left.apply_to_full(u_full, t, self.domain)
            if self.bc_right is not None:
                self.bc_right.apply_to_full(u_full, t, self.domain)
            
            # Adjust for Neumann BCs
            from .bc import apply_neumann_ghosts
            apply_neumann_ghosts(u_full, self.bc_left, self.bc_right, t, self.domain)
            
            return u_full, v_full
        else:
            # Periodic: state is already full
            u_full = interior[:nx]
            v_full = interior[nx:]
            return u_full, v_full
    
    # ------------------------------------------------------------------
    # Right-hand side functions
    # ------------------------------------------------------------------
    def _rhs_periodic(self, t: float, y: Array) -> Array:
        """
        RHS for periodic domains.
        
        y = [u, v] where v = u_t
        Returns [u_t, v_t] = [v, L[u] + M[v]]
        """
        nx = self.domain.nx
        u = y[:nx]
        v = y[nx:]
        
        # Apply spatial operator to u: L[u]
        L_u = self.spatial_operator.apply(u, self.domain, t)
        
        # Apply operator to v (u_t) if provided: M[v]
        if self.u_t_operator is not None:
            M_v = self.u_t_operator.apply(v, self.domain, t)
        else:
            M_v = np.zeros_like(v)
        
        # Return [u_t, v_t] = [v, L[u] + M[v]]
        return np.concatenate([v, L_u + M_v])
    
    def _rhs_nonperiodic(self, t: float, y: Array) -> Array:
        """
        RHS for non-periodic domains.
        
        y = [u_interior, v_interior]
        Returns [u_t_interior, v_t_interior]
        """
        u_full, v_full = self._reconstruct_full_from_interior(y, t)
        
        # Apply operators to full vectors
        L_u_full = self.spatial_operator.apply(u_full, self.domain, t)
        
        if self.u_t_operator is not None:
            M_v_full = self.u_t_operator.apply(v_full, self.domain, t)
        else:
            M_v_full = np.zeros_like(v_full)
        
        # Extract interior parts
        v_t_interior = v_full[1:-1]  # u_t = v
        L_u_interior = L_u_full[1:-1]
        M_v_interior = M_v_full[1:-1]
        
        return np.concatenate([v_t_interior, L_u_interior + M_v_interior])
    
    def rhs(self, t: float, y: Array) -> Array:
        """
        Unified RHS dispatch.
        
        Parameters
        ----------
        t:
            Current time.
        y:
            Extended state vector [u, v] (or [u_interior, v_interior] for non-periodic).
            
        Returns
        -------
        Array
            Time derivative [u_t, v_t] of same shape as y.
        """
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
        Solve the second-order PDE in time using SciPy's ``solve_ivp``.
        
        Parameters
        ----------
        t_span:
            Tuple ``(t0, tf)`` specifying the integration interval.
        t_eval:
            Optional array of times at which to store the solution.
        method:
            Name of the time integrator. For stiff systems (e.g., with damping),
            use 'BDF'. Default is 'RK45'.
        plot:
            If True, generate plots of the solution.
        plot_dir:
            Directory for saving plots.
        solve_ivp_kwargs:
            Additional keyword arguments passed directly to ``solve_ivp``.
            
        Returns
        -------
        scipy.integrate.OdeResult
            Result object with additional attributes:
            - `u`: u component (shape (nx, nt) or (nx-2, nt) for non-periodic)
            - `v`: v = u_t component (same shape as u)
        """
        t0, _ = t_span
        
        y0 = self.initial_full(t0)
        
        if self.domain.periodic:
            fun = self._rhs_periodic
        else:
            fun = self._rhs_nonperiodic
        
        result = solve_ivp(
            fun=fun,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            **solve_ivp_kwargs,
        )
        
        # Extract u and v components for convenience
        nx = self.domain.nx
        if self.domain.periodic:
            result.u = result.y[:nx, :]
            result.v = result.y[nx:, :]
        else:
            interior_size = nx - 2
            result.u = result.y[:interior_size, :]
            result.v = result.y[interior_size:, :]
        
        # Optional plotting
        if plot:
            import os
            
            base_dir = plot_dir or "test_plots"
            os.makedirs(base_dir, exist_ok=True)
            
            x = self.domain.x
            
            # Reconstruct full solutions for plotting
            if self.domain.periodic:
                u_full_solutions = result.u
            else:
                u_full_states = []
                for k, t in enumerate(result.t):
                    interior_k = result.y[:, k]
                    u_full, _ = self._reconstruct_full_from_interior(interior_k, t)
                    u_full_states.append(u_full)
                u_full_solutions = np.stack(u_full_states, axis=1)
            
            # Plot initial and final snapshots
            plot_1d(
                x,
                u_full_solutions[:, 0],
                title="Initial solution u(x,0)",
                savepath=os.path.join(base_dir, "solution1d_initial.png"),
                )
            plot_1d(
                x,
                u_full_solutions[:, -1],
                title="Final solution u(x,t)",
                savepath=os.path.join(base_dir, "solution1d_final.png"),
                )
            
            # Time series
            plot_1d_time_series(
                x,
                u_full_solutions,
                    result.t,
                prefix="solution1d",
                    out_dir=base_dir,
                )

        return result

