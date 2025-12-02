"""
Spatial discretization schemes for finite difference methods.

This module provides different discretization schemes optimized for different
types of PDE terms:
- Advection terms: Upwind schemes (first-order or higher-order) for stability
- Diffusion terms: Central differences (second-order) for accuracy
- Reaction terms: No spatial discretization needed

The module is designed to be modular, allowing different schemes to be applied
to different terms in the same PDE for optimal accuracy and stability.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from .domain import Domain1D

Array = np.ndarray


class DiscretizationScheme(Enum):
    """
    Enumeration of available discretization schemes.
    
    Schemes:
    - CENTRAL: Second-order central differences (default, good for diffusion)
    - UPWIND_FIRST: First-order upwind (stable for advection, less accurate)
    - UPWIND_SECOND: Second-order upwind (stable and more accurate for advection)
    - BACKWARD: First-order backward differences
    - FORWARD: First-order forward differences
    """
    CENTRAL = "central"
    UPWIND_FIRST = "upwind_first"
    UPWIND_SECOND = "upwind_second"
    BACKWARD = "backward"
    FORWARD = "forward"


def first_derivative(
    u: Array,
    dx: float,
    domain: Domain1D,
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL,
    velocity: Optional[Array] = None,
) -> Array:
    """
    Compute first derivative using specified scheme.
    
    Parameters
    ----------
    u : Array
        Function values on the grid
    dx : float
        Grid spacing
    domain : Domain1D
        Domain information (for periodic handling)
    scheme : DiscretizationScheme or str
        Discretization scheme to use. Default is CENTRAL.
    velocity : Array, optional
        Advection velocity (required for upwind schemes).
        Should have same shape as u.
        
    Returns
    -------
    Array
        First derivative approximation with same shape as u.
        
    Raises
    ------
    ValueError
        If upwind scheme is requested but velocity is not provided.
    """
    if isinstance(scheme, str):
        try:
            scheme = DiscretizationScheme(scheme)
        except ValueError:
            raise ValueError(
                f"Unknown scheme: {scheme}. "
                f"Valid schemes: {[s.value for s in DiscretizationScheme]}"
            )
    
    periodic = domain.periodic
    
    if scheme == DiscretizationScheme.CENTRAL:
        return _central_first_derivative(u, dx, periodic)
    elif scheme == DiscretizationScheme.UPWIND_FIRST:
        if velocity is None:
            raise ValueError(
                "Upwind scheme requires velocity array. "
                "Provide velocity parameter or use a different scheme."
            )
        return _upwind_first_derivative(u, dx, periodic, velocity)
    elif scheme == DiscretizationScheme.UPWIND_SECOND:
        if velocity is None:
            raise ValueError(
                "Upwind scheme requires velocity array. "
                "Provide velocity parameter or use a different scheme."
            )
        return _upwind_second_derivative(u, dx, periodic, velocity)
    elif scheme == DiscretizationScheme.BACKWARD:
        return _backward_first_derivative(u, dx, periodic)
    elif scheme == DiscretizationScheme.FORWARD:
        return _forward_first_derivative(u, dx, periodic)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")


def second_derivative(
    u: Array,
    dx: float,
    domain: Domain1D,
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL,
) -> Array:
    """
    Compute second derivative using specified scheme.
    
    For second derivatives, central differences are typically optimal.
    Other schemes are not commonly used but may be added in the future.
    
    Parameters
    ----------
    u : Array
        Function values on the grid
    dx : float
        Grid spacing
    domain : Domain1D
        Domain information
    scheme : DiscretizationScheme or str
        Discretization scheme (currently only CENTRAL is supported)
        
    Returns
    -------
    Array
        Second derivative approximation with same shape as u.
    """
    if isinstance(scheme, str):
        try:
            scheme = DiscretizationScheme(scheme)
        except ValueError:
            raise ValueError(
                f"Unknown scheme: {scheme}. "
                f"Valid schemes: {[s.value for s in DiscretizationScheme]}"
            )
    
    periodic = domain.periodic
    
    if scheme == DiscretizationScheme.CENTRAL:
        return _central_second_derivative(u, dx, periodic)
    else:
        # For now, only central differences for second derivatives
        # Could extend to other schemes if needed
        return _central_second_derivative(u, dx, periodic)


def third_derivative(
    u: Array,
    dx: float,
    domain: Domain1D,
    scheme: DiscretizationScheme | str = DiscretizationScheme.CENTRAL,
) -> Array:
    """
    Compute third derivative using specified scheme.
    
    Parameters
    ----------
    u : Array
        Function values on the grid
    dx : float
        Grid spacing
    domain : Domain1D
        Domain information
    scheme : DiscretizationScheme or str
        Discretization scheme (currently only CENTRAL is supported)
        
    Returns
    -------
    Array
        Third derivative approximation with same shape as u.
    """
    if isinstance(scheme, str):
        try:
            scheme = DiscretizationScheme(scheme)
        except ValueError:
            raise ValueError(
                f"Unknown scheme: {scheme}. "
                f"Valid schemes: {[s.value for s in DiscretizationScheme]}"
            )
    
    periodic = domain.periodic
    
    if scheme == DiscretizationScheme.CENTRAL:
        return _central_third_derivative(u, dx, periodic)
    else:
        # For now, only central differences for third derivatives
        return _central_third_derivative(u, dx, periodic)


# ============================================================================
# Implementation functions
# ============================================================================


def _central_first_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Second-order central first derivative in 1D.
    
    - For periodic domains, use a wrapped central stencil for *all* points.
    - For non-periodic domains, delegate to ``np.gradient`` with
      edge_order=2 (central in the interior, one-sided at boundaries).
      
    Mathematical formula (interior):
        u_x[i] ≈ (u[i+1] - u[i-1]) / (2*dx)
        
    Accuracy: O(dx²)
    """
    if u.size < 3:
        return np.zeros_like(u, dtype=float)
    
    du = np.zeros_like(u, dtype=float)
    
    if periodic:
        # Interior points: standard central stencil
        du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        # Periodic end points: wrap to the penultimate point
        du[0] = (u[1] - u[-2]) / (2.0 * dx)
        du[-1] = du[0]
        return du
    
    # Non-periodic: central in interior, one-sided at boundaries via gradient
    return np.gradient(u, dx, edge_order=2)


def _upwind_first_derivative(
    u: Array,
    dx: float,
    periodic: bool,
    velocity: Array,
) -> Array:
    """
    First-order upwind scheme for advection.
    
    Uses backward difference when velocity > 0, forward when velocity < 0.
    This ensures information flows in the correct direction, improving stability.
    
    Mathematical form:
    - If a > 0: u_x ≈ (u[i] - u[i-1]) / dx  (backward)
    - If a < 0: u_x ≈ (u[i+1] - u[i]) / dx  (forward)
    - If a = 0: use central difference
    
    Accuracy: O(dx) (first-order)
    Stability: Excellent for advection-dominated problems
    """
    if u.size < 2:
        return np.zeros_like(u, dtype=float)
    
    du = np.zeros_like(u, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    
    if velocity.shape != u.shape:
        raise ValueError(f"Velocity shape {velocity.shape} must match u shape {u.shape}")
    
    # Determine upwind direction based on velocity sign
    a_positive = velocity > 0
    a_negative = velocity < 0
    a_zero = velocity == 0
    
    if periodic:
        # Interior points
        # Backward difference where a > 0
        backward_mask = a_positive[1:-1]
        du[1:-1][backward_mask] = (u[1:-1][backward_mask] - u[:-2][backward_mask]) / dx
        
        # Forward difference where a < 0
        forward_mask = a_negative[1:-1]
        du[1:-1][forward_mask] = (u[2:][forward_mask] - u[1:-1][forward_mask]) / dx
        
        # Central difference where a = 0
        zero_mask = a_zero[1:-1]
        if np.any(zero_mask):
            du[1:-1][zero_mask] = (u[2:][zero_mask] - u[:-2][zero_mask]) / (2.0 * dx)
        
        # Boundaries with periodic wrapping
        if a_positive[0]:
            du[0] = (u[0] - u[-2]) / dx
        elif a_negative[0]:
            du[0] = (u[1] - u[0]) / dx
        else:
            du[0] = (u[1] - u[-2]) / (2.0 * dx)
        
        du[-1] = du[0]
    else:
        # Non-periodic: use backward/forward based on velocity
        # Interior
        backward_mask = a_positive[1:-1]
        forward_mask = a_negative[1:-1]
        zero_mask = a_zero[1:-1]
        
        du[1:-1][backward_mask] = (u[1:-1][backward_mask] - u[:-2][backward_mask]) / dx
        du[1:-1][forward_mask] = (u[2:][forward_mask] - u[1:-1][forward_mask]) / dx
        
        if np.any(zero_mask):
            du[1:-1][zero_mask] = (u[2:][zero_mask] - u[:-2][zero_mask]) / (2.0 * dx)
        
        # Left boundary: forward difference
        du[0] = (u[1] - u[0]) / dx
        
        # Right boundary: backward difference
        du[-1] = (u[-1] - u[-2]) / dx
    
    return du


def _upwind_second_derivative(
    u: Array,
    dx: float,
    periodic: bool,
    velocity: Array,
) -> Array:
    """
    Second-order upwind scheme (essentially non-oscillatory, ENO-like).
    
    Uses a three-point stencil biased in the upwind direction.
    More accurate than first-order upwind while maintaining stability.
    
    Mathematical form (when a > 0, backward):
        u_x[i] ≈ (3*u[i] - 4*u[i-1] + u[i-2]) / (2*dx)
        
    Mathematical form (when a < 0, forward):
        u_x[i] ≈ (-u[i+2] + 4*u[i+1] - 3*u[i]) / (2*dx)
    
    Accuracy: O(dx²) (second-order)
    Stability: Excellent for advection-dominated problems
    """
    if u.size < 3:
        # Fall back to first-order if not enough points
        return _upwind_first_derivative(u, dx, periodic, velocity)
    
    du = np.zeros_like(u, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    
    if velocity.shape != u.shape:
        raise ValueError(f"Velocity shape {velocity.shape} must match u shape {u.shape}")
    
    a_positive = velocity > 0
    a_negative = velocity < 0
    a_zero = velocity == 0
    
    if periodic:
        # Interior points
        # Second-order backward: (3u[i] - 4u[i-1] + u[i-2]) / (2*dx) when a > 0
        backward_mask = a_positive[1:-1]
        if np.any(backward_mask):
            indices = np.arange(1, len(u) - 1)
            backward_indices = indices[backward_mask]
            for idx in backward_indices:
                i_minus_2 = (idx - 2) % len(u)
                du[idx] = (3.0 * u[idx] - 4.0 * u[idx - 1] + u[i_minus_2]) / (2.0 * dx)
        
        # Second-order forward: (-u[i+2] + 4u[i+1] - 3u[i]) / (2*dx) when a < 0
        forward_mask = a_negative[1:-1]
        if np.any(forward_mask):
            indices = np.arange(1, len(u) - 1)
            forward_indices = indices[forward_mask]
            for idx in forward_indices:
                i_plus_2 = (idx + 2) % len(u)
                du[idx] = (-u[i_plus_2] + 4.0 * u[idx + 1] - 3.0 * u[idx]) / (2.0 * dx)
        
        # Central difference where a = 0
        zero_mask = a_zero[1:-1]
        if np.any(zero_mask):
            indices = np.arange(1, len(u) - 1)[zero_mask]
            du[indices] = (u[indices + 1] - u[indices - 1]) / (2.0 * dx)
        
        # Boundaries with periodic wrapping
        if a_positive[0]:
            du[0] = (3.0 * u[0] - 4.0 * u[-2] + u[-3]) / (2.0 * dx)
        elif a_negative[0]:
            du[0] = (-u[2] + 4.0 * u[1] - 3.0 * u[0]) / (2.0 * dx)
        else:
            du[0] = (u[1] - u[-2]) / (2.0 * dx)
        
        du[-1] = du[0]
    else:
        # Non-periodic: similar logic with boundary handling
        backward_mask = a_positive[1:-1]
        forward_mask = a_negative[1:-1]
        zero_mask = a_zero[1:-1]
        
        if np.any(backward_mask):
            indices = np.arange(1, len(u) - 1)[backward_mask]
            # Ensure we don't go out of bounds
            valid = indices >= 2
            if np.any(valid):
                valid_indices = indices[valid]
                du[valid_indices] = (
                    3.0 * u[valid_indices]
                    - 4.0 * u[valid_indices - 1]
                    + u[valid_indices - 2]
                ) / (2.0 * dx)
            # For indices < 2, fall back to first-order
            invalid = indices < 2
            if np.any(invalid):
                invalid_indices = indices[invalid]
                du[invalid_indices] = (u[invalid_indices] - u[invalid_indices - 1]) / dx
        
        if np.any(forward_mask):
            indices = np.arange(1, len(u) - 1)[forward_mask]
            # Ensure we don't go out of bounds
            valid = indices < len(u) - 2
            if np.any(valid):
                valid_indices = indices[valid]
                du[valid_indices] = (
                    -u[valid_indices + 2]
                    + 4.0 * u[valid_indices + 1]
                    - 3.0 * u[valid_indices]
                ) / (2.0 * dx)
            # For indices >= len(u) - 2, fall back to first-order
            invalid = indices >= len(u) - 2
            if np.any(invalid):
                invalid_indices = indices[invalid]
                du[invalid_indices] = (u[invalid_indices + 1] - u[invalid_indices]) / dx
        
        if np.any(zero_mask):
            indices = np.arange(1, len(u) - 1)[zero_mask]
            du[indices] = (u[indices + 1] - u[indices - 1]) / (2.0 * dx)
        
        # Boundaries: fall back to first-order
        du[0] = (u[1] - u[0]) / dx
        du[-1] = (u[-1] - u[-2]) / dx
    
    return du


def _backward_first_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    First-order backward difference.
    
    Mathematical form:
        u_x[i] ≈ (u[i] - u[i-1]) / dx
        
    Accuracy: O(dx) (first-order)
    """
    if u.size < 2:
        return np.zeros_like(u, dtype=float)
    
    du = np.zeros_like(u, dtype=float)
    
    if periodic:
        du[1:] = (u[1:] - u[:-1]) / dx
        du[0] = (u[0] - u[-2]) / dx
    else:
        du[1:] = (u[1:] - u[:-1]) / dx
        du[0] = (u[1] - u[0]) / dx  # Forward at left boundary
    
    return du


def _forward_first_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    First-order forward difference.
    
    Mathematical form:
        u_x[i] ≈ (u[i+1] - u[i]) / dx
        
    Accuracy: O(dx) (first-order)
    """
    if u.size < 2:
        return np.zeros_like(u, dtype=float)
    
    du = np.zeros_like(u, dtype=float)
    
    if periodic:
        du[:-1] = (u[1:] - u[:-1]) / dx
        du[-1] = (u[1] - u[0]) / dx
    else:
        du[:-1] = (u[1:] - u[:-1]) / dx
        du[-1] = (u[-1] - u[-2]) / dx  # Backward at right boundary
    
    return du


def _central_second_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Second-order central second derivative in 1D.
    
    - For periodic domains, use a wrapped central stencil.
    - For non-periodic domains, approximate via a second application of
      ``np.gradient`` (which is central in the interior).
      
    Mathematical formula (interior):
        u_xx[i] ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²
        
    Accuracy: O(dx²)
    """
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


def _central_third_derivative(u: Array, dx: float, periodic: bool) -> Array:
    """
    Second-order central third derivative in 1D.
    
    Uses the stencil: u_xxx ≈ (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2]) / (2*dx^3)
    
    Parameters
    ----------
    u:
        1D array of function values.
    dx:
        Grid spacing.
    periodic:
        Whether the domain is periodic.
        
    Returns
    -------
    Array
        Third derivative approximation with same shape as u.
        
    Accuracy: O(dx²)
    """
    if u.size < 5:
        return np.zeros_like(u, dtype=float)
    
    d3 = np.zeros_like(u, dtype=float)
    
    if periodic:
        # Interior points: central difference stencil
        # u_xxx[i] ≈ (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2]) / (2*dx^3)
        d3[2:-2] = (u[4:] - 2.0*u[3:-1] + 2.0*u[1:-3] - u[:-4]) / (2.0 * dx**3)
        
        # Handle boundaries with periodic wrapping
        # Point 0: wrap to end
        d3[0] = (u[2] - 2.0*u[1] + 2.0*u[-2] - u[-3]) / (2.0 * dx**3)
        # Point 1: wrap to end
        d3[1] = (u[3] - 2.0*u[2] + 2.0*u[0] - u[-2]) / (2.0 * dx**3)
        # Point -2: wrap to beginning
        d3[-2] = (u[0] - 2.0*u[-1] + 2.0*u[-3] - u[-4]) / (2.0 * dx**3)
        # Point -1: same as point 0 (periodic)
        d3[-1] = d3[0]
        return d3
    
    # Non-periodic: use one-sided differences at boundaries
    # Interior: central difference
    d3[2:-2] = (u[4:] - 2.0*u[3:-1] + 2.0*u[1:-3] - u[:-4]) / (2.0 * dx**3)
    
    # Left boundary: forward differences
    # Third-order forward: u_xxx[0] ≈ (-u[3] + 3*u[2] - 3*u[1] + u[0]) / dx^3
    d3[0] = (-u[3] + 3.0*u[2] - 3.0*u[1] + u[0]) / (dx**3)
    d3[1] = (-u[4] + 3.0*u[3] - 3.0*u[2] + u[1]) / (dx**3)
    
    # Right boundary: backward differences
    # Third-order backward: u_xxx[-1] ≈ (u[-1] - 3*u[-2] + 3*u[-3] - u[-4]) / dx^3
    d3[-2] = (u[-1] - 3.0*u[-2] + 3.0*u[-3] - u[-4]) / (dx**3)
    d3[-1] = (u[-1] - 3.0*u[-2] + 3.0*u[-3] - u[-4]) / (dx**3)
    
    return d3

