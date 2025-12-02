"""
Tests for spatial discretization schemes.

Tests verify that different discretization schemes work correctly for
various derivative orders and boundary conditions.
"""

import numpy as np
import pytest

from pde.domain import Domain1D
from pde.spatial_discretization import (
    DiscretizationScheme,
    first_derivative,
    second_derivative,
    third_derivative,
)


def test_central_first_derivative_periodic():
    """Test central first derivative on periodic domain."""
    domain = Domain1D(0.0, 2.0 * np.pi, 101, periodic=True)
    x = domain.x
    u = np.sin(x)
    
    du = first_derivative(u, domain.dx, domain, DiscretizationScheme.CENTRAL)
    du_exact = np.cos(x)
    
    # Check interior points (skip boundaries due to periodicity handling)
    error = np.abs(du[2:-2] - du_exact[2:-2])
    assert np.max(error) < 0.01, f"Max error: {np.max(error)}"


def test_central_first_derivative_nonperiodic():
    """Test central first derivative on non-periodic domain."""
    domain = Domain1D(0.0, 2.0 * np.pi, 101, periodic=False)
    x = domain.x
    u = np.sin(x)
    
    du = first_derivative(u, domain.dx, domain, DiscretizationScheme.CENTRAL)
    du_exact = np.cos(x)
    
    # Check interior points
    error = np.abs(du[2:-2] - du_exact[2:-2])
    assert np.max(error) < 0.01, f"Max error: {np.max(error)}"


def test_upwind_first_derivative_positive_velocity():
    """Test first-order upwind with positive velocity."""
    domain = Domain1D(0.0, 1.0, 101, periodic=False)
    x = domain.x
    u = np.exp(-x)  # Decaying function
    velocity = np.ones_like(x)  # Positive velocity
    
    du = first_derivative(
        u, domain.dx, domain, 
        DiscretizationScheme.UPWIND_FIRST, 
        velocity=velocity
    )
    du_exact = -np.exp(-x)
    
    # Upwind should be less accurate but stable
    error = np.abs(du[2:-2] - du_exact[2:-2])
    assert np.max(error) < 0.1, f"Max error: {np.max(error)}"


def test_upwind_first_derivative_negative_velocity():
    """Test first-order upwind with negative velocity."""
    domain = Domain1D(0.0, 1.0, 101, periodic=False)
    x = domain.x
    u = np.exp(x)  # Growing function
    velocity = -np.ones_like(x)  # Negative velocity
    
    du = first_derivative(
        u, domain.dx, domain,
        DiscretizationScheme.UPWIND_FIRST,
        velocity=velocity
    )
    du_exact = np.exp(x)
    
    error = np.abs(du[2:-2] - du_exact[2:-2])
    assert np.max(error) < 0.1, f"Max error: {np.max(error)}"


def test_upwind_second_derivative():
    """Test second-order upwind scheme."""
    domain = Domain1D(0.0, 2.0 * np.pi, 101, periodic=True)
    x = domain.x
    u = np.sin(x)
    velocity = np.ones_like(x)  # Positive velocity
    
    du = first_derivative(
        u, domain.dx, domain,
        DiscretizationScheme.UPWIND_SECOND,
        velocity=velocity
    )
    du_exact = np.cos(x)
    
    # Second-order upwind should be more accurate than first-order
    error = np.abs(du[3:-3] - du_exact[3:-3])
    assert np.max(error) < 0.05, f"Max error: {np.max(error)}"


def test_central_second_derivative():
    """Test central second derivative."""
    domain = Domain1D(0.0, 2.0 * np.pi, 101, periodic=True)
    x = domain.x
    u = np.sin(x)
    
    d2u = second_derivative(u, domain.dx, domain, DiscretizationScheme.CENTRAL)
    d2u_exact = -np.sin(x)
    
    error = np.abs(d2u[2:-2] - d2u_exact[2:-2])
    assert np.max(error) < 0.01, f"Max error: {np.max(error)}"


def test_central_third_derivative():
    """Test central third derivative."""
    domain = Domain1D(0.0, 2.0 * np.pi, 101, periodic=True)
    x = domain.x
    u = np.sin(x)
    
    d3u = third_derivative(u, domain.dx, domain, DiscretizationScheme.CENTRAL)
    d3u_exact = -np.cos(x)
    
    error = np.abs(d3u[3:-3] - d3u_exact[3:-3])
    assert np.max(error) < 0.1, f"Max error: {np.max(error)}"


def test_scheme_string_conversion():
    """Test that string scheme names work."""
    domain = Domain1D(0.0, 1.0, 51, periodic=False)
    u = np.sin(2 * np.pi * domain.x)
    
    # Test string conversion
    du1 = first_derivative(u, domain.dx, domain, "central")
    du2 = first_derivative(u, domain.dx, domain, DiscretizationScheme.CENTRAL)
    
    np.testing.assert_array_almost_equal(du1, du2)


def test_upwind_requires_velocity():
    """Test that upwind schemes require velocity."""
    domain = Domain1D(0.0, 1.0, 51, periodic=False)
    u = np.sin(2 * np.pi * domain.x)
    
    with pytest.raises(ValueError, match="velocity"):
        first_derivative(
            u, domain.dx, domain,
            DiscretizationScheme.UPWIND_FIRST,
            velocity=None
        )


def test_invalid_scheme():
    """Test that invalid scheme names raise errors."""
    domain = Domain1D(0.0, 1.0, 51, periodic=False)
    u = np.sin(2 * np.pi * domain.x)
    
    with pytest.raises(ValueError, match="Unknown scheme"):
        first_derivative(u, domain.dx, domain, "invalid_scheme")


def test_advection_operator_with_schemes():
    """Test that Advection operator works with different schemes."""
    from pde.operators import Advection
    
    domain = Domain1D(0.0, 1.0, 51, periodic=True)
    u = np.sin(2 * np.pi * domain.x)
    
    # Test with upwind_first (default)
    adv1 = Advection(a=1.0, scheme="upwind_first")
    result1 = adv1.apply(u, domain, 0.0)
    
    # Test with upwind_second
    adv2 = Advection(a=1.0, scheme="upwind_second")
    result2 = adv2.apply(u, domain, 0.0)
    
    # Test with central
    adv3 = Advection(a=1.0, scheme="central")
    result3 = adv3.apply(u, domain, 0.0)
    
    # All should produce results
    assert result1.shape == u.shape
    assert result2.shape == u.shape
    assert result3.shape == u.shape


def test_expression_operator_with_schemes():
    """Test that ExpressionOperator works with scheme selection."""
    from pde.operators import ExpressionOperator
    
    domain = Domain1D(0.0, 1.0, 51, periodic=True)
    u = np.sin(2 * np.pi * domain.x)
    
    # Test with upwind for ux
    op = ExpressionOperator(
        expr_string="-u*ux + nu*uxx",
        params={"nu": 0.1},
        schemes={"ux": "upwind_second", "uxx": "central"}
    )
    
    result = op.apply(u, domain, 0.0)
    assert result.shape == u.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

