"""Tests for the Poisson equation PINN model components."""

import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import Callable, Tuple

# Import model components to test with relative imports
from model.operators import laplacian, poisson_equation


def test_laplacian_sin():
    """Test that the Laplacian of sin(x)sin(y) is computed correctly.
    
    For u(x,y) = sin(x)sin(y), the Laplacian should be -2*sin(x)sin(y).
    """
    # Define test function
    def u(xy):
        x, y = xy[0], xy[1]
        return jnp.sin(x) * jnp.sin(y)
    
    # Compute Laplacian using our function
    lap_u_fn = laplacian(u)
    
    # Test at multiple points
    test_points = [
        jnp.array([0.0, 0.0]),
        jnp.array([jnp.pi/4, jnp.pi/4]),
        jnp.array([jnp.pi/2, jnp.pi/2]),
    ]
    
    for xy in test_points:
        # Compute Laplacian
        lap_u_computed = lap_u_fn(xy)
        
        # Expected Laplacian: -2*sin(x)sin(y)
        x, y = xy[0], xy[1]
        lap_u_expected = -2.0 * jnp.sin(x) * jnp.sin(y)
        
        # Check that they're close
        assert jnp.abs(lap_u_computed - lap_u_expected) < 1e-5, \
            f"Laplacian mismatch at {xy}: computed={lap_u_computed}, expected={lap_u_expected}"


def test_finite_difference_laplacian():
    """Test Laplacian calculation using finite differences."""
    # Define test function: u(x,y) = sin(2πx)sin(2πy)
    def u(xy):
        x, y = xy[0], xy[1]
        return jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    
    # Compute Laplacian using our autodiff function
    lap_u_fn = laplacian(u)
    
    # Compute Laplacian using finite differences
    def finite_diff_laplacian(f, x, h=1e-4):
        """Compute Laplacian using 2nd-order finite differences."""
        n_dims = len(x)
        result = 0.0
        
        for i in range(n_dims):
            # Create offset vectors
            e_i = jnp.zeros_like(x)
            e_i = e_i.at[i].set(1.0)
            
            # Compute second derivative in this dimension
            f_plus = f(x + h * e_i)
            f_center = f(x)
            f_minus = f(x - h * e_i)
            
            # 2nd-order central difference
            d2f_dx2 = (f_plus - 2 * f_center + f_minus) / (h * h)
            
            # Add to Laplacian
            result += d2f_dx2
        
        return result
    
    # Test at a specific point
    test_point = jnp.array([0.25, 0.75])
    
    # Compute Laplacian using both methods
    lap_u_autodiff = lap_u_fn(test_point)
    lap_u_fd = finite_diff_laplacian(u, test_point)
    
    # Expected Laplacian: -8π²sin(2πx)sin(2πy)
    x, y = test_point[0], test_point[1]
    lap_u_expected = -8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    
    # Check that both methods are close to expected
    assert jnp.abs(lap_u_autodiff - lap_u_expected) < 1e-5, \
        f"Autodiff Laplacian mismatch: computed={lap_u_autodiff}, expected={lap_u_expected}"
    
    assert jnp.abs(lap_u_fd - lap_u_expected) < 1e-3, \
        f"Finite difference Laplacian mismatch: computed={lap_u_fd}, expected={lap_u_expected}"


def test_poisson_residual():
    """Test the PDE residual calculation for a known solution."""
    # Test with:
    # - Solution: u(x,y) = sin(2πx)sin(2πy)
    # - Source term: f(x,y) = -8π²sin(2πx)sin(2πy)
    # This gives a PDE: ∇²u = -8π²sin(2πx)sin(2πy)
    
    # Define solution and source term
    def u_fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    
    def f_fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        return -8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    
    # Compute PDE residual
    test_point = jnp.array([0.3, 0.7])
    residual = poisson_equation(u_fn, test_point, f_fn)
    
    # Since u is the exact solution, residual should be close to zero
    assert jnp.abs(residual) < 1e-5, f"PDE residual is too large: {residual}"


if __name__ == "__main__":
    # Run tests manually
    test_laplacian_sin()
    test_finite_difference_laplacian()
    test_poisson_residual()
    print("All tests passed!") 