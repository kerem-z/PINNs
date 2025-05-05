"""Differential operators and domain sampling functions for PINNs."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, List, Tuple, Union, Dict


def grad(f: Callable, argnums: int = 0) -> Callable:
    """Compute gradient of a function with respect to its inputs.
    
    Args:
        f: Function to differentiate
        argnums: Which input argument to differentiate with respect to
        
    Returns:
        Gradient function
    """
    return jax.grad(f, argnums=argnums)


def laplacian(f: Callable, argnums: int = 0) -> Callable:
    """Compute the Laplacian of a function.
    
    For a function f(x), the Laplacian is ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ...
    
    Args:
        f: Function to differentiate
        argnums: Which input argument to differentiate with respect to
        
    Returns:
        Function that computes the Laplacian
    """
    def _laplacian_scalar(x):
        # Handle scalar output from f
        def _f_sum_output(*args):
            return jnp.sum(f(*args))
        
        hessian_diag = jax.hessian(_f_sum_output, argnums=argnums)(x)
        return jnp.trace(hessian_diag)
    
    def _laplacian_vector(x):
        # Vector-valued function case (e.g., f returns multiple values)
        y = f(x)
        laplacians = []
        
        for i in range(y.shape[-1]):
            # Extract each output component
            def _f_i(x):
                return f(x)[..., i]
            
            # Compute Laplacian of this component
            laplacians.append(_laplacian_scalar(_f_i, x))
            
        return jnp.stack(laplacians, axis=-1)
    
    def wrapped_laplacian(x):
        # Determine which implementation to use based on output shape
        y = f(x)
        if y.ndim == 0 or (y.ndim == 1 and y.shape[0] == 1):
            return _laplacian_scalar(x)
        else:
            return _laplacian_vector(x)
    
    return wrapped_laplacian


def sample_domain(
    bounds: List[List[float]],
    n_points: int, 
    rng_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Sample points uniformly from the interior of a hyper-rectangle.
    
    Args:
        bounds: List of [min, max] bounds for each dimension
        n_points: Number of points to sample
        rng_key: JAX random key
        
    Returns:
        Array of shape (n_points, n_dims) containing sampled points
    """
    n_dims = len(bounds)
    points = []
    
    # Split the PRNG key for each dimension
    keys = jax.random.split(rng_key, n_dims)
    
    for i, (low, high) in enumerate(bounds):
        # Sample uniformly from [low, high)
        dim_samples = jax.random.uniform(
            keys[i], 
            shape=(n_points,), 
            minval=low, 
            maxval=high
        )
        points.append(dim_samples)
    
    # Stack to get points with shape (n_points, n_dims)
    return jnp.stack(points, axis=1)


def sample_boundary(
    bounds: List[List[float]],
    n_points: int,
    rng_key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points from the boundary of a hyper-rectangle.
    
    In 2D, this samples from the 4 edges of the rectangle.
    
    Args:
        bounds: List of [min, max] bounds for each dimension
        n_points: Total number of boundary points (distributed evenly)
        rng_key: JAX random key
        
    Returns:
        Tuple of (points, normals) where:
            - points: Array of shape (n_points, n_dims) containing boundary points
            - normals: Array of shape (n_points, n_dims) containing outward unit normals
    """
    n_dims = len(bounds)
    
    if n_dims != 2:
        raise NotImplementedError(
            f"Boundary sampling currently only implemented for 2D domains, got {n_dims}D"
        )
    
    # For a 2D rectangle, we have 4 edges
    n_edges = 2 * n_dims
    points_per_edge = n_points // n_edges
    
    if points_per_edge * n_edges != n_points:
        raise ValueError(
            f"For even distribution, n_points ({n_points}) must be divisible by "
            f"number of edges ({n_edges})"
        )
    
    # Split the random key for each edge
    keys = jax.random.split(rng_key, n_edges)
    
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # List to collect boundary points and their normals
    all_points = []
    all_normals = []
    
    # Bottom edge (y = y_min)
    x_samples = jax.random.uniform(keys[0], (points_per_edge,), minval=x_min, maxval=x_max)
    edge_points = jnp.stack([x_samples, jnp.ones_like(x_samples) * y_min], axis=1)
    edge_normals = jnp.zeros_like(edge_points).at[:, 1].set(-1.0)  # Normal: (0, -1)
    all_points.append(edge_points)
    all_normals.append(edge_normals)
    
    # Right edge (x = x_max)
    y_samples = jax.random.uniform(keys[1], (points_per_edge,), minval=y_min, maxval=y_max)
    edge_points = jnp.stack([jnp.ones_like(y_samples) * x_max, y_samples], axis=1)
    edge_normals = jnp.zeros_like(edge_points).at[:, 0].set(1.0)  # Normal: (1, 0)
    all_points.append(edge_points)
    all_normals.append(edge_normals)
    
    # Top edge (y = y_max)
    x_samples = jax.random.uniform(keys[2], (points_per_edge,), minval=x_min, maxval=x_max)
    edge_points = jnp.stack([x_samples, jnp.ones_like(x_samples) * y_max], axis=1)
    edge_normals = jnp.zeros_like(edge_points).at[:, 1].set(1.0)  # Normal: (0, 1)
    all_points.append(edge_points)
    all_normals.append(edge_normals)
    
    # Left edge (x = x_min)
    y_samples = jax.random.uniform(keys[3], (points_per_edge,), minval=y_min, maxval=y_max)
    edge_points = jnp.stack([jnp.ones_like(y_samples) * x_min, y_samples], axis=1)
    edge_normals = jnp.zeros_like(edge_points).at[:, 0].set(-1.0)  # Normal: (-1, 0)
    all_points.append(edge_points)
    all_normals.append(edge_normals)
    
    # Concatenate all points and normals
    return jnp.concatenate(all_points, axis=0), jnp.concatenate(all_normals, axis=0)


def poisson_equation(u_fn: Callable, x: jnp.ndarray, f: Callable) -> jnp.ndarray:
    """Compute the residual of the Poisson equation: ∇²u = f.
    
    Args:
        u_fn: Neural network function approximating u(x)
        x: Input coordinates where to evaluate the equation
        f: Right-hand side function f(x) in ∇²u = f
        
    Returns:
        Residual of the Poisson equation at x
    """
    # Define a function that takes x as input and returns u(x)
    def u(x_input):
        return u_fn(x_input)
    
    # Compute Laplacian of u
    laplacian_u = laplacian(u)(x)
    
    # Residual: ∇²u - f
    return laplacian_u - f(x)


def dirichlet_bc(u_fn: Callable, x_boundary: jnp.ndarray, g: Callable) -> jnp.ndarray:
    """Compute the residual of Dirichlet boundary conditions: u = g on boundary.
    
    Args:
        u_fn: Neural network function approximating u(x)
        x_boundary: Boundary points
        g: Boundary condition function g(x)
        
    Returns:
        Residual of the boundary condition at x_boundary
    """
    # Predicted values on the boundary
    u_pred = u_fn(x_boundary)
    
    # Target values on the boundary
    u_target = g(x_boundary)
    
    # Residual: u - g
    return u_pred - u_target 