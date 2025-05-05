"""Plotting utilities for PINN solutions and analysis."""

import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp


def plot_solution_2d(
    u_fn: Callable,
    domain_bounds: List[List[float]],
    n_grid: int = 100,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "PINN Solution",
    colorbar: bool = True,
    cmap: str = "viridis",
    return_fig: bool = False,
    contour_levels: int = 50,
) -> Optional[plt.Figure]:
    """Plot a 2D solution obtained from a PINN.
    
    Args:
        u_fn: Function that computes u(x,y)
        domain_bounds: List of [min, max] for each dimension
        n_grid: Number of grid points in each dimension
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating a new figure
        title: Plot title
        colorbar: Whether to show a colorbar
        cmap: Colormap name
        return_fig: Whether to return the figure object
        contour_levels: Number of contour levels
        
    Returns:
        Figure object if return_fig is True, else None
    """
    # Create grid
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Combine into grid points
    xy = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Compute solution values
    u_values = jax.vmap(u_fn)(jnp.array(xy)).reshape(n_grid, n_grid)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot contour
    contour = ax.contourf(X, Y, u_values, levels=contour_levels, cmap=cmap)
    
    # Add colorbar
    if colorbar:
        fig.colorbar(contour, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make axes equal
    ax.set_aspect('equal')
    
    if return_fig:
        return fig


def plot_convergence(
    loss_history: Dict[str, List[float]],
    log_scale: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot loss convergence curves.
    
    Args:
        loss_history: Dictionary with loss components
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size
        return_fig: Whether to return the figure
        
    Returns:
        Figure object if return_fig is True, else None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each loss component
    for loss_name, values in loss_history.items():
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, label=loss_name)
    
    # Set y-axis to log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Add labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if return_fig:
        return fig


def plot_pde_residual(
    u_fn: Callable,
    f_fn: Callable,
    domain_bounds: List[List[float]],
    n_grid: int = 100,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "PDE Residual: ∇²u - f",
    colorbar: bool = True,
    cmap: str = "RdBu_r",
    return_fig: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    """Plot the residual of the PDE (∇²u - f).
    
    Args:
        u_fn: Neural network function approximating u(x)
        f_fn: Right-hand side function f(x) in ∇²u = f
        domain_bounds: List of [min, max] for each dimension
        n_grid: Number of grid points in each dimension
        figsize: Figure size
        title: Plot title
        colorbar: Whether to show a colorbar
        cmap: Colormap name
        return_fig: Whether to return the figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        Figure object if return_fig is True, else None
    """
    # Create grid
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Compute residual using JAX autodiff
    def residual_fn(x):
        from model.operators import poisson_equation
        return poisson_equation(u_fn, x, f_fn)
    
    # Combine into grid points
    xy = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Compute residual values
    residual_values = jax.vmap(residual_fn)(jnp.array(xy)).reshape(n_grid, n_grid)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color limits if not provided
    if vmin is None:
        vmin = -np.abs(residual_values).max()
    if vmax is None:
        vmax = np.abs(residual_values).max()
    
    # Plot contour
    contour = ax.contourf(
        X, Y, residual_values, cmap=cmap, levels=50, vmin=vmin, vmax=vmax
    )
    
    # Add colorbar
    if colorbar:
        fig.colorbar(contour, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make axes equal
    ax.set_aspect('equal')
    
    if return_fig:
        return fig 