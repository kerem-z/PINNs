"""Weights & Biases integration utilities for logging PINN training."""

import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Callable, Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp


def log_solution_plot(
    u_fn: Callable,
    domain_bounds: List[List[float]],
    step: int,
    n_grid: int = 100,
    name: str = "pinn_solution",
):
    """Create and log a solution plot to wandb.
    
    Args:
        u_fn: Function to evaluate the solution
        domain_bounds: Domain boundaries [x_min, x_max], [y_min, y_max]
        step: Current training step
        n_grid: Number of grid points in each dimension
        name: Name for the plot in wandb
    """
    # Import locally to avoid dependency loops
    from visualizations.plots import plot_solution_2d
    
    # Create figure
    fig = plot_solution_2d(
        u_fn=u_fn,
        domain_bounds=domain_bounds,
        n_grid=n_grid,
        title=f"PINN Solution (Step {step})",
        return_fig=True,
    )
    
    # Log to wandb
    wandb.log({name: wandb.Image(fig)}, step=step)
    
    # Close figure to free memory
    plt.close(fig)


def log_residual_plot(
    u_fn: Callable,
    f_fn: Callable,
    domain_bounds: List[List[float]],
    step: int,
    n_grid: int = 100,
    name: str = "pde_residual",
):
    """Create and log a PDE residual plot to wandb.
    
    Args:
        u_fn: Neural network function
        f_fn: Source term in PDE
        domain_bounds: Domain boundaries [x_min, x_max], [y_min, y_max]
        step: Current training step
        n_grid: Number of grid points in each dimension
        name: Name for the plot in wandb
    """
    # Import locally to avoid dependency loops
    from visualizations.plots import plot_pde_residual
    
    # Create figure
    fig = plot_pde_residual(
        u_fn=u_fn,
        f_fn=f_fn,
        domain_bounds=domain_bounds,
        n_grid=n_grid,
        title=f"PDE Residual (Step {step})",
        return_fig=True,
    )
    
    # Log to wandb
    wandb.log({name: wandb.Image(fig)}, step=step)
    
    # Close figure to free memory
    plt.close(fig)


def log_loss_metrics(
    loss_info: Dict[str, jnp.ndarray],
    step: int,
):
    """Log loss metrics to wandb.
    
    Args:
        loss_info: Dictionary with loss metrics
        step: Current training step
    """
    # Convert jax array metrics to Python values
    metrics = {k: float(v) for k, v in loss_info.items()}
    
    # Log metrics
    wandb.log(metrics, step=step)


def log_model_gradients(
    grads: Dict[str, Any],
    step: int,
    include_histograms: bool = True,
):
    """Log gradient statistics to wandb.
    
    Args:
        grads: Gradients dictionary
        step: Current training step
        include_histograms: Whether to log histogram of gradients
    """
    # Flatten nested dictionary of gradients
    flat_grads = {}
    
    def _flatten_dict(d, prefix=""):
        for k, v in d.items():
            new_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                _flatten_dict(v, new_key)
            else:
                flat_grads[new_key] = v

    _flatten_dict(grads)
    
    # Compute and log gradient statistics
    grad_stats = {}
    for name, g in flat_grads.items():
        if hasattr(g, 'shape'):  # Check if it's an array
            grad_stats[f"grad_norm/{name}"] = jnp.linalg.norm(g)
            grad_stats[f"grad_mean/{name}"] = jnp.mean(g)
            grad_stats[f"grad_std/{name}"] = jnp.std(g)
            
            # Log histogram
            if include_histograms:
                grad_stats[f"grad_hist/{name}"] = wandb.Histogram(g)
    
    # Log all statistics
    wandb.log(grad_stats, step=step) 