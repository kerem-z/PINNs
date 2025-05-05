"""Main script for training and evaluating PINNs for the Poisson equation.

This script orchestrates:
1. Configuration loading (via Hydra)
2. Weights & Biases initialization
3. Dataset construction
4. Training process
5. Evaluation and visualization
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import jax
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
import time
from tqdm import tqdm

# Import model components
from model import (
    init_model_and_state,
    sample_domain,
    sample_boundary,
    train_step, 
    eval_step,
)

# Import visualization utilities - keeping them separate
from visualizations.plots import (
    plot_solution_2d,
    plot_convergence,
    plot_pde_residual,
)
from visualizations.wandb_callbacks import (
    log_solution_plot,
    log_residual_plot,
    log_loss_metrics,
)


def get_source_and_boundary_terms():
    """Define the source term f(x,y) and boundary condition g(x,y) for the Poisson equation.
    
    Here we set up a simple example with known analytic solution:
    - Equation: ∇²u(x,y) = f(x,y) = -8π²sin(2πx)sin(2πy)
    - Boundary condition: u(x,y) = 0 on the boundary
    - Analytic solution: u(x,y) = sin(2πx)sin(2πy)
    
    Returns:
        Tuple of (f_fn, g_fn, exact_fn)
    """
    # Source term: -8π²sin(2πx)sin(2πy)
    def f_fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        return -8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    
    # Boundary condition (Dirichlet): u(x,y) = 0 on the boundary
    def g_fn(xy):
        return jnp.zeros_like(xy[..., 0])
    
    # Exact solution: sin(2πx)sin(2πy)
    def exact_fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    
    return f_fn, g_fn, exact_fn


@hydra.main(config_path="../configs", config_name="poisson", version_base=None)
def main(config: DictConfig):
    """Main function for training and evaluating PINN for Poisson equation.
    
    Args:
        config: Hydra configuration
    """
    # Print configuration
    print(OmegaConf.to_yaml(config))
    
    # Set random seed
    rng_key = jax.random.PRNGKey(config.data.seed)
    np.random.seed(config.data.seed)
    
    # Initialize wandb
    if not os.getenv("WANDB_DISABLED", ""):
        wandb.init(
            project=config.experiment.project,
            name=config.experiment.name,
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.experiment.tags,
            notes=config.experiment.get("notes", ""),
        )
    
    # Create problem definition
    f_fn, g_fn, exact_fn = get_source_and_boundary_terms()
    
    # Initialize model and optimization state
    model, state = init_model_and_state(config, rng_key)
    
    # Training loop
    loss_history = {
        'total_loss': [],
        'pde_loss': [],
        'bc_loss': [],
    }
    
    # Define domain and sampling points
    domain_bounds = config.data.domain_bounds
    n_domain = config.data.n_domain
    n_boundary = config.data.n_boundary
    
    # Sample initial points
    rng_key, sample_key = jax.random.split(rng_key)
    x_domain = sample_domain(domain_bounds, n_domain, sample_key)
    
    rng_key, sample_key = jax.random.split(rng_key)
    x_boundary, _ = sample_boundary(domain_bounds, n_boundary, sample_key)
    
    # Extract training settings
    epochs = config.training.epochs
    log_freq = config.training.log_freq
    
    # BC weight scheduling (cosine annealing)
    bc_weight_start = config.training.bc_weight.start
    bc_weight_end = config.training.bc_weight.end
    
    # Optionally enable early stopping
    early_stopping_patience = config.training.get("early_stopping_patience", None)
    if early_stopping_patience is not None:
        best_loss = float('inf')
        patience_counter = 0
    
    # Create a prediction function that uses the current model state
    def predict_fn(x):
        return state.apply_fn({'params': state.params}, x)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in tqdm(range(1, epochs+1)):
        # For each epoch, resample the domain points to focus on difficult regions
        if epoch > 1:  # Keep first epoch fixed for reproducibility
            rng_key, sample_key = jax.random.split(rng_key)
            x_domain = sample_domain(domain_bounds, n_domain, sample_key)
        
        # Calculate BC weight based on schedule
        # Cosine annealing from start to end
        progress = epoch / epochs
        bc_weight = bc_weight_end + 0.5 * (bc_weight_start - bc_weight_end) * (1 + jnp.cos(jnp.pi * progress))
        
        # Update state (training step)
        state, loss_info = train_step(state, x_domain, x_boundary, f_fn, g_fn, bc_weight)
        
        # Record losses
        for k, v in loss_info.items():
            if k in loss_history:
                loss_history[k].append(float(v))
        
        # Log to wandb at specified frequency
        if epoch % log_freq == 0 or epoch == 1 or epoch == epochs:
            # Log metrics
            log_loss_metrics(loss_info, epoch)
            
            # Log solution and residual plots
            log_solution_plot(predict_fn, domain_bounds, epoch)
            log_residual_plot(predict_fn, f_fn, domain_bounds, epoch)
            
            # Print progress
            print(f"Epoch {epoch}/{epochs}, Loss: {float(loss_info['total_loss']):.6f}, "
                  f"PDE Loss: {float(loss_info['pde_loss']):.6f}, "
                  f"BC Loss: {float(loss_info['bc_loss']):.6f}")
        
        # Check for early stopping
        if early_stopping_patience is not None:
            current_loss = float(loss_info['total_loss'])
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Print training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Final evaluation
    print("Final evaluation...")
    
    # Sample a dense grid for visualization and evaluation
    rng_key, eval_key = jax.random.split(rng_key)
    x_eval_domain = sample_domain(domain_bounds, 1000, eval_key)
    
    # Evaluate the model on the test domain
    eval_results = eval_step(state, x_eval_domain, x_boundary, f_fn, g_fn)
    print(f"Final PDE Loss: {float(eval_results['pde_loss']):.8f}")
    print(f"Final BC Loss: {float(eval_results['bc_loss']):.8f}")
    
    # Create summary visualization
    # 1. Solution plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot PINN solution
    plot_solution_2d(
        predict_fn, 
        domain_bounds, 
        ax=axes[0], 
        title="PINN Solution"
    )
    
    # Plot exact solution
    plot_solution_2d(
        exact_fn, 
        domain_bounds, 
        ax=axes[1], 
        title="Exact Solution"
    )
    
    # Plot PDE residual
    plot_pde_residual(
        predict_fn, 
        f_fn, 
        domain_bounds, 
        ax=axes[2], 
        title="PDE Residual"
    )
    
    fig.tight_layout()
    plt.savefig("final_results.png", dpi=300, bbox_inches="tight")
    
    if not os.getenv("WANDB_DISABLED", ""):
        wandb.log({"final_results": wandb.Image(fig)})
        wandb.finish()
    
    # Plot and save loss history
    fig = plot_convergence(loss_history, log_scale=True, return_fig=True)
    plt.savefig("loss_history.png", dpi=300, bbox_inches="tight")
    
    print("Done!")


if __name__ == "__main__":
    main() 