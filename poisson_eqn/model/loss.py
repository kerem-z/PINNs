"""Loss functions and training steps for PINNs."""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Dict, Tuple, Any, List, Mapping, Optional
from flax.training import train_state
import flax.linen as nn

from .operators import poisson_equation, dirichlet_bc, sample_domain, sample_boundary


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics."""
    batch_stats: Optional[Mapping[str, Any]] = None


def create_train_state(
    rng_key: jax.random.PRNGKey,
    model: nn.Module,
    input_shape: Tuple[int, ...],
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Create initial training state.
    
    Args:
        rng_key: JAX random key
        model: Flax model
        input_shape: Shape of input (without batch dimension)
        optimizer: Optax optimizer
    
    Returns:
        Initial train state
    """
    # Initialize the model
    variables = model.init(rng_key, jnp.ones(input_shape))
    
    # Extract parameters and batch stats (if any)
    if 'batch_stats' in variables:
        params = variables['params']
        batch_stats = variables['batch_stats']
    else:
        params = variables
        batch_stats = None
    
    # Create training state
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
    )


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared error between predictions and targets.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        Mean squared error
    """
    return jnp.mean((predictions - targets) ** 2)


def compute_poisson_loss(
    params: Dict,
    x_domain: jnp.ndarray,
    x_boundary: jnp.ndarray,
    state: TrainState,
    f_fn: Callable,  # Source term in Poisson equation: -∇²u = f
    g_fn: Callable,  # Boundary condition: u = g on boundary
    bc_weight: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the physics-informed loss for the Poisson equation.
    
    The loss consists of:
    1. PDE residual at interior points: MSE(∇²u + f)
    2. Boundary condition residual: MSE(u - g) on boundary
    
    Args:
        params: Model parameters
        x_domain: Interior domain points, shape (batch_size, input_dim)
        x_boundary: Boundary points, shape (batch_size, input_dim)
        state: Training state with apply_fn
        f_fn: Source term in Poisson equation
        g_fn: Boundary condition function
        bc_weight: Weight for boundary condition loss
        
    Returns:
        Tuple of (total_loss, loss_components)
    """
    # Create function to predict u given x
    def u_fn(x):
        return state.apply_fn({'params': params}, x)
    
    # Compute PDE residual: ∇²u - f
    pde_residual = poisson_equation(u_fn, x_domain, f_fn)
    pde_loss = mean_squared_error(pde_residual, jnp.zeros_like(pde_residual))
    
    # Compute boundary condition residual: u - g
    bc_residual = dirichlet_bc(u_fn, x_boundary, g_fn)
    bc_loss = mean_squared_error(bc_residual, jnp.zeros_like(bc_residual))
    
    # Total loss
    total_loss = pde_loss + bc_weight * bc_loss
    
    # Return total loss and individual components
    return total_loss, {
        'pde_loss': pde_loss,
        'bc_loss': bc_loss,
        'total_loss': total_loss,
    }


@jax.jit
def train_step(
    state: TrainState,
    x_domain: jnp.ndarray,
    x_boundary: jnp.ndarray,
    f_fn: Callable,
    g_fn: Callable,
    bc_weight: float = 1.0,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Execute a single training step.
    
    Args:
        state: Current training state
        x_domain: Interior domain points
        x_boundary: Boundary points
        f_fn: Source term in Poisson equation
        g_fn: Boundary condition function
        bc_weight: Weight for boundary condition loss
        
    Returns:
        Tuple of (new_state, metrics)
    """
    # Compute loss and gradients
    def loss_fn(params):
        return compute_poisson_loss(
            params, x_domain, x_boundary, state, f_fn, g_fn, bc_weight
        )
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, loss_info), grads = grad_fn(state.params)
    
    # Update state
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss_info


@jax.jit
def eval_step(
    state: TrainState,
    x_domain: jnp.ndarray,
    x_boundary: jnp.ndarray,
    f_fn: Callable,
    g_fn: Callable,
    bc_weight: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Evaluate the model without updating parameters.
    
    Args:
        state: Current training state
        x_domain: Interior domain points
        x_boundary: Boundary points
        f_fn: Source term in Poisson equation
        g_fn: Boundary condition function
        bc_weight: Weight for boundary condition loss
        
    Returns:
        Loss metrics dictionary
    """
    _, loss_info = compute_poisson_loss(
        state.params, x_domain, x_boundary, state, f_fn, g_fn, bc_weight
    )
    return loss_info


def create_optimizer(config) -> optax.GradientTransformation:
    """Create optimizer from configuration.
    
    Args:
        config: Configuration object with training parameters
        
    Returns:
        Optax optimizer
    """
    # Extract optimizer settings
    lr = config.training.lr
    optimizer_name = config.training.optimizer.lower()
    weight_decay = config.training.weight_decay
    b1 = config.training.optimizer_config.b1
    b2 = config.training.optimizer_config.b2
    eps = config.training.optimizer_config.eps
    
    # Optionally add gradient clipping
    if hasattr(config.training, 'clip_grad_norm') and config.training.clip_grad_norm > 0:
        clip_norm = config.training.clip_grad_norm
    else:
        clip_norm = None
    
    # Create optimizer chain
    tx_chain = []
    
    # Base optimizer
    if optimizer_name == 'adam':
        tx_chain.append(optax.adam(
            learning_rate=lr, 
            b1=b1, 
            b2=b2, 
            eps=eps
        ))
    elif optimizer_name == 'adamw':
        tx_chain.append(optax.adamw(
            learning_rate=lr, 
            b1=b1, 
            b2=b2, 
            eps=eps,
            weight_decay=weight_decay
        ))
    elif optimizer_name == 'sgd':
        tx_chain.append(optax.sgd(
            learning_rate=lr,
            momentum=b1 if b1 > 0 else None
        ))
    elif optimizer_name == 'rmsprop':
        tx_chain.append(optax.rmsprop(
            learning_rate=lr,
            decay=0.9,
            eps=eps
        ))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Add gradient clipping if requested
    if clip_norm is not None:
        tx_chain.append(optax.clip_by_global_norm(clip_norm))
    
    # Combine all transformations
    return optax.chain(*tx_chain)


def init_model_and_state(config, rng_key: jax.random.PRNGKey) -> Tuple[nn.Module, TrainState]:
    """Initialize the model and training state from configuration.
    
    Args:
        config: Configuration object
        rng_key: JAX random key
        
    Returns:
        Tuple of (model, train_state)
    """
    from .nets import create_pinn_model
    
    # Create model from config
    model = create_pinn_model(config)
    
    # Create optimizer
    optimizer = create_optimizer(config)
    
    # Initialize training state
    input_shape = (config.model.input_dim,)
    params_key, state_key = jax.random.split(rng_key)
    state = create_train_state(params_key, model, input_shape, optimizer)
    
    return model, state 