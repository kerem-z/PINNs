"""Physics-Informed Neural Network (PINN) for Poisson equation."""

# Re-export network components
from .nets import PINN, MLP, create_pinn_model

# Re-export differential operators and sampling functions
from .operators import (
    grad, 
    laplacian, 
    sample_domain, 
    sample_boundary,
    poisson_equation,
    dirichlet_bc,
)

# Re-export loss and training functions
from .loss import (
    mean_squared_error,
    compute_poisson_loss,
    train_step,
    eval_step,
    create_optimizer,
    init_model_and_state,
    TrainState,
)

# Re-export components for easier imports
# Example:
# from .nets import PINN
# from .operators import laplacian, sample_boundary
# from .loss import loss_fn, train_step 