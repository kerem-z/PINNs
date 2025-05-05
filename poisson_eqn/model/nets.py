"""Neural network model definitions for Physics-Informed Neural Networks."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, List, Optional, Tuple, Sequence


class MLP(nn.Module):
    """Standard multi-layer perceptron with residual connections."""
    features: Sequence[int]
    activation: Callable = nn.tanh
    
    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            y = nn.Dense(feat)(x)
            if i < len(self.features) - 1:  # Don't apply activation to final output
                y = self.activation(y)
            x = y
        return x


class PINN(nn.Module):
    """Physics-Informed Neural Network.
    
    A neural network that can be used to solve PDEs using the PINN method.
    """
    input_dim: int
    output_dim: int
    hidden_layers: List[int]
    activation: str = "tanh"
    
    def setup(self):
        # Convert activation function name to Flax function
        activation_map = {
            "tanh": nn.tanh,
            "relu": nn.relu,
            "gelu": nn.gelu,
            "swish": nn.swish,
            "sigmoid": nn.sigmoid,
            "softplus": nn.softplus,
        }
        act_fn = activation_map.get(self.activation, nn.tanh)
        
        # Define all hidden layers plus the output layer
        # Convert to list to ensure proper concatenation
        hidden_layers_list = list(self.hidden_layers)
        all_features = hidden_layers_list + [self.output_dim]
        self.net = MLP(features=all_features, activation=act_fn)
    
    def __call__(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor with shape (..., input_dim)
            
        Returns:
            Output tensor with shape (..., output_dim)
        """
        return self.net(x)
    
    
def create_pinn_model(config):
    """Helper to create a PINN model from configuration.
    
    Args:
        config: Configuration dictionary/object with model parameters
        
    Returns:
        Initialized PINN model
    """
    # Extract model configuration
    input_dim = config.model.input_dim
    output_dim = config.model.output_dim
    depth = config.model.depth
    width = config.model.width
    activation = config.model.activation
    
    # Create hidden layer dimensions (as a list, not tuple)
    hidden_layers = [width] * depth
    
    # Create model
    return PINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        activation=activation
    ) 