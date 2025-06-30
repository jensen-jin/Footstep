"""
Model factory for selecting different neural network architectures.
支持标准Actor-Critic和高级DWAQ架构的工厂模式
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def get_model_class(architecture: str = "standard"):
    """
    Get the appropriate model class based on architecture type.
    
    Args:
        architecture: Type of architecture ("standard" or "dwaq")
        
    Returns:
        Model class
    """
    if architecture.lower() == "standard":
        from .actor_critic import ActorCritic
        return ActorCritic
    elif architecture.lower() == "dwaq":
        from .actor_critic_DWAQ import ActorCritic_DWAQ
        return ActorCritic_DWAQ
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: 'standard', 'dwaq'")

def create_model(
    architecture: str,
    num_actor_obs: int,
    num_critic_obs: int, 
    num_actions: int,
    actor_hidden_dims: list = [512, 256, 128],
    critic_hidden_dims: list = [512, 256, 128],
    activation: str = "elu",
    init_noise_std: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Create a model instance with the specified architecture.
    
    Args:
        architecture: Architecture type ("standard" or "dwaq")
        num_actor_obs: Number of actor observations
        num_critic_obs: Number of critic observations  
        num_actions: Number of actions
        actor_hidden_dims: Hidden layer dimensions for actor
        critic_hidden_dims: Hidden layer dimensions for critic
        activation: Activation function name
        init_noise_std: Initial noise standard deviation
        **kwargs: Additional architecture-specific parameters
        
    Returns:
        Model instance
    """
    model_class = get_model_class(architecture)
    
    if architecture.lower() == "standard":
        # Standard ActorCritic parameters
        return model_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims, 
            activation=activation,
            init_noise_std=init_noise_std
        )
    
    elif architecture.lower() == "dwaq":
        # DWAQ specific parameters
        cenet_in_dim = kwargs.get('cenet_in_dim', num_actor_obs - 20)  # Default context encoding input
        cenet_out_dim = kwargs.get('cenet_out_dim', 20)  # Default context encoding output
        dropout_prob = kwargs.get('dropout_prob', 0.1)  # Default dropout probability
        
        return model_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            cenet_in_dim=cenet_in_dim,
            cenet_out_dim=cenet_out_dim,
            activation=activation,
            init_noise_std=init_noise_std,
            dropout_prob=dropout_prob
        )
    
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

def get_architecture_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available architectures.
    
    Returns:
        Dictionary with architecture information
    """
    return {
        "standard": {
            "name": "Standard Actor-Critic",
            "description": "Traditional actor-critic architecture with configurable hidden layers",
            "features": [
                "Separate actor and critic networks",
                "Configurable hidden dimensions", 
                "Multiple activation functions",
                "Gaussian policy distribution"
            ],
            "use_cases": [
                "General reinforcement learning tasks",
                "Single robot training",
                "Standard locomotion control"
            ],
            "parameters": {
                "required": ["num_actor_obs", "num_critic_obs", "num_actions"],
                "optional": ["actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std"]
            }
        },
        "dwaq": {
            "name": "DWAQ (Deep With Autoencoder Quantization)",
            "description": "Advanced architecture with encoder-decoder and context understanding",
            "features": [
                "Variational autoencoder elements",
                "Context encoding with latent space", 
                "Separate velocity encoding",
                "Reparameterization trick",
                "Dropout support for regularization"
            ],
            "use_cases": [
                "Multi-robot morphology training",
                "Enhanced context understanding",
                "Complex locomotion patterns",
                "Transfer learning scenarios"
            ],
            "parameters": {
                "required": ["num_actor_obs", "num_critic_obs", "num_actions", "cenet_in_dim", "cenet_out_dim"],
                "optional": ["activation", "init_noise_std", "dropout_prob"]
            }
        }
    }

def print_architecture_comparison():
    """Print a comparison of available architectures."""
    info = get_architecture_info()
    
    print("🧠 Available Neural Network Architectures")
    print("=" * 70)
    
    for arch_name, arch_info in info.items():
        print(f"\n📋 {arch_info['name']} ({arch_name})")
        print(f"   Description: {arch_info['description']}")
        
        print("   ✨ Features:")
        for feature in arch_info['features']:
            print(f"      • {feature}")
        
        print("   🎯 Use Cases:")
        for use_case in arch_info['use_cases']:
            print(f"      • {use_case}")
        
        print("   ⚙️ Parameters:")
        print(f"      Required: {', '.join(arch_info['parameters']['required'])}")
        print(f"      Optional: {', '.join(arch_info['parameters']['optional'])}")

def validate_architecture_config(architecture: str, config: Dict[str, Any]) -> bool:
    """
    Validate that a configuration is compatible with the specified architecture.
    
    Args:
        architecture: Architecture name
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    info = get_architecture_info()
    
    if architecture not in info:
        print(f"❌ Unknown architecture: {architecture}")
        return False
    
    arch_info = info[architecture]
    required_params = arch_info['parameters']['required']
    
    missing_params = []
    for param in required_params:
        if param not in config:
            missing_params.append(param)
    
    if missing_params:
        print(f"❌ Missing required parameters for {architecture}: {missing_params}")
        return False
    
    print(f"✅ Configuration is valid for {architecture} architecture")
    return True

__all__ = [
    'get_model_class',
    'create_model', 
    'get_architecture_info',
    'print_architecture_comparison',
    'validate_architecture_config'
]