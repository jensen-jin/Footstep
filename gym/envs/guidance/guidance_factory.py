#!/usr/bin/env python3
"""
Guidance Model Factory

This module provides factory functions to create guidance models
based on configuration and robot specifications.
"""

import numpy as np
from typing import Dict, Any, Optional, Type
from .base_guidance import BaseGuidanceModel, LegacyGuidanceWrapper
from .ipc3d_controller import IPC3D, IPC3DParams, create_ipc3d_from_robot_spec
from .ipc3d_guidance import IPC3DGuidanceModel


class GuidanceModelFactory:
    """Factory for creating guidance models."""
    
    _registered_models: Dict[str, Type[BaseGuidanceModel]] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseGuidanceModel]):
        """Register a guidance model class."""
        cls._registered_models[model_type] = model_class
        
    @classmethod
    def list_available_models(cls) -> list:
        """List all available guidance model types."""
        return list(cls._registered_models.keys())
        
    @classmethod
    def create_model(cls, model_type: str, robot_spec, guidance_config, **kwargs) -> BaseGuidanceModel:
        """
        Create guidance model instance.
        
        Args:
            model_type: Type of guidance model ('limp', 'ipc3d')
            robot_spec: Robot specification from registry
            guidance_config: Guidance configuration object
            **kwargs: Additional parameters
            
        Returns:
            Guidance model instance
        """
        if model_type == 'limp':
            return cls._create_limp_model(robot_spec, guidance_config, **kwargs)
        elif model_type == 'ipc3d':
            return cls._create_ipc3d_model(robot_spec, guidance_config, **kwargs)
        elif model_type in cls._registered_models:
            model_class = cls._registered_models[model_type]
            return model_class(model_type, guidance_config, robot_spec, **kwargs)
        else:
            raise ValueError(f"Unknown guidance model type: {model_type}. Available: {cls.list_available_models()}")
    
    @classmethod
    def _create_limp_model(cls, robot_spec, guidance_config, **kwargs) -> BaseGuidanceModel:
        """Create LIMP guidance model."""
        try:
            # Try to import existing LIMP implementation
            from gym.envs.humanoid.humanoid_gait_generator import LimpGaitGenerator
            
            # Create legacy wrapper
            limp_params = {
                'com_height': guidance_config.limp_com_height,
                'step_length': guidance_config.limp_step_length,
                'step_width': guidance_config.limp_step_width, 
                'step_time': guidance_config.limp_step_time,
                'dt': guidance_config.dt
            }
            
            legacy_guidance = LimpGaitGenerator(**limp_params)
            return LegacyGuidanceWrapper(legacy_guidance, 'limp', limp_params)
            
        except ImportError:
            print("Warning: LIMP implementation not found, creating placeholder")
            return cls._create_placeholder_model('limp', robot_spec, guidance_config)
    
    @classmethod  
    def _create_ipc3d_model(cls, robot_spec, guidance_config, **kwargs) -> BaseGuidanceModel:
        """Create IPC3D guidance model."""
        
        # Extract robot parameters
        mass = robot_spec.physical_params.get('mass', 42.0)
        com_height = robot_spec.physical_params.get('base_height', 0.62)
        
        # Create IPC3D parameters
        params = IPC3DParams(
            mass_cart=mass * 0.9,  # 90% of mass as cart
            mass_pole=mass * 0.1,  # 10% of mass as pole  
            pole_length=com_height,
            inertia=guidance_config.ipc3d_inertia,
            gravity=9.81,
            damping=guidance_config.ipc3d_damping,
            dt=guidance_config.dt,
            max_force=guidance_config.ipc3d_max_force,
            q_position=guidance_config.lqr_q_matrix[0],
            q_velocity=guidance_config.lqr_q_matrix[1],
            r_control=guidance_config.lqr_r_matrix[0],
            control_mode=1 if guidance_config.ipc3d_control_mode == 'velocity' else 0
        )
        
        # Create IPC3D guidance model
        return IPC3DGuidanceModel(params, robot_spec, guidance_config)
    
    @classmethod
    def _create_placeholder_model(cls, model_type: str, robot_spec, guidance_config) -> BaseGuidanceModel:
        """Create placeholder model for testing."""
        from .placeholder_guidance import PlaceholderGuidanceModel
        return PlaceholderGuidanceModel(model_type, guidance_config.__dict__)
    
    @classmethod
    def create_from_config_dict(cls, config: Dict[str, Any]) -> BaseGuidanceModel:
        """Create guidance model from configuration dictionary."""
        model_type = config.get('model_type', 'limp')
        
        # Create dummy robot spec for testing
        class DummyRobotSpec:
            def __init__(self):
                self.physical_params = {
                    'mass': config.get('robot_mass', 42.0),
                    'base_height': config.get('robot_height', 0.62)
                }
        
        # Create dummy guidance config
        class DummyGuidanceConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
                    
                # Set defaults
                self.dt = getattr(self, 'dt', 0.02)
                self.lqr_q_matrix = getattr(self, 'lqr_q_matrix', [10.0, 1.0])
                self.lqr_r_matrix = getattr(self, 'lqr_r_matrix', [0.1])
                self.ipc3d_inertia = getattr(self, 'ipc3d_inertia', 0.1)
                self.ipc3d_damping = getattr(self, 'ipc3d_damping', 0.1)
                self.ipc3d_max_force = getattr(self, 'ipc3d_max_force', 500.0)
                self.ipc3d_control_mode = getattr(self, 'ipc3d_control_mode', 'velocity')
        
        robot_spec = DummyRobotSpec()
        guidance_config = DummyGuidanceConfig(config)
        
        return cls.create_model(model_type, robot_spec, guidance_config)


# Register built-in models
def register_builtin_models():
    """Register built-in guidance models."""
    # IPC3D model is registered via import
    try:
        from .ipc3d_guidance import IPC3DGuidanceModel
        GuidanceModelFactory.register_model('ipc3d_direct', IPC3DGuidanceModel)
    except ImportError:
        pass
        
    # Register placeholder model for testing
    try:
        from .placeholder_guidance import PlaceholderGuidanceModel  
        GuidanceModelFactory.register_model('placeholder', PlaceholderGuidanceModel)
    except ImportError:
        pass


# Auto-register on import
register_builtin_models()


def create_guidance_model(model_type: str, robot_spec=None, guidance_config=None, **kwargs) -> BaseGuidanceModel:
    """
    Convenience function to create guidance model.
    
    Args:
        model_type: Type of guidance model
        robot_spec: Robot specification (optional)
        guidance_config: Guidance configuration (optional)
        **kwargs: Additional parameters
        
    Returns:
        Guidance model instance
    """
    return GuidanceModelFactory.create_model(model_type, robot_spec, guidance_config, **kwargs)