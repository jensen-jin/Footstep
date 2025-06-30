"""
Legacy compatibility adapter for existing configuration classes.

This module provides backward compatibility with existing hardcoded
configuration classes while transitioning to the new robot registry system.
"""

from typing import Type, Dict, Any, Optional
import inspect

from .factory import RobotConfigFactory
from ..base.legged_robot_config import LeggedRobotCfg


class LegacyConfigAdapter:
    """
    Adapter to maintain compatibility with existing configuration classes.
    
    This allows existing code to continue working while new code can use
    the robot registry system.
    """
    
    # Mapping from legacy config classes to robot names
    _legacy_mapping: Dict[str, str] = {
        'HumanoidControllerCfg': 'mit_humanoid_fixed_arms',
        'HumanoidVanillaCfg': 'mit_humanoid_fixed_arms_base',
        # Add more mappings as needed
    }
    
    @classmethod
    def register_legacy_mapping(cls, legacy_class_name: str, robot_name: str):
        """Register a mapping from legacy config class to robot name."""
        cls._legacy_mapping[legacy_class_name] = robot_name
    
    @classmethod
    def from_legacy_config(
        cls, 
        legacy_config_class: Type[LeggedRobotCfg],
        guidance_model: str = 'limp',
        **overrides
    ) -> LeggedRobotCfg:
        """
        Create a new config from a legacy configuration class.
        
        Args:
            legacy_config_class: The legacy configuration class
            guidance_model: Guidance model type ('limp' or 'ipc3d')
            **overrides: Additional overrides
            
        Returns:
            New configuration instance
        """
        class_name = legacy_config_class.__name__
        
        if class_name not in cls._legacy_mapping:
            raise ValueError(
                f"No robot mapping found for legacy config '{class_name}'. "
                f"Available mappings: {list(cls._legacy_mapping.keys())}"
            )
        
        robot_name = cls._legacy_mapping[class_name]
        
        print(f"🔄 Converting legacy config '{class_name}' to robot '{robot_name}'")
        
        return RobotConfigFactory.create_config(
            robot_name=robot_name,
            guidance_model=guidance_model,
            **overrides
        )
    
    @classmethod
    def extract_config_from_legacy(
        cls, 
        legacy_config_instance: LeggedRobotCfg
    ) -> Dict[str, Any]:
        """
        Extract configuration parameters from a legacy config instance.
        
        This can be used to convert existing configs to robot specifications.
        
        Args:
            legacy_config_instance: Instance of legacy config
            
        Returns:
            Dictionary of extracted parameters
        """
        extracted = {}
        
        # Extract basic parameters
        if hasattr(legacy_config_instance, 'env'):
            extracted['num_actuators'] = getattr(legacy_config_instance.env, 'num_actuators', None)
        
        if hasattr(legacy_config_instance, 'asset'):
            extracted['urdf_path'] = getattr(legacy_config_instance.asset, 'file', None)
            extracted['end_effectors'] = getattr(legacy_config_instance.asset, 'end_effectors', None)
        
        if hasattr(legacy_config_instance, 'init_state'):
            extracted['default_joint_angles'] = getattr(
                legacy_config_instance.init_state, 'default_joint_angles', {}
            )
            extracted['joint_pos_range'] = getattr(
                legacy_config_instance.init_state, 'dof_pos_range', {}
            )
        
        if hasattr(legacy_config_instance, 'control'):
            extracted['stiffness'] = getattr(legacy_config_instance.control, 'stiffness', {})
            extracted['damping'] = getattr(legacy_config_instance.control, 'damping', {})
        
        if hasattr(legacy_config_instance, 'rewards'):
            extracted['base_height_target'] = getattr(
                legacy_config_instance.rewards, 'base_height_target', None
            )
        
        return extracted
    
    @classmethod
    def create_robot_spec_from_legacy(
        cls,
        name: str,
        legacy_config_class: Type[LeggedRobotCfg],
        description: str = "",
        category: str = "unknown"
    ) -> 'RobotSpec':
        """
        Create a RobotSpec from a legacy configuration class.
        
        This is useful for migrating existing configs to the registry system.
        
        Args:
            name: Name for the new robot spec
            legacy_config_class: Legacy config class
            description: Description of the robot
            category: Robot category
            
        Returns:
            New RobotSpec instance
        """
        from .registry import RobotSpec
        
        # Create an instance to extract parameters
        legacy_instance = legacy_config_class()
        extracted = cls.extract_config_from_legacy(legacy_instance)
        
        # Build joint limits from position ranges
        joint_limits = {}
        if 'joint_pos_range' in extracted:
            for joint, range_vals in extracted['joint_pos_range'].items():
                if isinstance(range_vals, list) and len(range_vals) == 2:
                    joint_limits[joint] = range_vals
        
        # Build control config
        control_config = {}
        if 'stiffness' in extracted:
            control_config['stiffness'] = extracted['stiffness']
        if 'damping' in extracted:
            control_config['damping'] = extracted['damping']
        
        # Build physical params
        physical_params = {}
        if 'base_height_target' in extracted and extracted['base_height_target']:
            physical_params['base_height'] = extracted['base_height_target']
        if 'end_effectors' in extracted and extracted['end_effectors']:
            physical_params['end_effectors'] = extracted['end_effectors']
        
        return RobotSpec(
            name=name,
            urdf_path=extracted.get('urdf_path', ''),
            num_actuators=extracted.get('num_actuators', 0),
            default_joint_angles=extracted.get('default_joint_angles', {}),
            joint_limits=joint_limits,
            control_config=control_config,
            physical_params=physical_params,
            description=description or f"Migrated from {legacy_config_class.__name__}",
            category=category
        )


def create_legacy_wrapper(robot_name: str, guidance_model: str = 'limp'):
    """
    Create a wrapper class that mimics legacy config behavior.
    
    This can be used as a drop-in replacement for legacy config classes.
    
    Args:
        robot_name: Name of robot in registry
        guidance_model: Guidance model type
        
    Returns:
        Class that behaves like legacy config
    """
    
    class LegacyWrapper(LeggedRobotCfg):
        """Wrapper that provides legacy config interface."""
        
        def __init__(self):
            # Generate config using new system
            config = RobotConfigFactory.create_config(robot_name, guidance_model)
            
            # Copy all attributes from generated config
            for attr_name in dir(config):
                if not attr_name.startswith('_'):
                    setattr(self, attr_name, getattr(config, attr_name))
        
        @classmethod
        def get_robot_name(cls):
            """Get the robot name for this wrapper."""
            return robot_name
        
        @classmethod
        def get_guidance_model(cls):
            """Get the guidance model for this wrapper."""
            return guidance_model
    
    LegacyWrapper.__name__ = f"{robot_name}LegacyWrapper"
    LegacyWrapper.__qualname__ = f"{robot_name}LegacyWrapper"
    
    return LegacyWrapper


# Pre-create some common legacy wrappers
HumanoidControllerCfgNew = create_legacy_wrapper('mit_humanoid_fixed_arms', 'limp')
HumanoidControllerCfgIPC = create_legacy_wrapper('mit_humanoid_fixed_arms', 'ipc3d')