"""
Robot configuration factory for automated config generation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import copy

from .registry import RobotRegistry, RobotSpec
from ..base.legged_robot_config import LeggedRobotCfg


@dataclass
class GuidanceConfig:
    """Configuration for guidance models (LIPM/IPC3D)."""
    model_type: str = 'lipm'  # 'lipm' or 'ipc3d'
    
    # Common parameters
    dt: float = 0.02
    gravity: float = 9.81
    
    # LIPM specific parameters
    limp_com_height: Optional[float] = None
    limp_step_time: float = 0.34
    limp_step_length: float = 0.5
    limp_step_width: float = 0.3
    
    # IPC3D specific parameters
    ipc3d_mass_cart: float = 1.0
    ipc3d_mass_pole: float = 0.1
    ipc3d_pole_length: float = 0.5
    ipc3d_inertia: float = 0.1
    ipc3d_damping: float = 0.1
    ipc3d_max_force: float = 500.0
    ipc3d_control_mode: str = 'velocity'  # 'velocity' or 'position'
    
    # Control parameters
    lqr_q_matrix: list = field(default_factory=lambda: [10.0, 1.0])
    lqr_r_matrix: list = field(default_factory=lambda: [0.1])
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.model_type not in ['limp', 'ipc3d']:
            raise ValueError(f"Invalid guidance model type: {self.model_type}")
        
        # Auto-adjust parameters based on model type
        if self.model_type == 'ipc3d':
            # Set reasonable defaults for IPC3D if not specified
            if self.ipc3d_pole_length == 0.5 and self.limp_com_height:
                self.ipc3d_pole_length = self.limp_com_height


class RobotConfigFactory:
    """
    Factory for creating robot configurations from robot specifications.
    
    This factory automatically generates LeggedRobotCfg instances based on
    robot specifications and guidance model settings.
    """
    
    @staticmethod
    def create_config(
        robot_name: str,
        guidance_model: str = 'limp',
        guidance_params: Optional[Dict[str, Any]] = None,
        **overrides
    ) -> LeggedRobotCfg:
        """
        Create a robot configuration from robot specification.
        
        Args:
            robot_name: Name of registered robot
            guidance_model: Type of guidance model ('limp' or 'ipc3d')
            guidance_params: Optional parameters for guidance model
            **overrides: Override configuration parameters
            
        Returns:
            Configured LeggedRobotCfg instance
        """
        # Get robot specification
        robot_spec = RobotRegistry.get_robot(robot_name)
        
        # Create base configuration
        config = LeggedRobotCfg()
        
        # Configure asset
        config.asset.file = robot_spec.get_urdf_full_path()
        config.env.num_actuators = robot_spec.num_actuators
        
        # Configure initial state
        config.init_state.default_joint_angles = robot_spec.default_joint_angles.copy()
        
        # Configure joint limits
        RobotConfigFactory._configure_joint_limits(config, robot_spec)
        
        # Configure control parameters
        RobotConfigFactory._configure_control(config, robot_spec)
        
        # Configure physical parameters
        RobotConfigFactory._configure_physical_params(config, robot_spec)
        
        # Configure guidance model
        guidance_config = RobotConfigFactory._create_guidance_config(
            guidance_model, robot_spec, guidance_params or {}
        )
        config.guidance = guidance_config
        
        # Apply user overrides
        RobotConfigFactory._apply_overrides(config, overrides)
        
        print(f"✅ Created config for robot: {robot_name} with guidance: {guidance_model}")
        
        return config
    
    @staticmethod
    def _configure_joint_limits(config: LeggedRobotCfg, robot_spec: RobotSpec):
        """Configure joint position and velocity limits."""
        # Set DOF position ranges for reset
        config.init_state.dof_pos_range = {}
        config.init_state.dof_vel_range = {}
        
        for joint_name, limits in robot_spec.joint_limits.items():
            if len(limits) == 2:
                # Position limits
                config.init_state.dof_pos_range[joint_name] = limits.copy()
                # Default velocity limits (can be overridden)
                config.init_state.dof_vel_range[joint_name] = [-0.1, 0.1]
    
    @staticmethod
    def _configure_control(config: LeggedRobotCfg, robot_spec: RobotSpec):
        """Configure control parameters (stiffness, damping)."""
        control_cfg = robot_spec.control_config
        
        if 'stiffness' in control_cfg:
            config.control.stiffness = control_cfg['stiffness'].copy()
        
        if 'damping' in control_cfg:
            config.control.damping = control_cfg['damping'].copy()
        
        # Set other control parameters
        config.control.actuation_scale = control_cfg.get('actuation_scale', 1.0)
        config.control.decimation = control_cfg.get('decimation', 10)
    
    @staticmethod
    def _configure_physical_params(config: LeggedRobotCfg, robot_spec: RobotSpec):
        """Configure physical parameters from robot spec."""
        phys_params = robot_spec.physical_params
        
        # Configure base height target
        if 'base_height' in phys_params:
            config.rewards.base_height_target = phys_params['base_height']
            config.init_state.pos[2] = phys_params['base_height']
        
        # Configure mass properties
        if 'mass' in phys_params:
            # Store for potential use in domain randomization
            config.domain_rand.base_mass = phys_params['mass']
        
        # Configure rotor inertia if provided
        if 'rotor_inertia' in phys_params:
            config.asset.rotor_inertia = phys_params['rotor_inertia']
        
        # Configure end effectors
        if 'end_effectors' in phys_params:
            config.asset.end_effectors = phys_params['end_effectors']
        
        if 'foot_name' in phys_params:
            config.asset.foot_name = phys_params['foot_name']
    
    @staticmethod
    def _create_guidance_config(
        guidance_model: str,
        robot_spec: RobotSpec,
        guidance_params: Dict[str, Any]
    ) -> GuidanceConfig:
        """Create guidance configuration."""
        # Get robot-specific guidance parameters
        robot_guidance_params = robot_spec.physical_params.get('guidance_params', {})
        
        # Merge parameters (user params override robot defaults)
        merged_params = {**robot_guidance_params, **guidance_params}
        merged_params['model_type'] = guidance_model
        
        # Set robot-specific defaults
        if guidance_model == 'limp' and 'limp_com_height' not in merged_params:
            merged_params['limp_com_height'] = robot_spec.physical_params.get('base_height', 1.0)
        
        return GuidanceConfig(**merged_params)
    
    @staticmethod
    def _apply_overrides(config: LeggedRobotCfg, overrides: Dict[str, Any]):
        """Apply user-specified configuration overrides."""
        for section_name, section_overrides in overrides.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                RobotConfigFactory._apply_section_overrides(section, section_overrides)
            else:
                print(f"⚠️  Warning: Unknown config section '{section_name}' in overrides")
    
    @staticmethod
    def _apply_section_overrides(section_obj: Any, overrides: Dict[str, Any]):
        """Apply overrides to a configuration section."""
        for key, value in overrides.items():
            if hasattr(section_obj, key):
                if isinstance(value, dict) and hasattr(getattr(section_obj, key), '__dict__'):
                    # Nested section (e.g., rewards.weights)
                    RobotConfigFactory._apply_section_overrides(getattr(section_obj, key), value)
                else:
                    # Direct attribute
                    setattr(section_obj, key, value)
            else:
                print(f"⚠️  Warning: Unknown config attribute '{key}' in section")
    
    @staticmethod
    def list_available_robots() -> Dict[str, str]:
        """Get a summary of all available robots."""
        robots_info = RobotRegistry.get_robots_info()
        summary = {}
        for name, info in robots_info.items():
            summary[name] = f"{info['category']} - {info['description']}"
        return summary
    
    @staticmethod
    def get_robot_details(robot_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific robot."""
        robot_spec = RobotRegistry.get_robot(robot_name)
        return {
            'name': robot_spec.name,
            'category': robot_spec.category,
            'description': robot_spec.description,
            'num_actuators': robot_spec.num_actuators,
            'joint_names': robot_spec.get_joint_names(),
            'urdf_path': robot_spec.urdf_path,
            'tags': robot_spec.tags,
            'physical_params': robot_spec.physical_params
        }