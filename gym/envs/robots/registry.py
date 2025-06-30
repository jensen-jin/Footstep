"""
Robot registry system for managing robot specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import os
from pathlib import Path

@dataclass
class RobotSpec:
    """
    Robot specification containing all necessary information for configuration.
    
    This class defines a standardized format for robot models that can be
    used across different experiments and training scenarios.
    """
    name: str
    urdf_path: str
    num_actuators: int
    default_joint_angles: Dict[str, float]
    joint_limits: Dict[str, List[float]]
    control_config: Dict[str, Any]
    physical_params: Dict[str, Any]
    description: str
    category: str = "humanoid"  # humanoid, quadruped, simple
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate robot specification after initialization."""
        self._validate_spec()
    
    def _validate_spec(self):
        """Validate the robot specification."""
        # Check that joint names are consistent
        joint_names_angles = set(self.default_joint_angles.keys())
        joint_names_limits = set(self.joint_limits.keys())
        
        if joint_names_angles != joint_names_limits:
            missing_in_limits = joint_names_angles - joint_names_limits
            missing_in_angles = joint_names_limits - joint_names_angles
            
            error_msg = f"Joint name mismatch in robot '{self.name}':"
            if missing_in_limits:
                error_msg += f"\n  Missing limits for: {missing_in_limits}"
            if missing_in_angles:
                error_msg += f"\n  Missing angles for: {missing_in_angles}"
            
            raise ValueError(error_msg)
        
        # Check number of actuators consistency
        if len(self.default_joint_angles) != self.num_actuators:
            raise ValueError(
                f"Robot '{self.name}': num_actuators ({self.num_actuators}) "
                f"doesn't match joint count ({len(self.default_joint_angles)})"
            )
    
    def get_joint_names(self) -> List[str]:
        """Get list of joint names in consistent order."""
        return sorted(self.default_joint_angles.keys())
    
    def get_urdf_full_path(self) -> str:
        """Get full path to URDF file with environment variable expansion."""
        expanded_path = os.path.expandvars(self.urdf_path)
        
        # Handle relative paths from project root
        if not os.path.isabs(expanded_path):
            project_root = Path(__file__).parent.parent.parent.parent
            expanded_path = str(project_root / expanded_path)
        
        return expanded_path


class RobotRegistry:
    """
    Central registry for robot specifications.
    
    This class manages all available robot models and provides methods
    to register, retrieve, and list robots.
    """
    _robots: Dict[str, RobotSpec] = {}
    _categories: Dict[str, List[str]] = {}
    
    @classmethod
    def register(cls, robot_spec: RobotSpec):
        """
        Register a robot specification.
        
        Args:
            robot_spec: RobotSpec instance to register
            
        Raises:
            ValueError: If robot name already exists
        """
        if robot_spec.name in cls._robots:
            raise ValueError(f"Robot '{robot_spec.name}' is already registered")
        
        cls._robots[robot_spec.name] = robot_spec
        
        # Update category index
        category = robot_spec.category
        if category not in cls._categories:
            cls._categories[category] = []
        cls._categories[category].append(robot_spec.name)
        
        print(f"✅ Registered robot: {robot_spec.name} ({robot_spec.category})")
    
    @classmethod
    def get_robot(cls, name: str) -> RobotSpec:
        """
        Get robot specification by name.
        
        Args:
            name: Robot name
            
        Returns:
            RobotSpec for the requested robot
            
        Raises:
            ValueError: If robot not found
        """
        if name not in cls._robots:
            available = list(cls._robots.keys())
            raise ValueError(
                f"Robot '{name}' not found. Available robots: {available}"
            )
        return cls._robots[name]
    
    @classmethod
    def list_robots(cls, category: Optional[str] = None) -> List[str]:
        """
        List all registered robots, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of robot names
        """
        if category is None:
            return list(cls._robots.keys())
        
        if category not in cls._categories:
            return []
        
        return cls._categories[category].copy()
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """Get list of all available categories."""
        return list(cls._categories.keys())
    
    @classmethod
    def get_robots_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get summary information about all registered robots.
        
        Returns:
            Dictionary with robot info
        """
        info = {}
        for name, spec in cls._robots.items():
            info[name] = {
                'category': spec.category,
                'num_actuators': spec.num_actuators,
                'description': spec.description,
                'tags': spec.tags,
                'urdf_path': spec.urdf_path
            }
        return info
    
    @classmethod
    def find_robots_by_tag(cls, tag: str) -> List[str]:
        """Find robots that have a specific tag."""
        matching = []
        for name, spec in cls._robots.items():
            if tag in spec.tags:
                matching.append(name)
        return matching
    
    @classmethod
    def clear(cls):
        """Clear all registered robots (mainly for testing)."""
        cls._robots.clear()
        cls._categories.clear()