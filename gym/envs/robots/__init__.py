"""
Robot registry and factory system for unified robot model management.

This module provides a centralized registry for robot specifications and
automated configuration generation for different robot models.
"""

from .registry import RobotRegistry, RobotSpec
from .factory import RobotConfigFactory, GuidanceConfig
from .specifications import register_all_robots

# Auto-register all built-in robots
register_all_robots()

__all__ = [
    'RobotRegistry',
    'RobotSpec', 
    'RobotConfigFactory',
    'GuidanceConfig',
    'register_all_robots'
]