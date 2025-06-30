"""
Robot specifications for different robot types.
"""

from .humanoid import register_humanoid_robots
from .simple import register_simple_robots
from .push_robots import register_push_robots

def register_all_robots():
    """Register all available robots."""
    register_humanoid_robots()
    register_simple_robots()
    register_push_robots()  # Add push_robot_verify branch robots
    
__all__ = [
    'register_all_robots',
    'register_humanoid_robots', 
    'register_simple_robots',
    'register_push_robots'
]