"""
Robot specifications for different robot types.
"""

from .humanoid import register_humanoid_robots
from .simple import register_simple_robots

def register_all_robots():
    """Register all available robots."""
    register_humanoid_robots()
    register_simple_robots()
    
__all__ = [
    'register_all_robots',
    'register_humanoid_robots', 
    'register_simple_robots'
]