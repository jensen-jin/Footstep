#!/usr/bin/env python3
"""
Guidance Models Package

This package provides guidance models for robot locomotion:
- LIMP (Linear Inverted Pendulum Model) - existing implementation
- IPC3D (3D Inverted Pendulum on Cart) - new implementation

The guidance models generate reference trajectories for reinforcement learning.
"""

from .base_guidance import BaseGuidanceModel, GuidanceTrajectory
from .ipc3d_controller import IPC3D, IPC3DParams, create_ipc3d_from_robot_spec
from .guidance_factory import GuidanceModelFactory

__all__ = [
    'BaseGuidanceModel',
    'GuidanceTrajectory', 
    'IPC3D',
    'IPC3DParams',
    'create_ipc3d_from_robot_spec',
    'GuidanceModelFactory'
]