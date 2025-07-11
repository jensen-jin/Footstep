#!/usr/bin/env python3
"""
Guidance Models Package

This package provides guidance models for robot locomotion:
- LIMP (Linear Inverted Pendulum Model) - existing implementation
- IPC3D (3D Inverted Pendulum on Cart) - new implementation

The guidance models generate reference trajectories for reinforcement learning.
"""

from .base_guidance import BaseGuidanceModel, GuidanceTrajectory
from .ipc3d_controller import IPC3DParams, IPC_Plane
from .guidance_factory import GuidanceModelFactory
from .ipc3d_guidance import IPC3DGuidanceModel, IPC3DGuidanceModelParams

__all__ = [
    'BaseGuidanceModel',
    'GuidanceTrajectory', 
    'IPC3DParams',
    'IPC_Plane',
    'GuidanceModelFactory',
    'IPC3DGuidanceModel',
    'IPC3DGuidanceModelParams'
]