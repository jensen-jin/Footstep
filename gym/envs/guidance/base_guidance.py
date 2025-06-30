#!/usr/bin/env python3
"""
Base classes for guidance models.

This module defines the interface that all guidance models must implement.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass 
class GuidanceTrajectory:
    """Container for guidance trajectory data."""
    
    # Time-series data
    positions: np.ndarray        # [T, 3] - x, y, z positions
    velocities: np.ndarray       # [T, 3] - x, y, z velocities  
    accelerations: np.ndarray    # [T, 3] - x, y, z accelerations
    orientations: np.ndarray     # [T, 4] - quaternions [w, x, y, z]
    angular_velocities: np.ndarray # [T, 3] - angular velocities
    
    # Control data
    forces: np.ndarray           # [T, 3] - control forces
    torques: np.ndarray          # [T, 3] - control torques
    
    # Timing
    timestamps: np.ndarray       # [T] - time stamps
    dt: float                    # Time step
    
    # Metadata
    num_steps: int
    is_valid: bool = True
    
    def __post_init__(self):
        """Validate trajectory data."""
        self.num_steps = len(self.timestamps)
        
        # Check dimensions
        expected_shapes = [
            (self.positions, (self.num_steps, 3)),
            (self.velocities, (self.num_steps, 3)),
            (self.accelerations, (self.num_steps, 3)),
            (self.orientations, (self.num_steps, 4)),
            (self.angular_velocities, (self.num_steps, 3)),
            (self.forces, (self.num_steps, 3)),
            (self.torques, (self.num_steps, 3))
        ]
        
        for arr, expected_shape in expected_shapes:
            if arr.shape != expected_shape:
                print(f"Warning: Shape mismatch - expected {expected_shape}, got {arr.shape}")
                self.is_valid = False
    
    def to_torch(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert trajectory to PyTorch tensors."""
        return {
            'positions': torch.from_numpy(self.positions).float().to(device),
            'velocities': torch.from_numpy(self.velocities).float().to(device),
            'accelerations': torch.from_numpy(self.accelerations).float().to(device),
            'orientations': torch.from_numpy(self.orientations).float().to(device),
            'angular_velocities': torch.from_numpy(self.angular_velocities).float().to(device),
            'forces': torch.from_numpy(self.forces).float().to(device),
            'torques': torch.from_numpy(self.torques).float().to(device),
            'timestamps': torch.from_numpy(self.timestamps).float().to(device)
        }
        
    def get_state_at_time(self, t: float) -> Dict[str, np.ndarray]:
        """Get trajectory state at specific time (with interpolation)."""
        if t < self.timestamps[0] or t > self.timestamps[-1]:
            raise ValueError(f"Time {t} outside trajectory range [{self.timestamps[0]}, {self.timestamps[-1]}]")
        
        # Find surrounding time indices
        idx = np.searchsorted(self.timestamps, t)
        if idx == 0:
            idx = 1
        elif idx >= len(self.timestamps):
            idx = len(self.timestamps) - 1
            
        # Linear interpolation
        t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        
        def interpolate(arr):
            return arr[idx-1] + alpha * (arr[idx] - arr[idx-1])
        
        return {
            'position': interpolate(self.positions),
            'velocity': interpolate(self.velocities),
            'acceleration': interpolate(self.accelerations),
            'orientation': interpolate(self.orientations),
            'angular_velocity': interpolate(self.angular_velocities),
            'force': interpolate(self.forces),
            'torque': interpolate(self.torques)
        }


class BaseGuidanceModel(ABC):
    """Base class for all guidance models."""
    
    def __init__(self, model_type: str, config: Dict[str, Any]):
        """
        Initialize guidance model.
        
        Args:
            model_type: Type of guidance model ('limp', 'ipc3d', etc.)
            config: Configuration parameters
        """
        self.model_type = model_type
        self.config = config
        self.dt = config.get('dt', 0.02)
        self.is_initialized = False
        
        # Internal state
        self.current_state = {}
        self.target_state = {}
        self.control_history = []
        self.trajectory_cache = {}
        
    @abstractmethod
    def initialize(self, robot_state: Dict[str, Any]) -> bool:
        """
        Initialize the guidance model with robot state.
        
        Args:
            robot_state: Current robot state
            
        Returns:
            True if initialization successful
        """
        pass
        
    @abstractmethod
    def set_target(self, target: Dict[str, Any]):
        """
        Set target state/trajectory for the guidance model.
        
        Args:
            target: Target state or trajectory parameters
        """
        pass
        
    @abstractmethod
    def update(self, robot_state: Dict[str, Any], dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Update guidance model and compute control outputs.
        
        Args:
            robot_state: Current robot state
            dt: Time step (optional, uses default if None)
            
        Returns:
            Dictionary with guidance outputs (forces, torques, etc.)
        """
        pass
        
    @abstractmethod
    def generate_trajectory(self, duration: float, target_params: Dict[str, Any]) -> GuidanceTrajectory:
        """
        Generate reference trajectory.
        
        Args:
            duration: Trajectory duration in seconds
            target_params: Target parameters (velocity, position, etc.)
            
        Returns:
            Generated trajectory
        """
        pass
        
    @abstractmethod
    def get_control_reference(self, t: float) -> Dict[str, np.ndarray]:
        """
        Get control reference at specific time.
        
        Args:
            t: Time in seconds
            
        Returns:
            Control reference (positions, velocities, forces, etc.)
        """
        pass
        
    def reset(self):
        """Reset guidance model to initial state."""
        self.current_state.clear()
        self.target_state.clear()
        self.control_history.clear()
        self.trajectory_cache.clear()
        self.is_initialized = False
        
    def get_state(self) -> Dict[str, Any]:
        """Get current guidance model state."""
        return {
            'model_type': self.model_type,
            'current_state': self.current_state.copy(),
            'target_state': self.target_state.copy(),
            'is_initialized': self.is_initialized,
            'dt': self.dt
        }
        
    def set_dt(self, dt: float):
        """Set time step."""
        self.dt = dt
        
    def validate_robot_state(self, robot_state: Dict[str, Any]) -> bool:
        """Validate robot state has required fields."""
        required_fields = ['position', 'velocity', 'orientation']
        for field in required_fields:
            if field not in robot_state:
                print(f"Warning: Missing required field '{field}' in robot state")
                return False
        return True
        
    def compute_tracking_error(self, reference: Dict[str, np.ndarray], 
                             actual: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute tracking errors between reference and actual states."""
        errors = {}
        
        for key in reference:
            if key in actual:
                ref_val = np.asarray(reference[key])
                act_val = np.asarray(actual[key])
                
                if ref_val.shape == act_val.shape:
                    error = np.linalg.norm(ref_val - act_val)
                    errors[f'{key}_error'] = float(error)
                    
        return errors
        
    def log_control_history(self, control_output: Dict[str, Any]):
        """Log control output to history."""
        self.control_history.append({
            'timestamp': len(self.control_history) * self.dt,
            'control': control_output.copy()
        })
        
        # Limit history size to prevent memory issues
        max_history = 1000
        if len(self.control_history) > max_history:
            self.control_history = self.control_history[-max_history:]


class LegacyGuidanceWrapper(BaseGuidanceModel):
    """Wrapper for existing guidance implementations to match new interface."""
    
    def __init__(self, legacy_guidance, model_type: str, config: Dict[str, Any]):
        """
        Wrap legacy guidance model.
        
        Args:
            legacy_guidance: Existing guidance model instance
            model_type: Type identifier  
            config: Configuration
        """
        super().__init__(model_type, config)
        self.legacy_guidance = legacy_guidance
        self.is_initialized = True
        
    def initialize(self, robot_state: Dict[str, Any]) -> bool:
        """Initialize wrapper - already done in constructor."""
        return True
        
    def set_target(self, target: Dict[str, Any]):
        """Set target via legacy interface."""
        if hasattr(self.legacy_guidance, 'set_target'):
            self.legacy_guidance.set_target(target)
        self.target_state = target
        
    def update(self, robot_state: Dict[str, Any], dt: Optional[float] = None) -> Dict[str, Any]:
        """Update via legacy interface."""
        if hasattr(self.legacy_guidance, 'update'):
            return self.legacy_guidance.update(robot_state, dt or self.dt)
        return {}
        
    def generate_trajectory(self, duration: float, target_params: Dict[str, Any]) -> GuidanceTrajectory:
        """Generate trajectory via legacy interface or create dummy."""
        if hasattr(self.legacy_guidance, 'generate_trajectory'):
            return self.legacy_guidance.generate_trajectory(duration, target_params)
        
        # Create dummy trajectory
        num_steps = int(duration / self.dt)
        return GuidanceTrajectory(
            positions=np.zeros((num_steps, 3)),
            velocities=np.zeros((num_steps, 3)),
            accelerations=np.zeros((num_steps, 3)),
            orientations=np.tile([1, 0, 0, 0], (num_steps, 1)),
            angular_velocities=np.zeros((num_steps, 3)),
            forces=np.zeros((num_steps, 3)),
            torques=np.zeros((num_steps, 3)),
            timestamps=np.arange(num_steps) * self.dt,
            dt=self.dt,
            num_steps=num_steps
        )
        
    def get_control_reference(self, t: float) -> Dict[str, np.ndarray]:
        """Get control reference via legacy interface."""
        if hasattr(self.legacy_guidance, 'get_control_reference'):
            return self.legacy_guidance.get_control_reference(t)
        return {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'force': np.zeros(3)
        }