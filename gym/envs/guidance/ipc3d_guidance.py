#!/usr/bin/env python3
"""
IPC3D Guidance Model Implementation

This module implements the IPC3D guidance model that integrates
with the robot training framework.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

try:
    from .base_guidance import BaseGuidanceModel, GuidanceTrajectory
    from .ipc3d_controller import IPC3D, IPC3DParams
except ImportError:
    # Fallback for standalone testing
    from base_guidance import BaseGuidanceModel, GuidanceTrajectory
    from ipc3d_controller import IPC3D, IPC3DParams


class IPC3DGuidanceModel(BaseGuidanceModel):
    """IPC3D guidance model for robot locomotion."""
    
    def __init__(self, params: IPC3DParams, robot_spec=None, guidance_config=None):
        """
        Initialize IPC3D guidance model.
        
        Args:
            params: IPC3D controller parameters
            robot_spec: Robot specification (optional)
            guidance_config: Guidance configuration (optional)
        """
        config = guidance_config.__dict__ if guidance_config else {}
        super().__init__('ipc3d', config)
        
        self.params = params
        self.robot_spec = robot_spec
        self.controller = IPC3D(params)
        
        # Reference tracking
        self.reference_trajectory = None
        self.trajectory_start_time = 0.0
        
        # State tracking
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.base_orientation = np.array([1, 0, 0, 0])  # [w, x, y, z]
        
        # Control outputs
        self.control_forces = np.zeros(3)
        self.control_torques = np.zeros(3)
        
        # Gait parameters
        self.gait_phase = 0.0
        self.step_length = 0.5
        self.step_width = 0.3
        self.step_height = 0.1
        self.step_time = 0.5
        
        print(f"✅ IPC3D Guidance Model initialized")
        print(f"   Cart mass: {params.mass_cart:.1f} kg")
        print(f"   Pole length: {params.pole_length:.3f} m") 
        print(f"   Control mode: {'velocity' if params.control_mode == 1 else 'position'}")
        
    def initialize(self, robot_state: Dict[str, Any]) -> bool:
        """Initialize with robot state."""
        if not self.validate_robot_state(robot_state):
            return False
            
        # Extract initial state
        self.com_position = np.array(robot_state.get('position', [0, 0, 0]))
        self.com_velocity = np.array(robot_state.get('velocity', [0, 0, 0]))
        self.base_orientation = np.array(robot_state.get('orientation', [1, 0, 0, 0]))
        
        # Reset controller
        self.controller.reset()
        
        self.is_initialized = True
        return True
        
    def set_target(self, target: Dict[str, Any]):
        """Set target velocity or gait parameters."""
        self.target_state = target.copy()
        
        # Extract target velocities
        target_vel_x = target.get('velocity_x', 0.0)
        target_vel_z = target.get('velocity_z', 0.0)
        
        # Set controller target
        self.controller.set_desired_velocity(target_vel_x, target_vel_z)
        
        # Update gait parameters if provided
        if 'step_length' in target:
            self.step_length = target['step_length']
        if 'step_width' in target:
            self.step_width = target['step_width']
        if 'step_time' in target:
            self.step_time = target['step_time']
            
        print(f"IPC3D target set: vx={target_vel_x:.2f}, vz={target_vel_z:.2f} m/s")
        
    def update(self, robot_state: Dict[str, Any], dt: Optional[float] = None) -> Dict[str, Any]:
        """Update guidance model and compute control outputs."""
        if not self.is_initialized:
            self.initialize(robot_state)
            
        dt = dt or self.dt
        
        # Update current state
        self.com_position = np.array(robot_state.get('position', self.com_position))
        self.com_velocity = np.array(robot_state.get('velocity', self.com_velocity))
        self.base_orientation = np.array(robot_state.get('orientation', self.base_orientation))
        
        # Update controller
        controller_result = self.controller.step(n_times=1)
        
        # Get control forces
        fx, fz = self.controller.get_control_forces()
        self.control_forces = np.array([fx, 0.0, fz])
        
        # Compute torques based on orientation error
        self.control_torques = self._compute_orientation_torques(robot_state)
        
        # Update gait phase
        self.gait_phase += dt / self.step_time
        if self.gait_phase >= 1.0:
            self.gait_phase -= 1.0
            
        # Generate control output
        control_output = {
            'forces': self.control_forces.copy(),
            'torques': self.control_torques.copy(),
            'reference_position': self._get_reference_position(),
            'reference_velocity': self._get_reference_velocity(),
            'gait_phase': self.gait_phase,
            'controller_state': self.controller.get_state(),
            'tracking_error': self._compute_tracking_error(robot_state)
        }
        
        # Log control history
        self.log_control_history(control_output)
        
        return control_output
        
    def generate_trajectory(self, duration: float, target_params: Dict[str, Any]) -> GuidanceTrajectory:
        """Generate reference trajectory for given duration."""
        
        # Set target parameters
        self.set_target(target_params)
        
        # Time parameters
        num_steps = int(duration / self.dt)
        timestamps = np.arange(num_steps) * self.dt
        
        # Initialize arrays
        positions = np.zeros((num_steps, 3))
        velocities = np.zeros((num_steps, 3))
        accelerations = np.zeros((num_steps, 3))
        orientations = np.tile([1, 0, 0, 0], (num_steps, 1))
        angular_velocities = np.zeros((num_steps, 3))
        forces = np.zeros((num_steps, 3))
        torques = np.zeros((num_steps, 3))
        
        # Reset controller for trajectory generation
        self.controller.reset()
        
        # Generate trajectory
        current_pos = np.zeros(3)
        current_vel = np.zeros(3)
        
        for i in range(num_steps):
            t = timestamps[i]
            
            # Update controller
            result = self.controller.step(n_times=1)
            
            # Get control forces
            fx, fz = self.controller.get_control_forces()
            
            # Compute reference motion
            positions[i] = current_pos
            velocities[i] = current_vel
            forces[i] = [fx, 0.0, fz]
            
            # Simple integration for trajectory
            if i > 0:
                accelerations[i] = (velocities[i] - velocities[i-1]) / self.dt
                
            # Update position and velocity
            current_pos += current_vel * self.dt
            current_vel = np.array([
                self.controller.get_state()['x_cart_vel'],
                0.0,
                self.controller.get_state()['z_cart_vel']
            ])
            
            # Update gait phase
            gait_phase = (t / self.step_time) % 1.0
            
            # Add step motion
            positions[i, 1] = self._get_step_height(gait_phase)
            
        # Create trajectory object
        trajectory = GuidanceTrajectory(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            orientations=orientations,
            angular_velocities=angular_velocities,
            forces=forces,
            torques=torques,
            timestamps=timestamps,
            dt=self.dt,
            num_steps=num_steps
        )
        
        self.reference_trajectory = trajectory
        self.trajectory_start_time = 0.0
        
        return trajectory
        
    def get_control_reference(self, t: float) -> Dict[str, np.ndarray]:
        """Get control reference at specific time."""
        if self.reference_trajectory is None:
            # Generate default reference
            return {
                'position': self.com_position,
                'velocity': np.array(self.target_state.get('velocity', [0, 0, 0])),
                'force': self.control_forces,
                'torque': self.control_torques
            }
            
        # Get state from trajectory
        try:
            traj_state = self.reference_trajectory.get_state_at_time(t - self.trajectory_start_time)
            return {
                'position': traj_state['position'],
                'velocity': traj_state['velocity'],
                'force': traj_state['force'],
                'torque': traj_state['torque']
            }
        except ValueError:
            # Time outside trajectory range
            return {
                'position': self.com_position,
                'velocity': np.zeros(3),
                'force': np.zeros(3),
                'torque': np.zeros(3)
            }
            
    def _compute_orientation_torques(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Compute orientation control torques."""
        # Simple orientation control - keep upright
        current_orientation = robot_state.get('orientation', [1, 0, 0, 0])
        target_orientation = [1, 0, 0, 0]  # Upright
        
        # Compute orientation error (simplified)
        orientation_error = np.array(current_orientation[1:3])  # x, y components
        
        # Proportional control
        kp_orientation = 50.0
        torques = -kp_orientation * orientation_error
        
        return np.array([torques[0], torques[1], 0.0])
        
    def _get_reference_position(self) -> np.ndarray:
        """Get current reference position."""
        controller_state = self.controller.get_state()
        return np.array([
            controller_state['x_cart_pos'],
            self._get_step_height(self.gait_phase),
            controller_state['z_cart_pos']
        ])
        
    def _get_reference_velocity(self) -> np.ndarray:
        """Get current reference velocity."""
        controller_state = self.controller.get_state()
        return np.array([
            controller_state['x_cart_vel'],
            self._get_step_velocity(self.gait_phase),
            controller_state['z_cart_vel']
        ])
        
    def _get_step_height(self, phase: float) -> float:
        """Get step height for given gait phase."""
        # Simple sinusoidal step pattern
        if 0.2 <= phase <= 0.8:  # Swing phase
            swing_phase = (phase - 0.2) / 0.6
            return self.step_height * np.sin(np.pi * swing_phase)
        else:  # Stance phase
            return 0.0
            
    def _get_step_velocity(self, phase: float) -> float:
        """Get vertical velocity for given gait phase."""
        # Derivative of step height
        if 0.2 <= phase <= 0.8:  # Swing phase
            swing_phase = (phase - 0.2) / 0.6
            return (self.step_height * np.pi / 0.6) * np.cos(np.pi * swing_phase) / self.step_time
        else:
            return 0.0
            
    def _compute_tracking_error(self, robot_state: Dict[str, Any]) -> Dict[str, float]:
        """Compute tracking errors."""
        reference = self.get_control_reference(0.0)  # Current time
        actual = {
            'position': robot_state.get('position', np.zeros(3)),
            'velocity': robot_state.get('velocity', np.zeros(3))
        }
        
        return self.compute_tracking_error(reference, actual)
        
    def get_guidance_info(self) -> Dict[str, Any]:
        """Get detailed guidance model information."""
        controller_state = self.controller.get_state()
        
        return {
            'model_type': self.model_type,
            'controller_params': {
                'mass_cart': self.params.mass_cart,
                'mass_pole': self.params.mass_pole,
                'pole_length': self.params.pole_length,
                'control_mode': 'velocity' if self.params.control_mode == 1 else 'position'
            },
            'current_state': {
                'com_position': self.com_position,
                'com_velocity': self.com_velocity,
                'control_forces': self.control_forces,
                'gait_phase': self.gait_phase
            },
            'controller_state': controller_state,
            'target_state': self.target_state,
            'gait_params': {
                'step_length': self.step_length,
                'step_width': self.step_width,
                'step_height': self.step_height,
                'step_time': self.step_time
            }
        }
        
    def visualize_controller_state(self) -> Dict[str, Any]:
        """Get data for visualization."""
        controller_state = self.controller.get_state()
        
        return {
            'pendulum_x': {
                'cart_pos': controller_state['x_cart_pos'],
                'pole_angle': controller_state['x_pole_angle'],
                'cart_vel': controller_state['x_cart_vel'],
                'control_force': controller_state['control_x']
            },
            'pendulum_z': {
                'cart_pos': controller_state['z_cart_pos'],
                'pole_angle': controller_state['z_pole_angle'],
                'cart_vel': controller_state['z_cart_vel'],
                'control_force': controller_state['control_z']
            },
            'gait_info': {
                'phase': self.gait_phase,
                'step_height': self._get_step_height(self.gait_phase)
            }
        }