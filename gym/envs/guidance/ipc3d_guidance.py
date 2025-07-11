#!/usr/bin/env python3
"""
Simplified IPC3D Guidance Model Implementation

This module implements a simplified IPC guidance model with the following logic:
1. Generate COM trajectory using ipc3d_controller and orientation_manager
2. Adaptive cart trajectory sampling with distance/time constraints
3. Alternating footstep assignment with lateral offsets
4. Unified output dictionary format

Key improvements:
- Simplified trajectory generation
- Adaptive sampling based on velocity constraints
- Direct footstep assignment logic
- Clean separation of concerns
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

try:
    from .base_guidance import BaseGuidanceModel, GuidanceTrajectory
    from .ipc3d_controller import IPC3DParams, IPC_Plane
    from .ipc3d_orientation_manager import IPC3DOrientationManager, OrientationTrajectoryPoint
except ImportError:
    # Fallback for standalone testing
    from base_guidance import BaseGuidanceModel, GuidanceTrajectory
    from ipc3d_controller import IPC3DParams, IPC_Plane
    from ipc3d_orientation_manager import IPC3DOrientationManager, OrientationTrajectoryPoint

@dataclass
class IPC3DGuidanceModelParams:
    """Configuration parameters for IPC3D guidance model."""
    # Physical parameters
    mass_cart: float = 1.0
    mass_pole: float = 13.0
    pole_length: float = 0.559
    dt: float = 0.025
    control_mode: int = 1

    # Sampling constraints
    min_step_distance: float = 0.05
    max_step_distance: float = 0.58
    min_step_time: float = 0.25
    max_step_time: float = 0.7
    max_target_speed: float = 2.3

    # Footstep parameters
    step_width: float = 0.2

    # Orientation parameters
    future_horizon: float = 2.0
    footstep_duration: float = 0.5
    min_angular_velocity: float = 0.01

@dataclass
class HumanoidReferenceState:
    """Complete reference state for humanoid robot control."""
    # Time
    time: float
    
    # CoM reference
    com_position: np.ndarray      # [x, y, z] in world frame
    com_velocity: np.ndarray      # [vx, vy, vz] in world frame
    com_acceleration: np.ndarray  # [ax, ay, az] in world frame
    
    # Orientation reference
    pitch_angle: float            # Pitch angle (rad)
    pitch_angular_velocity: float # Pitch angular velocity (rad/s)
    yaw_angle: float              # Yaw angle (rad)
    yaw_angular_velocity: float   # Yaw angular velocity (rad/s)
    
    # Footstep reference
    left_foot_position: np.ndarray   # [x, y, z] in world frame
    right_foot_position: np.ndarray  # [x, y, z] in world frame
    left_foot_velocity: np.ndarray   # [vx, vy, vz] in world frame
    right_foot_velocity: np.ndarray  # [vx, vy, vz] in world frame
    
    # Gait information
    gait_phase: float             # 0-1 gait phase
    support_foot: str             # 'left' or 'right'


class SimplifiedIPCGuidanceModel(BaseGuidanceModel):
    """
    Simplified IPC guidance model with clear separation of concerns.
    
    Four-step process:
    1. Generate COM trajectory using ipc3d_controller and orientation_manager
    2. Adaptive cart sampling with distance/time constraints (0.05-0.58m, 0.25-0.7s)
    3. Alternating footstep assignment with lateral offsets
    4. Unified output dictionary
    """
    
    def __init__(self, params: IPC3DParams, robot_spec=None, guidance_config=None):
        """Initialize simplified IPC3D guidance model."""
        config = guidance_config.__dict__ if guidance_config else {}
        super().__init__('ipc3d_simplified', config)
        
        self.params = params
        self.robot_spec = robot_spec
        
        # Step 1: Initialize COM trajectory generators
        self.plane_controller = IPC_Plane(params)
        self.orientation_manager = IPC3DOrientationManager(
            future_horizon=2.0,
            footstep_duration=0.5,
            min_angular_velocity=0.01
        )
        
        # Current state
        self.current_time = 0.0
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.world_heading = 0.0
        self.world_angular_velocity = 0.0
        
        # World coordinate tracking
        self.world_position = np.zeros(3)
        self.world_velocity = np.zeros(3)
        
        # Command inputs
        self.desired_forward_velocity = 0.0
        self.desired_angular_velocity = 0.0
        
        # Step 2: Adaptive sampling constraints
        self.min_step_distance = 0.05    # Minimum step distance (m)
        self.max_step_distance = 0.58    # Maximum step distance (m)
        self.min_step_time = 0.25        # Minimum step time (s)
        self.max_step_time = 0.7         # Maximum step time (s)
        self.max_target_speed = 2.3      # Maximum target speed (m/s)
        
        # Step 3: Footstep management
        self.step_width = 0.2            # Lateral offset for feet (m)
        self.current_swing_foot = 'right' # Current swinging foot
        self.last_footstep_time = 0.0     # Last footstep generation time
        
        # Footstep positions (always z=0 for targets)
        self.left_foot_target = np.array([0.0, self.step_width/2, 0.0])
        self.right_foot_target = np.array([0.0, -self.step_width/2, 0.0])
        
        # Cart sampling state
        self.cart_samples = []           # List of cart sampling points
        self.adaptive_step_time = 0.5    # Current adaptive step time
        
        # print(f"✅ Simplified IPC Guidance Model initialized")
        # print(f"   Sampling constraints: {self.min_step_distance:.2f}-{self.max_step_distance:.2f}m, {self.min_step_time:.2f}-{self.max_step_time:.2f}s")
        # print(f"   Max target speed: {self.max_target_speed:.1f} m/s")
        
    def initialize(self, robot_state: Dict[str, Any]) -> bool:
        """Initialize with robot state."""
        if not self.validate_robot_state(robot_state):
            return False
            
        # Extract initial state
        self.com_position = np.array(robot_state.get('position', [0, 0, 0]))
        self.com_velocity = np.array(robot_state.get('velocity', [0, 0, 0]))
        
        # Initialize world tracking
        self.world_position = self.com_position.copy()
        self.world_velocity = self.com_velocity.copy()
        self.world_heading = self._extract_yaw_from_orientation(robot_state.get('orientation', [1, 0, 0, 0]))
        
        # Initialize orientation manager
        self.orientation_manager.update_current_heading(self.world_heading, 0.0)
        
        # Reset plane controller
        self.plane_controller.reset()
        
        self.is_initialized = True
        return True
    
    def set_target(self, target: Dict[str, Any]):
        """Set target velocities."""
        self.target_state = target.copy()
        
        self.desired_forward_velocity = target.get('desired_forward_velocity', 0.0)
        self.desired_angular_velocity = target.get('desired_angular_velocity', 0.0)
        
        # Update controllers
        self.plane_controller.set_desired_forward_velocity(self.desired_forward_velocity)
        self.orientation_manager.update_from_angular_velocity(
            self.desired_angular_velocity, self.current_time
        )
        
        # print(f"🎯 Target set: forward={self.desired_forward_velocity:.2f} m/s, angular={self.desired_angular_velocity:.2f} rad/s")
    
    def update(self, robot_state: Dict[str, Any], dt: Optional[float] = None) -> Dict[str, Any]:
        """Main update loop implementing the four-step process."""
        if not self.is_initialized:
            self.initialize(robot_state)
            
        dt = dt or self.dt
        self.current_time += dt
        
        # Update current state
        self.com_position = np.array(robot_state.get('position', self.com_position))
        self.com_velocity = np.array(robot_state.get('velocity', self.com_velocity))
        
        # Step 1: Generate COM trajectory using controllers
        com_trajectory, cart_trajectory = self._step1_generate_com_cart_trajectories()
        
        # Step 2: Adaptive cart trajectory sampling
        self._step2_adaptive_cart_sampling(cart_trajectory)
        
        # Step 3: Alternating footstep assignment
        self._step3_alternating_footstep_assignment()
        
        # Step 4: Generate unified output dictionary
        return self._step4_generate_unified_output(com_trajectory, cart_trajectory)
    
    def _step1_generate_com_cart_trajectories(self) -> Tuple[Dict, Dict]:
        """
        Step 1: Generate COM trajectory using ipc3d_controller and orientation_manager.
        
        Returns:
            Tuple of (com_trajectory, cart_trajectory) dictionaries
        """
        # Update plane controller
        self.plane_controller.step(n_times=1)
        plane_state = self.plane_controller.get_robot_state()
        
        # Update orientation manager
        current_yaw = self.world_heading
        self.orientation_manager.update_current_heading(current_yaw, self.current_time)
        orientation_info = self.orientation_manager.get_current_orientation_info(self.current_time)
        
        # Extract trajectories
        pitch_angle = plane_state['pitch_angle']
        pitch_angular_velocity = plane_state['pitch_angular_velocity']
        forward_velocity = plane_state['forward_velocity']
        control_force = plane_state['control_force']
        
        # Update world coordinate tracking
        dt = self.params.dt
        self.world_angular_velocity = self.desired_angular_velocity
        self.world_heading += self.world_angular_velocity * dt
        
        # Calculate world velocities
        world_velocity_x = forward_velocity * np.cos(self.world_heading)
        world_velocity_y = forward_velocity * np.sin(self.world_heading)
        
        # Update world position
        self.world_position[0] += world_velocity_x * dt
        self.world_position[1] += world_velocity_y * dt
        
        # COM trajectory (with vertical oscillation)
        com_height = self._get_com_height(self.current_time)
        com_position = np.array([
            self.world_position[0],
            self.world_position[1],
            com_height
        ])
        
        com_velocity = np.array([
            world_velocity_x,
            world_velocity_y,
            self._get_com_vertical_velocity(self.current_time)
        ])
        
        com_acceleration = np.array([
            control_force / self.params.mass_cart * np.cos(self.world_heading),
            control_force / self.params.mass_cart * np.sin(self.world_heading),
            0.0
        ])
        
        # Cart trajectory (on ground, offset by pole)
        pole_length = self.params.pole_length
        cart_offset = np.array([
            -pole_length * np.sin(pitch_angle) * np.cos(self.world_heading),
            -pole_length * np.sin(pitch_angle) * np.sin(self.world_heading),
            0.0
        ])
        
        cart_position = com_position + cart_offset
        cart_position[2] = 0.0  # Cart on ground
        
        cart_velocity = np.array([
            world_velocity_x - pole_length * np.cos(pitch_angle) * pitch_angular_velocity * np.cos(self.world_heading),
            world_velocity_y - pole_length * np.cos(pitch_angle) * pitch_angular_velocity * np.sin(self.world_heading),
            0.0
        ])
        
        return {
            'position': com_position,
            'velocity': com_velocity,
            'acceleration': com_acceleration,
            'pitch_angle': pitch_angle,
            'pitch_angular_velocity': pitch_angular_velocity,
            'yaw_angle': self.world_heading,
            'yaw_angular_velocity': self.world_angular_velocity
        }, {
            'position': cart_position,
            'velocity': cart_velocity
        }
    
    def _step2_adaptive_cart_sampling(self, cart_trajectory: Dict):
        """
        Step 2: Adaptive cart trajectory sampling with distance/time constraints.
        
        Ensures sampling points have distance 0.05-0.58m and time 0.25-0.7s,
        adapted to track target speed up to 2.3 m/s.
        """
        cart_position = cart_trajectory['position']
        cart_velocity = cart_trajectory['velocity']
        current_speed = np.linalg.norm(cart_velocity[:2])
        
        # Adaptive sampling based on current speed
        speed_ratio = min(current_speed / self.max_target_speed, 1.0)
        
        # Adaptive step time (faster speed = shorter step time)
        target_step_time = self.max_step_time - (self.max_step_time - self.min_step_time) * speed_ratio
        self.adaptive_step_time = np.clip(target_step_time, self.min_step_time, self.max_step_time)
        
        # Adaptive step distance based on speed and time
        target_step_distance = current_speed * self.adaptive_step_time
        target_step_distance = np.clip(target_step_distance, self.min_step_distance, self.max_step_distance)
        
        # Check if we should generate a new sample
        time_since_last = self.current_time - self.last_footstep_time
        
        if time_since_last >= self.adaptive_step_time:
            # Generate new sampling point
            if current_speed > 0.01:
                step_direction = cart_velocity[:2] / current_speed
            else:
                step_direction = np.array([np.cos(self.world_heading), np.sin(self.world_heading)])
            
            step_vector = step_direction * target_step_distance
            new_sample = cart_position.copy()
            new_sample[:2] += step_vector
            new_sample[2] = 0.0  # Always on ground
            
            self.cart_samples.append({
                'position': new_sample,
                'time': self.current_time,
                'distance': target_step_distance,
                'step_time': self.adaptive_step_time,
                'speed': current_speed
            })
            
            # Keep only recent samples
            if len(self.cart_samples) > 10:
                self.cart_samples.pop(0)
    
    def _step3_alternating_footstep_assignment(self):
        """
        Step 3: Assign cart sampling points alternately to left/right feet with lateral offsets.
        
        Constraints:
        - Left/right feet get appropriate lateral offsets relative to cart trajectory
        - Footstep z-axis always = 0
        - Target positions only update when assigned new sampling point
        """
        if not self.cart_samples:
            return
        
        # Check if we should assign a new footstep
        time_since_last = self.current_time - self.last_footstep_time
        
        if time_since_last >= self.adaptive_step_time and self.cart_samples:
            # Get latest cart sample
            latest_sample = self.cart_samples[-1]
            sampling_point = latest_sample['position']
            
            # Calculate lateral offsets in world frame
            lateral_offset = self.step_width / 2.0
            
            left_offset_local = np.array([0.0, lateral_offset, 0.0])
            right_offset_local = np.array([0.0, -lateral_offset, 0.0])
            
            left_offset_world = self._transform_to_world_frame(left_offset_local, self.world_heading)
            right_offset_world = self._transform_to_world_frame(right_offset_local, self.world_heading)
            
            # Assign to current swing foot
            if self.current_swing_foot == 'left':
                self.left_foot_target = sampling_point + left_offset_world
                self.left_foot_target[2] = 0.0  # Always on ground
                # Right foot target remains unchanged (support foot)
            else:
                self.right_foot_target = sampling_point + right_offset_world  
                self.right_foot_target[2] = 0.0  # Always on ground
                # Left foot target remains unchanged (support foot)
            
            # Switch swing foot for next step
            self.current_swing_foot = 'left' if self.current_swing_foot == 'right' else 'right'
            self.last_footstep_time = self.current_time
            
            # Debug output
            # if len(self.cart_samples) % 3 == 0:  # Every 3rd step
            #     print(f"🦶 Step {len(self.cart_samples)}: {self.current_swing_foot} assigned to "
            #           f"({sampling_point[0]:.2f}, {sampling_point[1]:.2f}) | "
            #           f"Distance: {latest_sample['distance']:.2f}m | "
            #           f"Time: {latest_sample['step_time']:.2f}s | "
            #           f"Speed: {latest_sample['speed']:.2f} m/s")
    
    def _step4_generate_unified_output(self, com_trajectory: Dict, cart_trajectory: Dict) -> Dict[str, Any]:
        """
        Step 4: Generate unified output dictionary with required format.
        
        Returns:
            Dictionary containing: time, com_position, com_velocity, com_acceleration,
            left_foot_position, right_foot_position
        """
        # Calculate footstep velocities
        current_speed = np.linalg.norm(com_trajectory['velocity'][:2])
        
        if self.current_swing_foot == 'left':
            # Right foot is support, left foot is swinging
            left_foot_velocity = self._calculate_swing_foot_velocity(com_trajectory['velocity'])
            right_foot_velocity = np.array([com_trajectory['velocity'][0], com_trajectory['velocity'][1], 0.0])
        else:
            # Left foot is support, right foot is swinging
            left_foot_velocity = np.array([com_trajectory['velocity'][0], com_trajectory['velocity'][1], 0.0])
            right_foot_velocity = self._calculate_swing_foot_velocity(com_trajectory['velocity'])
        
        # Unified output dictionary
        unified_output = {
            'time': self.current_time,
            'com_position': com_trajectory['position'],
            'com_velocity': com_trajectory['velocity'],
            'com_acceleration': com_trajectory['acceleration'],
            'left_foot_position': self.left_foot_target,
            'right_foot_position': self.right_foot_target,
            'left_foot_velocity': left_foot_velocity,
            'right_foot_velocity': right_foot_velocity,
            'pitch_angle': com_trajectory['pitch_angle'],
            'pitch_angular_velocity': com_trajectory['pitch_angular_velocity'],
            'yaw_angle': com_trajectory['yaw_angle'],
            'yaw_angular_velocity': com_trajectory['yaw_angular_velocity'],
            'support_foot': 'right' if self.current_swing_foot == 'left' else 'left',
            'current_speed': current_speed,
            'adaptive_step_time': self.adaptive_step_time,
            'cart_position': cart_trajectory['position'],
            'cart_velocity': cart_trajectory['velocity']
        }
        
        return unified_output
    
    # Helper methods
    def _extract_yaw_from_orientation(self, orientation: List[float]) -> float:
        """Extract yaw angle from quaternion [w, x, y, z]."""
        w, x, y, z = orientation
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw
    
    def _transform_to_world_frame(self, local_vec: np.ndarray, yaw: float) -> np.ndarray:
        """Transform vector from local to world frame."""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        world_vec = np.array([
            cos_yaw * local_vec[0] - sin_yaw * local_vec[1],
            sin_yaw * local_vec[0] + cos_yaw * local_vec[1],
            local_vec[2]
        ])
        
        return world_vec
    
    def _get_com_height(self, t: float) -> float:
        """Get COM height with small vertical oscillation."""
        base_height = 0.559  # Base COM height
        height_variation = 0.02  # Small oscillation
        return base_height + height_variation * np.sin(2 * np.pi * t / 0.5)
    
    def _get_com_vertical_velocity(self, t: float) -> float:
        """Get COM vertical velocity."""
        height_variation = 0.02
        return height_variation * (2 * np.pi / 0.5) * np.cos(2 * np.pi * t / 0.5)
    
    def _calculate_swing_foot_velocity(self, com_velocity: np.ndarray) -> np.ndarray:
        """Calculate swing foot velocity."""
        swing_velocity = com_velocity.copy()
        swing_velocity[2] = 0.0  # No vertical velocity for simplicity
        return swing_velocity
    
    def generate_complete_trajectory_output(self) -> Dict[str, Any]:
        """Generate complete trajectory output for backwards compatibility."""
        # Create a minimal robot state for update
        robot_state = {
            'position': self.com_position,
            'velocity': self.com_velocity,
            'orientation': [1, 0, 0, 0]  # Identity quaternion
        }
        
        output = self.update(robot_state)
        
        # Format for backwards compatibility
        complete_output = {
            'com_trajectory': {
                'position': output['com_position'],
                'velocity': output['com_velocity'],
                'acceleration': output['com_acceleration'],
                'pitch_angle': output['pitch_angle'],
                'pitch_angular_velocity': output['pitch_angular_velocity'],
                'yaw_angle': output['yaw_angle'],
                'yaw_angular_velocity': output['yaw_angular_velocity']
            },
            'com_reference_velocity': output['com_velocity'],
            'desired_footsteps': {
                'left_foot': output['left_foot_position'],
                'right_foot': output['right_foot_position'],
                'left_foot_velocity': output['left_foot_velocity'],
                'right_foot_velocity': output['right_foot_velocity'],
                'support_foot': output['support_foot'],
                'adaptive_step_time': output['adaptive_step_time'],
                'current_speed': output['current_speed']
            },
            'control_info': {
                'time': output['time'],
                'cart_position': output.get('cart_position'),
                'cart_velocity': output.get('cart_velocity'),
                'target_velocities': {
                    'forward': self.desired_forward_velocity,
                    'angular': self.desired_angular_velocity
                }
            }
        }
        
        return complete_output

    def get_control_reference(self, t: float) -> Dict[str, Any]:
        """Get control reference at specific time t."""
        saved_time = self.current_time
        self.current_time = t
        
        # Generate reference at time t
        robot_state = {
            'position': self.com_position,
            'velocity': self.com_velocity,
            'orientation': [1, 0, 0, 0]
        }
        
        output = self.update(robot_state)
        
        # Restore time
        self.current_time = saved_time
        
        return {
            'time': t,
            'com_position': output['com_position'],
            'com_velocity': output['com_velocity'],
            'com_acceleration': output['com_acceleration'],
            'orientation': self._yaw_to_quat(output['yaw_angle']),
            'angular_velocity': np.array([0.0, output['pitch_angular_velocity'], output['yaw_angular_velocity']]),
            'footsteps': {
                'left': output['left_foot_position'],
                'right': output['right_foot_position']
            },
            'support_foot': output['support_foot']
        }
    
    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        """Convert yaw angle to quaternion."""
        return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

    def generate_trajectory(self, duration: float, target_params: Dict[str, Any]) -> GuidanceTrajectory:
        """Generate reference trajectory for given duration (backwards compatibility)."""
        # Set target parameters
        self.set_target(target_params)
        
        # Generate trajectory by stepping through time
        saved_time = self.current_time
        self.current_time = 0.0
        
        num_steps = int(duration / self.dt)
        timestamps = []
        positions = []
        velocities = []
        accelerations = []
        orientations = []
        angular_velocities = []
        
        # Create minimal robot state
        robot_state = {
            'position': self.com_position,
            'velocity': self.com_velocity, 
            'orientation': [1, 0, 0, 0]
        }
        
        for i in range(num_steps):
            output = self.update(robot_state, dt=self.dt)
            
            timestamps.append(output['time'])
            positions.append(output['com_position'])
            velocities.append(output['com_velocity'])
            accelerations.append(output['com_acceleration'])
            orientations.append(self._yaw_to_quat(output['yaw_angle']))
            angular_velocities.append([0.0, output['pitch_angular_velocity'], output['yaw_angular_velocity']])
            
            # Update robot state for next step
            robot_state['position'] = output['com_position']
            robot_state['velocity'] = output['com_velocity']
        
        # Restore time
        self.current_time = saved_time
        
        # Convert to arrays
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)
        orientations = np.array(orientations)
        angular_velocities = np.array(angular_velocities)
        
        # Forces and torques (placeholder)
        forces = np.zeros((num_steps, 3))
        torques = np.zeros((num_steps, 3))
        
        # Create trajectory object
        trajectory_obj = GuidanceTrajectory(
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
        
        return trajectory_obj


# For backwards compatibility
IPCGuidanceModel = SimplifiedIPCGuidanceModel
IPC3DGuidanceModel = SimplifiedIPCGuidanceModel


if __name__ == "__main__":
    # Test the simplified guidance model
    params = IPC3DParams(
        mass_cart=1.0,
        mass_pole=13.0,
        pole_length=0.559,
        dt=0.025,
        control_mode=1
    )
    
    guidance = SimplifiedIPCGuidanceModel(params)
    
    robot_state = {
        'position': np.array([0.0, 0.0, 0.559]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'orientation': np.array([1.0, 0.0, 0.0, 0.0])
    }
    
    guidance.initialize(robot_state)
    guidance.set_target({
        'desired_forward_velocity': 1.0,
        'desired_angular_velocity': 0.1
    })
    
    print("🧪 Testing simplified guidance model...")
    for i in range(50):
        output = guidance.update(robot_state, dt=0.025)
        
        if i % 10 == 0:
            print(f"t={output['time']:.2f}s: "
                  f"COM=({output['com_position'][0]:.2f},{output['com_position'][1]:.2f}) "
                  f"L_foot=({output['left_foot_position'][0]:.2f},{output['left_foot_position'][1]:.2f}) "
                  f"R_foot=({output['right_foot_position'][0]:.2f},{output['right_foot_position'][1]:.2f}) "
                  f"Speed={output['current_speed']:.2f} m/s")
        
        # Update robot state for next iteration
        robot_state['position'] = output['com_position']
        robot_state['velocity'] = output['com_velocity']
    
    print("✅ Simplified guidance model test completed!")