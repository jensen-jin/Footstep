#!/usr/bin/env python3
"""
IPC3D Orientation Manager Implementation

This module implements orientation-aware trajectory generation for IPC3D
based on the research paper approach using Hermite splines for facing direction.

Key features:
- Hermite spline interpolation for smooth orientation transitions
- Global to relative coordinate system conversion
- Future trajectory planning with 1-2 footstep horizon
- Integration with IPC3D controller for relative velocity control

Reference: "The inverted pendulum model by itself cannot represent the facing 
direction of the character. Instead, we define the facing direction using a 
Hermite spline that spans a fixed-duration future time horizon..."
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrientationTrajectoryPoint:
    """Single point in orientation-aware trajectory."""
    position: np.ndarray  # [x, y] body position
    velocity: np.ndarray  # [vx, vy] body velocity
    heading: float        # Current heading angle (rad)
    time: float          # Time from start (s)


class IPC3DOrientationManager:
    """
    Manages orientation and reference trajectory generation for IPC3D.
    
    Implements the paper's approach of using Hermite splines for facing direction
    that spans a fixed-duration future time horizon containing 1-2 footsteps.
    """
    
    def __init__(self, 
                 future_horizon: float = 1.5,
                 footstep_duration: float = 0.5,
                 min_angular_velocity: float = 0.01):
        """
        Initialize orientation manager.
        
        Args:
            future_horizon: Time span for future trajectory (s)
            footstep_duration: Duration of one footstep cycle (s)  
            min_angular_velocity: Minimum angular velocity to trigger replan (rad/s)
        """
        self.future_horizon = future_horizon
        self.footstep_duration = footstep_duration
        self.min_angular_velocity = min_angular_velocity
        
        # Current orientation state
        self.current_heading = 0.0
        self.target_heading = 0.0
        self.heading_velocity = 0.0
        
        # Hermite spline parameters
        self.heading_spline = None
        self.spline_start_time = 0.0
        self.spline_duration = future_horizon
        
        # Trajectory cache
        self.reference_trajectory: List[OrientationTrajectoryPoint] = []
        self.last_update_time = 0.0
        
        print(f"✅ IPC3D Orientation Manager initialized")
        print(f"   Future horizon: {future_horizon:.1f}s ({future_horizon/footstep_duration:.1f} footsteps)")
        print(f"   Spline duration: {self.spline_duration:.1f}s")
        
    # 更新当前朝向和角速度 
    def update_current_heading(self, heading: float, current_time: float):
        """Update current robot heading from sensor data."""
        # Compute heading velocity
        # 检查self 是否存在 _last_heading_time 对象
        if hasattr(self, '_last_heading_time'):
            dt = current_time - self._last_heading_time
            if dt > 0:
                heading_diff = self._normalize_angle(heading - self._last_heading)
                self.heading_velocity = heading_diff / dt
        
        self.current_heading = self._normalize_angle(heading)
        self._last_heading = heading
        self._last_heading_time = current_time
    
       
    def update_target_heading(self, 
                            new_target_heading: float, 
                            current_time: float,
                            force_update: bool = False):
        """
        Update target heading and recreate Hermite spline.
        
        Args:
            new_target_heading: New desired heading (rad)
            current_time: Current simulation time (s)
            force_update: Force spline recreation even for small changes
        """
        new_target = self._normalize_angle(new_target_heading)
        
        # Check if update is needed
        heading_change = abs(self._normalize_angle(new_target - self.target_heading))
        if not force_update and heading_change < self.min_angular_velocity * 0.1:
            return  # Skip small changes
            
        self.target_heading = new_target
        self.spline_start_time = current_time
        
        # Create new Hermite spline
        self._create_heading_spline()
    # 输入角速度，更新目标朝向  计算方式:角速度*未来时间（default : 1s）
    def update_from_angular_velocity(self, angular_velocity: float, current_time: float):
        """
        Update target heading based on commanded angular velocity.
        
        This integrates angular velocity over the future horizon to determine
        the target heading, as described in the paper.
        """
        if abs(angular_velocity) < self.min_angular_velocity:
            return  # No significant angular command
            
        # Project angular velocity over future horizon
        future_heading_change = angular_velocity * self.future_horizon
        projected_target = self.current_heading + future_heading_change
        
        self.update_target_heading(projected_target, current_time)
        
    def _create_heading_spline(self):
        """
        Create Hermite spline for smooth heading interpolation.
        
        The spline smoothly interpolates from current to target heading
        with controlled derivatives at both ends.
        """
        # Ensure angle difference is in [-π, π] for shortest path
        angle_diff = self._normalize_angle(self.target_heading - self.current_heading)
        
        # Hermite spline control points
        p0 = self.current_heading                    # Start position  
        p1 = self.current_heading + angle_diff      # End position
        m0 = self.heading_velocity * 0.5            # Start tangent (damped current velocity)
        m1 = 0.0                                    # End tangent (smooth stop)
        
        self.heading_spline = {
            'p0': p0,
            'p1': p1, 
            'm0': m0,
            'm1': m1,
            'duration': self.spline_duration,
            'angle_diff': angle_diff
        }
    # 获取当前时间的朝向角度
    def get_heading_at_time(self, t_relative: float) -> float:
        """
        Get heading at relative time using Hermite interpolation.
        
        Args:
            t_relative: Time relative to spline start (s)
            
        Returns:
            Interpolated heading angle (rad)
        """
        if self.heading_spline is None:
            return self.current_heading
            
        # Clamp time to spline duration
        t = max(0.0, min(1.0, t_relative / self.heading_spline['duration']))
        
        # Hermite basis functions
        h00 = 2*t**3 - 3*t**2 + 1    # p0 coefficient
        h10 = t**3 - 2*t**2 + t      # m0 coefficient  
        h01 = -2*t**3 + 3*t**2       # p1 coefficient
        h11 = t**3 - t**2            # m1 coefficient
        
        # Compute interpolated heading
        heading = (h00 * self.heading_spline['p0'] + 
                  h10 * self.heading_spline['m0'] +
                  h01 * self.heading_spline['p1'] + 
                  h11 * self.heading_spline['m1'])
                  
        return self._normalize_angle(heading)
    # 获取当前时间的朝向角速度
    def get_heading_velocity_at_time(self, t_relative: float) -> float:
        """Get heading angular velocity at relative time."""
        if self.heading_spline is None or self.heading_spline['duration'] <= 0:
            return 0.0
            
        # Clamp time
        t = max(0.0, min(1.0, t_relative / self.heading_spline['duration']))
        
        # Derivative of Hermite basis functions
        dh00_dt = 6*t**2 - 6*t       # d/dt h00
        dh10_dt = 3*t**2 - 4*t + 1   # d/dt h10
        dh01_dt = -6*t**2 + 6*t      # d/dt h01  
        dh11_dt = 3*t**2 - 2*t       # d/dt h11
        
        # Compute heading velocity (scaled by duration)
        heading_vel = ((dh00_dt * self.heading_spline['p0'] + 
                       dh10_dt * self.heading_spline['m0'] +
                       dh01_dt * self.heading_spline['p1'] + 
                       dh11_dt * self.heading_spline['m1']) / 
                      self.heading_spline['duration'])
                      
        return heading_vel
    
    # TODO：这个函数将指令速度认为是全局坐标系下的速度，转换为相对于当前朝向的速度
    # 但是我们一般认为指令速度是相对于当前朝向的，所以这个函数可能需要调整 
    def convert_global_to_relative_velocity(self, 
                                          global_vel_x: float, 
                                          global_vel_y: float,
                                          current_time: float) -> Tuple[float, float]:
        """
        Convert global velocity commands to relative (body-frame) velocities.
        
        This implements the paper's concept: "the desired velocity is always 
        defined relative to the current orientation."
        
        Args:
            global_vel_x: Global X velocity command (m/s)
            global_vel_y: Global Y velocity command (m/s)  
            current_time: Current simulation time (s)
            
        Returns:
            (relative_vel_x, relative_vel_z): Body-frame velocities
        """
        # Get current heading from spline
        t_rel = current_time - self.spline_start_time
        current_heading = self.get_heading_at_time(t_rel)
        
        # Rotation matrix: Global → Body frame
        cos_h = np.cos(current_heading)
        sin_h = np.sin(current_heading)
        
        # Transform velocity vector
        relative_vel_x = cos_h * global_vel_x + sin_h * global_vel_y   # Forward velocity
        relative_vel_z = -sin_h * global_vel_x + cos_h * global_vel_y  # Lateral → Vertical
        
        return relative_vel_x, relative_vel_z
    # TODO ：同上，和前一个函数正好相反，可能也是用不到的
    def convert_relative_to_global_velocity(self,
                                          relative_vel_x: float,
                                          relative_vel_z: float, 
                                          current_time: float) -> Tuple[float, float]:
        """Convert relative velocities back to global frame."""
        # Get current heading
        t_rel = current_time - self.spline_start_time
        current_heading = self.get_heading_at_time(t_rel)
        
        # Rotation matrix: Body → Global frame  
        cos_h = np.cos(current_heading)
        sin_h = np.sin(current_heading)
        
        # Transform velocity vector
        global_vel_x = cos_h * relative_vel_x - sin_h * relative_vel_z
        global_vel_y = sin_h * relative_vel_x + cos_h * relative_vel_z
        
        return global_vel_x, global_vel_y
    #  获得朝向轨迹列表
    def generate_orientation_trajectory(self, 
                                      start_time: float,
                                      dt: float,
                                      num_points: int) -> List[OrientationTrajectoryPoint]:
        """
        输入起始时间、时间步长和点数，生成朝向轨迹。
        Args:
            start_time: 起始时间 (s)
            dt: 时间步长 (s)
            num_points: 轨迹点数量
        输出:
            List[OrientationTrajectoryPoint]: 朝向轨迹点列表
        生成的轨迹点将包含位置、速度、朝向和时间信息。
        """
        trajectory = []
        
        for i in range(num_points):
            t_future = i * dt
            t_rel = start_time + t_future - self.spline_start_time
            
            # Get orientation at this time
            heading = self.get_heading_at_time(t_rel)
            heading_vel = self.get_heading_velocity_at_time(t_rel)
            
            # Create trajectory point (position will be filled by IPC3D)
            point = OrientationTrajectoryPoint(
                position=np.zeros(2),  # To be filled by caller
                velocity=np.zeros(2),  # To be filled by caller
                heading=heading,
                # 从0到self.future_horizon的时间
                time=t_future
            )
            
            trajectory.append(point)
            
        self.reference_trajectory = trajectory
        self.last_update_time = start_time
        
        return trajectory        
    # 将角度标准化到[-π, π]范围内 
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_current_orientation_info(self, current_time: float) -> Dict[str, float]:
        """Get current orientation information for debugging/logging."""
        t_rel = current_time - self.spline_start_time
        
        return {
            'current_heading_deg': np.degrees(self.current_heading),
            'target_heading_deg': np.degrees(self.target_heading),
            'spline_heading_deg': np.degrees(self.get_heading_at_time(t_rel)),
            'heading_velocity_dps': np.degrees(self.heading_velocity),
            'spline_velocity_dps': np.degrees(self.get_heading_velocity_at_time(t_rel)),
            'time_since_spline_start': t_rel,
            'spline_progress': min(1.0, t_rel / self.spline_duration) if self.spline_duration > 0 else 0.0
        }

def test_orientation_manager():
    """Test the orientation manager functionality."""
    print("🧪 Testing IPC3D Orientation Manager")
    print("=" * 50)
    
    # Create manager
    manager = IPC3DOrientationManager(future_horizon=2.0, footstep_duration=0.5)
    
    # Test 1: Basic heading updates
    print("\n📍 Test 1: Basic heading updates")
    manager.update_current_heading(0.0, 0.0)
    manager.update_target_heading(np.pi/4, 0.0)  # 45 degrees
    
    # Sample trajectory
    for t in np.linspace(0, 2.0, 11):
        heading = manager.get_heading_at_time(t)
        print(f"t={t:.1f}s: heading={np.degrees(heading):.1f}°")
        
    # Test 2: Velocity conversion
    print("\n🔄 Test 2: Velocity conversion")
    # 创建新的管理器实例避免测试1的样条干扰
    manager2 = IPC3DOrientationManager(future_horizon=2.0, footstep_duration=0.5)
    manager2.update_current_heading(np.pi/6, 1.0)  # 30 degrees at t=1s
    # 设置相同的目标朝向，让样条稳定
    manager2.update_target_heading(np.pi/6, 1.0)  # 目标也是30度
    
    global_vx, global_vy = 1.0, 0.5  # Global velocities
    rel_vx, rel_vz = manager2.convert_global_to_relative_velocity(global_vx, global_vy, 1.0)
    
    print(f"Global velocity: ({global_vx:.2f}, {global_vy:.2f}) m/s")
    print(f"Relative velocity: ({rel_vx:.2f}, {rel_vz:.2f}) m/s")
    
    # Convert back
    back_vx, back_vy = manager2.convert_relative_to_global_velocity(rel_vx, rel_vz, 1.0)
    print(f"Converted back: ({back_vx:.2f}, {back_vy:.2f}) m/s")
    
    # Test 3: Angular velocity integration
    print("\n⚙️ Test 3: Angular velocity integration")
    # 使用新的管理器实例
    manager3 = IPC3DOrientationManager(future_horizon=1.5, footstep_duration=0.5)
    manager3.update_current_heading(np.pi/6, 2.0)  # 30 degrees at t=2s
    manager3.update_from_angular_velocity(0.5, 2.0)  # 0.5 rad/s at t=2s
    
    info = manager3.get_current_orientation_info(2.5)
    print("Orientation info:")
    for key, value in info.items():
        print(f"  {key}: {value:.2f}")
        
    print("\n✅ Orientation manager test completed!")


if __name__ == "__main__":
    test_orientation_manager()