#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced 3D/2D IPC3D Trajectory Visualization

This module provides comprehensive visualization of the simplified IPC3D guidance model with:
1. 3D visualization of COM trajectory and footsteps in 3D space
2. Top-view (bird's eye) visualization of cart position and footsteps
3. Custom trajectory scenario with specific speed and turning patterns

Features:
- Real-time 3D COM trajectory with height variation
- Top-view cart trajectory and footstep visualization
- Custom scenario: forward motion with left/right turns and deceleration
- Dynamic camera tracking and smooth animations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import time
from typing import Dict, List, Any

# Import the simplified guidance system
try:
    from .ipc3d_guidance import SimplifiedIPCGuidanceModel
    from .ipc3d_controller import IPC3DParams
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ipc3d_guidance import SimplifiedIPCGuidanceModel
    from ipc3d_controller import IPC3DParams


class Enhanced3DTrajectoryVisualizer:
    """Enhanced 3D/2D visualization of simplified IPC3D guidance system."""
    
    def __init__(self, guidance_model: SimplifiedIPCGuidanceModel, 
                 window_size: float = 15.0,
                 max_history: int = 1500):  # Increased for 35s simulation
        """
        Initialize the enhanced visualizer.
        
        Args:
            guidance_model: Simplified IPC3D guidance model
            window_size: Size of visualization window (meters)
            max_history: Maximum number of historical points to keep
        """
        self.guidance = guidance_model
        self.window_size = window_size
        self.max_history = max_history
        
        # Data storage
        self.history = {
            'time': [],
            'com_positions': [],
            'cart_positions': [],
            'left_foot_positions': [],
            'right_foot_positions': [],
            'cart_samples': [],
            'current_speed': [],
            'adaptive_step_time': [],
            'support_foot': [],
            'yaw_angle': [],
            'target_forward_vel': [],
            'target_angular_vel': []
        }
        
        # Robot state for simulation
        self.robot_state = {
            'position': np.array([0.0, 0.0, 0.559]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0])
        }
        
        # Initialize matplotlib with 1x2 layout
        self.fig = plt.figure(figsize=(18, 9))
        
        # Create 3D plot (left side)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        # Create top-view plot (right side)
        self.ax_top = self.fig.add_subplot(122)
        
        self.fig.suptitle('Enhanced IPC3D Guidance Visualization: 3D + Top View', fontsize=16)
        
        # Setup subplots
        self.setup_3d_plot()
        self.setup_topview_plot()
        
        # Animation
        self.animation = None
        self.frame_count = 0
        
        # Trajectory statistics
        self.total_distance = 0.0
        self.max_speed = 0.0
        
        print("Enhanced 3D/2D Trajectory Visualizer initialized")
        print("Window size: {}m".format(window_size))
        print("Max history: {} points".format(max_history))
    
    def setup_3d_plot(self):
        """Setup the 3D trajectory plot."""
        self.ax_3d.set_title('3D COM Trajectory and Footsteps', fontsize=14)
        self.ax_3d.set_xlabel('X Position (m)')
        self.ax_3d.set_ylabel('Y Position (m)')
        self.ax_3d.set_zlabel('Z Height (m)')
        
        # Initialize 3D trajectory lines
        self.com_line_3d, = self.ax_3d.plot([], [], [], 'b-', linewidth=3, label='COM Trajectory', alpha=0.8)
        
        # Footstep scatter plots
        self.left_foot_scatter = self.ax_3d.scatter([], [], [], c='green', s=100, 
                                                   marker='s', alpha=0.8, label='Left Footsteps')
        self.right_foot_scatter = self.ax_3d.scatter([], [], [], c='red', s=100, 
                                                    marker='s', alpha=0.8, label='Right Footsteps')
        
        # Current COM position marker (3D sphere)
        self.com_marker_3d = self.ax_3d.scatter([], [], [], c='blue', s=200, 
                                               marker='o', alpha=1.0, label='Current COM')
        
        # Ground plane (optional - can be added for reference)
        xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        zz = np.zeros_like(xx)
        self.ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        # Set initial 3D view
        self.ax_3d.view_init(elev=20, azim=45)
        self.ax_3d.set_xlim([-2, 8])
        self.ax_3d.set_ylim([-5, 5])
        self.ax_3d.set_zlim([0, 1.2])
        
        self.ax_3d.legend(loc='upper right')
    
    def setup_topview_plot(self):
        """Setup the top-view (bird's eye) plot."""
        self.ax_top.set_title('Top View: Cart Position and Footsteps', fontsize=14)
        self.ax_top.set_xlabel('X Position (m)')
        self.ax_top.set_ylabel('Y Position (m)')
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.set_aspect('equal')
        
        # Initialize top-view lines and markers
        self.cart_line_top, = self.ax_top.plot([], [], 'r-', linewidth=2, label='Cart Trajectory')
        self.left_foot_line_top, = self.ax_top.plot([], [], 'go', markersize=8, 
                                                   alpha=0.7, label='Left Footsteps')
        self.right_foot_line_top, = self.ax_top.plot([], [], 'ro', markersize=8, 
                                                     alpha=0.7, label='Right Footsteps')
        
        # Cart sampling points
        self.cart_samples_scatter_top = self.ax_top.scatter([], [], c='orange', s=60, 
                                                           alpha=0.8, label='Cart Samples')
        
        # Current position markers
        self.cart_marker_top = Circle((0, 0), 0.15, color='red', alpha=0.9)
        self.ax_top.add_patch(self.cart_marker_top)
        
        # Direction arrow for current heading
        self.direction_arrow = self.ax_top.annotate('', xy=(0, 0), xytext=(0, 0),
                                                   arrowprops=dict(arrowstyle='->', 
                                                                 color='black', lw=3))
        
        self.ax_top.set_xlim([-2, 8])
        self.ax_top.set_ylim([-5, 5])
        self.ax_top.legend(loc='upper right')
        
        # Add text box for real-time stats
        self.stats_text = self.ax_top.text(0.02, 0.98, '', transform=self.ax_top.transAxes,
                                          verticalalignment='top', fontfamily='monospace',
                                          fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                                facecolor="lightblue", alpha=0.8))
    
    def update_data(self, dt: float = 0.05):
        """Update simulation data."""
        # Get guidance output
        output = self.guidance.update(self.robot_state, dt=dt)
        
        # Update robot state
        self.robot_state['position'] = output['com_position']
        self.robot_state['velocity'] = output['com_velocity']
        
        # Store history
        self.history['time'].append(output['time'])
        self.history['com_positions'].append(output['com_position'].copy())
        self.history['cart_positions'].append(output.get('cart_position', output['com_position']).copy())
        self.history['left_foot_positions'].append(output['left_foot_position'].copy())
        self.history['right_foot_positions'].append(output['right_foot_position'].copy())
        self.history['current_speed'].append(output['current_speed'])
        self.history['adaptive_step_time'].append(output['adaptive_step_time'])
        self.history['support_foot'].append(output['support_foot'])
        self.history['yaw_angle'].append(output.get('yaw_angle', 0.0))
        self.history['target_forward_vel'].append(self.guidance.desired_forward_velocity)
        self.history['target_angular_vel'].append(self.guidance.desired_angular_velocity)
        
        # Store cart samples if available
        if hasattr(self.guidance, 'cart_samples') and self.guidance.cart_samples:
            self.history['cart_samples'] = [sample['position'] for sample in self.guidance.cart_samples]
        
        # Update statistics
        if len(self.history['com_positions']) > 1:
            prev_pos = self.history['com_positions'][-2]
            curr_pos = self.history['com_positions'][-1]
            distance = np.linalg.norm(curr_pos[:2] - prev_pos[:2])
            self.total_distance += distance
        
        self.max_speed = max(self.max_speed, output['current_speed'])
        
        # Limit history length
        if len(self.history['time']) > self.max_history:
            for key in self.history:
                if isinstance(self.history[key], list) and len(self.history[key]) > self.max_history:
                    self.history[key] = self.history[key][-self.max_history:]
    
    def update_3d_plot(self):
        """Update the 3D trajectory plot."""
        if not self.history['time']:
            return
        
        # Update 3D COM trajectory
        com_positions = np.array(self.history['com_positions'])
        if len(com_positions) > 0:
            self.com_line_3d.set_data_3d(com_positions[:, 0], com_positions[:, 1], com_positions[:, 2])
            
            # Update current COM marker
            current_com = com_positions[-1]
            self.com_marker_3d._offsets3d = ([current_com[0]], [current_com[1]], [current_com[2]])
        
        # Update footstep positions
        left_positions = np.array(self.history['left_foot_positions'])
        right_positions = np.array(self.history['right_foot_positions'])
        
        if len(left_positions) > 0:
            # Filter significant footstep movements
            left_steps = self._filter_significant_steps(left_positions)
            if len(left_steps) > 0:
                self.left_foot_scatter._offsets3d = (left_steps[:, 0], left_steps[:, 1], left_steps[:, 2])
        
        if len(right_positions) > 0:
            right_steps = self._filter_significant_steps(right_positions)
            if len(right_steps) > 0:
                self.right_foot_scatter._offsets3d = (right_steps[:, 0], right_steps[:, 1], right_steps[:, 2])
        
        # Update 3D axis limits to follow trajectory
        if len(com_positions) > 0:
            current_pos = com_positions[-1]
            margin = self.window_size / 3
            
            self.ax_3d.set_xlim([current_pos[0] - margin, current_pos[0] + margin])
            self.ax_3d.set_ylim([current_pos[1] - margin, current_pos[1] + margin])
            
            # Optionally rotate view slightly for dynamic effect
            if self.frame_count % 20 == 0:  # Every 20 frames
                azim = 45 + np.sin(self.frame_count * 0.01) * 10  # Gentle oscillation
                self.ax_3d.view_init(elev=20, azim=azim)
    
    def update_topview_plot(self):
        """Update the top-view plot."""
        if not self.history['time']:
            return
        
        # Update cart trajectory
        cart_positions = np.array(self.history['cart_positions'])
        if len(cart_positions) > 0:
            self.cart_line_top.set_data(cart_positions[:, 0], cart_positions[:, 1])
            
            # Update current cart marker
            current_cart = cart_positions[-1]
            self.cart_marker_top.center = (current_cart[0], current_cart[1])
            
            # Update direction arrow
            if len(self.history['yaw_angle']) > 0:
                yaw = self.history['yaw_angle'][-1]
                arrow_length = 0.5
                arrow_end = (current_cart[0] + arrow_length * np.cos(yaw),
                           current_cart[1] + arrow_length * np.sin(yaw))
                self.direction_arrow.xy = arrow_end
                self.direction_arrow.xytext = (current_cart[0], current_cart[1])
        
        # Update footstep positions
        left_positions = np.array(self.history['left_foot_positions'])
        right_positions = np.array(self.history['right_foot_positions'])
        
        if len(left_positions) > 0:
            left_steps = self._filter_significant_steps(left_positions)
            if len(left_steps) > 0:
                self.left_foot_line_top.set_data(left_steps[:, 0], left_steps[:, 1])
        
        if len(right_positions) > 0:
            right_steps = self._filter_significant_steps(right_positions)
            if len(right_steps) > 0:
                self.right_foot_line_top.set_data(right_steps[:, 0], right_steps[:, 1])
        
        # Update cart sampling points
        if self.history['cart_samples']:
            cart_samples = np.array(self.history['cart_samples'])
            self.cart_samples_scatter_top.set_offsets(cart_samples[:, :2])
        
        # Update axis limits to follow trajectory
        if len(cart_positions) > 0:
            current_pos = cart_positions[-1]
            margin = self.window_size / 2
            
            self.ax_top.set_xlim([current_pos[0] - margin, current_pos[0] + margin])
            self.ax_top.set_ylim([current_pos[1] - margin, current_pos[1] + margin])
        
        # Update statistics text
        self._update_stats_display()
    
    def _filter_significant_steps(self, positions: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """Filter positions to show only significant footstep movements."""
        if len(positions) < 2:
            return positions
        
        # Calculate movement distances
        movements = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        significant_indices = [0]  # Always include first position
        
        for i, movement in enumerate(movements):
            if movement > threshold:
                significant_indices.append(i + 1)
        
        return positions[significant_indices]
    
    def _update_stats_display(self):
        """Update the statistics display in top-view plot."""
        if not self.history['time']:
            return
        
        current_time = self.history['time'][-1]
        current_speed = self.history['current_speed'][-1]
        step_time = self.history['adaptive_step_time'][-1]
        
        target_forward = self.history['target_forward_vel'][-1]
        target_angular = self.history['target_angular_vel'][-1]
        
        yaw_deg = np.degrees(self.history['yaw_angle'][-1]) if self.history['yaw_angle'] else 0.0
        
        # Calculate estimated footsteps
        left_steps = len(self._filter_significant_steps(np.array(self.history['left_foot_positions'])))
        right_steps = len(self._filter_significant_steps(np.array(self.history['right_foot_positions'])))
        
        stats_text = """Time: {:.1f}s
Current Speed: {:.2f} m/s
Target Speed: {:.2f} m/s
Target Angular: {:.2f} rad/s
Heading: {:.1f}°
Step Time: {:.2f}s
Total Distance: {:.2f}m
Max Speed: {:.2f} m/s
Steps: L={}, R={}""".format(
            current_time, current_speed, target_forward, target_angular,
            yaw_deg, step_time, self.total_distance, self.max_speed,
            left_steps, right_steps
        )
        
        self.stats_text.set_text(stats_text)
    
    def animate(self, frame):
        """Animation function."""
        # Update data
        self.update_data(dt=0.05)
        
        # Update both plots
        self.update_3d_plot()
        self.update_topview_plot()
        
        self.frame_count += 1
        
        # Update main title with current time
        if self.history['time']:
            current_time = self.history['time'][-1]
            current_speed = self.history['current_speed'][-1]
            self.fig.suptitle('Enhanced IPC3D Guidance - Time: {:.1f}s | Speed: {:.2f} m/s'.format(
                current_time, current_speed), fontsize=16)
        
        return [self.com_line_3d, self.cart_line_top, self.left_foot_line_top, 
                self.right_foot_line_top, self.stats_text]
    
    def start_animation(self, interval: int = 50):
        """Start the real-time animation."""
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, interval=interval, blit=False, repeat=True
        )
        plt.tight_layout()
        plt.show()
    
    def save_animation(self, filename: str, duration: float = 35.0, fps: int = 20):
        """Save animation to file."""
        frames = int(duration * fps)
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, frames=frames, interval=1000//fps, blit=False
        )
        
        print("Saving enhanced animation to {} ({} frames, {} FPS)...".format(filename, frames, fps))
        self.animation.save(filename, writer='pillow', fps=fps)
        print("Animation saved successfully!")


def create_custom_trajectory_scenario():
    """
    Create custom trajectory scenario with specific speed and turning patterns:
    - 0-10s: Forward 1.0 m/s, left turn 30° at 4-6s
    - 10-20s: Forward 1.5 m/s
    - 20-30s: Forward 0.8 m/s, right turn 45° at 24-26s
    - 30-35s: Decelerate to stop
    """
    # Create guidance model
    params = IPC3DParams(
        mass_cart=1.0,
        mass_pole=13.0,
        pole_length=0.559,
        dt=0.025,
        control_mode=1
    )
    
    guidance = SimplifiedIPCGuidanceModel(params)
    
    # Initialize with robot state
    robot_state = {
        'position': np.array([0.0, 0.0, 0.559]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'orientation': np.array([1.0, 0.0, 0.0, 0.0])
    }
    
    guidance.initialize(robot_state)
    
    # Set initial target
    guidance.set_target({
        'desired_forward_velocity': 1.0,
        'desired_angular_velocity': 0.0
    })
    
    return guidance


class TrajectoryController:
    """Controls the guidance trajectory according to the specified timeline."""
    
    def __init__(self, guidance_model):
        self.guidance = guidance_model
        self.current_scenario = "start"
        
    def update_trajectory_commands(self, current_time: float):
        """Update guidance commands based on current time."""
        
        if 0 <= current_time < 4:
            # Phase 1: Forward motion 1.0 m/s
            if self.current_scenario != "forward_1":
                self.guidance.set_target({
                    'desired_forward_velocity': 1.0,
                    'desired_angular_velocity': 0.0
                })
                self.current_scenario = "forward_1"
                print(f"t={current_time:.1f}s: Phase 1 - Forward 1.0 m/s")
                
        elif 4 <= current_time < 6:
            # Phase 2: Left turn 30° (0.26 rad/s for 2s)
            if self.current_scenario != "left_turn":
                angular_vel = np.radians(30) / 2.0  # 30° over 2 seconds
                self.guidance.set_target({
                    'desired_forward_velocity': 1.0,
                    'desired_angular_velocity': angular_vel
                })
                self.current_scenario = "left_turn"
                print(f"t={current_time:.1f}s: Phase 2 - Left turn 30° (angular: {angular_vel:.3f} rad/s)")
                
        elif 6 <= current_time < 10:
            # Phase 3: Continue forward after turn
            if self.current_scenario != "forward_after_left":
                self.guidance.set_target({
                    'desired_forward_velocity': 1.0,
                    'desired_angular_velocity': 0.0
                })
                self.current_scenario = "forward_after_left"
                print(f"t={current_time:.1f}s: Phase 3 - Forward 1.0 m/s (after left turn)")
                
        elif 10 <= current_time < 20:
            # Phase 4: Increase speed to 1.5 m/s
            if self.current_scenario != "forward_fast":
                self.guidance.set_target({
                    'desired_forward_velocity': 1.5,
                    'desired_angular_velocity': 0.0
                })
                self.current_scenario = "forward_fast"
                print(f"t={current_time:.1f}s: Phase 4 - Fast forward 1.5 m/s")
                
        elif 20 <= current_time < 24:
            # Phase 5: Reduce speed to 0.8 m/s
            if self.current_scenario != "forward_medium":
                self.guidance.set_target({
                    'desired_forward_velocity': 0.8,
                    'desired_angular_velocity': 0.0
                })
                self.current_scenario = "forward_medium"
                print(f"t={current_time:.1f}s: Phase 5 - Medium forward 0.8 m/s")
                
        elif 24 <= current_time < 26:
            # Phase 6: Right turn 45° (-0.39 rad/s for 2s)
            if self.current_scenario != "right_turn":
                angular_vel = -np.radians(45) / 2.0  # -45° over 2 seconds
                self.guidance.set_target({
                    'desired_forward_velocity': 0.8,
                    'desired_angular_velocity': angular_vel
                })
                self.current_scenario = "right_turn"
                print(f"t={current_time:.1f}s: Phase 6 - Right turn 45° (angular: {angular_vel:.3f} rad/s)")
                
        elif 26 <= current_time < 30:
            # Phase 7: Continue forward after right turn
            if self.current_scenario != "forward_after_right":
                self.guidance.set_target({
                    'desired_forward_velocity': 0.8,
                    'desired_angular_velocity': 0.0
                })
                self.current_scenario = "forward_after_right"
                print(f"t={current_time:.1f}s: Phase 7 - Forward 0.8 m/s (after right turn)")
                
        elif 30 <= current_time <= 35:
            # Phase 8: Decelerate to stop
            if self.current_scenario != "decelerate":
                # Linear deceleration from 0.8 to 0 over 5 seconds
                decel_progress = (current_time - 30) / 5.0
                target_speed = 0.8 * (1 - decel_progress)
                target_speed = max(0.0, target_speed)
                
                self.guidance.set_target({
                    'desired_forward_velocity': target_speed,
                    'desired_angular_velocity': 0.0
                })
                
                if current_time == 30:
                    self.current_scenario = "decelerate"
                    print(f"t={current_time:.1f}s: Phase 8 - Decelerate to stop")


def main():
    """Main function to run the enhanced visualization."""
    print("Starting Enhanced 3D/2D IPC3D Trajectory Visualization...")
    print("Custom trajectory scenario:")
    print("  0-4s: Forward 1.0 m/s")
    print("  4-6s: Left turn 30° while moving")
    print("  6-10s: Continue forward 1.0 m/s")
    print("  10-20s: Fast forward 1.5 m/s") 
    print("  20-24s: Medium forward 0.8 m/s")
    print("  24-26s: Right turn 45° while moving")
    print("  26-30s: Continue forward 0.8 m/s")
    print("  30-35s: Decelerate to stop")
    print("Press Ctrl+C to stop the animation.")
    
    try:
        # Create custom scenario
        guidance = create_custom_trajectory_scenario()
        
        # Create trajectory controller
        controller = TrajectoryController(guidance)
        
        # Create enhanced visualizer
        visualizer = Enhanced3DTrajectoryVisualizer(guidance, window_size=20.0)
        
        # Store original animate method
        original_animate = visualizer.animate
        
        # Custom animation function with trajectory control
        def controlled_animate(frame):
            # Update trajectory commands based on current time
            if visualizer.history['time']:
                current_time = visualizer.history['time'][-1]
                controller.update_trajectory_commands(current_time)
            
            # Call original animation function
            return original_animate(frame)
        
        # Override animation function
        visualizer.animate = controlled_animate
        
        # Start animation
        visualizer.start_animation(interval=50)  # 20 FPS
        
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()