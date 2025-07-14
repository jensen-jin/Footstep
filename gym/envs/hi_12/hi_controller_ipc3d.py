"""
HI-12 Humanoid Controller with IPC3D Guidance and External Force Disturbance
=============================================================================

This module integrates:
1. HI-12 humanoid robot controller (12-DOF)
2. IPC3D trajectory guidance model 
3. External force disturbance training system
4. Hierarchical control architecture for robust locomotion

Architecture:
- High-level: IPC3D guidance trajectory planning
- Mid-level: HI-12 gait pattern generation  
- Low-level: PPO reinforcement learning policy
- Disturbance: Random external force perturbations

Author: Advanced Robotics Research Team
Date: 2025
"""

# IMPORTANT: Import Isaac Gym before torch to avoid import order conflicts
from isaacgym import gymtorch, gymapi

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F

# Import base controller and components
from .hi_controller import HiController, get_euler_xyz_tensor
from .hi_controller_config import HiControllerCfg
from gym.envs.guidance.ipc3d_guidance import SimplifiedIPCGuidanceModel, IPC3DGuidanceModelParams
from gym.envs.guidance.ipc3d_controller import IPC3DParams
from gym.envs.guidance.ipc3d_orientation_manager import IPC3DOrientationManager


class HiControllerIPC3DCfg(HiControllerCfg):
    """Extended configuration for HI-12 with IPC3D guidance and force disturbance."""
    
    class guidance:
        """IPC3D guidance model configuration - 继承自IPC3DGuidanceModelParams。"""
        enable_ipc3d = True
        
        # 基础物理参数（与guidance保持一致）
        mass_cart = 1.0          # Cart mass (kg) - 与guidance一致
        mass_pole = 13.0         # Pole mass (kg) - 与guidance一致  
        pole_length = 0.559      # Pole length (m) - 与guidance一致
        dt = 0.025              # Time step (s) - 与guidance一致
        control_mode = 1         # 1=velocity control, 0=position control
        
        # 采样约束参数（与guidance一致）
        min_step_distance = 0.05    # Minimum step distance (m)
        max_step_distance = 0.58    # Maximum step distance (m) 
        min_step_time = 0.25        # Minimum step time (s)
        max_step_time = 0.7         # Maximum step time (s)
        max_target_speed = 2.3      # Maximum target speed (m/s)
        
        # 足迹参数（与guidance一致）
        step_width = 0.2            # Lateral offset for feet (m)
        
        # 方向参数（与guidance一致）
        future_horizon = 2.0        # Future planning horizon (s)
        footstep_duration = 0.5     # Footstep duration (s)
        min_angular_velocity = 0.01 # Minimum angular velocity threshold
        
        # 更新频率控制
        guidance_update_freq = 5 # Update every N simulation steps
        
    class domain_rand(HiControllerCfg.domain_rand):
        """Enhanced domain randomization with force disturbance."""
        
        # Force push system  
        use_force_push = True
        max_push_force_xy = 120.0     # Maximum push force for HI-12 (N)
        push_duration = 20            # Push duration (simulation steps)
        push_interval = 200           # Interval between pushes (steps)
        push_debug = False            # Enable debug output
        
        # Curriculum learning for push forces
        curriculum_push = True
        push_force_schedule = {
            0: 0.0,       # 0-15k steps: gentle push
            3000: 20,   # 15k-40k steps: moderate push  
            5000: 40.0,  # 40k+ steps: strong push
            7000: 70.0   # 80k+ steps: maximum challenge
        }
        
        # Multi-point push system for HI-12
        enable_multi_point_push = True
        push_body_parts = [
            "base_link",      # Primary push point (torso)
            "l_thigh_link",   # Left thigh disturbance
            "r_thigh_link",   # Right thigh disturbance  
            "l_calf_link",    # Left lower leg
            "r_calf_link"     # Right lower leg
        ]
        push_body_weights = {
            "base_link": 0.5,      # 50% torso pushes
            "l_thigh_link": 0.15,  # 15% left thigh
            "r_thigh_link": 0.15,  # 15% right thigh
            "l_calf_link": 0.1,    # 10% left calf
            "r_calf_link": 0.1     # 10% right calf
        }
        
        # 3D force directions
        push_3d_enabled = True
        push_z_scale = 0.3        # Scale down vertical forces
        
        # Adaptive push timing based on gait phase
        adaptive_push_timing = True
        push_during_stance = 0.7   # Probability during stance phase
        push_during_swing = 0.4    # Probability during swing phase
        
    class rewards(HiControllerCfg.rewards):
        """Enhanced reward system for IPC3D guidance training."""
        
        class weights(HiControllerCfg.rewards.weights):
            # 主要IPC3D跟踪奖励（控制在合理范围，保持同一数量级）
            ipc3d_trajectory_tracking = 2.0      # 质心跟踪稍大：从8.0降低到2.0
            footstep_placement_accuracy = 0.5     # 落足点权重相对小：从4.0降低到0.5
            
            # 原有奖励权重调整（降低以避免与IPC3D冲突）
            tracking_lin_vel_base_x = 2.0         # 从6.0降低
            tracking_lin_vel_base_y = 1.0         # 从2.0保持
            command_yaw_vel = 3.0                 # 从9降低
            
            # 新增IPC3D特定奖励（大幅降低权重防止数值爆炸）
            ipc3d_force_consistency = 0.2         # 控制力一致性：从1.5降低到0.2
            ipc3d_orientation_tracking = 0.1      # 姿态跟踪：从2.0大幅降低到0.1
            
            # 干扰恢复奖励（增强）
            stability_recovery = 2.5              # 快速恢复
            balance_maintenance = 1.5             # 平衡维持
            contact_stability = 1.0               # 接触稳定性
            
            # 能效奖励（增强）
            torques = 1e-4                       # 从1e-4保持
            dof_vel = 1e-3                       # 从1e-3保持
            action_smoothness = 0.5              # 增加平滑性要求
            
            # 保持重要的安全性奖励
            base_height = 3.0                    # 保持原有权重
            base_z_orientation = 2.0             # 保持姿态稳定
            joint_regularization = 2.0           # 保持关节规则化
            contact_schedule = 4.0               # 保持接触调度
            feet_slip = -8.0                     # 保持防滑倒
            feet_distance = 8.0                  # 保持足距离
            ankle_roll_posture_roll = 3.0        # 保持踝关节控制
            ankle_roll_posture_pitch = 3.0       # 保持踝关节控制
            ankle_roll_action_zero = 1.0         # 保持踝关节优化
            
            # 兼容性奖励（保持向后兼容）
            trajectory_tracking = 0.0            # 由ipc3d_trajectory_tracking替代
            guidance_consistency = 0.0           # 由ipc3d_force_consistency替代
            step_location_error = 0.0            # 由footstep_placement_accuracy替代
            
        # Reward scaling parameters
        trajectory_sigma = 0.25        # Gaussian scaling for trajectory error
        stability_sigma = 0.3          # Gaussian scaling for stability
        force_recovery_time = 2.0      # Time window for recovery evaluation (s)


class HiControllerIPC3D(HiController):
    """
    HI-12 Humanoid Controller with IPC3D Guidance Integration.
    
    This class extends the base HI-12 controller to include:
    - IPC3D trajectory guidance model
    - External force disturbance handling
    - Enhanced reward system for robust locomotion
    """
    
    cfg: HiControllerIPC3DCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize HI-12 controller with IPC3D guidance."""
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize IPC3D guidance system
        self._init_ipc3d_guidance()
        
        # Initialize force disturbance system  
        self._init_force_disturbance_system()
        
        # Initialize additional tracking buffers
        self._init_guidance_buffers()
        
        # Force overwrite any problematic attributes that might exist in base classes
        self._force_tensor_initialization()
        
        print(f"🤖 HI-12 + IPC3D简化控制器初始化完成")
        print(f"   机器人: HI-12 ({self.cfg.env.num_actuators} DOF)")
        print(f"   引导: IPC3D简化模型 (mass_cart={self.cfg.guidance.mass_cart}kg, pole_length={self.cfg.guidance.pole_length}m)")
        print(f"   干扰: 外部推力 ({self.cfg.domain_rand.max_push_force_xy}N max)")
        
    def _init_ipc3d_guidance(self):
        """Initialize IPC3D guidance model with simplified parameters."""
        if not self.cfg.guidance.enable_ipc3d:
            return
            
        # 创建IPC3DParams (plane controller需要这个)
        ipc3d_params = IPC3DParams(
            mass_cart=self.cfg.guidance.mass_cart,
            mass_pole=self.cfg.guidance.mass_pole,
            pole_length=self.cfg.guidance.pole_length,
            dt=self.cfg.guidance.dt,
            control_mode=self.cfg.guidance.control_mode,
            # IPC3DParams特有的参数 - 使用默认值
            inertia=self.cfg.guidance.mass_pole * (self.cfg.guidance.pole_length ** 2),  # I = m*l²
            gravity=9.81,
            damping=3.5,  # 系统阻尼 - 默认值
            max_force=1000.0,  # 最大控制力
            # LQR权重参数 - 使用默认值
            q_cart_position=0.01,
            q_cart_velocity=8.0,
            q_pole_angle=20.0,
            q_pole_angular_velocity=25.0,
            r_control=0.04
        )
        
        # 创建SimplifiedIPCGuidanceModel实例
        self.ipc3d_guidance = SimplifiedIPCGuidanceModel(
            params=ipc3d_params,
            robot_spec=None,
            guidance_config=self.cfg.guidance  # 传递guidance配置
        )
        
        # IPC3D状态跟踪
        self.guidance_update_counter = 0
        self.last_guidance_update_time = 0.0      
        
    def _init_force_disturbance_system(self):
        """Initialize the force disturbance system for HI-12."""
        if not self.cfg.domain_rand.use_force_push:
            return
            
        # Force disturbance state
        self.push_active = False
        self.push_step_count = 0
        self.last_push_step = 0
        
        # Get push body indices for HI-12
        self.push_body_indices = {}
        for body_name in self.cfg.domain_rand.push_body_parts:
            if body_name in self.rigid_body_idx:
                self.push_body_indices[body_name] = self.rigid_body_idx[body_name]

        # Current push state
        self.current_push_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_push_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
    def _init_guidance_buffers(self):
        """Initialize additional buffers for IPC3D guidance tracking."""
        # IPC3D trajectory buffers
        self.ipc3d_desired_trajectory = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # [x, y, z] desired position
        
        self.ipc3d_desired_velocity = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False  
        )  # [vx, vy, vz] desired velocity
        
        self.ipc3d_control_forces = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # [fx, fy, fz] IPC3D control forces
        
        # Trajectory tracking errors - 确保2D形状
        self.trajectory_tracking_error = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Create aliases for observation configuration compatibility
        self.ipc3d_trajectory_error = self.trajectory_tracking_error
        
        # Initialize additional observation buffers needed by configuration
        self.stability_score = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # [num_envs, 1] - 确保2D形状
        
        self.push_state = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # [push_active, steps_since_push, force_magnitude, time_since_push]
        
        # 确保所有基础观测有正确的2D形状
        # 注意：这些可能会被父类的初始化覆盖，但我们尝试确保正确的形状
        if not hasattr(self, 'phase_sin') or self.phase_sin.dim() != 2:
            self.phase_sin = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            )
        
        if not hasattr(self, 'phase_cos') or self.phase_cos.dim() != 2:
            self.phase_cos = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            )
            
        if not hasattr(self, 'full_step_period_obs') or self.full_step_period_obs.dim() != 2:
            self.full_step_period_obs = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            )
        
        # Stability tracking for disturbance recovery
        self.stability_before_push = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Force overwrite any existing recovery_start_time attribute
        self.recovery_start_time = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        
        # Guidance consistency tracking
        self.guidance_consistency_score = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Add base_euler_xyz observation buffer (required by configuration)
        self.base_euler_xyz = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # [roll, pitch, yaw] base orientation in Euler angles

    # 保证数据类型一致性    
    def _force_tensor_initialization(self):
        """Force initialization of critical tensors to prevent type conflicts."""
        # Critical: Ensure recovery_start_time is always a tensor
        if not isinstance(getattr(self, 'recovery_start_time', None), torch.Tensor):
            self.recovery_start_time = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
            )
        
        # Ensure other critical tensors are properly typed
        if not isinstance(getattr(self, 'stability_before_push', None), torch.Tensor):
            self.stability_before_push = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )
            
        # Verify tensor types
        assert isinstance(self.recovery_start_time, torch.Tensor), f"recovery_start_time is {type(self.recovery_start_time)}, expected torch.Tensor"
        assert isinstance(self.stability_before_push, torch.Tensor), f"stability_before_push is {type(self.stability_before_push)}, expected torch.Tensor"
        
        # Tensor initialization verified silently
        
    def _verify_tensor_types(self):
        """Verify that critical tensors are still tensors and not corrupted."""
        if not isinstance(self.recovery_start_time, torch.Tensor):
            # print(f"🚨 Critical: recovery_start_time corrupted to {type(self.recovery_start_time)}")
            self.recovery_start_time = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
            )
        
        if not isinstance(self.stability_before_push, torch.Tensor):
            # print(f"🚨 Critical: stability_before_push corrupted to {type(self.stability_before_push)}")
            self.stability_before_push = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )

    def reset_envs(self, env_ids):
        """Reset environments and guidance states."""
        super().reset_envs(env_ids)
        
        # Ensure critical tensors are still tensors before reset
        self._verify_tensor_types()
        
        # Reset base_euler_xyz observation
        self.base_euler_xyz[env_ids] = get_euler_xyz_tensor(self.base_quat[env_ids])
        
        if self.cfg.guidance.enable_ipc3d:
            # Reset IPC3D guidance states
            self.ipc3d_desired_trajectory[env_ids] = 0.0
            self.ipc3d_desired_velocity[env_ids] = 0.0
            self.ipc3d_control_forces[env_ids] = 0.0
            self.trajectory_tracking_error[env_ids] = 0.0
            self.guidance_consistency_score[env_ids] = 0.0
            
        if self.cfg.domain_rand.use_force_push:
            # Reset disturbance states with proper tensor values
            self.stability_before_push[env_ids] = 0.0
            # Ensure recovery_start_time is a tensor before assignment
            if isinstance(self.recovery_start_time, torch.Tensor):
                self.recovery_start_time[env_ids] = 0
            else:
                # print(f"⚠️  recovery_start_time is {type(self.recovery_start_time)}, reinitializing...")
                self.recovery_start_time = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
                )
                self.recovery_start_time[env_ids] = 0
            
        # Reset completed silently

    def _post_physics_step_callback(self):
        """Enhanced post-physics callback with IPC3D guidance and force disturbance."""
        super()._post_physics_step_callback()
        
        # Compute base_euler_xyz observation (required by configuration)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        
        # Update IPC3D guidance trajectory
        if self.cfg.guidance.enable_ipc3d:
            self._update_ipc3d_guidance()
        
        # Handle force disturbance system
        if self.cfg.domain_rand.use_force_push:
            self._maybe_push_robot_hi12()
            
        # Update tracking metrics
        self._update_guidance_tracking()
        
    def _convert_commands_to_ipc3d_format(self):
        """简化的命令转换，直接输出guidance模型期望的格式。
        
        Returns:
            dict: guidance模型需要的目标速度字典
        """
        # 直接使用命令作为目标速度，简化处理
        forward_velocity = self.commands[:, 0]  # X方向速度作为前向速度
        angular_velocity = self.commands[:, 2] if self.commands.shape[1] > 2 else torch.zeros_like(forward_velocity)
        
        # 返回guidance模型期望的格式
        return {
            'desired_forward_velocity': forward_velocity,
            'desired_angular_velocity': angular_velocity
        }
    
    def _update_ipc3d_guidance(self):
        """Update IPC3D guidance trajectory with orientation-aware control."""
        self.guidance_update_counter += 1
        
        # Update at specified frequency to reduce computational load
        if self.guidance_update_counter % self.cfg.guidance.guidance_update_freq != 0:
            return
            
        current_time = self.common_step_counter * self.cfg.sim.dt
        
        # 获取转换后的IPC3D格式命令
        ipc3d_commands = self._convert_commands_to_ipc3d_format()
        
        # Process each environment
        for env_idx in range(self.num_envs):
            # Extract current robot state
            current_state = {
                'position': self.base_pos[env_idx].cpu().numpy(),
                'velocity': self.base_lin_vel[env_idx].cpu().numpy(),
                'orientation': self.base_quat[env_idx].cpu().numpy(),
                'height': self.base_height[env_idx].cpu().numpy()
            }
            
            # Update orientation manager with current robot heading
            current_heading = float(self.base_euler_xyz[env_idx, 2].cpu())  # Yaw angle
            self.ipc3d_guidance.orientation_manager.update_current_heading(current_heading, current_time)
            
            # 使用转换后的IPC3D命令（替换原来的全局命令处理）
            forward_velocity = float(ipc3d_commands['desired_forward_velocity'][env_idx].cpu())
            angular_velocity = float(ipc3d_commands['desired_angular_velocity'][env_idx].cpu())
            
            # Update target heading based on angular velocity command
            if abs(angular_velocity) > self.ipc3d_guidance.orientation_manager.min_angular_velocity:
                self.ipc3d_guidance.orientation_manager.update_from_angular_velocity(angular_velocity, current_time)
            
            # Create target velocity for IPC3D (现在使用简化的格式)
            target_velocity = {
                'desired_forward_velocity': forward_velocity,  # 前向速度
                'desired_angular_velocity': angular_velocity  # 角速度
            }
            
            # Compute IPC3D trajectory for this environment
            try:
                # Initialize guidance if needed
                if not self.ipc3d_guidance.is_initialized:
                    self.ipc3d_guidance.initialize(current_state)
                
                # Set target for IPC3D controller (现在使用转换后的相对速度)
                self.ipc3d_guidance.set_target(target_velocity)
                
                # 使用简化的guidance模型更新方法
                guidance_output = self.ipc3d_guidance.update(current_state, dt=self.cfg.guidance.dt)
                
                if guidance_output:
                    # 更新期望轨迹缓存
                    self.ipc3d_desired_trajectory[env_idx] = torch.tensor(
                        guidance_output.get('com_position', [0, 0, 0]), device=self.device
                    )
                    self.ipc3d_desired_velocity[env_idx] = torch.tensor(
                        guidance_output.get('com_velocity', [0, 0, 0]), device=self.device  
                    )
                    self.ipc3d_control_forces[env_idx] = torch.tensor(
                        guidance_output.get('com_acceleration', [0, 0, 0]), device=self.device
                    )
                    
            except Exception as e:
                # 错误处理
                print(f"⚠️ IPC3D计算错误 env {env_idx}: {e}")
                continue
                
    def _maybe_push_robot(self):
        """Override base class method to prevent dual push systems."""
        # Use our specialized HI-12 push system instead
        self._maybe_push_robot_hi12()
        
    def _maybe_push_robot_hi12(self):
        """HI-12 specific external force disturbance system."""
        # Check if it's time for a new push
        if self.common_step_counter % self.cfg.domain_rand.push_interval == 0:
            self._initiate_push_sequence()
            
        # Apply continuous force if push is active
        if getattr(self, 'push_active', False):
            self._apply_continuous_push()
            self.push_step_count += 1
            
            # End push sequence
            if self.push_step_count >= self.cfg.domain_rand.push_duration:
                self._end_push_sequence()
                
    def _initiate_push_sequence(self):
        """Initiate a new force push sequence."""
        self.push_active = True
        self.push_step_count = 0
        self.last_push_step = self.common_step_counter
        
        # Record stability before push for recovery evaluation
        self.stability_before_push = self._compute_current_stability()
        self.recovery_start_time = self.common_step_counter
        
        # Get curriculum-based push force
        current_max_force = self._get_curriculum_push_force()
        
        # Generate push forces for each environment
        self._generate_push_forces(current_max_force)
        
        # Select push targets for each environment
        self._select_push_targets()
        
        if self.cfg.domain_rand.push_debug:
            # avg_force = torch.mean(torch.norm(self.current_push_force, dim=1))
            # print(f"🔥 Push initiated: {avg_force:.1f}N average force at step {self.common_step_counter}")
            pass
            
    def _get_curriculum_push_force(self):
        """Get curriculum learning based push force intensity."""
        step = self.common_step_counter
        schedule = self.cfg.domain_rand.push_force_schedule
        
        current_force = 40.0  # Default minimum
        for threshold_step, force_magnitude in sorted(schedule.items()):
            if step >= threshold_step:
                current_force = force_magnitude
                
        return current_force
        
    def _generate_push_forces(self, max_force):
        """Generate 3D push forces for all environments."""
        for env_idx in range(self.num_envs):
            # Random force magnitude
            force_magnitude = torch.rand(1, device=self.device) * max_force
            
            if self.cfg.domain_rand.push_3d_enabled:
                # 3D force direction with reduced Z component
                direction = torch.randn(3, device=self.device)
                direction[2] *= self.cfg.domain_rand.push_z_scale  # Reduce vertical force
                direction = direction / torch.norm(direction)
            else:
                # 2D force in XY plane
                direction = torch.zeros(3, device=self.device)
                xy_dir = torch.randn(2, device=self.device)
                direction[:2] = xy_dir / torch.norm(xy_dir)
                
            self.current_push_force[env_idx] = direction * force_magnitude
            
    def _select_push_targets(self):
        """Select random body parts to push for each environment."""
        if not self.cfg.domain_rand.enable_multi_point_push:
            # Default to base_link
            self.current_push_target[:] = self.push_body_indices.get("base_link", 0)
            return
            
        # Weighted random selection based on configuration
        body_names = list(self.push_body_indices.keys())
        weights = [self.cfg.domain_rand.push_body_weights.get(name, 0.1) for name in body_names]
        weight_tensor = torch.tensor(weights, device=self.device)
        
        for env_idx in range(self.num_envs):
            # Random selection based on weights
            selected_idx = torch.multinomial(weight_tensor, 1).item()
            selected_body = body_names[selected_idx]
            self.current_push_target[env_idx] = self.push_body_indices[selected_body]
            
    def _apply_continuous_push(self):
        """Apply continuous external forces during push sequence."""
        if not self.push_active:
            return
            
        # Check if base_link exists
        if "base_link" not in self.rigid_body_idx:
            return
            
        base_index = self.rigid_body_idx["base_link"]
        
        # Create force tensor with correct shape (num_envs, num_bodies, 3)
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        # Apply gait-aware timing if enabled
        current_forces = self.current_push_force.clone()
        if hasattr(self.cfg.domain_rand, 'adaptive_push_timing') and self.cfg.domain_rand.adaptive_push_timing:
            current_forces = self._apply_gait_aware_timing_simple(current_forces)
        
        # Apply current push force to base_link only
        forces[:, base_index, :] = current_forces
        
        # Get force application positions - use rigid_body_state positions
        # Shape: (num_envs, num_bodies, 13) -> extract positions (num_envs, num_bodies, 3)
        force_positions = self.rigid_body_state[:, :, 0:3].contiguous()
        
        # Flatten tensors to match Isaac Gym API expectations: (num_envs*num_bodies, 3)
        forces_flat = forces.view(-1, 3).contiguous()
        positions_flat = force_positions.view(-1, 3).contiguous()
        
        # Apply forces using Isaac Gym API
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces_flat),
            gymtorch.unwrap_tensor(positions_flat),
            gymapi.ENV_SPACE
        )
        
    def _apply_gait_aware_timing_simple(self, forces):
        """Apply simplified gait phase aware force timing."""
        # Detect stance vs swing phase (simplified)
        stance_mask = torch.sum(self.foot_contact.float(), dim=1) > 1.0  # Both feet in contact
        
        # Random probability checks
        random_vals = torch.rand(self.num_envs, device=self.device)
        
        # Apply forces based on gait phase probabilities (default 0.7 if not configured)
        stance_prob = getattr(self.cfg.domain_rand, 'push_during_stance', 0.7)
        swing_prob = getattr(self.cfg.domain_rand, 'push_during_swing', 0.3)
        
        stance_apply = stance_mask & (random_vals < stance_prob)
        swing_apply = ~stance_mask & (random_vals < swing_prob)
        
        # Combine timing masks
        apply_mask = stance_apply | swing_apply
        forces = forces * apply_mask.unsqueeze(-1).float()
        
        return forces
        
    def _end_push_sequence(self):
        """End the current push sequence."""
        self.push_active = False
        
        if self.cfg.domain_rand.push_debug:
            # print(f"✅ Push sequence ended at step {self.common_step_counter}")
            pass
    def _compute_current_stability(self):
        """Compute current stability metric for recovery evaluation."""
        # Combine multiple stability indicators
        
        # 1. Base orientation stability  
        orientation_stability = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) / 0.3)
        
        # 2. Velocity stability
        velocity_magnitude = torch.norm(self.base_lin_vel, dim=1)
        velocity_stability = torch.exp(-velocity_magnitude / 2.0)
        
        # 3. Contact stability
        contact_ratio = torch.sum(self.foot_contact.float(), dim=1) / len(self.feet_ids)
        
        # Combined stability score
        stability = 0.4 * orientation_stability + 0.3 * velocity_stability + 0.3 * contact_ratio
        
        return stability
        
    def _update_guidance_tracking(self):
        """Update guidance tracking metrics and errors."""
        if not self.cfg.guidance.enable_ipc3d:
            return
            
        # Compute trajectory tracking error
        trajectory_error = torch.norm(
            self.base_pos - self.ipc3d_desired_trajectory, dim=1
        )
        self.trajectory_tracking_error = trajectory_error
        
        # Compute guidance consistency (velocity alignment)
        if torch.norm(self.ipc3d_desired_velocity).sum() > 0:
            velocity_error = torch.norm(
                self.base_lin_vel - self.ipc3d_desired_velocity, dim=1
            )
            self.guidance_consistency_score = torch.exp(-velocity_error / 1.0)
        else:
            self.guidance_consistency_score = torch.ones(self.num_envs, device=self.device)
            
    # ==================================================================================
    # REWARD SYSTEM - Enhanced for IPC3D Guidance and Force Disturbance Training
    # ==================================================================================
    
    def _prepare_reward_function(self):
        """Prepare enhanced reward function with IPC3D guidance and disturbance recovery."""
        # Call parent class first to initialize base reward system
        super()._prepare_reward_function()
        
        # Note: Base class uses eval_reward() system, not reward_functions list
        # Our reward methods are automatically called if weights are defined in config
        
        # print(f"✅ Enhanced reward system prepared:")
        # print(f"   Base rewards: {len(self.reward_names) if hasattr(self, 'reward_names') else 0}")
        # print(f"   IPC3D guidance rewards: {'Enabled' if self.cfg.guidance.enable_ipc3d else 'Disabled'}")
        # print(f"   Disturbance recovery rewards: {'Enabled' if self.cfg.domain_rand.use_force_push else 'Disabled'}")
        
        # The reward methods below will be automatically called by eval_reward() 
        # if their corresponding weights are defined in the configuration
        
    def _reward_ipc3d_trajectory_tracking(self):
        """增强的IPC3D轨迹跟踪奖励 - 主要奖励（数值稳定版本）。
        
        结合COM位置跟踪和COM速度跟踪，为机器人提供精确的轨迹引导。
        这是连接IPC3D引导和强化学习的核心奖励函数。
        
        Returns:
            torch.Tensor: 轨迹跟踪奖励，值范围[0, 1]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        try:
            # 数值稳定性检查输入
            if (torch.any(torch.isnan(self.base_pos)) or torch.any(torch.isinf(self.base_pos)) or
                torch.any(torch.isnan(self.base_lin_vel)) or torch.any(torch.isinf(self.base_lin_vel)) or
                torch.any(torch.isnan(self.ipc3d_desired_trajectory)) or torch.any(torch.isinf(self.ipc3d_desired_trajectory)) or
                torch.any(torch.isnan(self.ipc3d_desired_velocity)) or torch.any(torch.isinf(self.ipc3d_desired_velocity))):
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
            # COM位置跟踪奖励
            com_position_error = torch.norm(
                self.base_pos - self.ipc3d_desired_trajectory, dim=1
            )
            com_position_error = torch.clamp(com_position_error, 0.0, 10.0)  # 限制误差范围
            position_reward = torch.exp(-com_position_error / 0.15)  # 15cm容忍度
            
            # COM速度跟踪奖励
            com_velocity_error = torch.norm(
                self.base_lin_vel - self.ipc3d_desired_velocity, dim=1
            )
            com_velocity_error = torch.clamp(com_velocity_error, 0.0, 10.0)  # 限制误差范围
            velocity_reward = torch.exp(-com_velocity_error / 0.3)   # 0.3m/s容忍度
            
            # 组合奖励：70%位置 + 30%速度
            combined_reward = 0.7 * position_reward + 0.3 * velocity_reward
            
            # 最终数值稳定性检查和范围限制
            combined_reward = torch.clamp(combined_reward, 0.0, 1.0)
            
            # 检查NaN/inf
            if torch.any(torch.isnan(combined_reward)) or torch.any(torch.isinf(combined_reward)):
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
            return combined_reward
            
        except Exception as e:
            # 任何异常都返回零奖励
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
                if self.debug_counter % 1000 == 0:
                    print(f"⚠️ Trajectory tracking reward error: {e}")
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
    def _reward_trajectory_tracking(self):
        """Reward for tracking IPC3D desired trajectory (backwards compatibility)."""
        # 保持向后兼容性，调用新的增强方法
        return self._reward_ipc3d_trajectory_tracking()
        
    def _reward_guidance_consistency(self):
        """Reward for consistency with IPC3D guidance forces/velocities."""
        return self.guidance_consistency_score
        
    def _reward_footstep_placement_accuracy(self):
        """简化的足迹位置准确性奖励。
        
        基于简化的guidance输出，计算足迹位置精度。
        
        Returns:
            torch.Tensor: 足迹位置准确性奖励
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        try:
            # 使用存储的足迹目标（如果有）
            if hasattr(self, 'current_footstep_targets') and self.current_footstep_targets:
                left_target = self.current_footstep_targets['left_foot']
                right_target = self.current_footstep_targets['right_foot']
                
                # 获取当前足迹位置
                if hasattr(self, 'feet_ids') and len(self.feet_ids) >= 2:
                    left_current = self.rigid_body_pos[:, self.feet_ids[0], :2]  # 只考虑XY
                    right_current = self.rigid_body_pos[:, self.feet_ids[1], :2]
                    
                    # 转换目标为tensor
                    left_target_tensor = torch.tensor(left_target[:2], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
                    right_target_tensor = torch.tensor(right_target[:2], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
                    
                    # 计算误差
                    left_error = torch.norm(left_current - left_target_tensor, dim=1)
                    right_error = torch.norm(right_current - right_target_tensor, dim=1)
                    
                    # 平均误差
                    footstep_errors = (left_error + right_error) / 2.0
                    
                    # 高斯奖励：15cm容忍度（放宽一些）
                    placement_reward = torch.exp(-footstep_errors / 0.15)
                    return torch.clamp(placement_reward, 0.0, 1.0)
            
            # 如果没有足迹目标，返回中性奖励
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device) * 0.5
            
        except Exception as e:
            # 错误处理：返回中性奖励
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device) * 0.5
        
    def _reward_step_location_error(self):
        """Penalty for inaccurate foot placement relative to IPC3D guidance (backwards compatibility)."""
        # 向后兼容性：转换为奖励格式（负数权重会将其转换为惩罚）
        placement_reward = self._reward_footstep_placement_accuracy()
        # 返回错误格式（高错误 = 低奖励）
        return 1.0 - placement_reward
        
    def _reward_stability_recovery(self):
        """Reward for quick recovery from external force disturbances."""
        if not self.cfg.domain_rand.use_force_push:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # Check if currently recovering from a push
        is_recovering = (self.common_step_counter - self.recovery_start_time) < (
            self.cfg.rewards.force_recovery_time / self.cfg.sim.dt
        )
        
        # Compute current stability
        current_stability = self._compute_current_stability()
        
        # Reward based on stability improvement during recovery
        recovery_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        recovery_mask = is_recovering & (self.stability_before_push > 0)
        if recovery_mask.any():
            # Reward stability improvement
            stability_improvement = current_stability[recovery_mask] - self.stability_before_push[recovery_mask]
            recovery_reward[recovery_mask] = torch.clamp(stability_improvement, 0, 1)
            
        return recovery_reward
        
    def _reward_balance_maintenance(self):
        """Reward for maintaining balance under disturbances."""
        return self._compute_current_stability()
        
    def _reward_contact_stability(self):
        """Reward for maintaining proper foot contact patterns."""
        # Reward proper contact patterns (not airborne, not excessive contact forces)
        
        # 1. Penalize being completely airborne
        any_contact = torch.sum(self.foot_contact.float(), dim=1) > 0
        airborne_penalty = (~any_contact).float() * -1.0
        
        # 2. Reward balanced contact (not always both feet down)
        contact_count = torch.sum(self.foot_contact.float(), dim=1)
        balanced_contact = torch.where(
            contact_count == 1.0,  # Single support preferred during walking
            torch.ones_like(contact_count),
            torch.where(
                contact_count == 2.0,  # Double support acceptable
                torch.ones_like(contact_count) * 0.8,
                torch.zeros_like(contact_count)  # No contact or >2 contacts penalized
            )
        )
        
        # 3. Penalize excessive contact forces
        max_contact_force = self.cfg.rewards.max_contact_force
        force_penalties = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        for foot_idx in range(len(self.feet_ids)):
            foot_forces = torch.norm(self.contact_forces[:, self.feet_ids[foot_idx], :], dim=1)
            force_penalties += torch.clamp((foot_forces - max_contact_force) / max_contact_force, 0, 1)
            
        force_penalties = force_penalties / len(self.feet_ids)
        
        # Combined contact stability
        contact_stability = balanced_contact - force_penalties + airborne_penalty
        
        return torch.clamp(contact_stability, -1, 1)
        
    def _reward_ipc3d_force_consistency(self):
        """简化的控制力一致性奖励（数值稳定版本）。
        
        基于简化的guidance输出，奖励机器人加速度与质心轨迹加速度一致。
        
        Returns:
            torch.Tensor: 控制力一致性奖励，范围限制在[-1.0, 1.0]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        try:
            # 使用存储的guidance输出（如果有）
            if hasattr(self, 'current_guidance_output') and self.current_guidance_output:
                desired_acceleration = self.current_guidance_output.get('com_acceleration')
                if desired_acceleration is None:
                    return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                
                # 转换为tensor并进行数值稳定性检查
                if isinstance(desired_acceleration, np.ndarray):
                    if np.any(np.isnan(desired_acceleration)) or np.any(np.isinf(desired_acceleration)):
                        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                    desired_acceleration = torch.from_numpy(desired_acceleration).to(self.device)
                elif not isinstance(desired_acceleration, torch.Tensor):
                    return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                
                # 检查tensor数值稳定性
                if torch.any(torch.isnan(desired_acceleration)) or torch.any(torch.isinf(desired_acceleration)):
                    return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                
                # 确保维度匹配
                if desired_acceleration.dim() == 1 and self.num_envs > 1:
                    desired_acceleration = desired_acceleration.unsqueeze(0).repeat(self.num_envs, 1)
                elif desired_acceleration.dim() == 1:
                    desired_acceleration = desired_acceleration.unsqueeze(0)
                
                # 计算机器人当前加速度（仅使用xy分量）
                if hasattr(self, 'last_base_lin_vel'):
                    # 数值稳定性检查
                    if (torch.any(torch.isnan(self.base_lin_vel)) or torch.any(torch.isinf(self.base_lin_vel)) or
                        torch.any(torch.isnan(self.last_base_lin_vel)) or torch.any(torch.isinf(self.last_base_lin_vel))):
                        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                    
                    robot_acceleration = (self.base_lin_vel - self.last_base_lin_vel) / self.cfg.sim.dt
                    robot_acceleration_xy = robot_acceleration[:, :2]  # 只考虑水平加速度
                    
                    # 限制加速度范围
                    robot_acceleration_xy = torch.clamp(robot_acceleration_xy, -50.0, 50.0)
                else:
                    robot_acceleration_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
                
                # 计算误差（只使用xy分量）
                desired_acceleration_xy = desired_acceleration[:, :2]
                # 限制期望加速度范围
                desired_acceleration_xy = torch.clamp(desired_acceleration_xy, -50.0, 50.0)
                
                force_consistency_error = torch.norm(
                    robot_acceleration_xy - desired_acceleration_xy, dim=1
                )
                
                # 限制误差范围
                force_consistency_error = torch.clamp(force_consistency_error, 0.0, 100.0)
                
                # 高斯奖励 - 2.0 m/s^2容忍度
                consistency_reward = torch.exp(-force_consistency_error / 2.0)
                
                # 最终数值稳定性检查
                consistency_reward = torch.clamp(consistency_reward, -1.0, 1.0)
                
                # 检查NaN/inf
                if torch.any(torch.isnan(consistency_reward)) or torch.any(torch.isinf(consistency_reward)):
                    return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                
                return consistency_reward
                
            else:
                # 如果没有guidance输出，返回零奖励
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                
        except Exception as e:
            # 发生错误时返回零奖励，避免训练中断
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
                if self.debug_counter % 100 == 0:  # 每100步打印一次
                    print(f"⚠️ Force consistency reward error: {e}")
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
    def _reward_ipc3d_orientation_tracking(self):
        """姿态跟踪奖励（数值稳定版本）。
        
        奖励机器人维持与IPC3D引导一致的姿态，特别是偏航角。
        添加了完整的数值稳定性检查。
        
        Returns:
            torch.Tensor: 姿态跟踪奖励，范围限制在[-2.0, 2.0]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        try:
            # 获取IPC3D轨迹信息
            try:
                ipc3d_output = self.ipc3d_guidance.generate_complete_trajectory_output()
                desired_yaw = ipc3d_output['com_trajectory'].get('yaw_angle', 0.0)
            except Exception as e:
                # 如果IPC3D输出不可用，使用命令角速度估算
                dt = self.cfg.sim.dt
                desired_yaw = self.base_euler_xyz[:, 2] + self.commands[:, 2] * dt
                
            # 计算偏航角跟踪误差
            current_yaw = self.base_euler_xyz[:, 2]
            
            # 数值稳定性检查
            if torch.any(torch.isnan(current_yaw)) or torch.any(torch.isinf(current_yaw)):
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
            # 确保desired_yaw为合适的tensor格式
            if isinstance(desired_yaw, (int, float)):
                desired_yaw = torch.full_like(current_yaw, desired_yaw)
            elif isinstance(desired_yaw, np.ndarray):
                desired_yaw = torch.from_numpy(desired_yaw).to(self.device)
            
            # 数值稳定性检查desired_yaw
            if torch.any(torch.isnan(desired_yaw)) or torch.any(torch.isinf(desired_yaw)):
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                
            # 计算角度差（使用数值稳定的方法）
            yaw_error = torch.abs(current_yaw - desired_yaw)
            # 处理角度循环（-π到π），确保数值稳定
            yaw_error = torch.min(yaw_error, 2 * np.pi - yaw_error)
            yaw_error = torch.clamp(yaw_error, 0.0, np.pi)  # 限制在合理范围
            
            # 姿态稳定性奖励（roll和pitch保持小）
            roll_pitch_error = torch.norm(self.base_euler_xyz[:, :2], dim=1)
            roll_pitch_error = torch.clamp(roll_pitch_error, 0.0, np.pi)  # 限制范围
            
            # 组合奖励（使用更保守的参数）
            yaw_reward = torch.exp(-yaw_error / 0.5)  # 增大容忍度到0.5 rad
            stability_reward = torch.exp(-roll_pitch_error / 0.3)  # 增大容忍度到0.3 rad
            
            # 组合并限制最终奖励
            combined_reward = 0.6 * yaw_reward + 0.4 * stability_reward
            
            # 最终数值稳定性检查和裁剪
            combined_reward = torch.clamp(combined_reward, -2.0, 2.0)
            
            # 检查NaN/inf
            if torch.any(torch.isnan(combined_reward)) or torch.any(torch.isinf(combined_reward)):
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
            return combined_reward
            
        except Exception as e:
            # 任何异常都返回零奖励，避免训练崩溃
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
                if self.debug_counter % 1000 == 0:  # 每1000步打印一次
                    print(f"⚠️ Orientation tracking reward error: {e}")
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    
    def _reward_action_smoothness(self):
        """Reward for smooth, non-jerky actions."""
        if not hasattr(self, 'last_actions'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # Compute action differences (acceleration)
        action_diff = torch.norm(self.actions - self.last_actions, dim=1)
        
        # Penalize large action changes (jerky motion)
        smoothness_reward = torch.exp(-action_diff / 0.5)
        
        return smoothness_reward
        
    def _compute_rewards(self):
        """Compute all rewards with enhanced weighting for IPC3D guidance and numerical stability monitoring."""
        # Call parent reward computation first
        super()._compute_rewards()
        
        # Global numerical stability check for IPC3D rewards
        if hasattr(self, 'rew_buf'):
            # Check for any abnormal reward values
            if torch.any(torch.isnan(self.rew_buf)) or torch.any(torch.isinf(self.rew_buf)):
                # Replace invalid values with zeros
                nan_mask = torch.isnan(self.rew_buf) | torch.isinf(self.rew_buf)
                self.rew_buf[nan_mask] = 0.0
                
                # Debug output for monitoring
                if not hasattr(self, 'debug_counter'):
                    self.debug_counter = 0
                self.debug_counter += 1
                if self.debug_counter % 1000 == 0:
                    print(f"⚠️ Detected and fixed {torch.sum(nan_mask).item()} invalid reward values")
            
            # Check for extremely large reward values
            large_reward_mask = torch.abs(self.rew_buf) > 1000.0
            if torch.any(large_reward_mask):
                # Clamp extremely large rewards
                self.rew_buf = torch.clamp(self.rew_buf, -1000.0, 1000.0)
                
                if hasattr(self, 'debug_counter'):
                    self.debug_counter += 1
                    if self.debug_counter % 1000 == 0:
                        print(f"⚠️ Clamped {torch.sum(large_reward_mask).item()} extremely large reward values")
        
        # Store current states for consistency rewards
        if hasattr(self, 'actions'):
            if not hasattr(self, 'last_actions'):
                self.last_actions = torch.zeros_like(self.actions)
            else:
                self.last_actions = self.actions.clone()
                
        # Store base linear velocity for force consistency calculation
        if hasattr(self, 'base_lin_vel'):
            if not hasattr(self, 'last_base_lin_vel'):
                self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
            else:
                self.last_base_lin_vel = self.base_lin_vel.clone()
                
    def _update_command_curriculum(self, env_ids):
        """Enhanced command curriculum considering IPC3D guidance performance."""
        super()._update_command_curriculum(env_ids)
        
        # Additional curriculum based on IPC3D tracking performance
        if self.cfg.guidance.enable_ipc3d and hasattr(self, 'trajectory_tracking_error'):
            avg_tracking_error = torch.mean(self.trajectory_tracking_error[env_ids])
            
            # If tracking is good, increase command difficulty
            if avg_tracking_error < 0.1:  # Less than 10cm average error
                # Gradually increase command ranges
                current_range = self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0]
                if current_range < 2.0:  # Don't exceed 2 m/s range
                    expansion = 0.05
                    self.command_ranges["lin_vel_x"][0] = max(self.command_ranges["lin_vel_x"][0] - expansion, -1.5)
                    self.command_ranges["lin_vel_x"][1] = min(self.command_ranges["lin_vel_x"][1] + expansion, 1.5)
                    
        # print(f"📈 Command curriculum updated for {len(env_ids)} environments")
        # print(f"   Current lin_vel_x range: [{self.command_ranges['lin_vel_x'][0]:.2f}, {self.command_ranges['lin_vel_x'][1]:.2f}]")
        
    # ==================================================================================
    # OBSERVATION SPACE - Enhanced with IPC3D Guidance Information  
    # ==================================================================================
    
    def _get_obs_ipc3d_desired_trajectory(self):
        """获取IPC3D期望轨迹作为观测值。
        
        基于简化的guidance输出，返回质心位置目标。
        
        Returns:
            torch.Tensor: IPC3D期望轨迹 [num_envs, 3]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        try:
            if hasattr(self, 'current_guidance_output') and self.current_guidance_output:
                desired_position = self.current_guidance_output.get('com_position')
                if desired_position is not None:
                    if isinstance(desired_position, np.ndarray):
                        desired_position = torch.from_numpy(desired_position).to(self.device)
                    if desired_position.dim() == 1:
                        desired_position = desired_position.unsqueeze(0).repeat(self.num_envs, 1)
                    return desired_position[:, :3]  # 确保是3维
            
            # 如果没有guidance输出，返回零向量
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
            
        except Exception:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
    def _get_obs_ipc3d_desired_velocity(self):
        """获取IPC3D期望速度作为观测值。
        
        基于简化的guidance输出，返回质心速度目标。
        
        Returns:
            torch.Tensor: IPC3D期望速度 [num_envs, 3]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        try:
            if hasattr(self, 'current_guidance_output') and self.current_guidance_output:
                desired_velocity = self.current_guidance_output.get('com_velocity')
                if desired_velocity is not None:
                    if isinstance(desired_velocity, np.ndarray):
                        desired_velocity = torch.from_numpy(desired_velocity).to(self.device)
                    if desired_velocity.dim() == 1:
                        desired_velocity = desired_velocity.unsqueeze(0).repeat(self.num_envs, 1)
                    return desired_velocity[:, :3]  # 确保是3维
            
            # 如果没有guidance输出，返回零向量
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
            
        except Exception:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
    def _get_obs_ipc3d_control_forces(self):
        """获取IPC3D控制力作为观测值。
        
        基于简化的guidance输出，返回质心加速度（作为控制力代理）。
        
        Returns:
            torch.Tensor: IPC3D控制力 [num_envs, 3]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        try:
            if hasattr(self, 'current_guidance_output') and self.current_guidance_output:
                desired_acceleration = self.current_guidance_output.get('com_acceleration')
                if desired_acceleration is not None:
                    if isinstance(desired_acceleration, np.ndarray):
                        desired_acceleration = torch.from_numpy(desired_acceleration).to(self.device)
                    if desired_acceleration.dim() == 1:
                        desired_acceleration = desired_acceleration.unsqueeze(0).repeat(self.num_envs, 1)
                    return desired_acceleration[:, :3]  # 确保是3维
            
            # 如果没有guidance输出，返回零向量
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
            
        except Exception:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
    def _get_obs_ipc3d_trajectory_error(self):
        """获取轨迹跟踪误差作为观测值。
        
        计算当前质心位置与期望位置的误差。
        
        Returns:
            torch.Tensor: 轨迹跟踪误差 [num_envs, 1]
        """
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        
        try:
            if hasattr(self, 'current_guidance_output') and self.current_guidance_output:
                desired_position = self.current_guidance_output.get('com_position')
                if desired_position is not None:
                    if isinstance(desired_position, np.ndarray):
                        desired_position = torch.from_numpy(desired_position).to(self.device)
                    if desired_position.dim() == 1:
                        desired_position = desired_position.unsqueeze(0).repeat(self.num_envs, 1)
                    
                    # 计算位置误差（只考虑xy平面）
                    current_position = self.base_pos[:, :2]
                    desired_position_xy = desired_position[:, :2]
                    error = torch.norm(current_position - desired_position_xy, dim=1)
                    # 更新trajectory_tracking_error缓冲区
                    self.trajectory_tracking_error[:, 0] = error
                    return self.trajectory_tracking_error
            
            # 如果没有guidance输出，返回零误差
            return torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            
        except Exception:
            return torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        
    def _get_obs_stability_score(self):
        """Get current stability score as observation."""
        stability = self._compute_current_stability()
        # 更新stability_score缓冲区
        self.stability_score[:, 0] = stability
        return self.stability_score
        
    def _get_obs_push_state(self):
        """Get external push state information."""
        push_state = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        
        if self.cfg.domain_rand.use_force_push:
            # [push_active, steps_since_push, current_force_magnitude, time_since_push]
            push_state[:, 0] = float(getattr(self, 'push_active', False))
            push_state[:, 1] = (self.common_step_counter - getattr(self, 'last_push_step', 0)) / 100.0  # Normalized
            
            if hasattr(self, 'current_push_force'):
                push_state[:, 2] = torch.norm(self.current_push_force, dim=1) / self.cfg.domain_rand.max_push_force_xy
                
            push_state[:, 3] = (self.common_step_counter - self.recovery_start_time) / (
                self.cfg.rewards.force_recovery_time / self.cfg.sim.dt
            )
            
        return push_state
    
    def _generate_orientation_aware_trajectory(self, env_idx: int, current_time: float, ipc3d_trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate orientation-aware reference trajectory that transforms IPC3D relative 
        trajectories back to global coordinates considering Hermite spline heading interpolation.
        
        This method implements the paper's approach where the trajectory is generated in 
        body-relative coordinates but then transformed to global coordinates for execution.
        
        Args:
            env_idx: Environment index
            current_time: Current simulation time
            ipc3d_trajectory: IPC3D trajectory in relative coordinates
            
        Returns:
            Dictionary containing global trajectory with position, velocity, and forces
        """
        try:
            # Extract relative trajectory from IPC3D
            relative_position = ipc3d_trajectory.get('position', np.zeros(3))
            relative_velocity = ipc3d_trajectory.get('velocity', np.zeros(3))
            control_forces = ipc3d_trajectory.get('forces', np.zeros(3))
            
            # Get current robot state
            current_position = self.base_pos[env_idx].cpu().numpy()
            current_velocity = self.base_lin_vel[env_idx].cpu().numpy()
            current_heading = float(self.base_euler_xyz[env_idx, 2].cpu())
            
            # Update orientation manager with current heading
            self.ipc3d_guidance.orientation_manager.update_current_heading(current_heading, current_time)
            
            # Generate future trajectory points over the planning horizon
            trajectory_duration = self.ipc3d_guidance.orientation_manager.future_horizon
            dt = self.cfg.sim.dt * self.cfg.guidance.guidance_update_freq
            num_points = int(trajectory_duration / dt)
            
            # Generate orientation trajectory using Hermite spline
            orientation_trajectory = self.ipc3d_guidance.orientation_manager.generate_orientation_trajectory(
                start_time=current_time,
                dt=dt,
                num_points=num_points
            )
            
            # Transform relative trajectory to global coordinates
            global_trajectory_points = []
            
            for i, orientation_point in enumerate(orientation_trajectory):
                # Get heading at this future time
                future_heading = orientation_point.heading
                
                # Transform relative position to global coordinates
                if len(relative_position) >= 2:
                    # Apply rotation transformation: global = R * relative
                    cos_h = np.cos(future_heading)
                    sin_h = np.sin(future_heading)
                    
                    # Transform position (relative to current position)
                    rel_x = relative_position[0] if len(relative_position) > 0 else 0.0
                    rel_z = relative_position[2] if len(relative_position) > 2 else 0.0
                    
                    global_x = current_position[0] + (cos_h * rel_x - sin_h * rel_z)
                    global_y = current_position[1] + (sin_h * rel_x + cos_h * rel_z)
                    global_z = current_position[2] + relative_position[1] if len(relative_position) > 1 else current_position[2]
                    
                    # Transform velocity (relative to current velocity)
                    rel_vx = relative_velocity[0] if len(relative_velocity) > 0 else 0.0
                    rel_vz = relative_velocity[2] if len(relative_velocity) > 2 else 0.0
                    
                    global_vx = cos_h * rel_vx - sin_h * rel_vz
                    global_vy = sin_h * rel_vx + cos_h * rel_vz
                    global_vz = relative_velocity[1] if len(relative_velocity) > 1 else 0.0
                    
                    # Store transformed point
                    global_point = {
                        'position': np.array([global_x, global_y, global_z]),
                        'velocity': np.array([global_vx, global_vy, global_vz]),
                        'heading': future_heading,
                        'time': orientation_point.time
                    }
                    global_trajectory_points.append(global_point)
                else:
                    # Fallback for incomplete relative trajectory
                    global_trajectory_points.append({
                        'position': current_position.copy(),
                        'velocity': current_velocity.copy(),
                        'heading': future_heading,
                        'time': orientation_point.time
                    })
            
            # Return the transformed trajectory (use first point for immediate reference)
            if global_trajectory_points:
                reference_point = global_trajectory_points[0]
                
                # Include control forces (transformed to global frame)
                global_forces = np.zeros(3)
                if len(control_forces) >= 2:
                    cos_h = np.cos(reference_point['heading'])
                    sin_h = np.sin(reference_point['heading'])
                    
                    fx_rel = control_forces[0] if len(control_forces) > 0 else 0.0
                    fz_rel = control_forces[2] if len(control_forces) > 2 else 0.0
                    
                    global_forces[0] = cos_h * fx_rel - sin_h * fz_rel
                    global_forces[1] = sin_h * fx_rel + cos_h * fz_rel
                    global_forces[2] = control_forces[1] if len(control_forces) > 1 else 0.0
                
                return {
                    'position': reference_point['position'],
                    'velocity': reference_point['velocity'],
                    'forces': global_forces,
                    'heading': reference_point['heading'],
                    'trajectory_points': global_trajectory_points  # Full trajectory for debugging
                }
            else:
                # Fallback: return current state
                return {
                    'position': current_position,
                    'velocity': current_velocity,
                    'forces': np.zeros(3),
                    'heading': current_heading,
                    'trajectory_points': []
                }
                
        except Exception as e:
            # Handle errors gracefully
            # print(f"⚠️ Error generating orientation-aware trajectory for env {env_idx}: {e}")
            return None