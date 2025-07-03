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
from gym.envs.guidance.ipc3d_guidance import IPC3DGuidanceModel
from gym.envs.guidance.ipc3d_controller import IPC3DParams


class HiControllerIPC3DCfg(HiControllerCfg):
    """Extended configuration for HI-12 with IPC3D guidance and force disturbance."""
    
    class guidance:
        """IPC3D guidance model configuration."""
        enable_ipc3d = True
        
        # Physical parameters (matched to HI-12)
        cart_mass = 15.0          # HI-12 robot mass (kg)
        pole_length = 0.421     # HI-12 COM height (m)
        control_mode = 1         # 1=velocity control, 0=position control
        max_force = 300.0        # Maximum control force (N)
        
        # Control parameters
        dt = 0.02                # Control time step (s)
        damping = 0.1            # System damping
        
        # LQR weights for IPC3D
        q_position = 10.0        # Position tracking weight
        q_velocity = 1.0         # Velocity tracking weight  
        r_control = 0.1          # Control effort weight
        
        # Trajectory planning
        step_length = 0.4        # Desired step length (m)
        step_width = 0.2         # Desired step width (m)
        step_height = 0.08       # Step height (m)
        step_time = 0.5          # Step duration (s)
        
        # Update frequency
        guidance_update_freq = 5 # Update every N simulation steps
        
    class domain_rand(HiControllerCfg.domain_rand):
        """Enhanced domain randomization with force disturbance."""
        
        # Force push system  
        use_force_push = True
        max_push_force_xy = 120.0     # Maximum push force for HI-12 (N)
        push_duration = 20            # Push duration (simulation steps)
        push_interval = 150           # Interval between pushes (steps)
        push_debug = False            # Enable debug output
        
        # Curriculum learning for push forces
        curriculum_push = True
        push_force_schedule = {
            # 0: 40.0,       # 0-15k steps: gentle push
            # 15000: 80.0,   # 15k-40k steps: moderate push  
            # 40000: 120.0,  # 40k+ steps: strong push
            # 80000: 150.0   # 80k+ steps: maximum challenge
            0 : 5,         # 0-15k steps: gentle push
            2000 : 10,   # 15k-40k steps: moderate push
            5000 : 20,  # 40k+ steps: strong push
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
            # Original HI-12 rewards (reduced weights)
            tracking_lin_vel = 0.5      # Reduced from default
            tracking_ang_vel = 0.3      # Reduced from default
            lin_vel_z = -1.0           # Penalize vertical motion
            ang_vel_xy = -0.5          # Penalize roll/pitch rotation
            orientation = -0.5         # Penalize orientation deviation
            base_height = -0.5         # Maintain proper height
            
            # IPC3D guidance rewards (new)
            trajectory_tracking = 1.5   # Track IPC3D desired trajectory
            guidance_consistency = 1.0  # Consistency with IPC3D forces
            step_location_error = -1.0  # Penalize step location errors
            
            # Disturbance recovery rewards (new)
            stability_recovery = 2.0    # Reward fast recovery from pushes
            balance_maintenance = 1.0   # Reward maintaining balance
            contact_stability = 0.8     # Reward proper foot contact
            
            # Energy efficiency (enhanced)
            torques = -0.0002          # Penalize high torques
            dof_vel = -0.0001          # Penalize high joint velocities
            action_smoothness = -0.5   # Penalize jerky actions
            
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
        
        # print("🤖 HI-12 + IPC3D + Force Disturbance Controller Initialized")
        # print(f"   Robot: HI-12 ({self.cfg.env.num_actuators} DOF)")
        # print(f"   Guidance: IPC3D (mass={self.cfg.guidance.cart_mass}kg)")
        # print(f"   Disturbance: Force push enabled ({self.cfg.domain_rand.max_push_force_xy}N max)")
        
    def _init_ipc3d_guidance(self):
        """Initialize IPC3D guidance model and related buffers."""
        if not self.cfg.guidance.enable_ipc3d:
            return
            
        # Configure IPC3D parameters for HI-12
        self.ipc3d_params = IPC3DParams(
            mass_cart=self.cfg.guidance.cart_mass,
            mass_pole=3.0,  # Estimated distributed mass
            pole_length=self.cfg.guidance.pole_length,
            gravity=9.81,
            damping=self.cfg.guidance.damping,
            dt=self.cfg.guidance.dt,
            max_force=self.cfg.guidance.max_force,
            q_position=self.cfg.guidance.q_position,
            q_velocity=self.cfg.guidance.q_velocity,
            r_control=self.cfg.guidance.r_control,
            control_mode=self.cfg.guidance.control_mode
        )
        
        # Create IPC3D guidance model
        self.ipc3d_guidance = IPC3DGuidanceModel(
            params=self.ipc3d_params,
            robot_spec=None,
            guidance_config=self.cfg.guidance
        )
        
        # IPC3D state tracking
        self.guidance_update_counter = 0
        
        # print(f"✅ IPC3D guidance initialized:")
        # print(f"   Control mode: {'Velocity' if self.cfg.guidance.control_mode else 'Position'}")
        # print(f"   Update frequency: every {self.cfg.guidance.guidance_update_freq} steps")
        
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
                # print(f"   Push body registered: {body_name}")
            else:
                # print(f"   Warning: Push body {body_name} not found in HI-12 model")
                pass

        # Current push state
        self.current_push_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_push_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # print(f"✅ Force disturbance system initialized:")
        # print(f"   Push bodies: {len(self.push_body_indices)} registered")
        # print(f"   Max force: {self.cfg.domain_rand.max_push_force_xy}N")
        
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
        
        # Trajectory tracking errors
        self.trajectory_tracking_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
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
        
        # print("✅ Guidance tracking buffers initialized")
        
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
        
    def _update_ipc3d_guidance(self):
        """Update IPC3D guidance trajectory and control forces."""
        self.guidance_update_counter += 1
        
        # Update at specified frequency to reduce computational load
        if self.guidance_update_counter % self.cfg.guidance.guidance_update_freq != 0:
            return
            
        # Process each environment
        for env_idx in range(self.num_envs):
            # Extract current robot state
            current_state = {
                'position': self.base_pos[env_idx].cpu().numpy(),
                'velocity': self.base_lin_vel[env_idx].cpu().numpy(),
                'orientation': self.base_quat[env_idx].cpu().numpy(),
                'height': self.base_height[env_idx].cpu().numpy()
            }
            
            # Get target velocity from commands
            target_velocity = {
                'velocity_x': float(self.commands[env_idx, 0].cpu()),
                'velocity_y': float(self.commands[env_idx, 1].cpu()) if self.commands.shape[1] > 1 else 0.0,
                'angular_velocity': float(self.commands[env_idx, 2].cpu()) if self.commands.shape[1] > 2 else 0.0
            }
            
            # Compute IPC3D trajectory for this environment
            try:
                # Initialize guidance if needed
                if not self.ipc3d_guidance.is_initialized:
                    self.ipc3d_guidance.initialize(current_state)
                
                # Set target for IPC3D controller
                self.ipc3d_guidance.set_target(target_velocity)
                
                # Compute next trajectory point
                dt = self.cfg.sim.dt * self.cfg.guidance.guidance_update_freq
                trajectory = self.ipc3d_guidance.compute_trajectory(
                    current_state=current_state,
                    target_velocity=target_velocity,
                    dt=dt
                )
                
                if trajectory is not None:
                    # Update desired trajectory
                    self.ipc3d_desired_trajectory[env_idx] = torch.tensor(
                        trajectory.get('position', [0, 0, 0]), device=self.device
                    )
                    self.ipc3d_desired_velocity[env_idx] = torch.tensor(
                        trajectory.get('velocity', [0, 0, 0]), device=self.device  
                    )
                    self.ipc3d_control_forces[env_idx] = torch.tensor(
                        trajectory.get('forces', [0, 0, 0]), device=self.device
                    )
                    
            except Exception:
                # Handle IPC3D computation errors gracefully
                # Use previous trajectory or zero as fallback
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
        
    def _reward_trajectory_tracking(self):
        """Reward for tracking IPC3D desired trajectory."""
        # Gaussian reward based on tracking error
        tracking_reward = torch.exp(-self.trajectory_tracking_error / self.cfg.rewards.trajectory_sigma)
        return tracking_reward
        
    def _reward_guidance_consistency(self):
        """Reward for consistency with IPC3D guidance forces/velocities."""
        return self.guidance_consistency_score
        
    def _reward_step_location_error(self):
        """Penalty for inaccurate foot placement relative to IPC3D guidance."""
        if not hasattr(self, 'step_commands') or not hasattr(self, 'foot_states'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # Compute foot placement error relative to step commands
        # This is a simplified version - in practice you'd want more sophisticated step tracking
        step_errors = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # For each foot, compute placement error
        for foot_idx in range(len(self.feet_ids)):
            if hasattr(self, 'step_commands'):
                # Get current foot position in base frame
                foot_pos_world = self.foot_states[:, foot_idx, :3]
                foot_pos_base = foot_pos_world - self.base_pos
                
                # Compare with step command (if available)
                if hasattr(self, 'step_commands'):
                    target_pos = self.step_commands[:, foot_idx, :3] if self.step_commands.shape[2] >= 3 else self.step_commands[:, foot_idx, :2]
                    if target_pos.shape[1] == 2:
                        # Only XY comparison
                        error = torch.norm(foot_pos_base[:, :2] - target_pos, dim=1)
                    else:
                        # Full XYZ comparison
                        error = torch.norm(foot_pos_base - target_pos, dim=1)
                    step_errors += error
                    
        # Average error across feet
        step_errors = step_errors / len(self.feet_ids)
        
        # Return error values (higher = worse, will be penalized by negative weight)
        return step_errors
        
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
        """Compute all rewards with enhanced weighting for IPC3D guidance."""
        # Call parent reward computation first
        super()._compute_rewards()
        
        # Store current actions for smoothness reward
        if hasattr(self, 'actions'):
            if not hasattr(self, 'last_actions'):
                self.last_actions = torch.zeros_like(self.actions)
            else:
                self.last_actions = self.actions.clone()
                
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
    
    def _get_obs_ipc3d_trajectory(self):
        """Get IPC3D desired trajectory as observation."""
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        return self.ipc3d_desired_trajectory
        
    def _get_obs_ipc3d_velocity(self):
        """Get IPC3D desired velocity as observation."""  
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        return self.ipc3d_desired_velocity
        
    def _get_obs_ipc3d_forces(self):
        """Get IPC3D control forces as observation."""
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        return self.ipc3d_control_forces
        
    def _get_obs_trajectory_error(self):
        """Get trajectory tracking error as observation."""
        if not self.cfg.guidance.enable_ipc3d:
            return torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        return self.trajectory_tracking_error.unsqueeze(1)
        
    def _get_obs_stability_score(self):
        """Get current stability score as observation."""
        stability = self._compute_current_stability()
        return stability.unsqueeze(1)
        
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