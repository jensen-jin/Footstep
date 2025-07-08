#!/usr/bin/env python3
"""
IPC3D (3D Inverted Pendulum on Cart) Controller Implementation

Based on FlexLoco's IPC3D algorithm, this module provides:
- 3D inverted pendulum dynamics simulation
- SDRE (State-Dependent Riccati Equation) control
- LQR-based control synthesis
- Dual-axis independent control for X and Z directions

Reference: FlexLocoIPCDNNwalk/work/gym_walk/IPC3d_approx.lua
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.linalg import solve_continuous_are
import scipy.linalg


@dataclass
class IPC3DParams:
    """Parameters for 3D Inverted Pendulum on Cart model."""
    
    # Physical parameters - 专门针对1kg:15kg速度跟踪优化
    mass_cart: float = 1.0       # Cart mass (kg) - robot body mass  
    mass_pole: float = 15.0      # Pole mass (kg) - distributed mass
    pole_length: float = 0.56    # Pole length (m) - COM height
    inertia: float = 4.7         # Pole moment of inertia (kg⋅m²) - 基于15kg*0.56²计算的合理惯性矩
    gravity: float = 9.81        # Gravity (m/s²)
    damping: float = 3        # System damping - 降低阻尼提升响应性
    
    # Control parameters
    dt: float = 1/50.0           # Time step (s) - 提高仿真频率
    max_force: float = 800.0     # Maximum control force (N) - 提高控制力上限
    
    # LQR weights - 解耦的权重参数设计
    # 小车位置/速度控制权重
    q_cart_position: float = 0.01        # Cart position tracking weight (低权重用于速度控制)
    q_cart_velocity: float = 10     # Cart velocity tracking weight
    
    # 摆杆角度/角速度控制权重
    q_pole_angle: float = 25.0          # Pole angle stabilization weight
    q_pole_angular_velocity: float = 30.0  # Pole angular velocity damping weight
    
    # 控制努力权重 - 静差消除优化后参数
    r_control: float = 0.05              # 降低控制努力权重，允许更大控制力
    
    # Control mode: 0=position, 1=velocity
    control_mode: int = 1        # Default to velocity control

# SDRE（状态依赖黎卡提方程 控制器）基类
class NonlinearController:
    """Base class for nonlinear controllers using SDRE method."""
    
    def __init__(self, state_dim: int, control_dim: int):
        self.dim = state_dim
        self.cdim = control_dim
        
        # System matrices D*ddx + C*dx + G*x = H*u
        self.D = np.zeros((state_dim, state_dim))
        self.C = np.zeros((state_dim, state_dim))  
        self.G = np.zeros((state_dim, state_dim))
        self.H = np.zeros((state_dim, control_dim))
        
        # Linearized system matrices dx = A*x + B*u
        self.A = np.zeros((2 * state_dim, 2 * state_dim))
        self.B = np.zeros((2 * state_dim, control_dim))
        
        # Gradient of G with respect to theta (for nonlinear terms)
        self.dGdTheta_0 = np.zeros(state_dim)
        
    def _update_sdre(self, x: np.ndarray):
        """Update SDRE linearization around current state."""       
        try:
            D_inv = np.linalg.inv(self.D)
        except np.linalg.LinAlgError:
            print("⚠️ D matrix inversion failed, using pseudo-inverse")
            # Use pseudo-inverse if D is singular
            D_inv = np.linalg.pinv(self.D)
        
        # Upper block: [0 I]
        self.A[:self.dim, :self.dim] = 0
        self.A[:self.dim, self.dim:] = np.eye(self.dim)
        
        # Lower block: [-D^-1*G -D^-1*C]
        self.A[self.dim:, :self.dim] = -D_inv @ self.G
        self.A[self.dim:, self.dim:] = -D_inv @ self.C
        
        # Add small regularization to position states to ensure observability
        try:
            cond_num = np.linalg.cond(self.A)
            if cond_num > 1e10 or np.isnan(cond_num) or np.isinf(cond_num):
                epsilon = 1e-4
                self.A[0, 0] = -epsilon  # Small position damping
                self.A[1, 1] = -epsilon  # Small angle damping
                # print(f"🔧 Adding regularization: condition number = {cond_num}")
        except (np.linalg.LinAlgError, ValueError):
            # If condition number calculation fails, add regularization
            epsilon = 1e-4
            self.A[0, 0] = -epsilon
            self.A[1, 1] = -epsilon
        
        # Control matrix B = [0; D^-1*H]
        self.B[:self.dim, :] = 0
        self.B[self.dim:, :] = D_inv @ self.H
        
    def update_sdre(self, theta: np.ndarray, dtheta: np.ndarray):
        """Update SDRE matrices - must be implemented by subclasses."""
        raise NotImplementedError
    # 线性化 + 求解控制量 + 积分 返回加速度
    def one_step(self, x: np.ndarray, u: np.ndarray, dt: float, 
                 max_force: float, Q: np.ndarray, R: np.ndarray, 
                 K: np.ndarray, xd: np.ndarray, tau: Optional[float] = None) -> float:
        """Perform one integration step with LQR control."""
        
        # Check for numerical issues in state
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("⚠️ State contains NaN or Inf, resetting to zero")
            x[:] = 0.0
            u[:] = 0.0
            return 0.0
        
        # Apply control limits
        u_limited = np.clip(u, -max_force, max_force)
        
        # Update SDRE linearization
        dim = self.dim
        try:
            self.update_sdre(x[:dim], x[dim:])
        except Exception as e:
            print(f"⚠️ SDRE update failed: {e}, using previous matrices")
        
        # Calculate control force with numerical checks
        error = x - xd
        if np.any(np.isnan(error)) or np.any(np.isinf(error)):
            error = np.zeros_like(error)
        
        try:
            u_control = K @ error
            
            # 前馈补偿（仅对速度控制模式）
            if hasattr(self, 'feedforward_gain') and len(xd) > 2 and abs(xd[2]) > 1e-6:
                total_mass = self.M + self.m
                current_velocity = x[2] if len(x) > 2 else 0.0
                velocity_error = xd[2] - current_velocity
                
                # 1. 阻尼补偿
                damping_compensation = self.damping_compensation_gain * self.b * xd[2]
                
                # 2. 动态质量补偿
                acceleration_feedforward = self.mass_compensation_gain * total_mass * velocity_error
                
                # 3. 自适应前馈增益
                error_magnitude = abs(velocity_error)
                gain_min, gain_max = self.adaptive_gain_range
                
                if velocity_error > 0.1:  # 超调情况
                    adaptive_gain = gain_min * 0.5
                elif velocity_error < -0.1:  # 不足情况
                    adaptive_gain = gain_max
                else:
                    adaptive_gain = gain_min + (gain_max - gain_min) * min(error_magnitude, 1.0)
                
                # 4. 综合前馈控制
                base_feedforward = adaptive_gain * xd[2] * total_mass * 0.3
                feedforward = base_feedforward + damping_compensation + acceleration_feedforward
                
                u_control[0] += feedforward
            
                    
        except Exception as e:
            print(f"⚠️ Control calculation failed: {e}")
            u_control = np.zeros(self.cdim)
        
        # Check control output
        if np.any(np.isnan(u_control)) or np.any(np.isinf(u_control)):
            u_control = np.zeros_like(u_control)
        
        u_control = np.clip(u_control, -max_force, max_force)
        
        # System dynamics: dx = A*x + B*u with numerical checks
        try:
            dx = self.A @ x.reshape(-1, 1) + self.B @ u_control.reshape(-1, 1)
            dx = dx.flatten()
        except Exception:
            dx = np.zeros_like(x)
        
        # Check dynamics output
        if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
            dx = np.zeros_like(x)
        
        # Limit integration step size for stability
        max_step = 0.1  # Maximum change per step
        dx = np.clip(dx * dt, -max_step, max_step)
        
        # Euler integration
        x[:] = x + dx
        
        # 角度范围限制：保持在合理的物理范围内
        if len(x) > dim:  # 确保有角度状态
            angle_idx = 1  # 角度状态索引
            if angle_idx < dim:
                # 限制角度在合理范围内，围绕π平衡点
                x[angle_idx] = np.clip(x[angle_idx], np.pi/2, 3*np.pi/2)
        
        # Update control input
        u[:] = u_control
        
        # Return acceleration for compatibility
        return dx[dim:] if len(dx) > dim else 0.0

# 二维IPC model SRDE 控制器实现
class IPCController(NonlinearController):
    """2D Inverted Pendulum on Cart controller."""
    
    def __init__(self, M: float, m: float, b: float, I: float, g: float, l: float):
        super().__init__(2, 1)  # 2 states (cart pos, pole angle), 1 control input
        
        self.M = M  # Cart mass
        self.m = m  # Pole mass  
        self.b = b  # Damping
        self.I = I  # Pole inertia
        self.g = g  # Gravity
        self.l = l  # Pole length
        
        # Gravitational term gradient
        self.dGdTheta_0 = np.array([0, -m * g * l])
        self.H = np.array([[1], [0]])  # Force applied to cart
        
    def set_inertia(self, l: float, I: float):
        """Update pole length and inertia."""
        self.l = l
        self.I = I
        self.dGdTheta_0[1] = -self.m * self.g * self.l
    # 更新 SDRE 矩阵
    def update_sdre(self, x: np.ndarray, dx: np.ndarray):
        """Update SDRE matrices for current state."""
        M, m, b, I, g, l = self.M, self.m, self.b, self.I, self.g, self.l
        
        theta = x[1]  # Pole angle
        dtheta = dx[1]  # Pole angular velocity
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Mass matrix D
        self.D[0, 0] = M + m
        self.D[0, 1] = -m * l * cos_theta  
        self.D[1, 0] = -m * l * cos_theta
        self.D[1, 1] = I + m * l * l
        
        # Damping matrix C  
        self.C[0, 0] = b
        self.C[0, 1] = m * l * dtheta * sin_theta
        self.C[1, 0] = 0
        self.C[1, 1] = 0
        
        # Gravitational forces G
        self.G[0, 0] = 0
        self.G[0, 1] = 0  
        self.G[1, 0] = 0
        
        # 重力恢复力项 - 原来的角度约定：θ=π为竖直平衡点
        gravity_term = -m * g * l * sin_theta
        
        # 处理接近竖直位置的奇异情况 (θ ≈ 0 或 θ ≈ 2π)
        if abs(theta) < 1e-4 or abs(theta - 2*np.pi) < 1e-4:
            # 使用小角度近似避免数值不稳定
            effective_theta = np.sign(theta) * max(abs(theta), 1e-4) if theta != 0 else 1e-4
            gravity_term = -m * g * l * effective_theta
        
        self.G[1, 1] = gravity_term
        
        # Update linearized system
        self._update_sdre(x)

# 单轴 IPC 控制器实现
class IPCView:
    """Single-axis IPC controller with position/velocity control modes."""
    
    def __init__(self, M: float, m: float, b: float, I: float, g: float, l: float,
                 dt: float, q: float, qd: Optional[float] = None):
        """
        Initialize IPC controller for single axis.
        
        Args:
            M: Cart mass
            m: Pole mass
            b: Damping coefficient
            I: Pole moment of inertia  
            g: Gravity
            l: Pole length
            dt: Time step
            q: LQR position/velocity weight
            qd: LQR velocity weight (defaults to q)
        """
        # 初始化 IPC 控制器
        self.ipc = IPCController(M, m, b, I, g, l)
        self.l = l
        self.dt = dt
        self.t = 0.0
        
        # Initialize state: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
        self.x = np.zeros(4)
        self.xd = np.zeros(4)  # Desired state
        self.U = np.zeros(1)   # Control input
        
        # 精确前馈控制相关变量 - 静差消除优化版本
        self.prev_desired_vel = 0.0  # 上一次期望速度
        self.feedforward_gain = 0.3              # 优化的基本前馈增益
        self.damping_compensation_gain = 2.0     # 提升阻尼补偿消除静差
        self.mass_compensation_gain = 0.25       # 提升质量补偿消除静差
        self.adaptive_gain_range = (0.8, 1.2)    # 优化的自适应增益范围
        
        # LQR matrices
        self.Q = np.zeros((4, 4))
        self.R = np.eye(1)  # Will be updated with r_control parameter
        self.r_control = 0.1   # 静差消除优化的控制努力权重
        
        # Set control mode and gains
        qd = qd if qd is not None else q
        self.mode = -1  # Initialize to invalid mode to force update
        self.K = np.zeros((1, 4))  # Initialize K matrix
        self.set_lqr_gains(1, q, qd, q, qd, self.r_control)
        
        self.max_force = 800.0
    
    # 设置LQR增益 和 控制模式 - 解耦版本
    def set_lqr_gains(self, mode: int, 
                      cart_pos_weight: float, cart_vel_weight: float,
                      pole_angle_weight: float, pole_angvel_weight: float,
                      r_control: float = 0.3):
        """Set LQR gains with decoupled parameters for position/velocity/angle/angular_velocity control."""
        # Update control effort weight
        self.r_control = r_control
        self.R = np.eye(1) * r_control  # 修正：使用r_control参数而非固定值1.0
        
        if self.mode != mode:
            self.Q.fill(0)
            if mode == 0:  # Position control
                self.Q[0, 0] = cart_pos_weight     # Cart position tracking
                self.Q[1, 1] = pole_angle_weight   # Pole angle stabilization
                self.Q[2, 2] = cart_vel_weight     # Cart velocity damping
                self.Q[3, 3] = pole_angvel_weight  # Pole angular velocity damping
            else:  # Velocity control - 静差消除优化
                # 大幅提升速度跟踪权重以消除静差
                self.Q[0, 0] = cart_pos_weight * 0.005   # 保持较低位置权重
                self.Q[1, 1] = pole_angle_weight * 1.8   # 轻微降低角度权重，平衡跟踪与稳定性
                self.Q[2, 2] = cart_vel_weight * 3.0     # 大幅提升速度权重消除静差
                self.Q[3, 3] = pole_angvel_weight * 1.3  # 保持角速度阻尼
            
            self.mode = mode
            
            # Update SDRE and compute LQR gain
            self.ipc.update_sdre(self.x[:2], self.x[2:])
            self.K = self._solve_lqr(self.ipc.A, self.ipc.B, self.Q, self.R)
    # 求解LQR 问题的最优增益矩阵 K
    def _solve_lqr(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve LQR problem to find optimal gain matrix K with improved numerical stability."""
        try:
            # Check matrix conditions first
            if np.any(np.isnan(A)) or np.any(np.isinf(A)):
                raise ValueError("A matrix contains NaN or Inf")
            if np.any(np.isnan(B)) or np.any(np.isinf(B)):
                raise ValueError("B matrix contains NaN or Inf")
            if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
                raise ValueError("Q matrix contains NaN or Inf")
            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                raise ValueError("R matrix contains NaN or Inf")
            
            # 数值稳定性改进：检查并修复A矩阵条件数
            A_cond = np.linalg.cond(A)
            if A_cond > 1e12 or np.isinf(A_cond):
                # 添加小的正则化项改善条件数
                epsilon = 1e-6
                A_regularized = A.copy()
                # 对角线添加小阻尼，提升数值稳定性
                A_regularized[2, 2] += epsilon  # 小车速度阻尼
                A_regularized[3, 3] += epsilon  # 摆杆角速度阻尼
                A = A_regularized
            
            if np.linalg.norm(B) < 1e-12:
                raise ValueError("B matrix is too small")
            
            # 确保Q矩阵正半定
            if not np.allclose(Q, Q.T):
                Q = (Q + Q.T) / 2  # 强制对称
            
            Q_eigs = np.linalg.eigvals(Q)
            if np.any(Q_eigs < -1e-10):
                Q = Q + (abs(np.min(Q_eigs)) + 1e-6) * np.eye(Q.shape[0])
            
            # 确保R矩阵正定
            if not np.allclose(R, R.T):
                R = (R + R.T) / 2  # 强制对称
            
            R_eigs = np.linalg.eigvals(R)
            if np.any(R_eigs <= 1e-10):
                R = R + (abs(np.min(R_eigs)) + 1e-6) * np.eye(R.shape[0])
            
            # 求解连续时间代数Riccati方程
            P = solve_continuous_are(A, B, Q, R)
            
            # Check if P is valid
            if np.any(np.isnan(P)) or np.any(np.isinf(P)):
                raise ValueError("ARE solution contains NaN or Inf")
            
            R_inv = np.linalg.inv(R)
            K = R_inv @ B.T @ P
            
            # Final check of K
            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                raise ValueError("K matrix contains NaN or Inf")
            
            return K
            
        except Exception as e:
            print(f"⚠️ LQR solve failed: {e}")
            print(f"   A shape: {A.shape}, condition: {np.linalg.cond(A):.2e}")
            print(f"   B shape: {B.shape}, B norm: {np.linalg.norm(B):.2e}")
            print(f"   Q shape: {Q.shape}, Q trace: {np.trace(Q):.2e}")
            print(f"   R shape: {R.shape}, R trace: {np.trace(R):.2e}")
            
            # 返回保守的默认增益矩阵
            default_K = np.zeros((self.ipc.cdim, 2 * self.ipc.dim))
            # 基本的PD控制作为后备
            if self.ipc.cdim == 1 and self.ipc.dim == 2:
                default_K[0, 0] = -1.0   # 位置增益
                default_K[0, 1] = -10.0  # 角度增益  
                default_K[0, 2] = -2.0   # 速度增益
                default_K[0, 3] = -5.0   # 角速度增益
            return default_K
    # 设置期望速度或位置      
    def set_param(self, speed: float):
        """Set desired velocity or position."""
        self.xd.fill(0)
        if self.mode == 0:  # Position control
            self.xd[0] = speed  # Desired cart position
        else:  # Velocity control - 精确参考轨迹
            # 更精确的移动参考策略：根据当前速度调整预视距离
            current_velocity = self.x[2] if len(self.x) > 2 else 0.0
            velocity_error = abs(current_velocity - speed)
            
            # 自适应预视距离：误差大时缩短，误差小时延长
            if velocity_error > 0.5:
                preview_time = 3.0  # 大误差时短期预视
            elif velocity_error > 0.2:
                preview_time = 5.0  # 中等误差
            else:
                preview_time = 8.0  # 小误差时长期预视
            
            self.xd[0] = self.x[0] + speed * preview_time  # 动态预视位置
            self.xd[2] = speed  # Desired cart velocity
            # 更新前馈控制相关变量
            self.prev_desired_vel = speed
    # 设置最大控制力      
    def set_maximum_force(self, force: float):
        """Set maximum control force."""
        self.max_force = force
        
    #    
    def one_step(self, tau: Optional[float] = None) -> float:
        """Perform one simulation step with SDRE control."""
        
        ddtheta = self.ipc.one_step(
            self.x, self.U, self.dt, self.max_force,
            self.Q, self.R, self.K, self.xd, tau
        )

        self.t += self.dt
        return ddtheta
        
    def one_step_raw(self, U: np.ndarray):
        """Perform one simulation step with raw control input."""
        dim = self.ipc.dim
        self.ipc.update_sdre(self.x[:dim], self.x[dim:])
        
        # Apply control limits
        U_limited = np.clip(U, -self.max_force, self.max_force)
        
        # System dynamics
        dx = self.ipc.A @ self.x + self.ipc.B @ U_limited.reshape(-1, 1)
        
        # Euler integration
        self.x += dx.flatten() * self.dt
        self.t += self.dt

class IPC_Plane:
    """2D平面倒立摆小车控制器 - 专注于机器人前向运动和俯仰平衡"""
    
    def __init__(self, params: IPC3DParams):
        """
        初始化2D平面倒立摆控制器
        
        Args:
            params: IPC3D参数配置
        """
        self.params = params
        
        # 基于IPCView创建前向运动控制器 - 使用解耦参数
        self.forward_controller = IPCView(
            params.mass_cart, params.mass_pole, params.damping,
            params.inertia, params.gravity, params.pole_length,
            params.dt, params.q_cart_position, params.q_cart_velocity  # 使用解耦参数
        )
        
        # 设置控制模式为速度控制 - 使用解耦参数
        self.forward_controller.set_lqr_gains(
            params.control_mode, 
            params.q_cart_position, params.q_cart_velocity,
            params.q_pole_angle, params.q_pole_angular_velocity,
            params.r_control  # 传递控制努力权重
        )
        
        # 设置最大控制力
        self.forward_controller.set_maximum_force(params.max_force)
        
        # 期望前向速度
        self.desired_forward_velocity = 0.0
        
    def set_desired_forward_velocity(self, vel_forward: float):
        """
        设置期望的前向速度
        
        Args:
            vel_forward: 期望前向速度 (m/s)
        """
        self.desired_forward_velocity = vel_forward
        self.forward_controller.set_param(vel_forward)
        
    def step(self, n_times: int = 1) -> Dict[str, Any]:
        """
        执行控制步骤
        
        Args:
            n_times: 仿真步数
            
        Returns:
            包含控制器状态的字典
        """
        for _ in range(n_times):
            self.forward_controller.one_step()
            
        return {
            'forward_state': self.forward_controller.x.copy(),
            'control_force': self.forward_controller.U.copy(),
            'time': self.forward_controller.t
        }
        
    def get_robot_forward_position(self) -> float:
        """获取机器人前向位置 (m)"""
        return float(self.forward_controller.x[0])
        
    def get_robot_forward_velocity(self) -> float:
        """获取机器人前向速度 (m/s)"""
        return float(self.forward_controller.x[2])
        
    def get_robot_pitch_angle(self) -> float:
        """获取机器人俯仰角 (rad) - 转换为用户期望的角度约定"""
        # 内部使用θ=π为平衡点，转换为θ=0为平衡点
        internal_angle = float(self.forward_controller.x[1])
        # 转换：内部π -> 用户0，内部π±δ -> 用户±δ
        user_angle = internal_angle - np.pi
        # 限制到(-π/2, π/2)范围
        user_angle = np.clip(user_angle, -np.pi/2, np.pi/2)
        return user_angle
        
    def get_robot_pitch_angular_velocity(self) -> float:
        """获取机器人俯仰角速度 (rad/s)"""
        return float(self.forward_controller.x[3])
        
    def get_forward_control_force(self) -> float:
        """获取前向控制力 (N)"""
        return float(self.forward_controller.U[0])
        
    def get_robot_state(self) -> Dict[str, float]:
        """
        获取完整的机器人状态
        
        Returns:
            包含机器人所有状态信息的字典
        """
        return {
            'forward_position': self.get_robot_forward_position(),
            'forward_velocity': self.get_robot_forward_velocity(),
            'pitch_angle': self.get_robot_pitch_angle(),
            'pitch_angular_velocity': self.get_robot_pitch_angular_velocity(),
            'control_force': self.get_forward_control_force(),
            'time': self.forward_controller.t
        }
        
    def reset(self):
        """重置控制器到初始状态 - 倒立摆平衡点（竖直状态）"""
        # 保持原来的物理模型：θ=π为竖直平衡点
        self.forward_controller.x[0] = 0.0      # cart position = 0
        self.forward_controller.x[1] = np.pi    # pole angle = π (竖直向上平衡点)
        self.forward_controller.x[2] = 0.0      # cart velocity = 0
        self.forward_controller.x[3] = 0.0      # pole angular velocity = 0
        self.forward_controller.U.fill(0)
        self.forward_controller.t = 0
        self.desired_forward_velocity = 0.0


if __name__ == "__main__":
    print("🤖 Testing Inverted Pendulum Controllers")
    print("=" * 50)
    
    # Test parameters
    params = IPC3DParams()
    print(f"Parameters: cart_mass={params.mass_cart}kg, pole_mass={params.mass_pole}kg")
    print(f"           pole_length={params.pole_length}m, max_force={params.max_force}N")
    print(f"           control_mode={'velocity' if params.control_mode == 1 else 'position'}")
    print()
    
    # Test 1: IPC_Plane 2D平面控制器 - 30秒仿真
    print("🚗 Testing IPC_Plane (2D Planar Controller) - 30 Second Simulation")
    print("-" * 60)
    
    plane_controller = IPC_Plane(params)
    
    # 设置期望前向速度 - 测试修复后的跟踪性能
    target_velocity = 1.0  # m/s
    plane_controller.set_desired_forward_velocity(target_velocity)
    print(f"🎯 Target forward velocity: {target_velocity} m/s")
    print(f"📊 Mass configuration: Cart={params.mass_cart}kg, Pole={params.mass_pole}kg")
    print(f"⚙️  Control parameters: Inertia={params.inertia}kg⋅m², Damping={params.damping}")
    
    print("\n30-Second 2D Planar simulation results:")
    print("Time  Forward_Pos  Forward_Vel  Pitch_Angle  Pitch_AngVel  Control_Force")
    print("-" * 75)
    
    # 30秒仿真 = 30 * 40 = 1200步
    total_steps = int(30.0 / params.dt)  # 30秒
    
    for i in range(total_steps):
        plane_controller.step()
        
        # 每2秒输出一次结果
        if i % (int(2.0 / params.dt)) == 0:
            robot_state = plane_controller.get_robot_state()
            print(f"{robot_state['time']:4.1f}  "
                  f"{robot_state['forward_position']:10.3f}  "
                  f"{robot_state['forward_velocity']:10.3f}  "
                  f"{robot_state['pitch_angle']:10.3f}  "
                  f"{robot_state['pitch_angular_velocity']:11.3f}  "
                  f"{robot_state['control_force']:12.1f}")
    
    # 最终状态分析
    final_state = plane_controller.get_robot_state()
    print(f"\n📈 Final Results after 30 seconds:")
    print(f"   Forward Position: {final_state['forward_position']:.3f} m")
    print(f"   Forward Velocity: {final_state['forward_velocity']:.3f} m/s (Target: {target_velocity} m/s)")
    print(f"   Pitch Angle: {final_state['pitch_angle']:.3f} rad ({np.degrees(final_state['pitch_angle']):.1f}°)")
    print(f"   Control Force: {final_state['control_force']:.1f} N")
    
    # 评估跟踪性能
    velocity_error = abs(final_state['forward_velocity'] - target_velocity)
    if velocity_error < 0.1:
        print(f"✅ Excellent velocity tracking! Error: {velocity_error:.3f} m/s")
    elif velocity_error < 0.5:
        print(f"✔️  Good velocity tracking. Error: {velocity_error:.3f} m/s")
    else:
        print(f"⚠️  Poor velocity tracking. Error: {velocity_error:.3f} m/s")
    
    print("✅ IPC_Plane 30-second test completed!")
    print()
    