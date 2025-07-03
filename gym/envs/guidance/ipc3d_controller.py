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
    
    # Physical parameters
    mass_cart: float = 42.0      # Cart mass (kg) - robot body mass
    mass_pole: float = 5.0       # Pole mass (kg) - distributed mass
    pole_length: float = 0.62    # Pole length (m) - COM height
    inertia: float = 0.1         # Pole moment of inertia (kg⋅m²)
    gravity: float = 9.81        # Gravity (m/s²)
    damping: float = 0.1         # System damping
    
    # Control parameters
    dt: float = 0.02             # Time step (s)
    max_force: float = 500.0     # Maximum control force (N)
    
    # LQR weights
    q_position: float = 10.0     # Position tracking weight
    q_velocity: float = 1.0      # Velocity tracking weight
    r_control: float = 0.1       # Control effort weight
    
    # Control mode: 0=position, 1=velocity
    control_mode: int = 1        # Default to velocity control


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
        # Convert second-order system to first-order: [x; dx]
        # dx = [dx; ddx] = [0 I; -D^-1*G -D^-1*C] * [x; dx] + [0; D^-1*H] * u
        
        # Debug mass matrix (disabled)
        # D_det = np.linalg.det(self.D)
        # D_cond = np.linalg.cond(self.D)
        # if D_det == 0 or D_cond > 1e12:
        #     print(f"⚠️ Mass matrix D is singular or ill-conditioned!")
        #     print(f"   D determinant: {D_det:.2e}")
        #     print(f"   D condition: {D_cond:.2e}")
        #     print(f"   D matrix:\n{self.D}")
        
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
        if np.linalg.cond(self.A) > 1e12:
            epsilon = 1e-6
            self.A[0, 0] = -epsilon  # Small position damping
            # print(f"🔧 Adding position regularization: A[0,0] = {-epsilon}")
        
        # Control matrix B = [0; D^-1*H]
        self.B[:self.dim, :] = 0
        self.B[self.dim:, :] = D_inv @ self.H
        
        # Debug A matrix for the first call (disabled)
        # if np.allclose(x, 0):  # Initial state
        #     print(f"🔍 A matrix construction:")
        #     print(f"   A matrix shape: {self.A.shape}")
        #     print(f"   A matrix:\n{self.A}")
        #     print(f"   A condition: {np.linalg.cond(self.A):.2e}")
        #     print(f"   A rank: {np.linalg.matrix_rank(self.A)}")
        #     print(f"   A eigenvalues: {np.linalg.eigvals(self.A)}")
        
    def update_sdre(self, theta: np.ndarray, dtheta: np.ndarray):
        """Update SDRE matrices - must be implemented by subclasses."""
        raise NotImplementedError
        
    def one_step(self, x: np.ndarray, u: np.ndarray, dt: float, 
                 max_force: float, Q: np.ndarray, R: np.ndarray, 
                 K: np.ndarray, xd: np.ndarray, tau: Optional[float] = None) -> float:
        """Perform one integration step with LQR control."""
        
        # Apply control limits
        u_limited = np.clip(u, -max_force, max_force)
        
        # Update SDRE linearization
        dim = self.dim
        self.update_sdre(x[:dim], x[dim:])
        
        # Calculate control force
        error = x - xd
        u_control = K @ error
        u_control = np.clip(u_control, -max_force, max_force)
        
        # System dynamics: dx = A*x + B*u
        dx = self.A @ x.reshape(-1, 1) + self.B @ u_control.reshape(-1, 1)
        
        # Euler integration
        x[:] = x + dx.flatten() * dt
        
        # Update control input
        u[:] = u_control
        
        # Return acceleration for compatibility
        return dx[dim:].flatten() if len(dx) > dim else 0.0


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
        
        # Debug D matrix on first few calls (disabled)
        # if abs(theta) < 0.01 and abs(dtheta) < 0.01:  # Initial state
        #     print(f"🔍 Initial D matrix calculation:")
        #     print(f"   D matrix:\n{self.D}")
        #     print(f"   D determinant: {np.linalg.det(self.D):.6f}")
        
        # Damping matrix C  
        self.C[0, 0] = b
        self.C[0, 1] = m * l * dtheta * sin_theta
        self.C[1, 0] = 0
        self.C[1, 1] = 0
        
        # Gravitational forces G
        self.G[0, 0] = 0
        self.G[0, 1] = 0  
        self.G[1, 0] = 0
        
        # Handle singularity at theta=0 by adding small regularization
        gravity_term = -m * g * l * sin_theta
        if abs(theta) < 1e-4:  # Near vertical (theta ≈ 0)
            # Use small angle approximation: sin(theta) ≈ theta, but ensure non-zero
            effective_theta = np.sign(theta) * max(abs(theta), 1e-4) if theta != 0 else 1e-4
            gravity_term = -m * g * l * effective_theta
            # print(f"🔧 Near-vertical regularization: theta={theta:.6f} → {effective_theta:.6f}")
        
        self.G[1, 1] = gravity_term
        
        # Update linearized system
        self._update_sdre(x)


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
        self.ipc = IPCController(M, m, b, I, g, l)
        self.l = l
        self.dt = dt
        self.t = 0.0
        
        # Initialize state: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
        self.x = np.zeros(4)
        self.xd = np.zeros(4)  # Desired state
        self.U = np.zeros(1)   # Control input
        
        # LQR matrices
        self.Q = np.zeros((4, 4))
        self.R = np.eye(1)
        
        # Set control mode and gains
        qd = qd if qd is not None else q
        self.mode = -1  # Initialize to invalid mode to force update
        self.K = np.zeros((1, 4))  # Initialize K matrix
        self.set_lqr_gains(1, q, qd)  # Set to velocity control by default
        
        self.max_force = 500.0
        
    def set_lqr_gains(self, mode: int, q: float, qd: float):
        """Set LQR gains for position (mode=0) or velocity (mode=1) control."""
        if self.mode != mode:
            self.Q.fill(0)
            if mode == 0:  # Position control
                self.Q[0, 0] = q    # Cart position
                self.Q[1, 1] = 0    # Pole angle  
                self.Q[2, 2] = qd   # Cart velocity
                self.Q[3, 3] = 0    # Pole angular velocity
            else:  # Velocity control  
                self.Q[0, 0] = 0    # Cart position
                self.Q[1, 1] = q    # Pole angle
                self.Q[2, 2] = qd   # Cart velocity  
                self.Q[3, 3] = 0    # Pole angular velocity
            
            self.mode = mode
            
            # Update SDRE and compute LQR gain
            self.ipc.update_sdre(self.x[:2], self.x[2:])
            self.K = self._solve_lqr(self.ipc.A, self.ipc.B, self.Q, self.R)
    
    def _solve_lqr(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve LQR problem to find optimal gain matrix K."""
        try:
            P = solve_continuous_are(A, B, Q, R)
            R_inv = np.linalg.inv(R)
            K = R_inv @ B.T @ P
            return K
        except Exception as e:
            print(f"⚠️ LQR solve failed: {e}")
            print(f"   A shape: {A.shape}, condition: {np.linalg.cond(A):.2e}")
            print(f"   B shape: {B.shape}, B norm: {np.linalg.norm(B):.2e}")
            print(f"   Q shape: {Q.shape}, Q trace: {np.trace(Q):.2e}")
            print(f"   R shape: {R.shape}, R trace: {np.trace(R):.2e}")
            return np.zeros((self.ipc.cdim, 2 * self.ipc.dim))
            
    def set_param(self, speed: float):
        """Set desired velocity or position."""
        self.xd.fill(0)
        if self.mode == 0:  # Position control
            self.xd[0] = speed  # Desired cart position
            # print(f"🎯 Target position set: {speed:.3f} m")
        else:  # Velocity control
            self.xd[2] = speed  # Desired cart velocity
            # print(f"🎯 Target velocity set: {speed:.3f} m/s")
            
    def set_maximum_force(self, force: float):
        """Set maximum control force."""
        self.max_force = force
        
    def one_step(self, tau: Optional[float] = None) -> float:
        """Perform one simulation step with SDRE control."""
        # Debug info before control step (disabled)
        # if self.t < 0.1:  # Only print for first few steps to debug
        #     print(f"🔄 Step t={self.t:.3f}s:")
        #     print(f"   State: pos={self.x[0]:.3f}, ang={self.x[1]:.3f}, vel={self.x[2]:.3f}, avel={self.x[3]:.3f}")
        #     print(f"   Target: {self.xd}")
        #     print(f"   K shape: {self.K.shape}, K norm: {np.linalg.norm(self.K):.3f}")
        
        ddtheta = self.ipc.one_step(
            self.x, self.U, self.dt, self.max_force,
            self.Q, self.R, self.K, self.xd, tau
        )
        
        # if self.t < 0.1:  # Debug after control step (disabled)
        #     print(f"   Control force: {self.U[0]:.3f} N")
        #     if isinstance(ddtheta, (float, int)):
        #         print(f"   Angular accel: {ddtheta:.3f} rad/s²")
        #     else:
        #         print(f"   Angular accel: {ddtheta}")
        
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


class IPC3D:
    """3D Inverted Pendulum Controller with dual-axis control."""
    
    def __init__(self, params: IPC3DParams):
        """Initialize 3D IPC controller."""
        self.params = params
        
        # Create independent controllers for X and Z axes
        self.pend_x = IPCView(
            params.mass_cart, params.mass_pole, params.damping, 
            params.inertia, params.gravity, params.pole_length,
            params.dt, params.q_velocity, params.q_position
        )
        
        self.pend_z = IPCView(
            params.mass_cart, params.mass_pole, params.damping,
            params.inertia, params.gravity, params.pole_length, 
            params.dt, params.q_velocity, params.q_position
        )
        
        # Set control mode
        self.pend_x.set_lqr_gains(params.control_mode, params.q_velocity, params.q_position)
        self.pend_z.set_lqr_gains(params.control_mode, params.q_velocity, params.q_position)
        
        # Set force limits
        self.pend_x.set_maximum_force(params.max_force)
        self.pend_z.set_maximum_force(params.max_force)
        
        # Desired velocity
        self.desired_vel = np.zeros(3)  # [x, y, z]
        
    def set_desired_velocity(self, vel_x: float, vel_z: float):
        """Set desired velocity for X and Z axes."""
        self.desired_vel[0] = vel_x
        self.desired_vel[2] = vel_z
        
        self.pend_x.set_param(vel_x)
        self.pend_z.set_param(vel_z)
        
    def step(self, n_times: int = 1, ef: Optional[np.ndarray] = None, 
             ef_height: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform simulation steps.
        
        Args:
            n_times: Number of simulation steps
            ef: End-effector position (not used in basic implementation)
            ef_height: End-effector height (not used in basic implementation)
            
        Returns:
            Dictionary with controller states and outputs
        """
        for _ in range(n_times):
            # Independent control for X and Z axes
            self.pend_x.one_step()
            self.pend_z.one_step()
            
        return {
            'x_state': self.pend_x.x.copy(),
            'z_state': self.pend_z.x.copy(), 
            'x_control': self.pend_x.U.copy(),
            'z_control': self.pend_z.U.copy(),
            'time': self.pend_x.t
        }
        
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current controller state."""
        return {
            'x_cart_pos': self.pend_x.x[0],
            'x_pole_angle': self.pend_x.x[1], 
            'x_cart_vel': self.pend_x.x[2],
            'x_pole_vel': self.pend_x.x[3],
            'z_cart_pos': self.pend_z.x[0],
            'z_pole_angle': self.pend_z.x[1],
            'z_cart_vel': self.pend_z.x[2], 
            'z_pole_vel': self.pend_z.x[3],
            'control_x': self.pend_x.U[0],
            'control_z': self.pend_z.U[0]
        }
        
    def reset(self):
        """Reset controller to initial state."""
        self.pend_x.x.fill(0)
        self.pend_z.x.fill(0)
        self.pend_x.U.fill(0)
        self.pend_z.U.fill(0)
        self.pend_x.t = 0
        self.pend_z.t = 0
        
    def get_control_forces(self) -> Tuple[float, float]:
        """Get current control forces for X and Z axes."""
        return float(self.pend_x.U[0]), float(self.pend_z.U[0])


def create_ipc3d_from_robot_spec(robot_spec, guidance_config) -> IPC3D:
    """Create IPC3D controller from robot specification and guidance config."""
    
    # Extract physical parameters from robot spec
    mass = robot_spec.physical_params.get('mass', 42.0)
    com_height = robot_spec.physical_params.get('base_height', 0.62)
    
    # Create parameters
    params = IPC3DParams(
        mass_cart=mass * 0.9,  # 90% of mass as cart
        mass_pole=mass * 0.1,  # 10% of mass as pole
        pole_length=com_height,
        inertia=guidance_config.ipc3d_inertia,
        gravity=9.81,
        damping=guidance_config.ipc3d_damping,
        dt=guidance_config.dt,
        max_force=guidance_config.ipc3d_max_force,
        q_position=guidance_config.lqr_q_matrix[0],
        q_velocity=guidance_config.lqr_q_matrix[1], 
        r_control=guidance_config.lqr_r_matrix[0],
        control_mode=1 if guidance_config.ipc3d_control_mode == 'velocity' else 0
    )
    
    return IPC3D(params)


if __name__ == "__main__":
    # Test IPC3D controller
    params = IPC3DParams()
    controller = IPC3D(params)
    
    print("🚗 Testing IPC3D Controller")
    print("=" * 40)
    print(f"Parameters: cart_mass={params.mass_cart}kg, pole_mass={params.mass_pole}kg")
    print(f"           pole_length={params.pole_length}m, max_force={params.max_force}N")
    print(f"           control_mode={'velocity' if params.control_mode == 1 else 'position'}")
    
    # Set desired velocity
    controller.set_desired_velocity(1.0, 0.5)  # 1 m/s in X, 0.5 m/s in Z
    print(f"Target set: X={1.0} m/s, Z={0.5} m/s")
    
    # Run simulation
    print("\nRunning simulation:")
    for i in range(100):
        result = controller.step()
        
        if i % 20 == 0:
            state = controller.get_state()
            fx, fz = controller.get_control_forces()
            print(f"Step {i:3d}: X_vel={state['x_cart_vel']:.3f}, Z_vel={state['z_cart_vel']:.3f}")
            print(f"         Control: Fx={fx:.1f}N, Fz={fz:.1f}N")
    
    print("✅ IPC3D test completed!")