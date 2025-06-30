"""
Simple robot specifications (cartpole, pendulum, etc.).
"""

from ..registry import RobotRegistry, RobotSpec


def register_simple_robots():
    """Register simple control robots."""
    
    # Classic CartPole
    RobotRegistry.register(RobotSpec(
        name="cartpole",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/urdf/cartpole.urdf",
        num_actuators=1,
        
        default_joint_angles={
            'cart_joint': 0.0,
        },
        
        joint_limits={
            'cart_joint': [-2.4, 2.4],  # Cart position limits
        },
        
        control_config={
            'stiffness': {'cart_joint': 0.0},  # Force control
            'damping': {'cart_joint': 0.1},
            'actuation_scale': 10.0,
            'decimation': 1
        },
        
        physical_params={
            'base_height': 0.0,  # Cart on ground
            'mass': 1.1,  # Cart + pole mass
            'guidance_params': {
                'ipc3d_mass_cart': 1.0,
                'ipc3d_mass_pole': 0.1,
                'ipc3d_pole_length': 0.5,
                'ipc3d_friction': 0.1,
                'ipc3d_control_mode': 'velocity'
            }
        },
        
        category="simple",
        description="Classic cart-pole system for control experiments",
        tags=["classic", "control", "inverted_pendulum", "1dof"]
    ))
    
    # Simple Pendulum
    RobotRegistry.register(RobotSpec(
        name="pendulum",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/pendulum/urdf/pendulum.urdf",
        num_actuators=1,
        
        default_joint_angles={
            'pendulum_joint': 0.0,
        },
        
        joint_limits={
            'pendulum_joint': [-3.14159, 3.14159],  # Full rotation
        },
        
        control_config={
            'stiffness': {'pendulum_joint': 0.0},  # Torque control
            'damping': {'pendulum_joint': 0.1},
            'actuation_scale': 2.0,
            'decimation': 1
        },
        
        physical_params={
            'base_height': 1.0,  # Pendulum pivot height
            'mass': 1.0,
            'guidance_params': {
                'limp_com_height': 1.0,
                'pendulum_length': 1.0,
                'pendulum_mass': 1.0,
                'gravity': 9.81
            }
        },
        
        category="simple",
        description="Simple pendulum for swing-up control",
        tags=["pendulum", "control", "swing_up", "1dof"]
    ))
    
    # Point Mass (for testing)
    RobotRegistry.register(RobotSpec(
        name="point_mass",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/rom/urdf/point_mass.urdf",
        num_actuators=3,  # 3DOF position control
        
        default_joint_angles={
            'mass_x': 0.0,
            'mass_y': 0.0,
            'mass_z': 1.0,
        },
        
        joint_limits={
            'mass_x': [-5.0, 5.0],
            'mass_y': [-5.0, 5.0],
            'mass_z': [0.1, 3.0],
        },
        
        control_config={
            'stiffness': {'mass_x': 1000.0, 'mass_y': 1000.0, 'mass_z': 1000.0},
            'damping': {'mass_x': 100.0, 'mass_y': 100.0, 'mass_z': 100.0},
            'actuation_scale': 1.0,
            'decimation': 1
        },
        
        physical_params={
            'base_height': 1.0,
            'mass': 1.0,
            'guidance_params': {
                'limp_com_height': 1.0,
                'point_mass': 1.0
            }
        },
        
        category="simple",
        description="Point mass for basic dynamics testing",
        tags=["testing", "point_mass", "3dof"]
    ))
    
    # SLIP model (Spring Loaded Inverted Pendulum)
    RobotRegistry.register(RobotSpec(
        name="slip",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/rom/urdf/slip.urdf",
        num_actuators=2,  # Leg angle and spring compression
        
        default_joint_angles={
            'leg_angle': 0.0,
            'leg_compression': 0.0,
        },
        
        joint_limits={
            'leg_angle': [-1.57, 1.57],  # ±90 degrees
            'leg_compression': [-0.2, 0.1],  # Spring compression range
        },
        
        control_config={
            'stiffness': {'leg_angle': 50.0, 'leg_compression': 1000.0},
            'damping': {'leg_angle': 5.0, 'leg_compression': 50.0},
            'actuation_scale': 1.0,
            'decimation': 1
        },
        
        physical_params={
            'base_height': 1.0,
            'mass': 80.0,  # Human-like mass
            'end_effectors': ['foot'],
            'guidance_params': {
                'limp_com_height': 1.0,
                'slip_leg_length': 1.0,
                'slip_spring_stiffness': 10000.0,
                'slip_mass': 80.0
            }
        },
        
        category="simple",
        description="Spring Loaded Inverted Pendulum for running dynamics",
        tags=["slip", "running", "spring", "biomechanics"]
    ))