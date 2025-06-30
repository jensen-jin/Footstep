"""
Robot specifications for push_robot_verify branch robots.
包含HI-12, HI-CL-12, PI-CL-12机器人规格定义
"""

from ..registry import RobotRegistry, RobotSpec

def create_hi12_spec():
    """Create HI-12DOF robot specification"""
    return RobotSpec(
        name="hi_12",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_12dof_250108_4/urdf/hi_12dof_250108_4_rl.urdf",
        num_actuators=12,
        default_joint_angles={
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.4,
            "r_calf_joint": -0.8,
            "r_ankle_pitch_joint": 0.4,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.4,
            "l_calf_joint": -0.8,
            "l_ankle_pitch_joint": 0.4,
            "l_ankle_roll_joint": 0.0,
        },
        joint_limits={
            "r_hip_pitch_joint": [-1.25, 1.75],
            "r_hip_roll_joint": [-0.5, 0.12],
            "r_thigh_joint": [-0.6, 0.3],
            "r_calf_joint": [-0.65, 1.65],
            "r_ankle_pitch_joint": [-0.5, 1.3],
            "r_ankle_roll_joint": [-0.15, 0.15],
            "l_hip_pitch_joint": [-1.25, 1.75],
            "l_hip_roll_joint": [-0.12, 0.5],
            "l_thigh_joint": [-0.3, 0.6],
            "l_calf_joint": [-0.65, 1.65],
            "l_ankle_pitch_joint": [-0.5, 1.3],
            "l_ankle_roll_joint": [-0.15, 0.15],
        },
        control_config={
            "stiffness_range": [40.0, 60.0],
            "damping_range": [0.4, 4.0],
            "max_torque": 80.0,
            "control_type": "torque"
        },
        physical_params={
            "base_height": 0.5596,
            "total_mass": 6.0,
            "foot_separation": 0.199,
            "added_mass_range": [-2.2, 2.2],
            "episode_length": 17.0,
            "max_velocity": 3.0
        },
        description="12-DOF humanoid robot with enhanced motor parameters and extended episode training",
        category="humanoid",
        tags=["bipedal", "12dof", "enhanced_control", "extended_training"]
    )

def create_hi_cl12_spec():
    """Create HI-CL-12DOF robot specification"""
    return RobotSpec(
        name="hi_cl_12", 
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_cl_23_240925/urdf/hi_cl_23_240925_rl.urdf",
        num_actuators=12,
        default_joint_angles={
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.4,
            "r_calf_joint": -0.8,
            "r_ankle_pitch_joint": 0.4,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.4,
            "l_calf_joint": -0.8,
            "l_ankle_pitch_joint": 0.4,
            "l_ankle_roll_joint": 0.0,
        },
        joint_limits={
            "r_hip_pitch_joint": [-1.25, 1.75],
            "r_hip_roll_joint": [-0.5, 0.12],
            "r_thigh_joint": [-0.6, 0.3],
            "r_calf_joint": [-0.65, 1.65],
            "r_ankle_pitch_joint": [-0.5, 1.3],
            "r_ankle_roll_joint": [-0.15, 0.15],
            "l_hip_pitch_joint": [-1.25, 1.75],
            "l_hip_roll_joint": [-0.12, 0.5],
            "l_thigh_joint": [-0.3, 0.6],
            "l_calf_joint": [-0.65, 1.65],
            "l_ankle_pitch_joint": [-0.5, 1.3],
            "l_ankle_roll_joint": [-0.15, 0.15],
        },
        control_config={
            "stiffness": 40.0,  # Uniform stiffness
            "damping": 0.4,     # Uniform damping  
            "max_torque": 80.0,
            "control_type": "torque",
            "curriculum_learning": True
        },
        physical_params={
            "base_height": 0.68,  # Tallest robot
            "total_mass": 7.0,
            "foot_separation": 0.25,
            "added_mass_range": [-1.2, 1.2],
            "episode_length": 5.0,
            "max_velocity": 2.5  # High-speed locomotion
        },
        description="12-DOF humanoid robot optimized for high-velocity locomotion with curriculum learning",
        category="humanoid",
        tags=["bipedal", "12dof", "curriculum_learning", "high_velocity"]
    )

def create_pi_cl12_spec():
    """Create PI-CL-12DOF robot specification"""  
    return RobotSpec(
        name="pi_cl_12",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/urdf/pi_12dof_release_v1_rl.urdf", 
        num_actuators=12,
        default_joint_angles={
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.4,
            "r_calf_joint": -0.8,
            "r_ankle_pitch_joint": 0.4,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.4,
            "l_calf_joint": -0.8,
            "l_ankle_pitch_joint": 0.4,
            "l_ankle_roll_joint": 0.0,
        },
        joint_limits={
            "r_hip_pitch_joint": [-1.25, 1.75],
            "r_hip_roll_joint": [-0.5, 0.12],
            "r_thigh_joint": [-0.6, 0.3],
            "r_calf_joint": [-0.65, 1.65],
            "r_ankle_pitch_joint": [-0.5, 1.3],
            "r_ankle_roll_joint": [-0.15, 0.15],
            "l_hip_pitch_joint": [-1.25, 1.75],
            "l_hip_roll_joint": [-0.12, 0.5],
            "l_thigh_joint": [-0.3, 0.6],
            "l_calf_joint": [-0.65, 1.65],
            "l_ankle_pitch_joint": [-0.5, 1.3],
            "l_ankle_roll_joint": [-0.15, 0.15],
        },
        control_config={
            "stiffness": 40.0,   # Uniform stiffness
            "damping": 1.0,      # Higher damping for stability
            "max_torque": 60.0,  # Lower torque for precision
            "control_type": "torque",
            "precision_mode": True
        },
        physical_params={
            "base_height": 0.3453,  # Most compact robot  
            "total_mass": 3.5,
            "foot_separation": 0.14,
            "added_mass_range": [-1.0, 1.0],
            "episode_length": 5.0,
            "max_velocity": 0.3  # Conservative for precision
        },
        description="Compact 12-DOF humanoid robot optimized for precision control and stability",
        category="humanoid", 
        tags=["bipedal", "12dof", "compact", "precision_control", "conservative"]
    )

def register_push_robots():
    """Register all push_robot_verify branch robots"""
    print("🤖 Registering push_robot_verify branch robots...")
    
    # Register HI-12DOF robot
    hi12_spec = create_hi12_spec()
    RobotRegistry.register(hi12_spec)
    
    # Register HI-CL-12DOF robot  
    hi_cl12_spec = create_hi_cl12_spec()
    RobotRegistry.register(hi_cl12_spec)
    
    # Register PI-CL-12DOF robot
    pi_cl12_spec = create_pi_cl12_spec() 
    RobotRegistry.register(pi_cl12_spec)
    
    print(f"✅ Successfully registered {len([hi12_spec, hi_cl12_spec, pi_cl12_spec])} push_robot_verify robots")

__all__ = [
    'register_push_robots',
    'create_hi12_spec',
    'create_hi_cl12_spec', 
    'create_pi_cl12_spec'
]