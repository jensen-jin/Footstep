"""
Humanoid robot specifications.

This module contains specifications for various humanoid robots,
extracted from existing configuration files.
"""

from ..registry import RobotRegistry, RobotSpec


def register_humanoid_robots():
    """Register all humanoid robots."""
    
    # MIT Humanoid with Fixed Arms (SF Update)
    RobotRegistry.register(RobotSpec(
        name="mit_humanoid_fixed_arms",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/humanoid/urdf/humanoid_fixed_arms_sf_update.urdf",
        num_actuators=10,
        
        default_joint_angles={
            '01_right_hip_yaw': 0.,
            '02_right_hip_abad': 0.1,
            '03_right_hip_pitch': -0.667751,
            '04_right_knee': 1.4087,
            '05_right_ankle': -0.708876,
            '06_left_hip_yaw': 0.,
            '07_left_hip_abad': 0.1,
            '08_left_hip_pitch': -0.667751,
            '09_left_knee': 1.4087,
            '10_left_ankle': -0.708876,
        },
        
        joint_limits={
            '01_right_hip_yaw': [-0.1, 0.1],
            '02_right_hip_abad': [-0.1, 0.3],
            '03_right_hip_pitch': [-0.8, -0.4],
            '04_right_knee': [1.3, 1.5],
            '05_right_ankle': [-0.9, -0.5],
            '06_left_hip_yaw': [-0.1, 0.1],
            '07_left_hip_abad': [-0.1, 0.3],
            '08_left_hip_pitch': [-0.8, -0.4],
            '09_left_knee': [1.3, 1.5],
            '10_left_ankle': [-0.9, -0.5],
        },
        
        control_config={
            'stiffness': {
                '01_right_hip_yaw': 30.,
                '02_right_hip_abad': 30.,
                '03_right_hip_pitch': 30.,
                '04_right_knee': 30.,
                '05_right_ankle': 30.,
                '06_left_hip_yaw': 30.,
                '07_left_hip_abad': 30.,
                '08_left_hip_pitch': 30.,
                '09_left_knee': 30.,
                '10_left_ankle': 30.,
            },
            'damping': {
                '01_right_hip_yaw': 1.,
                '02_right_hip_abad': 1.,
                '03_right_hip_pitch': 1.,
                '04_right_knee': 1.,
                '05_right_ankle': 1.,
                '06_left_hip_yaw': 1.,
                '07_left_hip_abad': 1.,
                '08_left_hip_pitch': 1.,
                '09_left_knee': 1.,
                '10_left_ankle': 1.
            },
            'actuation_scale': 1.0,
            'decimation': 10
        },
        
        physical_params={
            'base_height': 0.62,
            'mass': 42.0,  # Estimated mass in kg
            'end_effectors': ['right_foot', 'left_foot'],
            'foot_name': 'foot',
            'rotor_inertia': [
                0.01188,    # RIGHT LEG
                0.01188,
                0.01980,
                0.07920,
                0.04752,
                0.01188,    # LEFT LEG
                0.01188,
                0.01980,
                0.07920,
                0.04752,
            ],
            'guidance_params': {
                'limp_com_height': 0.62,
                'limp_step_length': 0.5,
                'limp_step_width': 0.3,
                'limp_step_time': 0.34,
                'ipc3d_pole_length': 0.62,  # COM height as pole length
                'ipc3d_mass_cart': 42.0,    # Robot mass
                'ipc3d_mass_pole': 5.0,     # Estimated leg mass
            }
        },
        
        category="humanoid",
        description="MIT humanoid robot with fixed arms - optimized for walking",
        tags=["mit", "fixed_arms", "walking", "bipedal"]
    ))
    
    # MIT Humanoid Full Body
    RobotRegistry.register(RobotSpec(
        name="mit_humanoid_full",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/rom/urdf/humanoid_full.urdf",
        num_actuators=16,  # Including arms
        
        # Default configuration (needs to be filled from actual URDF)
        default_joint_angles={
            # Legs (same as fixed arms)
            '01_right_hip_yaw': 0.,
            '02_right_hip_abad': 0.1,
            '03_right_hip_pitch': -0.667751,
            '04_right_knee': 1.4087,
            '05_right_ankle': -0.708876,
            '06_left_hip_yaw': 0.,
            '07_left_hip_abad': 0.1,
            '08_left_hip_pitch': -0.667751,
            '09_left_knee': 1.4087,
            '10_left_ankle': -0.708876,
            # Arms (estimated)
            '11_right_shoulder_pitch': 0.0,
            '12_right_shoulder_roll': 0.0,
            '13_right_elbow': 0.0,
            '14_left_shoulder_pitch': 0.0,
            '15_left_shoulder_roll': 0.0,
            '16_left_elbow': 0.0,
        },
        
        joint_limits={
            # Legs (same as fixed arms)
            '01_right_hip_yaw': [-0.1, 0.1],
            '02_right_hip_abad': [-0.1, 0.3],
            '03_right_hip_pitch': [-0.8, -0.4],
            '04_right_knee': [1.3, 1.5],
            '05_right_ankle': [-0.9, -0.5],
            '06_left_hip_yaw': [-0.1, 0.1],
            '07_left_hip_abad': [-0.1, 0.3],
            '08_left_hip_pitch': [-0.8, -0.4],
            '09_left_knee': [1.3, 1.5],
            '10_left_ankle': [-0.9, -0.5],
            # Arms (estimated limits)
            '11_right_shoulder_pitch': [-1.5, 1.5],
            '12_right_shoulder_roll': [-1.0, 1.0],
            '13_right_elbow': [-2.0, 0.1],
            '14_left_shoulder_pitch': [-1.5, 1.5],
            '15_left_shoulder_roll': [-1.0, 1.0],
            '16_left_elbow': [-0.1, 2.0],
        },
        
        control_config={
            'stiffness': {
                # Legs - same as fixed arms
                '01_right_hip_yaw': 30., '02_right_hip_abad': 30., '03_right_hip_pitch': 30.,
                '04_right_knee': 30., '05_right_ankle': 30.,
                '06_left_hip_yaw': 30., '07_left_hip_abad': 30., '08_left_hip_pitch': 30.,
                '09_left_knee': 30., '10_left_ankle': 30.,
                # Arms - lower stiffness
                '11_right_shoulder_pitch': 15., '12_right_shoulder_roll': 15., '13_right_elbow': 15.,
                '14_left_shoulder_pitch': 15., '15_left_shoulder_roll': 15., '16_left_elbow': 15.,
            },
            'damping': {
                # Legs - same as fixed arms
                '01_right_hip_yaw': 1., '02_right_hip_abad': 1., '03_right_hip_pitch': 1.,
                '04_right_knee': 1., '05_right_ankle': 1.,
                '06_left_hip_yaw': 1., '07_left_hip_abad': 1., '08_left_hip_pitch': 1.,
                '09_left_knee': 1., '10_left_ankle': 1.,
                # Arms - lower damping
                '11_right_shoulder_pitch': 0.5, '12_right_shoulder_roll': 0.5, '13_right_elbow': 0.5,
                '14_left_shoulder_pitch': 0.5, '15_left_shoulder_roll': 0.5, '16_left_elbow': 0.5,
            },
            'actuation_scale': 1.0,
            'decimation': 10
        },
        
        physical_params={
            'base_height': 0.77,  # Slightly taller with arms
            'mass': 50.0,  # Heavier with arms
            'end_effectors': ['right_foot', 'left_foot', 'right_hand', 'left_hand'],
            'foot_name': 'foot',
            'guidance_params': {
                'limp_com_height': 0.77,
                'limp_step_length': 0.5,
                'limp_step_width': 0.3,
                'limp_step_time': 0.34,
                'ipc3d_pole_length': 0.77,
                'ipc3d_mass_cart': 50.0,
                'ipc3d_mass_pole': 7.0,
            }
        },
        
        category="humanoid",
        description="MIT humanoid robot with full body including arms",
        tags=["mit", "full_body", "arms", "bipedal", "manipulation"]
    ))
    
    # MIT Humanoid Fixed Arms (base version)
    RobotRegistry.register(RobotSpec(
        name="mit_humanoid_fixed_arms_base",
        urdf_path="{LEGGED_GYM_ROOT_DIR}/resources/robots/rom/urdf/humanoid_fixed_arms.urdf",
        num_actuators=10,
        
        # Use same config as updated version but different URDF
        default_joint_angles={
            '01_right_hip_yaw': 0.,
            '02_right_hip_abad': 0.,
            '03_right_hip_pitch': -0.2,
            '04_right_knee': 0.6,
            '05_right_ankle': 0.,
            '06_left_hip_yaw': 0.,
            '07_left_hip_abad': 0.,
            '08_left_hip_pitch': -0.2,
            '09_left_knee': 0.6,
            '10_left_ankle': 0.,
        },
        
        joint_limits={
            '01_right_hip_yaw': [-0.1, 0.1],
            '02_right_hip_abad': [-0.1, 0.1],
            '03_right_hip_pitch': [-0.2, 0.2],
            '04_right_knee': [0.6, 0.7],
            '05_right_ankle': [-0.3, 0.0],
            '06_left_hip_yaw': [-0.1, 0.1],
            '07_left_hip_abad': [-0.1, 0.1],
            '08_left_hip_pitch': [-0.2, 0.2],
            '09_left_knee': [0.6, 0.7],
            '10_left_ankle': [-0.3, 0.0],
        },
        
        control_config={
            'stiffness': {f'0{i+1}_right_hip_yaw' if i == 0 else f'0{i+1}_right_hip_abad' if i == 1 else f'0{i+1}_right_hip_pitch' if i == 2 else f'0{i+1}_right_knee' if i == 3 else f'0{i+1}_right_ankle' if i == 4 else f'0{i+1}_left_hip_yaw' if i == 5 else f'0{i+1}_left_hip_abad' if i == 6 else f'0{i+1}_left_hip_pitch' if i == 7 else f'0{i+1}_left_knee' if i == 8 else f'{i+1}_left_ankle': 30. for i in range(10)},
            'damping': {f'0{i+1}_right_hip_yaw' if i == 0 else f'0{i+1}_right_hip_abad' if i == 1 else f'0{i+1}_right_hip_pitch' if i == 2 else f'0{i+1}_right_knee' if i == 3 else f'0{i+1}_right_ankle' if i == 4 else f'0{i+1}_left_hip_yaw' if i == 5 else f'0{i+1}_left_hip_abad' if i == 6 else f'0{i+1}_left_hip_pitch' if i == 7 else f'0{i+1}_left_knee' if i == 8 else f'{i+1}_left_ankle': 1. for i in range(10)},
            'actuation_scale': 1.0,
            'decimation': 10
        },
        
        physical_params={
            'base_height': 0.77,
            'mass': 42.0,
            'end_effectors': ['right_foot', 'left_foot'],
            'foot_name': 'foot',
            'guidance_params': {
                'limp_com_height': 0.77,
                'limp_step_length': 0.5,
                'limp_step_width': 0.3,
                'limp_step_time': 0.34,
                'ipc3d_pole_length': 0.77,
                'ipc3d_mass_cart': 42.0,
                'ipc3d_mass_pole': 5.0,
            }
        },
        
        category="humanoid",
        description="MIT humanoid robot with fixed arms - base version",
        tags=["mit", "fixed_arms", "base", "bipedal"]
    ))