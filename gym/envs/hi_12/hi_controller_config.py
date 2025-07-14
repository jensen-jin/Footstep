"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg


class HiControllerCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 8  # 优化：从17降为8，提高训练稳定性
        frame_stack = 1
        # 更新观测空间大小（增加了IPC3D观测）
        # ipc3d_desired_trajectory: 3
        # ipc3d_desired_velocity: 3  
        # ipc3d_trajectory_error: 1
        # 原有观测: 36 + 新增: 7 = 43
        num_single_obs = 43
        num_observations = frame_stack * num_single_obs

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = "plane"  # 'plane' 'heightfield' 'trimesh'
        measure_heights = False  # True, False
        measured_points_x_range = [-0.8, 0.8]
        measured_points_x_num_sample = 33
        measured_points_y_range = [-0.8, 0.8]
        measured_points_y_num_sample = 33
        selected = True  # True, False
        terrain_kwargs = {"type": "stepping_stones"}
        # terrain_kwargs = {'type': 'random_uniform'}
        # terrain_kwargs = {'type': 'gap'}
        # difficulty = 0.35 # For gap terrain
        # platform_size = 5.5 # For gap terrain
        difficulty = 5.0  # For rough terrain
        terrain_length = 18.0  # For rough terrain
        terrain_width = 18.0  # For rough terrain
        # terrain types: [pyramid_sloped, random_uniform, stairs down, stairs up, discrete obstacles, stepping_stones, gap, pit]
        terrain_proportions = [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        # reset_mode = 'reset_to_range' # 'reset_to_basic'
        reset_mode = "reset_to_basic"  # 'reset_to_basic'
        pos = [0.0, 0.0, 0.5596]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.68, 0.70],  # z
            [-torch.pi / 10, torch.pi / 10],  # roll
            [-torch.pi / 10, torch.pi / 10],  # pitch
            [-torch.pi / 10, torch.pi / 10],  # yaw
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-0.5, 0.5],  # x
            [-0.5, 0.5],  # y
            [-0.5, 0.5],  # z
            [-0.5, 0.5],  # roll
            [-0.5, 0.5],  # pitch
            [-0.5, 0.5],  # yaw
        ]

        default_joint_angles = {
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
        }

        dof_pos_range = {
            "r_hip_pitch_joint": [-1.0, 1.8],  #ok
            "r_hip_roll_joint": [-0.5, 0.12],  #ok
            "r_thigh_joint": [-0.6, 0.3],      #ok
            "r_calf_joint": [-0.8, 1.5],       #ok
            "r_ankle_pitch_joint": [-0.45, 1.15], #ok
            "r_ankle_roll_joint": [-0.15, 0.15], #ok
            "l_hip_pitch_joint": [-1.0, 1.8], #ok
            "l_hip_roll_joint": [-0.12, 0.5], #ok
            "l_thigh_joint": [-0.3, 0.6], #ok
            "l_calf_joint": [-0.8, 1.5], #ok
            "l_ankle_pitch_joint": [-0.45, 1.15], #ok
            "l_ankle_roll_joint": [-0.15, 0.15], #ok
        }

        dof_vel_range = {
            "r_hip_pitch_joint": [-0.1, 0.1],
            "r_hip_roll_joint": [-0.1, 0.1],
            "r_thigh_joint": [-0.1, 0.1],
            "r_calf_joint": [-0.1, 0.1],
            "r_ankle_pitch_joint": [-0.1, 0.1],
            "r_ankle_roll_joint": [-0.1, 0.1],
            "l_hip_pitch_joint": [-0.1, 0.1],
            "l_hip_roll_joint": [-0.1, 0.1],
            "l_thigh_joint": [-0.1, 0.1],
            "l_calf_joint": [-0.1, 0.1],
            "l_ankle_pitch_joint": [-0.1, 0.1],
            "l_ankle_roll_joint": [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            "r_hip_pitch_joint": 35.0,  # 优化：从40.0降为35.0，减少过度刚性
            "r_hip_roll_joint": 35.0,   # 优化：从40.0降为35.0
            "r_thigh_joint": 35.0,      # 优化：从40.0降为35.0
            "r_calf_joint": 50.0,       # 优化：从60.0降为50.0
            "r_ankle_pitch_joint": 35.0, # 优化：从40.0降为35.0
            "r_ankle_roll_joint": 12.0, # 优化：从10.0增为12.0，增加稳定性
            
            "l_hip_pitch_joint": 35.0,  # 优化：从40.0降为35.0
            "l_hip_roll_joint": 35.0,   # 优化：从40.0降为35.0
            "l_thigh_joint": 35.0,      # 优化：从40.0降为35.0
            "l_calf_joint": 50.0,       # 优化：从60.0降为50.0
            "l_ankle_pitch_joint": 35.0, # 优化：从40.0降为35.0
            "l_ankle_roll_joint": 12.0, # 优化：从10.0增为12.0，增加稳定性
        }
        damping = {
            "r_hip_pitch_joint": 2,
            "r_hip_roll_joint": 2,
            "r_thigh_joint": 2,
            "r_calf_joint": 4,
            "r_ankle_pitch_joint": 2,  #影响抬脚高度， 1.5抬脚明显矮于 2
            #"r_ankle_pitch_joint": 1.5,
            "r_ankle_roll_joint": 0.4,
            "l_hip_pitch_joint": 2,
            "l_hip_roll_joint": 2,
            "l_thigh_joint": 2,
            "l_calf_joint": 4,
            "l_ankle_pitch_joint": 2,
            #"l_ankle_pitch_joint": 1.5,
            "l_ankle_roll_joint": 0.4,
        }

        actuation_scale = 1.0
        exp_avg_decay = 0.3  # 优化：从0.5改为0.3，更平滑的控制
        decimation = 10

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 5.0  # 5.

        succeed_step_radius = 0.03
        succeed_step_angle = 10
        apex_height_percentage = 0.15

        sample_angle_offset = 20
        sample_radius_offset = 0.05

        dstep_length = 0.2
        dstep_width = 0.199
        
        # IPC3D命令转换配置
        use_ipc3d_commands = True
        command_conversion_mode = "global_to_relative"  # 全局到相对坐标转换
        
        # IPC3D特定命令参数
        ipc3d_command_scaling = {
            'forward_velocity_scale': 1.0,
            'angular_velocity_scale': 1.0,
            'lateral_velocity_scale': 0.5  # 减少侧向速度影响
        }

        class ranges(LeggedRobotCfg.commands.ranges):
            # TRAINING STEP COMMAND RANGES #
            sample_period = [25, 32]  # 优化：从[20,28]调整为[25,32]，略慢的步态
            dstep_width = [0.18, 0.22]  # 优化：从[0.199,0.2]扩展为[0.18,0.22]

            # IPC3D优化的命令范围 - 符合物理约束
            lin_vel_x = [-0.6, 1.0]  # 减少后退速度，符合IPC3D前向优化
            lin_vel_y = 0.2          # 基类期望标量，会转换为[-0.2, 0.2]
            yaw_vel = 1.0            # 基类期望标量，会转换为[-1.0, 1.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True  # True, False
        friction_range = [0.1, 0.35]

        randomize_base_mass = True  # True, False
        added_mass_range = [-2.2, 2.2]

        #push_robots = False #推力关闭，需要时打开 
        push_robots = True
        push_interval_s = 5  # 优化：从3增加为5，降低推动频率
        #push_interval = 10 
        max_push_vel_xy = 0.6
        
        max_push_force_xy = 150  # 优化：从720大幅降为150，更现实的推力
    
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]

        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]

        # Add DR for rotor inertia and angular damping

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_12dof_250108_4/urdf/hi_12dof_250108_4_rl_2.urdf"
        keypoints = ["base_link"]
        foot_name = "ankle_roll"
        end_effectors = ["r_" + foot_name, "l_" + foot_name]
        terminate_after_contacts_on = [
            "base_link",
            "r_thigh_link",
            "r_calf_link",
            "r_ankle_pitch_link",
            # "r_ankle_roll_link",
            "l_thigh_link",
            "l_calf_link",
            "l_ankle_pitch_link",
            # "l_ankle_roll_link",
        ]

        disable_gravity = False
        disable_actuations = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = True
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

        angular_damping = 0.1
        rotor_inertia = [
            0.0001188,  # RIGHT LEG
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,  # LEFT LEG
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
        ]
        apply_humanoid_jacobian = False  # True, False

    class motor:
        max_vel = 60 * 2 * torch.pi / 60.0 /20*36
        mid_vel = 10 * 2 * torch.pi / 60.0 /20*36
        mid_tor = 17.5 /36*20
        k = ((max_vel - mid_vel)/(0-mid_tor))
        b = max_vel
        
    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5596
        base_height_range = 0.01
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 9.81*20

        curriculum = False
        only_positive_rewards = False
        tracking_sigma = 0.25
        min_dist_feet = 0.199 - 0.02  # 0.3038 0.195 0.109
        max_dist_feet = 0.199 + 0.02
        class weights(LeggedRobotCfg.rewards.weights):
            # * Regularization rewards * #
            actuation_rate = 1e-3
            actuation_rate2 = 4e-3
            torques = 1e-4
            dof_vel = 1e-3
            lin_vel_z = 20 * 0.1
            ang_vel_xy = 1e-2
            dof_pos_limits = 10
            torque_limits = 1e-1
            motor_limit = 0.1
            # * Floating base rewards * #
            base_height = 3.0  # 优化：从4.0降为3.0，平衡重要性
            # base_heading = 8.0
            base_z_orientation = 2.0  # 优化：增加姿态稳定性权重
            # tracking_lin_vel_world_x = 8.0
            # tracking_lin_vel_world_y = 2.0
            command_yaw_vel = 9
            tracking_lin_vel_base_x = 6.0  # 优化：从18.0大幅降为6.0，避免过度优化
            tracking_lin_vel_base_y = 2.0
            
            # * Stepping rewards * #
            joint_regularization = 2.0  # 优化：从4.0降为2.0，但保持重要性
            contact_schedule = 4.0  # 优化：保持接触调度的重要性

            # * Other * #
            feet_slip = -8.0
            feet_slip_dyaw = -1.0
            feet_distance = 8.0  # 优化：从10.0略降为8.0
            # feet_x_dis = 1.0
            ankle_roll_posture_roll = 3.0  # 优化：从5.0降为3.0
            ankle_roll_posture_pitch = 3.0  # 优化：从5.0降为3.0
            ankle_roll_action_zero = 1.0
            #jiangbo add 暂时不用这些
            #not_fallen = 5.0
            #com_within_support = 2.0
            #base_upright = 3.0
            #antidisturbance = 2.0 
        class termination_weights(LeggedRobotCfg.rewards.termination_weights):
            termination = 1.0

    class scaling(LeggedRobotCfg.scaling):
        base_height = 1.0
        base_lin_vel = 1.0  # .5
        base_ang_vel = 1.0  # 2.
        projected_gravity = 1.0
        foot_states_right = 1.0
        foot_states_left = 1.0
        dof_pos = 1.0
        dof_vel = 1.0  # .1
        dof_pos_target = dof_pos  # scale by range of motion

        # Action scales
        commands = 1.0
        clip_actions = 10.0

    class sim:
        dt =  0.001
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HiControllerRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = 4

    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = "elu"
        normalize_obs = True  # True, False

        single_actor_obs = [
            "phase_sin",
            "phase_cos",
            "commands",
            "dof_pos",
            "dof_vel",
            "base_ang_vel",
            "base_euler_xyz",
            "projected_gravity",  # 优化：增加重力方向感知
            "full_step_period_obs",
            "base_height",        # 优化：增加基座高度感知
            "foot_states_right",  # 优化：增加右足状态感知
            "foot_states_left",   # 优化：增加左足状态感知
        ]
        
        actor_obs = single_actor_obs + [
            # IPC3D相关观测
            "ipc3d_desired_trajectory",      # IPC3D目标轨迹
            "ipc3d_desired_velocity",        # IPC3D目标速度
            "ipc3d_trajectory_error",        # 轨迹跟踪误差
        ]
        critic_obs = [
            "base_height",
            "base_lin_vel_world",  # "base_lin_vel",
            "base_euler_xyz",
            "base_ang_vel",
            "projected_gravity",
            "foot_states_right",
            "foot_states_left",
            "step_commands_right",
            "step_commands_left",
            "commands",
            "phase_sin",
            "phase_cos",
            "dof_pos",
            "dof_vel",
            "full_step_period_obs",
            # IPC3D相关观测（critic需要更多信息进行价值评估）
            "ipc3d_desired_trajectory",
            "ipc3d_desired_velocity",
            "ipc3d_control_forces",          # IPC3D控制力
            "ipc3d_trajectory_error",
            "stability_score",               # 稳定性评分
            "push_state",                   # 外部推力状态
        ]

        actions = ["dof_pos_target"]

        class noise:
            base_height = 0.05
            base_lin_vel = 0.05
            base_lin_vel_world = 0.05
            base_heading = 0.01
            base_ang_vel = 0.15
            projected_gravity = 0.05
            base_euler_xyz = 0.15
            foot_states_right = 0.01
            foot_states_left = 0.01
            step_commands_right = 0.05
            step_commands_left = 0.05
            commands = 0.1
            dof_pos = 0.05
            dof_vel = 0.5  #0.5 抬脚更高，0.1抬脚低
            #dof_vel = 0.1  #jiangbo
            foot_contact = 0.1

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        class PPO:
            # algorithm training hyperparameters
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4  # minibatch size = num_envs*nsteps/nminibatches
            learning_rate = 3.0e-5  # 优化：从1.0e-5增加为3.0e-5，加快学习速度
            #learning_rate = 5.0e-5 #ft时使用 小一点
            schedule = "adaptive"  # could be adaptive, fixed
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 25
        max_iterations = 2001
        run_name = "sf"
        experiment_name = "Hi_Controller"
        save_interval = 25
        plot_input_gradients = False
        plot_parameter_gradients = False
