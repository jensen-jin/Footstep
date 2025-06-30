#!/usr/bin/env python
"""
人形机器人训练效果播放脚本
解决设备不匹配和参数问题
"""

import os
import sys

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU

# Isaac Gym相关导入
import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil

# 项目导入
import numpy as np
import torch
from gym.envs.humanoid.humanoid_controller import HumanoidController
from gym.envs.humanoid.humanoid_controller_config import HumanoidControllerCfg
from gym.utils.helpers import parse_sim_params
from learning.modules.actor_critic import ActorCritic

def create_args():
    """创建完整的参数对象"""
    class Args:
        def __init__(self):
            # 基础参数
            self.task = "humanoid_controller"
            self.headless = False
            self.sim_device = "cpu"
            self.rl_device = "cpu"
            self.graphics_device_id = 0
            self.num_envs = 1
            self.seed = 42
            
            # Isaac Gym 参数
            self.physics_engine = gymapi.SIM_PHYSX
            self.sim_device_type = "cpu"
            self.compute_device_id = 0
            self.graphics_device_id = 0
            self.num_threads = 0
            self.use_gpu = False
            self.use_gpu_pipeline = False
            self.subscenes = 0
            self.slices = 0
            
            # 训练参数
            self.experiment_name = "Humanoid_Controller"
            self.run_name = None
            self.load_run = -1
            self.checkpoint = -1  # Load latest checkpoint
            self.resume = False
            self.load_files = False
            
            # 其他参数
            self.max_iterations = None
            self.disable_wandb = True
            self.disable_local_saving = True
            self.wandb_project = None
            self.wandb_entity = None
            self.wandb_sweep_id = None
            self.wandb_sweep_config = None
            self.sampling_method = None
            self.record = False
            
    return Args()

def find_latest_model(logs_dir="logs"):
    """查找最新的训练模型"""
    import glob
    
    # 查找所有模型文件
    model_pattern = os.path.join(logs_dir, "**", "model_*.pt")
    model_files = glob.glob(model_pattern, recursive=True)
    
    if not model_files:
        return None
    
    # 按修改时间排序，取最新的
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def play_humanoid():
    """播放人形机器人训练效果"""
    
    print("🤖 启动人形机器人播放器...")
    
    # 创建参数
    args = create_args()
    
    # 创建仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1/60  # 60 FPS
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # PhysX参数
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    
    try:
        # 创建环境配置
        print("📋 加载环境配置...")
        cfg = HumanoidControllerCfg()
        cfg.env.num_envs = args.num_envs
        cfg.sim.dt = 1/60  # 60 FPS
        
        # 创建环境
        print("🌍 创建仿真环境...")
        env = HumanoidController(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless
        )
        
        print("✅ 环境创建成功！")
        
        # 查找最新模型
        model_path = find_latest_model()
        if model_path is None:
            print("❌ 未找到训练模型！请先进行训练。")
            print("训练命令: python gym/scripts/train.py --task=humanoid_controller")
            return False
            
        print(f"🧠 加载模型: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 创建策略网络
        actor_critic = ActorCritic(
            num_actor_obs=cfg.env.num_observations,
            num_critic_obs=cfg.env.num_observations,
            num_actions=cfg.env.num_actions,
            actor_hidden_dims=cfg.policy.actor_hidden_dims,
            critic_hidden_dims=cfg.policy.critic_hidden_dims,
            activation=cfg.policy.activation,
            init_noise_std=cfg.policy.init_noise_std
        ).to('cpu')
        
        # 加载模型权重
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        actor_critic.eval()
        
        print("✅ 模型加载成功！")
        print("🎮 开始播放...")
        print("\n" + "="*60)
        print("控制说明：")
        print("ESC: 退出")
        print("R: 重置环境")
        print("V: 停止/开始同步")
        print("WASD: 控制机器人移动 (如果支持)")
        print("="*60 + "\n")
        
        # 重置环境
        obs = env.reset()
        
        # 播放循环
        step_count = 0
        max_steps = 10000  # 最大步数
        
        while step_count < max_steps:
            # 获取策略动作
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to('cpu')
                actions = actor_critic.act_inference(obs_tensor)
                actions = actions.squeeze(0).numpy()
            
            # 执行动作
            obs, rewards, dones, infos = env.step(actions)
            
            # 检查是否需要重置
            if dones.any():
                print(f"🔄 环境重置 (步数: {step_count})")
                obs = env.reset()
                step_count = 0
            
            step_count += 1
            
            # 每1000步显示一次进度
            if step_count % 1000 == 0:
                print(f"📊 步数: {step_count}, 奖励: {rewards.mean():.3f}")
        
        print("✅ 播放完成！")
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断播放")
        return True
        
    except Exception as e:
        print(f"❌ 播放过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 人形机器人训练效果播放器")
    print("="*50)
    
    success = play_humanoid()
    
    if success:
        print("🎉 播放成功完成！")
    else:
        print("💥 播放失败！")
        sys.exit(1)