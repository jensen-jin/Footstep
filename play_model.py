#!/usr/bin/env python
"""
通用模型播放脚本，解决设备不匹配问题
"""

import os
import sys

# 强制使用CPU（如果需要）
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import isaacgym
from gym.envs import *
from gym.utils import task_registry
import argparse
import torch

def play_model(experiment_name, checkpoint, num_envs=1, device='cpu'):
    """播放训练好的模型"""
    
    print(f"🎮 Playing model from experiment: {experiment_name}")
    print(f"📋 Checkpoint: {checkpoint}")
    print(f"🖥️  Device: {device}")
    print(f"🔢 Environments: {num_envs}")
    
    # 创建参数对象
    class Args:
        def __init__(self):
            self.task = 'humanoid_controller'
            self.experiment_name = experiment_name
            self.checkpoint = checkpoint
            self.num_envs = num_envs
            self.sim_device = device
            self.rl_device = device
            self.headless = False
            self.resume = False
            self.load_run = -1
            self.load_files = False
            self.run_name = None
            self.seed = None
            self.max_iterations = None
            self.wandb_project = None
            self.wandb_entity = None
            self.wandb_sweep_id = None
            self.wandb_sweep_config = None
            self.disable_wandb = True
            self.disable_local_saving = True
            self.sampling_method = None
            self.record = False
            
    args = Args()
    
    try:
        # 加载环境配置
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        print("✅ Environment created successfully")
        
        # 加载训练配置和策略
        policy_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        print("✅ Policy loaded successfully")
        
        # 运行播放
        print("🚀 Starting playback...")
        env.reset()
        
        # 播放循环
        for i in range(1000):  # 播放1000步
            with torch.no_grad():
                obs = env.get_observations()
                actions = policy_runner.alg.actor(obs)
                env.step(actions)
                
            if i % 100 == 0:
                print(f"Step {i}/1000")
                
        print("✅ Playback completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during playback: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trained humanoid model")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--checkpoint", type=int, required=True, help="Checkpoint number")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda:0'], 
                       help="Device to use for simulation")
    
    args = parser.parse_args()
    
    success = play_model(args.experiment, args.checkpoint, args.num_envs, args.device)
    
    if success:
        print("🎉 Model playback successful!")
    else:
        print("💥 Model playback failed!")
        sys.exit(1)