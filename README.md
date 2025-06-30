# Footstep - 人形机器人足迹规划与强化学习训练平台

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-red.svg)](https://pytorch.org)
[![Isaac Gym](https://img.shields.io/badge/Isaac%20Gym-Preview%204-green.svg)](https://developer.nvidia.com/isaac-gym)

基于论文《Integrating Model-Based Footstep Planning with Model-Free Reinforcement Learning for Dynamic Legged Locomotion》(IROS 2024)的开源实现，结合Isaac Gym物理仿真和PyTorch强化学习，训练具有先进引导模型的人形机器人。

**论文链接**: [https://arxiv.org/abs/2408.02662](https://arxiv.org/abs/2408.02662)  
**演示视频**: [https://youtu.be/Z0E9AKt6RFo](https://youtu.be/Z0E9AKt6RFo)

## 🚀 项目特色

- **统一机器人管理系统**: 支持13种机器人模型（MIT人形机器人、倒立摆小车等）
- **双重引导模型**: LIMP（线性倒立摆）+ IPC3D（3D倒立摆控制）
- **先进算法**: IPC3D基于SDRE非线性控制，完整Python移植FlexLoco
- **GPU加速训练**: Isaac Gym支持数千个并行环境
- **生产就绪**: 完整的测试套件和配置管理

## 🛠 安装指南

### 系统要求

- **操作系统**: Ubuntu 20.04/22.04 (推荐) 或 Windows 10/11
- **Python**: 3.8 (必需)
- **GPU**: NVIDIA GPU + CUDA 11.3 (推荐，也支持CPU训练)
- **内存**: 至少16GB RAM, 8GB GPU内存

### 快速安装

#### 1. 创建conda环境
```bash
# 创建Python 3.8环境
conda create -n footstep python=3.8 -y
conda activate footstep
```

#### 2. 安装PyTorch
```bash
# 安装CUDA 11.3版本的PyTorch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### 3. 安装Isaac Gym
```bash
# 下载Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
# 解压到 ~/isaac-gym

cd ~/isaac-gym/python
pip install -e .
```

#### 4. 安装项目
```bash
# 克隆项目
git clone <repository_url>
cd Footstep

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### 5. 验证安装
```bash
# 测试基础组件
python simple_test.py

# 测试IPC3D算法
python test_ipc3d_standalone.py

# 列出可用机器人
python -m gym.envs.robots.unified_trainer list
```
## ⚡ 快速开始

### 第一次训练

```bash
# 激活环境
conda activate footstep

# 快速测试训练（CPU模式，适合验证安装）
python gym/scripts/train.py \
    --task=humanoid_controller \
    --num_envs=16 \
    --max_iterations=10 \
    --sim_device=cpu \
    --rl_device=cpu \
    --headless

# GPU训练（生产模式）
python gym/scripts/train.py \
    --task=humanoid_controller \
    --num_envs=1024 \
    --max_iterations=3000 \
    --headless
```

### 播放训练结果

```bash
# 播放最新训练的模型
python gym/scripts/play.py --task=humanoid_controller
```

## 📚 用户指南

### 1. LIPM演示动画
LIMP模型相关代码在`LIPM`文件夹中，基于[BipedalWalkingRobots](https://github.com/chauby/BipedalWalkingRobots)修改用于质心速度跟踪任务。

```bash
# 运行3D LIPM动画
python LIPM/demo_LIPM_3D_vt.py

# 运行LIPM分析和绘图  
python LIPM/demo_LIPM_3D_vt_analysis.py
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/edff8522-b9d5-42d3-80af-37c0f0d50758">
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/2e4e1600-2b34-4181-aaea-63e2288e85e7">
  <img width = "60%" src="https://github.com/user-attachments/assets/236682a4-8beb-4e46-b2a5-e4fe76a71978">
  <img width = "60%" src="https://github.com/user-attachments/assets/bd0f33c4-7ba8-4403-9647-2d1e61091263">
</div>

### 2. MIT人形机器人训练

#### 基础训练
```bash
# 默认GPU训练
python gym/scripts/train.py --task=humanoid_controller

# CPU训练（适合测试或无GPU环境）
python gym/scripts/train.py --task=humanoid_controller --sim_device=cpu --rl_device=cpu

# 无头模式（提高性能）
python gym/scripts/train.py --task=humanoid_controller --headless
```

#### 训练参数说明
| 参数 | 说明 |
|------|------|
| `--task` | 训练任务类型（humanoid_controller） |
| `--num_envs` | 并行环境数量（默认4096，CPU可设为16-64） |
| `--max_iterations` | 最大训练迭代数（推荐3000+） |
| `--sim_device` | 物理仿真设备（cuda:0 或 cpu） |
| `--rl_device` | 强化学习计算设备（cuda:0 或 cpu） |
| `--headless` | 无图形界面模式 |
| `--experiment_name` | 实验名称 |
| `--resume` | 从检查点恢复训练 |
| `--seed` | 随机种子 |

**性能提示**: 训练开始后按`v`键停止渲染以提高性能。

#### 恢复训练
```bash
# 从最新检查点恢复
python gym/scripts/train.py \
    --task=humanoid_controller \
    --resume \
    --load_run=-1 \
    --checkpoint=-1
```

#### 播放训练结果
```bash
# 播放最新模型
python gym/scripts/play.py --task=humanoid_controller

# 播放特定检查点
python gym/scripts/play.py \
    --task=humanoid_controller \
    --load_run=<run_name> \
    --checkpoint=<iteration>
```

**注意**: 通常需要约3000次迭代才能获得表现良好的策略。训练模型保存在 `gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。

<div align="center">
  <img src="https://github.com/user-attachments/assets/52acb865-057a-48bf-8b6c-4ec7f3415bca">
</div>

### 3. 统一机器人管理系统

项目提供统一的机器人注册系统，支持多种机器人模型和引导算法的组合：

```bash
# 列出可用机器人
python -m gym.envs.robots.unified_trainer list

# 查看机器人详情
python -m gym.envs.robots.unified_trainer info mit_humanoid_fixed_arms

# 使用特定机器人+引导模型训练
python -m gym.envs.robots.unified_trainer train \
    --robot mit_humanoid_fixed_arms \
    --guidance ipc3d \
    --experiment my_experiment

# 运行对比研究
python -m gym.envs.robots.unified_trainer compare \
    --robots mit_humanoid_fixed_arms mit_humanoid_full \
    --guidance limp ipc3d \
    --name guidance_comparison
```

#### 可用机器人模型
- **mit_humanoid_fixed_arms**: MIT人形机器人固定臂版本（10自由度，推荐）
- **mit_humanoid_full**: 带手臂的完整人形机器人（16自由度）
- **cartpole**: 经典倒立摆小车（控制基准）
- **pendulum**: 单摆系统（简单测试）
- 另外9种机器人变体...

#### 引导模型
1. **LIMP (线性倒立摆模型)**
   - 快速、传统的足迹规划
   - 适合平地行走

2. **IPC3D (3D倒立摆控制)** 🆕
   - 基于SDRE的先进非线性控制
   - 更适合复杂地形和动态平衡
   - 基于FlexLoco实现的双轴控制

### 4. 部署到机器人硬件

本项目不包含MIT人形机器人硬件部署代码栈。请查看[Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software)获取硬件代码栈。

```bash
# 导出训练策略用于硬件部署
# 在play.py中设置EXPORT_POLICY=True，然后运行：
python gym/scripts/play.py --task=humanoid_controller

# 这将生成policy.onnx文件用于C++代码
```

## 🔧 故障排除

### 常见问题

#### 1. Isaac Gym问题
```bash
# libpython错误
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# GPU架构不匹配（RTX 40xx系列）
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
export CUDA_LAUNCH_BLOCKING=1

# CUDA/GPU问题 - 切换到CPU模式
python gym/scripts/train.py --task=humanoid_controller --sim_device=cpu --rl_device=cpu
```

#### 2. 内存问题
```bash
# 减少环境数量
--num_envs=512  # GPU模式
--num_envs=16   # CPU模式或内存不足时
```

#### 3. RTX 40xx GPU支持
```bash
# 使用提供的设置脚本
./setup_rtx40xx.sh

# 或使用优化训练脚本
./train_rtx4060.sh
```

### 系统要求
- **操作系统**: Ubuntu 20.04/22.04 (推荐)
- **GPU**: NVIDIA GeForce RTX 30xx/40xx 或 RTX A系列
- **测试环境**: Ubuntu 22.04 + RTX 3090/4060

---
### Acknowledgement ###
We would appreciate it if you would cite it in academic publications:
```
@inproceedings{lee2024integrating,
  title={Integrating model-based footstep planning with model-free reinforcement learning for dynamic legged locomotion},
  author={Lee, Ho Jae and Hong, Seungwoo and Kim, Sangbae},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11248--11255},
  year={2024},
  organization={IEEE}
}
```
