# CLAUDE.md

本文件为Claude Code (claude.ai/code)在此代码库中工作时提供指导。

## 项目概述

这是一个人形机器人研究代码库，实现了"基于模型的足迹规划与无模型强化学习集成的动态腿式运动"（IROS 2024）。该项目结合Isaac Gym仿真和基于PyTorch的强化学习，用于训练具有先进引导模型的人形机器人。

## 开发命令

### 安装和设置
```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装Isaac Gym（需单独下载）
# 从NVIDIA Developer下载Isaac Gym Preview 4
cd <isaacgym_location>/python
pip install -e .

# 安装此包
pip install -e .
```

### 训练命令
```bash
# 使用默认设置训练人形机器人控制器
python gym/scripts/train.py --task=humanoid_controller

# 使用特定参数训练
python gym/scripts/train.py --task=humanoid_controller --num_envs=4096 --headless

# 从检查点恢复训练
python gym/scripts/train.py --task=humanoid_controller --resume --load_run=-1 --checkpoint=-1

# 使用CPU仿真运行（较慢）
python gym/scripts/train.py --task=humanoid_controller --sim_device=cpu --rl_device=cpu

# 启用力量推动训练（新功能）
python gym/scripts/train.py --task=humanoid_controller --enable_force_push

# 使用DWAQ神经网络架构
python gym/scripts/train.py --task=humanoid_controller --architecture=dwaq

# 使用新机器人配置
python gym/scripts/train.py --task=humanoid_controller --robot=hi_12
python gym/scripts/train.py --task=humanoid_controller --robot=hi_cl_12  
python gym/scripts/train.py --task=humanoid_controller --robot=pi_cl_12
```

### 播放/测试训练模型
```bash
# 播放训练策略（默认加载最新）
python gym/scripts/play.py --task=humanoid_controller

# 播放特定检查点
python gym/scripts/play.py --task=humanoid_controller --load_run=<run_name> --checkpoint=<iteration>

# 导出策略用于硬件部署
# 在play.py中设置EXPORT_POLICY=TRUE生成policy.onnx
```

### 机器人注册系统（统一接口）
```bash
# 列出可用机器人
python -m gym.envs.robots.unified_trainer list

# 获取机器人信息
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

### LIPM演示
```bash
# 运行3D LIPM动画
python LIPM/demo_LIPM_3D_vt.py

# 运行LIPM分析和绘图
python LIPM/demo_LIMP_3D_vt_analysis.py
```

### 测试
```bash
# 运行环境测试
python gym/tests/test_env.py

# 测试机器人注册系统
python test_robot_registry.py

# 测试IPC3D实现
python test_ipc3d_implementation.py

# 测试推动系统集成（新功能）
python test_push_integration.py

# 测试DWAQ架构集成（新功能）
python test_dwaq_integration.py

# 测试完整训练流程集成（新功能）
python test_training_flow_integration.py

# 运行pytest
pytest
```

## 架构概述

### 核心结构
- **gym/**: 主要Isaac Gym环境和训练基础架构
  - **envs/**: 任务环境（人形机器人、倒立摆小车、单摆）
    - **base/**: 所有机器人和任务的基类
    - **humanoid/**: 人形机器人特定控制器和配置
    - **robots/**: 统一机器人注册系统（13个机器人模型）
    - **guidance/**: 引导模型（LIPM、IPC3D）
  - **scripts/**: 训练和评估脚本
  - **utils/**: 日志记录、接口、地形生成工具

- **learning/**: 基于PyTorch的强化学习算法
  - **algorithms/**: PPO实现
  - **modules/**: 演员-评论家网络（标准和DWAQ架构）
  - **runners/**: 训练循环管理
  - **storage/**: 经验回放缓冲区

- **LIPM/**: 线性倒立摆模型演示
- **resources/**: 机器人URDF文件、轨迹和3D网格

### 机器人注册系统
项目具有统一的机器人管理系统：

```python
from gym.envs.robots import RobotConfigFactory, RobotRegistry

# 为任何机器人+引导组合创建配置
config = RobotConfigFactory.create_config(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d'
)

# 列出所有可用机器人
robots = RobotRegistry.list_robots()  # 13种不同的机器人模型
```

可用机器人包括：
- **mit_humanoid_fixed_arms**: 主要步行人形机器人（10自由度）
- **mit_humanoid_full**: 带手臂的完整人形机器人（16自由度）
- **hi_12**: HI-12自由度人形机器人（新增）
- **hi_cl_12**: HI-CL-12自由度人形机器人（新增）
- **pi_cl_12**: PI-CL-12自由度人形机器人（新增）
- **cartpole**: 经典控制基准
- **pendulum**: 用于测试的简单单摆
- 另外6种机器人变体

### 引导模型
实现了两个主要引导模型：

1. **LIMP（线性倒立摆模型）**:
   - 快速、传统的足迹规划
   - 适合平地行走

2. **IPC3D（3D倒立摆控制）**:
   - 基于SDRE的先进非线性控制
   - 更适合复杂地形和动态平衡
   - 基于FlexLoco实现的双轴控制

```python
from gym.envs.guidance import GuidanceModelFactory

# 创建引导模型
guidance = GuidanceModelFactory.create_model(
    model_type='ipc3d',
    robot_spec=robot_spec,
    guidance_config=config
)
```

## 新功能集成 (v2.0)

### 力量推动系统
项目现在集成了高级的力量推动系统，用于更真实的机器人训练：

```python
# 配置力量推动系统
config.domain_rand.use_force_push = True  # 启用力量推动
config.domain_rand.max_push_force_xy = 100.0  # 最大推力（牛顿）
config.domain_rand.push_duration = 10  # 推动持续时间（步数）
config.domain_rand.push_interval = 100  # 推动间隔
```

**主要特性：**
- **基于力的推动**：使用Isaac Gym的`apply_rigid_body_force_at_pos_tensors`
- **课程学习**：推力强度随训练步数逐渐增加
- **可配置推动点**：可推动机器人的特定部位
- **真实物理**：比基于速度的推动更符合物理规律
- **横纵方向独立控制**：可分别设置X和Y方向的推力强度和范围

#### 横纵方向独立推力控制
支持独立设置X和Y方向的推力参数：

```python
# 启用横纵方向独立控制
config.domain_rand.separate_xy_forces = True
config.domain_rand.max_push_force_x = 120.0  # X方向最大推力
config.domain_rand.max_push_force_y = 60.0   # Y方向最大推力
config.domain_rand.push_force_x_range = [-0.8, 1.2]  # X方向力量范围
config.domain_rand.push_force_y_range = [-1.0, 1.0]  # Y方向力量范围
```

**应用场景：**
- **前后运动训练**：重点训练前进/后退稳定性
- **侧向稳定性**：加强左右平衡能力
- **非对称扰动**：模拟真实环境中的方向性干扰
- **特定方向训练**：针对特定行走方向的鲁棒性训练

### DWAQ神经网络架构
集成了高级的DWAQ（Deep With Autoencoder Quantization）架构：

```python
from learning.modules.model_factory import create_model

# 创建DWAQ模型
model = create_model(
    architecture="dwaq",
    num_actor_obs=45,
    num_critic_obs=45,
    num_actions=12,
    cenet_in_dim=25,
    cenet_out_dim=20
)
```

**DWAQ特性：**
- **编码器-解码器架构**：分离上下文理解和动作生成
- **变分自编码器元素**：使用重参数化技巧
- **分离速度编码**：独立处理上下文和速度信息
- **Dropout支持**：正则化防止过拟合
- **多机器人泛化**：更好地适应不同机器人形态

### 改进的LIMP算法
修复了关键的坐标变换错误并增强了可视化：

```python
# LIMP算法的符号修正
offset_y = np.sin(theta) * original_offset_x + np.cos(theta) * original_offset_y  # 修正符号错误
```

**改进内容：**
- **坐标变换修正**：修复了符号错误，提高计算精度
- **3D可视化**：实时轨迹跟踪和分析
- **中文注释**：增强文档理解
- **课程学习演示**：集成渐进式学习展示

### 模型工厂系统
统一的神经架构选择接口：

```python
from learning.modules.model_factory import (
    get_model_class, get_architecture_info, print_architecture_comparison
)

# 获取架构信息
info = get_architecture_info()
print_architecture_comparison()

# 验证配置
is_valid = validate_architecture_config("dwaq", config)
```

**功能：**
- **架构选择**：在标准和DWAQ架构间切换
- **配置验证**：自动检查参数兼容性
- **架构比较**：详细对比不同神经网络
- **动态创建**：运行时选择最适合的架构

## 关键配置文件

- **configs/robot_training_config.yaml**: 批量实验配置
- **gym/envs/humanoid/humanoid_controller_config.py**: 人形机器人特定设置
- **gym/envs/robots/specifications/**: 机器人模型定义
  - **push_robots.py**: 新增机器人规格（HI-12, HI-CL-12, PI-CL-12）
- **gym/envs/base/legged_robot_config.py**: 基础机器人配置（包含力量推动参数）
- **learning/modules/**: 神经网络模块
  - **actor_critic.py**: 标准演员-评论家架构
  - **actor_critic_DWAQ.py**: DWAQ高级架构
  - **model_factory.py**: 架构选择工厂
- **requirements.txt**: Python依赖

## 开发说明

### 模型训练
- 目标约3000次迭代以获得良好的人形机器人策略
- 训练期间按'v'键禁用渲染以提高性能
- 模型保存到`gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`
- 使用`--headless`进行无可视化的快速训练

### Isaac Gym集成
- 基于Isaac Gym Preview 4构建（兼容Preview 3）
- GPU加速物理仿真，支持数千个并行环境
- 需要支持CUDA的NVIDIA GPU
- 默认设置使用4096个并行环境

### 硬件部署
- 训练策略可导出为ONNX格式用于C++部署
- 兼容MIT的Cheetah-Software硬件栈
- 在play.py中设置`EXPORT_POLICY=TRUE`进行策略导出

### IPC3D实现细节
- FlexLoco的Lua IPC3D算法的完整Python移植
- SDRE（状态相关Riccati方程）非线性控制
- 分离X和Z动力学的双轴控制
- 实时轨迹生成和跟踪
- 与机器人注册表集成，便于实验

## 常见故障排除

### Isaac Gym问题
```bash
# 缺少libpython错误
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# CUDA/GPU问题
python gym/scripts/train.py --task=humanoid_controller --sim_device=cpu
```

### 内存问题
- 对于复杂机器人或有限GPU内存，减少`num_envs`
- 使用`mit_humanoid_fixed_arms`（10自由度）而不是`mit_humanoid_full`（16自由度）
- 训练期间监控GPU内存使用

### 性能优化
- 训练时使用`--headless`模式
- 训练期间用'v'键禁用渲染
- 根据硬件调整回合长度和环境数量

该代码库支持传统的LIMP引导和新的IPC3D引导模型，提供统一接口用于训练任何机器人与任何引导模型的组合。

## 版本历史

### v2.0 (当前版本)
- ✅ **力量推动系统**：基于物理的机器人推动训练
- ✅ **DWAQ神经架构**：高级编码器-解码器网络  
- ✅ **新机器人支持**：HI-12, HI-CL-12, PI-CL-12配置
- ✅ **LIMP算法修复**：坐标变换符号错误修正
- ✅ **模型工厂系统**：统一架构选择接口
- ✅ **课程学习集成**：渐进式推力训练
- ✅ **增强测试套件**：全面集成测试覆盖

### v1.0 (基础版本)
- ✅ **IPC3D实现**：SDRE非线性控制引导
- ✅ **机器人注册系统**：13种机器人统一接口
- ✅ **Isaac Gym集成**：GPU加速并行仿真
- ✅ **PPO强化学习**：基础演员-评论家训练
- ✅ **LIMP引导**：传统线性倒立摆模型

## 新功能使用指南

### 启用力量推动训练
```python
# 在配置文件中设置
config.domain_rand.use_force_push = True
config.domain_rand.max_push_force_xy = 100.0  # 可根据机器人调整
config.domain_rand.push_interval = 100  # 每100步推动一次
```

### 选择神经网络架构
```python
# 使用标准架构（默认）
model = create_model("standard", num_actor_obs, num_critic_obs, num_actions)

# 使用DWAQ架构（推荐用于多机器人训练）
model = create_model("dwaq", num_actor_obs, num_critic_obs, num_actions,
                     cenet_in_dim=25, cenet_out_dim=20)
```

### 新机器人配置
```python
from gym.envs.robots.specifications.push_robots import hi_12, hi_cl_12, pi_cl_12

# 获取机器人规格
robot_spec = hi_12  # 或 hi_cl_12, pi_cl_12
```