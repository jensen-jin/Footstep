# Robot Registry System - Usage Guide

## 概述

这个机器人注册表系统为Footstep项目提供了统一的机器人模型管理和配置生成功能。它支持：

- ✅ **统一管理**：13种机器人模型的集中注册
- ✅ **引导模型选择**：LIPM和IPC3D引导模型自由切换  
- ✅ **自动配置生成**：根据机器人规格自动生成训练配置
- ✅ **向后兼容**：与现有代码完全兼容
- ✅ **批量实验**：支持大规模对比研究

## 核心组件

### 1. Robot Registry (机器人注册表)
```python
from gym.envs.robots import RobotRegistry

# 列出所有可用机器人
robots = RobotRegistry.list_robots()
print(robots)  # ['mit_humanoid_fixed_arms', 'cartpole', ...]

# 按类别筛选
humanoid_robots = RobotRegistry.list_robots('humanoid')

# 获取机器人详细信息
robot_spec = RobotRegistry.get_robot('mit_humanoid_fixed_arms')
```

### 2. Robot Config Factory (配置工厂)
```python
from gym.envs.robots import RobotConfigFactory

# 创建MIT人形机器人配置 + IPC3D引导
config = RobotConfigFactory.create_config(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d'
)

# 带参数覆盖的配置
config = RobotConfigFactory.create_config(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d',
    env={'num_envs': 2048},
    terrain={'mesh_type': 'heightfield'}
)
```

### 3. Unified Trainer (统一训练器)
```python
from gym.envs.robots.unified_trainer import UnifiedTrainer

trainer = UnifiedTrainer()

# 训练单个机器人
trainer.train_single_robot(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d',
    experiment_name='test_ipc3d'
)

# 对比研究
trainer.run_comparison_study(
    robots=['mit_humanoid_fixed_arms'],
    guidance_models=['limp', 'ipc3d'],
    base_experiment_name='guidance_comparison'
)
```

## 可用机器人模型

### 人形机器人 (Humanoid)
1. **mit_humanoid_fixed_arms** - MIT人形机器人固定臂版本 (推荐)
   - 10个自由度
   - 优化用于步行
   - 支持LIPM和IPC3D引导

2. **mit_humanoid_full** - MIT人形机器人全身版本
   - 16个自由度 (包含手臂)
   - 支持复杂动作
   - 更高计算需求

3. **mit_humanoid_fixed_arms_base** - MIT人形机器人基础版本
   - 兼容旧版URDF

### 简单机器人 (Simple)
1. **cartpole** - 经典倒立摆小车
   - 1个自由度
   - 控制理论验证
   - 适合IPC3D引导

2. **pendulum** - 单摆
   - 1个自由度  
   - 摆起控制

3. **point_mass** - 质点模型
   - 3个自由度
   - 基础动力学测试

4. **slip** - 弹簧腿倒立摆
   - 2个自由度
   - 跑步动力学

## 引导模型

### LIPM (线性倒立摆模型)
- **适用场景**：平地行走，简单步态
- **特点**：计算快速，理论成熟
- **参数**：COM高度，步长，步宽，步频

### IPC3D (3D倒立摆控制)
- **适用场景**：复杂地形，动态平衡
- **特点**：精确3D动力学，更真实
- **参数**：质量分布，极点长度，摩擦系数

## 使用方式

### 1. 命令行训练
```bash
# 基础训练
python -m gym.envs.robots.unified_trainer train \
    --robot mit_humanoid_fixed_arms \
    --guidance ipc3d \
    --experiment my_test

# 对比研究
python -m gym.envs.robots.unified_trainer compare \
    --robots mit_humanoid_fixed_arms mit_humanoid_full \
    --guidance limp ipc3d \
    --name robot_guidance_comparison

# 列出可用机器人
python -m gym.envs.robots.unified_trainer list

# 查看机器人详情
python -m gym.envs.robots.unified_trainer info mit_humanoid_fixed_arms
```

### 2. 配置文件训练
```yaml
# config.yaml
experiments:
  - robot: "mit_humanoid_fixed_arms"
    guidance: "ipc3d"  
    name: "improved_walking"
    overrides:
      env:
        num_envs: 4096
      terrain:
        mesh_type: "heightfield"
```

```bash
python -m gym.envs.robots.unified_trainer batch --config config.yaml
```

### 3. Python API
```python
# 直接使用配置工厂
from gym.envs.robots import RobotConfigFactory

config = RobotConfigFactory.create_config(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d'
)

# 传递给现有训练代码
# train_robot(config)  # 您的训练函数

# 或使用统一训练器
from gym.envs.robots.unified_trainer import UnifiedTrainer

trainer = UnifiedTrainer()
trainer.train_single_robot(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d'
)
```

## 向后兼容

### 现有代码迁移
```python
# 旧代码
from gym.envs.humanoid.humanoid_controller_config import HumanoidControllerCfg
config = HumanoidControllerCfg()

# 新代码 (推荐)
from gym.envs.robots import RobotConfigFactory
config = RobotConfigFactory.create_config(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d'  # 新功能！
)

# 兼容方式
from gym.envs.robots.legacy_adapter import LegacyConfigAdapter
config = LegacyConfigAdapter.from_legacy_config(
    HumanoidControllerCfg,
    guidance_model='ipc3d'
)
```

## 添加新机器人

### 1. 创建Robot Specification
```python
from gym.envs.robots import RobotRegistry, RobotSpec

# 定义新机器人
new_robot = RobotSpec(
    name="my_robot",
    urdf_path="path/to/robot.urdf",
    num_actuators=12,
    default_joint_angles={...},
    joint_limits={...},
    control_config={...},
    physical_params={...},
    description="My custom robot",
    category="custom"
)

# 注册机器人
RobotRegistry.register(new_robot)
```

### 2. 添加到规格文件
将机器人定义添加到 `gym/envs/robots/specifications/` 下的相应文件中。

## 实验配置示例

### 单机器人多引导对比
```python
trainer = UnifiedTrainer()
trainer.run_comparison_study(
    robots=['mit_humanoid_fixed_arms'],
    guidance_models=['limp', 'ipc3d'],
    base_experiment_name='guidance_comparison'
)
```

### 多机器人同引导对比  
```python
trainer.run_comparison_study(
    robots=['mit_humanoid_fixed_arms', 'mit_humanoid_full'],
    guidance_models=['ipc3d'],
    base_experiment_name='robot_comparison'
)
```

### 大规模网格搜索
```python
trainer.run_comparison_study(
    robots=['mit_humanoid_fixed_arms', 'mit_humanoid_full', 'cartpole'],
    guidance_models=['limp', 'ipc3d'],
    base_experiment_name='full_comparison'
)
```

## 最佳实践

1. **机器人选择**：
   - 步行任务：`mit_humanoid_fixed_arms`
   - 全身任务：`mit_humanoid_full`
   - 控制验证：`cartpole`

2. **引导模型选择**：
   - 平地快速训练：`limp`
   - 复杂地形/高精度：`ipc3d`

3. **实验设计**：
   - 先用简单机器人验证算法
   - 再用目标机器人对比引导模型
   - 最后进行大规模参数扫描

4. **性能优化**：
   - 复杂机器人减少环境数量
   - IPC3D引导可能需要更多迭代
   - 使用配置文件管理大规模实验

## 故障排除

### 常见问题

1. **机器人未找到**
   ```
   ValueError: Robot 'xxx' not found
   ```
   - 检查机器人名称拼写
   - 运行 `list` 命令查看可用机器人

2. **URDF文件未找到**
   - 确保设置了 `LEGGED_GYM_ROOT_DIR` 环境变量
   - 检查URDF文件路径是否正确

3. **配置冲突**
   - 检查参数覆盖是否正确
   - 验证关节名称匹配

### 调试模式
```python
# 详细信息
trainer = UnifiedTrainer(verbose=True)
config = RobotConfigFactory.create_config(
    robot_name='mit_humanoid_fixed_arms',
    guidance_model='ipc3d',
    debug=True
)
```

## 下一步计划

1. **算法集成**：
   - 完成IPC3D算法实现
   - 集成FlexLoco的SDRE控制器

2. **性能优化**：
   - 加速配置生成
   - 优化内存使用

3. **可视化增强**：
   - 实时训练监控
   - 引导轨迹可视化

4. **扩展支持**：
   - 四足机器人
   - 自定义环境