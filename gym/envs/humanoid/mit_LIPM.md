  humanoid_controller.py 训练解析
  ```
    python gym/scripts/train.py --task=humanoid_controller --sim_device=cpu --rl_device=cpu --max_iterations=3000 --num_envs=256 --headless
  ```
  1. 程序入口和执行链条

  主脚本: gym/scripts/train.py
  - 程序从 main() 函数开始，调用 train(args)
  - 参数解析：使用 get_args() 处理命令行参数
  - 主要执行步骤：
    a. 解析命令行参数（任务类型、设备、迭代次数等）
    b. 通过 task_registry.make_env() 创建环境
    c. 通过 task_registry.make_alg_runner() 创建策略运行器
    d. 设置日志记录（TensorBoard、WandB）
    e. 开始训练循环：policy_runner.learn()

  任务注册中心: gym/utils/task_registry.py
  - 统一的环境和算法工厂
  - 环境创建：make_env() 实例化具体任务类
  - 算法创建：make_alg_runner() 创建PPO训练算法

  2. 配置系统

  环境配置: gym/envs/humanoid/humanoid_controller_config.py
  - HumanoidControllerCfg：仿真参数、奖励函数、观测定义
  - HumanoidControllerRunnerCfg：PPO超参数、神经网络架构
  - 关键配置：
    - 环境数量：256个并行环境
    - 执行器：10个关节
    - 回合长度：5秒
    - 观测空间：基座状态、足部位置、步进命令、相位信息、关节状态
    - 动作空间：关节位置目标（10自由度）

  3. 环境初始化

  主环境类: gym/envs/humanoid/humanoid_controller.py
  - HumanoidController 继承自 LeggedRobot
  - 初始化流程：
    a. 设置Isaac Gym仿真环境
    b. 初始化机器人缓冲区（状态、命令、观测）
    c. 设置步进规划系统（3D LIMP/XCoM轨迹生成）
    d. 初始化奖励函数

  基础类: gym/envs/base/legged_robot.py
  - 通用机器人功能
  - 物理集成：Isaac Gym张量管理
  - 状态缓冲区：关节位置/速度、基座姿态、接触力

  4. 期望轨迹生成机制


  这是系统的核心部分，采用分层控制结构：

  高层轨迹规划

  方法: _generate_step_command_by_3DLIMP_XCoM() (第424-494行)
  算法: 3D线性倒立摆模型（LIMP）结合扩展捕获点（XCoM）
  输入: 基座速度命令（来自强化学习策略）
  输出: 足部着地位置和方向

  数学基础:
  # XCoM计算 (第468-471行)
  x_f = x_0*torch.cosh(T*w) + vx_0*torch.sinh(T*w)/w
  vx_f = x_0*w*torch.sinh(T*w) + vx_0*torch.cosh(T*w)
  eICP_x = x_f_world + vx_f/w  # 扩展瞬时捕获点

  轨迹生成流程:
  1. 速度命令随机生成: 在指定范围内随机采样目标速度
  2. LIMP动力学求解: 使用线性倒立摆模型计算最优足部着地点
  3. 捕获点控制: 确保机器人重心轨迹动态稳定
  4. 避碰调整: 防止双足碰撞
  5. 地形适应: 根据地形高度调整步进命令

  替代引导模型

  - IPC3D控制器: 使用SDRE（状态相关Riccati方程）的高级非线性控制
  - Raibert启发式: 经典步进算法作为备用方案

  5. 强化学习算法

  训练运行器: learning/runners/on_policy_runner.py
  - OnPolicyRunner 类：主要训练协调器
  - 关键组件：
    - Actor-Critic网络：learning/modules/actor_critic.py
    - PPO算法：learning/algorithms/ppo.py
    - 经验存储：回合缓冲区管理

  神经网络架构: learning/modules/actor_critic.py
  - Actor：[256, 256, 256] 隐藏层，输出关节位置目标
  - Critic：[256, 256, 256] 隐藏层，估计价值函数
  - 激活函数：ELU
  - 观测维度：45维（从观测列表计算得出）
  - 动作维度：10维（关节位置目标）

  6. 训练循环

  主训练循环:
  def learn(self, num_learning_iterations=3000):
      for iteration in range(num_learning_iterations):
          # 1. 收集回合数据（每个环境24步）
          for step in range(num_steps_per_env):
              obs = env.get_observations()
              actions = policy.act(obs)
              env.step(actions)

          # 2. 计算优势和回报
          # 3. 使用PPO更新策略
          # 4. 记录指标并保存检查点

  环境步进:
  def step(self):
      # 1. 对关节施加扭矩
      # 2. 仿真物理（每个控制步骤10次物理步骤）
      # 3. 刷新传感器数据
      # 4. 更新机器人状态
      # 5. 计算奖励
      # 6. 检查终止条件

  7. 观测空间定义

  Actor观测（45维向量）：
  1. 基座高度（1维）：垂直位置
  2. 基座线速度（3维）：世界坐标系速度
  3. 基座航向角（1维）：方向角度
  4. 基座角速度（3维）：本体坐标系旋转
  5. 投影重力（3维）：基座坐标系中的重力
  6. 足部状态（8维）：左右足在基座坐标系中的位置和方向
  7. 步进命令（8维）：基座坐标系中的期望足部位置
  8. 命令（3维）：高层速度命令
  9. 相位信息（2维）：步态相位的正弦/余弦
  10. 关节状态（20维）：关节位置和速度

  8. 动作空间和扭矩计算

  动作空间: 10维关节位置目标
  - 关节：左右腿的髋关节（yaw/abad/pitch）、膝关节、踝关节

  扭矩计算:
  def _compute_torques(self):
      q = self.dof_pos.clone()  # 当前位置
      q_des = self.desired_pos_target.clone()  # 目标位置

      if self.cfg.asset.apply_humanoid_jacobian:
          torques = apply_coupling(q, qd, q_des, qd_des, kp, kd, tau_ff)
      else:
          torques = kp*(q_des - q) + kd*(qd_des - qd)  # PD控制

      return torch.clip(torques, -self.torque_limits, self.torque_limits)

  9. 奖励函数

  主要奖励组件：
  1. _reward_tracking_lin_vel_world(): 跟踪期望速度（权重：4.0）
  2. _reward_base_height(): 维持站立高度（权重：1.0）
  3. _reward_base_heading(): 跟踪航向方向（权重：3.0）
  4. _reward_contact_schedule(): 正确的足部接触时序（权重：3.0）
  5. _reward_joint_regularization(): 惩罚极端关节角度（权重：1.0）

  10. 高层规划与底层强化学习的关系

  分层控制结构：
  1. 高层：基于XCoM的步进命令生成
    - 输入：期望基座速度
    - 输出：最优足部着地位置
    - 算法：求解3D LIMP动力学进行捕获点控制
  2. 底层：步进执行的强化学习策略
    - 输入：当前状态 + 步进命令
    - 输出：关节位置目标
    - 算法：PPO学习最优关节协调

  集成方式：
  - 步进命令：每35-36个时间步生成一次（0.35-0.36秒）
  - 策略频率：每个时间步运行（0.01秒，100Hz）
  - 轨迹跟踪：强化学习策略学习跟踪生成的步进命令同时保持平衡

  11. 力推系统（v2.0功能）

  高级扰动训练：
  def _maybe_push_robot(self):
      # 基于课程的力施加
      if separate_xy_forces:
          force_x = self._compute_push_force_x()  # 独立X方向力
          force_y = self._compute_push_force_y()  # 独立Y方向力

      # 使用Isaac Gym API施加力
      self.gym.apply_rigid_body_force_at_pos_tensors(...)