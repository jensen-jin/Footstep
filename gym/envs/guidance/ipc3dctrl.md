  这个文件实现了一个基于**IPC3D（3D倒立摆控制）**算法的高级机器人运动控制系统。让我按照代码结构详细讲解：

  🎯 1. 总体架构概述

  设计理念：将人形机器人的平衡控制抽象为3D倒立摆系统
  - Cart（小车）：代表机器人的质心/身体
  - Pole（杆子）：代表机器人上半身的质量分布
  - 控制目标：通过控制"小车"运动来维持"杆子"平衡

  📊 2. 核心数据结构

● IPC3DParams参数说明：

  物理参数：
  - mass_cart = 42.0kg：机器人主体质量
  - mass_pole = 5.0kg：上半身分布质量
  - pole_length = 0.62m：重心高度
  - inertia = 0.1 kg⋅m²：转动惯量

  控制参数：
  - dt = 0.02s：控制周期（50Hz）
  - max_force = 500N：最大控制力
  - q_position/velocity：LQR权重矩阵
  - control_mode = 1：速度控制模式

  🧮 3. 数学基础：NonlinearController

● Read(ipc3d_controller.py)
  ⎿  Read 30 lines (ctrl+r to expand)

● SDRE（状态相关Riccati方程）方法：

  系统方程：
  D*ddx + C*dx + G*x = H*u
  - D：质量矩阵（Mass matrix）
  - C：阻尼矩阵（Damping matrix）
  - G：重力/刚度矩阵（Gravity matrix）
  - H：控制输入矩阵
  - u：控制力

  状态空间变换：
  [x]   →   [x ]     [0  I] [x ]   [0    ]
  [dx]      [ddx] = [-D⁻¹G -D⁻¹C][dx] + [D⁻¹H] u

● Read(ipc3d_controller.py)
  ⎿  Read 25 lines (ctrl+r to expand)

● 关键数值稳定性处理：

  1. 矩阵求逆保护：D矩阵奇异时使用伪逆
  2. 条件数检查：A矩阵条件数>1e12时添加正则化
  3. 位置状态正则化：添加小阻尼项避免不可控性

  🎛️ 4. 2D倒立摆控制器：IPCController

● Read(ipc3d_controller.py)
  ⎿  Read 30 lines (ctrl+r to expand)

● 2D倒立摆物理模型：

  状态变量 x = [cart_pos, pole_angle]
  控制输入 u = [horizontal_force]

● Read(ipc3d_controller.py)
  ⎿  Read 30 lines (ctrl+r to expand)

● 倒立摆系统矩阵：

  质量矩阵D：
  D = [M+m        -ml*cos(θ)]
      [-ml*cos(θ)  I+ml²   ]

  阻尼矩阵C：
  C = [b           ml*θ̇*sin(θ)]
      [0           0         ]

● Read(ipc3d_controller.py)
  ⎿  Read 20 lines (ctrl+r to expand)

● 重力矩阵G：
  G = [0                    0]
      [0  -mgl*sin(θ)]  ← 重力恢复力

  关键数值处理：
  - 奇异性处理：θ≈0时，sin(θ)≈0导致系统不可控
  - 正则化方案：使用小角度近似sin(θ)≈θ，确保最小值1e-4
  - 物理意义：避免完全直立时的控制盲点

  🎮 5. 单轴控制器：IPCView

● Read(ipc3d_controller.py)
  ⎿  Read 30 lines (ctrl+r to expand)

● IPCView设计思想：
  - 状态向量：[位置, 角度, 速度, 角速度] (4维)
  - 控制目标：位置控制 OR 速度控制
  - LQR求解：最优控制增益矩阵K

● Read(ipc3d_controller.py)
  ⎿  Read 25 lines (ctrl+r to expand)

● LQR权重矩阵设计：

  位置控制模式（mode=0）：
  Q = diag([q, 0, qd, 0])  ← 重点控制位置和速度

  速度控制模式（mode=1）：
  Q = diag([0, q, qd, 0])  ← 重点控制角度和速度

  控制律：u = -K*(x - xd)，其中K通过连续时间Riccati方程求解

● Read(ipc3d_controller.py)
  ⎿  Read 25 lines (ctrl+r to expand)

● LQR求解算法：
  1. 连续时间ARE：ATP + PA - PBR⁻¹BᵀP + Q = 0
  2. 最优增益：K = R⁻¹BᵀP
  3. 异常处理：求解失败时返回零矩阵并详细诊断

  🚀 6. 3D双轴控制器：IPC3D

● Read(ipc3d_controller.py)
  ⎿  Read 30 lines (ctrl+r to expand)

● IPC3D核心设计：

  解耦控制策略：
  - X轴控制器：控制前后运动（pitch平衡）
  - Z轴控制器：控制左右运动（roll平衡）
  - 独立运行：两个2D倒立摆并行工作

  物理解释：
  - 将3D平衡问题分解为两个2D问题
  - X轴：sagittal平面的前后平衡
  - Z轴：frontal平面的左右平衡

● Read(ipc3d_controller.py)
  ⎿  Read 25 lines (ctrl+r to expand)

● 接口设计：
  - set_desired_velocity(vx, vz)：设置X/Z向目标速度
  - step(n_times)：执行n步仿真
  - Y轴（垂直）暂不控制，假设固定高度

● Read(ipc3d_controller.py)
  ⎿  Read 25 lines (ctrl+r to expand)

● 状态输出：
  state = {
      'x_cart_pos': X轴位置,     'z_cart_pos': Z轴位置,
      'x_pole_angle': X轴角度,   'z_pole_angle': Z轴角度,
      'x_cart_vel': X轴速度,     'z_cart_vel': Z轴速度,
      'x_pole_vel': X轴角速度,   'z_pole_vel': Z轴角速度,
      'control_x': X轴控制力,    'control_z': Z轴控制力
  }

  🧪 7. 测试代码分析

● Read(ipc3d_controller.py)
  ⎿  Read 11 lines (ctrl+r to expand)

● 测试流程：
  1. 创建默认参数的IPC3D控制器
  2. 设置目标速度：X=1.0m/s, Z=0.5m/s
  3. 运行100步仿真（2秒）
  4. 每20步输出状态和控制力

  🎯 8. 核心算法流程总结

  完整控制流程：

  1. 物理建模：
  机器人 → 3D倒立摆 → 两个2D倒立摆
  2. 实时SDRE：
  当前状态(θ,θ̇) → 更新D,C,G矩阵 → 线性化A,B矩阵
  3. LQR求解：
  solve_ARE(A,B,Q,R) → 最优增益K
  4. 控制执行：
  误差e = x - xd → 控制力u = -K*e → 系统积分
  5. 双轴协调：
  X轴控制器 || Z轴控制器 → 合成3D运动

  关键技术创新：

  - 奇异性处理：近垂直正则化避免θ=0时不可控
  - 数值稳定性：矩阵条件数检查和自适应正则化
  - 解耦设计：3D→2D降维简化控制复杂度
  - 实时优化：SDRE+LQR实现最优非线性控制

