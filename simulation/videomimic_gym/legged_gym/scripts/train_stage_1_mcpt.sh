#!/bin/bash

# =============================================================================
# VideoMimic 阶段1训练脚本 - MoCap预训练
# =============================================================================
# 此脚本用于训练VideoMimic的第一阶段：动作捕捉预训练
# 使用AMASS数据集进行深度模仿学习，为后续阶段奠定基础

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# 切换到项目根目录（向上两级）
cd "$SCRIPT_DIR/../.."

# =============================================================================
# 启动训练命令
# =============================================================================
torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--env.deepmimic.use_amass=True \
--multi_gpu \
--task=g1_deepmimic \
--headless \
--env.terrain.n_rows=16 \
--num_envs=4096 \
--wandb_note "videomimic_stage_1" \
--env.deepmimic.truncate_rollout_length=500 \
--env.noise.add_noise=True \
--env.deepmimic.link_pos_error_threshold=0.5 \
--env.rewards.scales.action_rate=-25.0 \
--env.deepmimic.amass_terrain_difficulty=2 \
--env.domain_rand.p_gain_rand=True \
--env.domain_rand.d_gain_rand=True \
--env.domain_rand.push_robots=True \
--env.domain_rand.control_delays=True \
--env.domain_rand.control_delay_min=0 \
--env.domain_rand.control_delay_max=5 \
--env.noise.offset_scales.gravity=0.02 \
--env.noise.offset_scales.dof_pos=0.005 \
--env.deepmimic.randomize_terrain_offset=True \
--env.asset.use_alt_files=True \
--env.noise.init_noise_scales.root_xy=0.1 \
--env.noise.init_noise_scales.root_z=0.02 \
--env.domain_rand.randomize_base_mass=True \
--env.asset.terminate_after_large_feet_contact_forces=True \
--env.noise.init_noise_scales.dof_pos=0.01

# =============================================================================
# 参数详细说明
# =============================================================================

# 基础训练配置 - 多GPU并行训练设置
# --nproc-per-node 2                    : 使用2个GPU进行并行训练
# --env.deepmimic.use_amass=True        : 启用AMASS数据集进行深度模仿学习
# --multi_gpu                           : 启用多GPU训练模式
# --task=g1_deepmimic                   : 使用G1机器人的深度模仿任务
# --headless                            : 无头模式运行（不显示图形界面，提高效率）

# 环境规模配置 - 大规模并行仿真
# --env.terrain.n_rows=16               : 地形网格行数，创建16x16的地形布局
# --num_envs=4096                       : 并行环境数量，4096个环境同时训练
# --wandb_note "videomimic_stage_1"     : Weights & Biases实验记录标签

# 训练控制参数 - Episode长度和数据处理
# --env.deepmimic.truncate_rollout_length=500 : 截断rollout长度，每个episode最多500步
# --env.noise.add_noise=True                   : 启用噪声添加，提高模型鲁棒性
# --env.deepmimic.link_pos_error_threshold=0.5 : 关节位置误差阈值，超过0.5米则重置

# 奖励函数调优 - 动作平滑性约束
# --env.rewards.scales.action_rate=-25.0        : 动作变化率惩罚权重，鼓励平滑动作

# AMASS数据集配置 - 地形难度设置
# --env.deepmimic.amass_terrain_difficulty=2    : AMASS地形难度等级，2级中等难度

# 域随机化配置 - 提高泛化能力和鲁棒性
# --env.domain_rand.p_gain_rand=True           : 随机化比例增益，模拟不同控制器参数
# --env.domain_rand.d_gain_rand=True           : 随机化微分增益，模拟不同控制器参数
# --env.domain_rand.push_robots=True           : 随机推动机器人，模拟外部干扰
# --env.domain_rand.control_delays=True         : 启用控制延迟，模拟真实控制延迟
# --env.domain_rand.control_delay_min=0         : 最小控制延迟0ms
# --env.domain_rand.control_delay_max=5         : 最大控制延迟5ms
# --env.domain_rand.randomize_base_mass=True    : 随机化基座质量，模拟负载变化

# 噪声配置 - 传感器噪声和初始化噪声
# --env.noise.offset_scales.gravity=0.02        : 重力噪声偏移2%，模拟重力变化
# --env.noise.offset_scales.dof_pos=0.005       : 关节位置噪声偏移0.5%，模拟传感器噪声
# --env.noise.init_noise_scales.root_xy=0.1     : 根部XY位置初始噪声10cm
# --env.noise.init_noise_scales.root_z=0.02      : 根部Z位置初始噪声2cm
# --env.noise.init_noise_scales.dof_pos=0.01    : 关节位置初始噪声1%

# 地形和资产配置 - 环境多样性
# --env.deepmimic.randomize_terrain_offset=True : 随机化地形偏移，增加地形多样性
# --env.asset.use_alt_files=True                : 使用备用资产文件（如球体碰撞几何）
# --env.asset.terminate_after_large_feet_contact_forces=True : 大脚部接触力时终止episode

# =============================================================================
# 训练目标说明
# =============================================================================
# 此阶段的核心目标：
# 1. 基础动作学习：使用AMASS数据集学习基本的人类动作模式
# 2. 深度模仿基础：建立深度模仿学习的基础模型，准确跟踪参考动作
# 3. 鲁棒性提升：通过域随机化和噪声注入提高泛化能力
# 4. 平滑性约束：通过动作平滑性惩罚确保自然运动
# 5. 地形适应：为后续的地形适应和强化学习阶段做准备
# 
# 关键训练特性：
# - 多GPU并行训练（2个GPU）提升训练效率
# - 大量并行环境（4096个）增加数据收集速度
# - 域随机化全面覆盖：控制器参数、外力、延迟、质量等
# - 多层次噪声注入：传感器噪声、初始化噪声、重力噪声
# - 动作平滑性约束确保自然运动模式
# - 地形多样性支持复杂场景训练
