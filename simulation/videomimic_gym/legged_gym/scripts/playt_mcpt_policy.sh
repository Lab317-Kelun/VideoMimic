#!/bin/bash

# =============================================================================
# VideoMimic MCPT策略推理脚本
# =============================================================================
# 此脚本用于运行VideoMimic的MCPT（Motion Capture Pre-training）策略推理
# 加载预训练的模型，在仿真环境中展示机器人动作

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# 切换到项目根目录（向上两级）
cd "$SCRIPT_DIR/../.."

# =============================================================================
# 启动推理命令
# =============================================================================
python legged_gym/scripts/play.py \
--task=g1_deepmimic \
--env.deepmimic.use_amass=True \
--env.deepmimic.amass_terrain_difficulty=1 \
--env.deepmimic.amass_replay_data_path="lafan_single_walk/*.pkl" \
--load_run 20250410_063030_g1_deepmimic \
--num_envs 1 \
--env.deepmimic.use_human_videos=False \
--headless \
--env.deepmimic.respawn_z_offset=0.1 \
--env.deepmimic.link_pos_error_threshold=10.0 \
--env.viser.enable=True

# =============================================================================
# 参数详细说明
# =============================================================================

# 基础任务配置
# --task=g1_deepmimic                           : 使用G1机器人的深度模仿任务
# --env.deepmimic.use_amass=True                : 启用AMASS数据集进行推理

# AMASS数据集配置
# --env.deepmimic.amass_terrain_difficulty=1     : AMASS地形难度等级，1级简单难度
# --env.deepmimic.amass_replay_data_path="lafan_single_walk/*.pkl" : 使用Lafan单步行走数据

# 模型加载配置
# --load_run 20250410_063030_g1_deepmimic       : 加载指定时间戳的训练模型

# 环境配置
# --num_envs 1                                  : 只运行1个环境进行推理
# --env.deepmimic.use_human_videos=False        : 不使用人类视频输入
# --headless                                     : 无头模式运行（不显示图形界面）

# 重置和容错配置
# --env.deepmimic.respawn_z_offset=0.1          : 重生时Z轴偏移0.1米
# --env.deepmimic.link_pos_error_threshold=10.0 : 关节位置误差阈值10米（推理时放宽）

# 可视化配置
# --env.viser.enable=True                        : 启用Viser可视化界面（localhost:8080）

# =============================================================================
# 推理说明
# =============================================================================
# 此脚本的功能：
# 1. 加载预训练的MCPT模型
# 2. 在仿真环境中运行机器人
# 3. 展示学习到的人类动作模式
# 4. 通过Viser界面可视化结果
# 
# 使用步骤：
# 1. 确保已训练好模型并保存
# 2. 修改 --load_run 参数为实际的模型路径
# 3. 运行脚本
# 4. 在浏览器中访问 localhost:8080 查看可视化结果
# 
# 注意事项：
# - 确保模型文件存在
# - 推理时使用单个环境
# - 可以调整地形难度和数据集路径