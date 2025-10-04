#!/bin/bash

# =============================================================================
# VideoMimic 阶段2训练脚本 - 地形适应训练
# =============================================================================
# 此脚本用于训练VideoMimic的第二阶段：地形适应训练（Terrain Tracking）
# 在阶段1的基础上增加地形感知能力，使用人类视频数据
# 从阶段1的检查点继续训练

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# 切换到项目根目录（向上两级）
cd "$SCRIPT_DIR/../.."

# =============================================================================
# 阶段1检查点配置
# =============================================================================
# 指定要加载的阶段1训练检查点（MCPT预训练的结果）
LOAD_RUN=20250410_063030_g1_deepmimic
# 格式：日期时间_任务名
# 需要替换为你实际的阶段1训练输出目录名

# =============================================================================
# 启动训练命令
# =============================================================================
torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--task=g1_deepmimic_proj_heightfield \
--multi_gpu \
--headless \
--env.terrain.n_rows=1 \
--num_envs=4096 \
--wandb_note "videomimic_stage_2" \
--env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml \
--load_run ${LOAD_RUN} \
--resume \
--train.policy.re_init_std=True \
--train.policy.init_noise_std=0.5 \
--train.algorithm.learning_rate=2e-5 \
--train.algorithm.schedule=fixed \
--env.deepmimic.amass_terrain_difficulty=1 \
--env.deepmimic.upsample_data=True \
--env.deepmimic.use_human_videos=True \
--env.deepmimic.link_pos_error_threshold=0.5 \
--train.runner.save_interval=500 \
--env.deepmimic.respawn_z_offset=0.1 \
--env.deepmimic.randomize_terrain_offset=False \
--env.terrain.cast_mesh_to_heightfield=False \
--env.deepmimic.truncate_rollout_length=500 \
--train.runner.load_model_strict=False \
--env.deepmimic.use_amass=False \
--env.rewards.scales.termination=-2000 \
--env.rewards.scales.alive=200.0 \
--env.rewards.scales.ankle_action=-3.0 \
--env.rewards.scales.action_rate=-3.0

# =============================================================================
# 参数详细说明
# =============================================================================

# 基础训练配置 - 多GPU和任务设置
# --nproc-per-node 2                                : 使用2个GPU进行并行训练
# --task=g1_deepmimic_proj_heightfield              : 使用投影地形高度场任务（阶段2专用）
# --multi_gpu                                       : 启用多GPU训练模式
# --headless                                        : 无头模式运行（不显示图形界面）

# 环境配置
# --env.terrain.n_rows=1                            : 地形网格行数为1（人类视频已有地形）
# --num_envs=4096                                   : 并行环境数量4096个
# --wandb_note "videomimic_stage_2"                 : Weights & Biases实验标签

# 数据源配置 - 从AMASS切换到人类视频
# --env.deepmimic.human_motion_source=...           : 指定人类动作数据源YAML文件
#                                                     包含123个人类视频动作
# --env.deepmimic.use_amass=False                   : 禁用AMASS数据（阶段1用的）
# --env.deepmimic.use_human_videos=True             : 启用人类视频数据（阶段2核心）
# --env.deepmimic.upsample_data=True                : 上采样数据以匹配仿真频率

# 检查点加载配置 - 从阶段1继续训练
# --load_run ${LOAD_RUN}                            : 加载阶段1的检查点
# --resume                                          : 恢复训练（继续之前的迭代计数）
# --train.runner.load_model_strict=False            : 非严格加载（允许网络结构变化）
#                                                     因为阶段2增加了地形嵌入层

# 策略网络重新初始化 - 探索新任务
# --train.policy.re_init_std=True                   : 重新初始化策略噪声标准差
# --train.policy.init_noise_std=0.5                 : 初始噪声标准差0.5（鼓励探索）
#                                                     帮助策略适应新的地形感知输入

# 训练算法配置
# --train.algorithm.learning_rate=2e-5              : 学习率2e-5（较低，微调）
#                                                     阶段1是1e-3，这里降低100倍
# --train.algorithm.schedule=fixed                  : 固定学习率（不自适应调整）

# 地形配置
# --env.deepmimic.amass_terrain_difficulty=1        : AMASS地形难度1（虽然不用AMASS了）
# --env.terrain.cast_mesh_to_heightfield=False      : 不将网格转换为高度场
#                                                     保持原始三角网格（更精确）
# --env.deepmimic.randomize_terrain_offset=False    : 不随机化地形偏移
#                                                     人类视频的地形已对齐

# 终止和重置配置
# --env.deepmimic.link_pos_error_threshold=0.5      : 关节位置误差阈值0.5米
#                                                     相比阶段1的0.5米，保持一致
# --env.deepmimic.truncate_rollout_length=500       : 截断rollout长度500步
# --env.deepmimic.respawn_z_offset=0.1              : 重生时垂直偏移0.1米
#                                                     防止脚部与地面相交

# 训练控制
# --train.runner.save_interval=500                  : 每500次迭代保存一次检查点

# 奖励函数调整 - 适应地形任务
# --env.rewards.scales.termination=-2000            : 终止惩罚-2000（严重惩罚失败）
#                                                     相比阶段1的-500，增加4倍
#                                                     因为地形任务更难，要更严格
# --env.rewards.scales.alive=200.0                  : 存活奖励200（鼓励持续存活）
#                                                     阶段1是0.0，阶段2增加存活奖励
# --env.rewards.scales.ankle_action=-3.0            : 踝关节动作惩罚-3.0
#                                                     限制踝关节过度动作
# --env.rewards.scales.action_rate=-3.0             : 动作变化率惩罚-3.0
#                                                     相比阶段1的-25.0，降低惩罚
#                                                     允许更快的动作变化以适应地形

# =============================================================================
# 训练目标说明
# =============================================================================
# 此阶段的核心目标：
# 1. 地形感知：增加地形高度场输入，学习适应复杂地形
# 2. 保持跟踪：继续跟踪参考动作，但在地形上执行
# 3. 人类视频：使用真实的人类在地形上的运动数据
# 4. 鲁棒性：通过地形多样性提高策略的泛化能力
# 5. 为蒸馏准备：训练好的策略将作为阶段3的教师
# 
# 关键训练特性：
# - 从阶段1检查点继续（迁移学习）
# - 增加地形感知能力（投影地形嵌入）
# - 使用人类视频数据（123个复杂动作）
# - 降低学习率（微调而非从头训练）
# - 调整奖励权重（适应地形挑战）
# - 策略噪声重初始化（探索地形适应）

# =============================================================================
# 与阶段1的主要区别
# =============================================================================
# 1. 任务配置：
#    - 阶段1: g1_deepmimic（无地形感知）
#    - 阶段2: g1_deepmimic_proj_heightfield（有地形感知）
# 
# 2. 数据源：
#    - 阶段1: AMASS动作捕捉数据（标准动作）
#    - 阶段2: 人类视频数据（真实场景动作 + 地形）
# 
# 3. 观测空间：
#    - 阶段1: 本体感知 + 根方向 + 参考关节（无地形）
#    - 阶段2: 本体感知 + 根方向 + 参考关节 + 地形高度场
# 
# 4. 学习率：
#    - 阶段1: 1e-3（从头训练）
#    - 阶段2: 2e-5（微调）
# 
# 5. 奖励权重：
#    - termination: -500 → -2000（更严格）
#    - alive: 0.0 → 200.0（新增存活奖励）
#    - action_rate: -25.0 → -3.0（更宽松，允许快速调整）

# =============================================================================
# 预期输出
# =============================================================================
# 训练完成后，检查点将保存在：
# logs/g1_deepmimic/YYYYMMDD_HHMMSS_g1_deepmimic_proj_heightfield/
# 
# 这个检查点将作为阶段3蒸馏的教师策略使用
、






# =============================================================================
# 如何使用自己的地形和动作数据
# =============================================================================

# 步骤1：准备数据目录结构
# ----------------------------------------
# 你的数据应该按以下结构组织：
# 
# ../data/videomimic_captures/
# └── your_motion_name/
#     ├── retarget_poses_g1.h5        # 动作数据（H5格式）
#     └── background_mesh.obj         # 地形网格（OBJ格式）
#
# 或者使用PKL格式：
# ../data/videomimic_captures/
# └── your_motion_name/
#     ├── motion_data.pkl             # 动作数据（PKL格式）
#     └── background_mesh.obj         # 地形网格
#
# 注意：
# - 每个动作文件夹必须同时包含动作数据和地形网格
# - 地形网格必须是OBJ格式的三角网格
# - 动作数据可以是H5或PKL格式

# 步骤2：动作数据格式要求
# ----------------------------------------
# H5格式必须包含以下字段（精确维度要求）：
# ============================================================
# 
# 【必需字段】
# 
# 1. 'root_pos': numpy.ndarray, shape=(N, 3), dtype=float32 或 float64
#    - N: 动作帧数（例如300帧）
#    - 3: [X, Y, Z] 世界坐标系下的位置（单位：米）
#    - 示例：shape=(300, 3)，表示300帧动作，每帧有XYZ三个坐标
#    - 数值范围：通常 X,Y ∈ [-10, 10], Z ∈ [0, 2]（地面以上）
#
# 2. 'root_quat': numpy.ndarray, shape=(N, 4), dtype=float32 或 float64
#    - N: 动作帧数（与root_pos一致）
#    - 4: 四元数 [W, X, Y, Z]（WXYZ格式，注意不是XYZW！）
#    - 示例：shape=(300, 4)
#    - 数值要求：必须是单位四元数（模长=1），即 W²+X²+Y²+Z²=1
#    - 坐标系：Z轴向上，右手坐标系
#
# 3. 'joints': numpy.ndarray, shape=(N, 23), dtype=float32 或 float64
#    - N: 动作帧数（与root_pos一致）
#    - 23: G1机器人的23个关节角度（单位：弧度）
#    - 示例：shape=(300, 23)
#    - 数值范围：每个关节 ∈ [-π, π]（-3.14159 到 3.14159）
#    - 关节顺序（必须严格按照此顺序）：
#      索引 0-5:   左腿（left_hip_yaw, left_hip_roll, left_hip_pitch, 
#                        left_knee, left_ankle_pitch, left_ankle_roll）
#      索引 6-11:  右腿（right_hip_yaw, right_hip_roll, right_hip_pitch,
#                        right_knee, right_ankle_pitch, right_ankle_roll）
#      索引 12:    躯干（torso）
#      索引 13-15: 腰部（waist_yaw, waist_pitch, waist_roll）
#      索引 16-19: 左臂（left_shoulder_pitch, left_shoulder_roll, 
#                        left_shoulder_yaw, left_elbow）
#      索引 20-22: 右臂（right_shoulder_pitch, right_shoulder_roll,
#                        right_shoulder_yaw, right_elbow）
#
# 【可选但强烈推荐的字段】
#
# 4. 'link_pos': numpy.ndarray, shape=(N, num_links, 3), dtype=float32 或 float64
#    - N: 动作帧数
#    - num_links: 跟踪的身体部位数量（通常13个）
#    - 3: [X, Y, Z] 世界坐标下的位置（单位：米）
#    - 示例：shape=(300, 13, 3)
#    - 如果不提供，训练时link_pos_tracking奖励会失效（但不会报错）
#    - 13个跟踪部位（按顺序）：
#      0: pelvis（骨盆）
#      1: left_hip_pitch_link（左髋俯仰）
#      2: left_knee_link（左膝）
#      3: left_ankle_roll_link（左踝滚转）
#      4: right_hip_pitch_link（右髋俯仰）
#      5: right_knee_link（右膝）
#      6: right_ankle_roll_link（右踝滚转）
#      7: left_shoulder_pitch_link（左肩俯仰）
#      8: left_elbow_link（左肘）
#      9: left_wrist_yaw_link（左腕偏航）
#      10: right_shoulder_pitch_link（右肩俯仰）
#      11: right_elbow_link（右肘）
#      12: right_wrist_yaw_link（右腕偏航）
#
# 5. 'link_quat': numpy.ndarray, shape=(N, num_links, 4), dtype=float32 或 float64
#    - N: 动作帧数
#    - num_links: 身体部位数量（与link_pos一致，通常13个）
#    - 4: 四元数 [W, X, Y, Z]（WXYZ格式）
#    - 示例：shape=(300, 13, 4)
#    - 每个四元数必须是单位四元数（模长=1）
#    - 如果不提供，link方向相关的奖励会失效
#
# 6. 'contacts': 字典（dict），包含接触信息
#    - 'left_foot': numpy.ndarray, shape=(N,), dtype=bool 或 int 或 float
#      * N: 动作帧数
#      * 值: 0=不接触地面，1=接触地面
#      * 示例：shape=(300,)，表示300帧中每帧左脚是否接触
#    - 'right_foot': numpy.ndarray, shape=(N,), dtype=bool 或 int 或 float
#      * N: 动作帧数
#      * 值: 0=不接触，1=接触
#      * 示例：shape=(300,)
#    - 如果不提供，feet_contact_matching奖励会失效（但不影响训练）
#
# 【H5属性（attributes）】
#
# 7. '/joint_names': 字符串列表或numpy数组（存储为H5属性）
#    - 类型：list[str] 或 numpy.ndarray of strings
#    - 长度：23（与joints的第二维度一致）
#    - 示例：['left_hip_yaw_joint', 'left_hip_roll_joint', ...]
#    - 必须按照上面joints中定义的顺序
#    - 在H5中存储方式：data.attrs['/joint_names'] = joint_names_list
#
# 8. '/link_names': 字符串列表（存储为H5属性）
#    - 类型：list[str] 或 numpy.ndarray of strings
#    - 长度：13（如果有link_pos/link_quat的话）
#    - 示例：['pelvis', 'left_hip_pitch_link', 'left_knee_link', ...]
#    - 必须按照上面link_pos中定义的顺序
#    - 在H5中存储方式：data.attrs['/link_names'] = link_names_list
#
# 9. '/fps': 浮点数（存储为H5属性）
#    - 类型：float 或 int
#    - 示例值：30.0, 60.0
#    - 表示动作数据的帧率（帧/秒）
#    - 在H5中存储方式：data.attrs['/fps'] = 30.0
#    - 如果不提供，会使用default_data_fps_override或default_data_fps
#

# 步骤3：地形网格格式要求
# ----------------------------------------
# - 格式：Wavefront OBJ格式（.obj文件）
# - 坐标系：Z轴向上，与动作数据对齐
# - 单位：米（与动作数据一致）
# - 建议：使用三角化网格，确保没有退化三角形
# - 地形原点应与动作起始位置对齐

# 步骤4：创建YAML配置文件
# ----------------------------------------
# 在 resources/data_config/ 创建你的配置文件，例如 my_motions.yaml：
#
# - folder_path: "your_motion_name"                    # 数据文件夹名（相对于data_root）
#   teacher_checkpoint_run_name: "20250410_063030_g1_deepmimic"  # 使用阶段1检查点
#   human_video_data_pattern: "retarget_poses_g1.h5"   # 动作文件名
#   human_video_terrain_pattern: "background_mesh.obj" # 地形文件名
#   default_data_fps_override: 60                      # 数据帧率（如果H5中没有）
#
# - folder_path: "another_motion"                      # 可以添加多个动作
#   teacher_checkpoint_run_name: "20250410_063030_g1_deepmimic"
#   human_video_data_pattern: "retarget_poses_g1.h5"
#   human_video_terrain_pattern: "background_mesh.obj"
#   default_data_fps_override: 30
#
# 字段说明：
# - folder_path: 数据文件夹名（必须）
# - teacher_checkpoint_run_name: 教师检查点（可选，用于多教师蒸馏）
# - human_video_data_pattern: 动作文件名（可选，默认retarget_poses_g1.h5）
# - human_video_terrain_pattern: 地形文件名（可选，默认background_mesh.obj）
# - default_data_fps_override: 强制指定帧率（可选）

# 步骤5：修改训练脚本使用自己的数据
# ----------------------------------------
# 将第33行的YAML文件路径改为你的配置文件：
# --env.deepmimic.human_motion_source=resources/data_config/my_motions.yaml \
#
# 如果数据在不同目录，还需要修改数据根目录：
# --env.deepmimic.data_root=../data/my_custom_data \
#
# 如果你的数据帧率与默认不同，可以设置：
# --env.deepmimic.default_data_fps=30 \

# =============================================================================
# 如何可视化训练和推理
# =============================================================================

# 方法1：训练时可视化（不推荐，会降低训练速度）
# ----------------------------------------
# 移除训练脚本中的 --headless 参数
# 这会启动Isaac Gym的图形界面显示训练过程
# 
# 注意：
# - 训练速度会大幅降低（约10-100倍）
# - 只建议用于调试少量环境（如--num_envs=16）
# - 无法在远程服务器上使用（除非有X11转发）

# 方法2：推理时可视化 - Viser Web界面（推荐）★
# ----------------------------------------
# Viser是一个Web可视化工具，默认已集成在代码中
# 
# 使用play.py脚本进行可视化推理：
# 
# cd simulation/videomimic_gym
# python legged_gym/scripts/play.py \
#   --task=g1_deepmimic_proj_heightfield \
#   --load_run=YYYYMMDD_HHMMSS_g1_deepmimic_proj_heightfield \
#   --checkpoint=5000 \
#   --num_envs=1 \
#   --env.viser.enable=True \
#   --headless \
#   --env.deepmimic.use_human_videos=True \
#   --env.deepmimic.human_motion_source=resources/data_config/my_motions.yaml
#
# 参数说明：
# - --task: 使用阶段2的任务配置
# - --load_run: 检查点目录名（logs/g1_deepmimic/下的文件夹）
# - --checkpoint: 检查点编号（如5000表示model_5000.pt）
# - --num_envs: 可视化环境数量（建议1个，多了会卡）
# - --env.viser.enable=True: ★启用Viser可视化（关键参数）
# - --headless: 不启动Isaac Gym图形界面（Viser足够了）
# - --env.deepmimic.use_human_videos: 使用人类视频数据
# - --env.deepmimic.human_motion_source: 指定你的YAML配置文件
#
# 启动后：
# 1. 终端会显示："Viser server started at http://localhost:8080"
# 2. 打开浏览器访问 http://localhost:8080
# 3. 你会看到一个3D交互界面，可以：
#    - 旋转、缩放、平移视角
#    - 播放/暂停仿真（Play/Pause按钮）
#    - 切换显示机器人/地形/关键点
#    - 查看机器人关节状态
#    - 导出训练好的策略（Export按钮）
#
# 如果在远程服务器上：
# 在本地电脑运行SSH隧道转发：
# ssh -L 8080:localhost:8080 user@remote_server
# 然后在本地浏览器访问 http://localhost:8080

# 方法3：推理时可视化 - Isaac Gym图形界面
# ----------------------------------------
# 如果你有本地GPU和显示器，可以使用Isaac Gym的图形界面：
# 
# python legged_gym/scripts/play.py \
#   --task=g1_deepmimic_proj_heightfield \
#   --load_run=YYYYMMDD_HHMMSS_g1_deepmimic_proj_heightfield \
#   --num_envs=1 \
#   --env.deepmimic.use_human_videos=True \
#   --env.deepmimic.human_motion_source=resources/data_config/my_motions.yaml
#
# 注意：移除 --headless 和 --env.viser.enable 参数
# 
# 控制：
# - 鼠标拖拽：旋转视角
# - 鼠标滚轮：缩放
# - WASD键：移动相机
# - V键：切换相机模式

# 方法4：可视化自己的动作数据（不训练）
# ----------------------------------------
# 如果只是想查看自己的动作数据是否正确加载：
#
# python legged_gym/scripts/play.py \
#   --task=g1_deepmimic_proj_heightfield \
#   --load_run=20250410_063030_g1_deepmimic \
#   --num_envs=1 \
#   --env.viser.enable=True \
#   --headless \
#   --env.deepmimic.use_human_videos=True \
#   --env.deepmimic.human_motion_source=resources/data_config/my_motions.yaml
#
# 这会使用阶段1的检查点（纯MCPT，没有地形适应）
# 可以检查：
# - 动作数据是否正确加载
# - 地形网格是否正确显示
# - 机器人初始位置是否与地形对齐
# - 参考动作轨迹是否合理

# 方法5：导出可视化视频
# ----------------------------------------
# play.py会自动录制视频（如果启用）
# 视频保存在：logs/.../videos/
#
# 或者使用Viser的录制功能：
# 在Viser Web界面中点击"Record"按钮即可录制

# =============================================================================
# 常见问题排查
# =============================================================================

# 问题1：地形和机器人位置不对齐
# ----------------------------------------
# 原因：动作数据的坐标原点与地形网格不一致
# 解决：
# - 检查动作数据的root_pos起始位置
# - 调整地形网格的位置，或使用--env.deepmimic.respawn_z_offset调整

# 问题2：Viser界面打不开
# ----------------------------------------
# 原因：端口被占用或防火墙阻止
# 解决：
# - 检查8080端口是否被占用：lsof -i:8080
# - 更换端口：--env.viser.port=8081
# - 检查防火墙设置

# 问题3：数据加载失败
# ----------------------------------------
# 原因：数据格式不正确或路径错误
# 解决：
# - 检查YAML文件中的folder_path是否正确
# - 确认H5/PKL文件包含所有必需字段
# - 查看终端输出的错误信息

# 问题4：机器人穿透地形或悬空
# ----------------------------------------
# 原因：地形网格质量问题或物理参数不匹配
# 解决：
# - 检查地形网格是否有退化三角形
# - 调整--env.deepmimic.respawn_z_offset参数
# - 确认地形网格坐标系正确（Z轴向上）

# 问题5：训练不收敛或表现很差
# ----------------------------------------
# 原因：动作数据质量问题或参数设置不当
# 解决：
# - 确认动作数据的关节角度在合理范围内（-π到π）
# - 检查动作数据的帧率设置是否正确
# - 适当降低--env.deepmimic.link_pos_error_threshold
# - 增加--num_envs提高数据收集效率

# =============================================================================
# 完整示例：使用自己数据的训练命令
# =============================================================================
# 
# # 1. 准备数据
# mkdir -p ../data/my_custom_motions
# # 将你的动作和地形放入：
# # ../data/my_custom_motions/walking/
# #   ├── retarget_poses_g1.h5
# #   └── background_mesh.obj
# 
# # 2. 创建YAML配置
# cat > resources/data_config/my_config.yaml << 'EOF'
# - folder_path: "walking"
#   teacher_checkpoint_run_name: "20250410_063030_g1_deepmimic"
#   human_video_data_pattern: "retarget_poses_g1.h5"
#   human_video_terrain_pattern: "background_mesh.obj"
#   default_data_fps_override: 30
# EOF
# 
# # 3. 修改训练脚本或直接运行命令
# torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
#   --task=g1_deepmimic_proj_heightfield \
#   --multi_gpu \
#   --headless \
#   --num_envs=4096 \
#   --load_run=20250410_063030_g1_deepmimic \
#   --resume \
#   --env.deepmimic.use_human_videos=True \
#   --env.deepmimic.use_amass=False \
#   --env.deepmimic.human_motion_source=resources/data_config/my_config.yaml \
#   --env.deepmimic.data_root=../data/my_custom_motions \
#   --env.deepmimic.default_data_fps=30 \
#   --train.algorithm.learning_rate=2e-5
# 
# # 4. 可视化训练结果
# python legged_gym/scripts/play.py \
#   --task=g1_deepmimic_proj_heightfield \
#   --load_run=YYYYMMDD_HHMMSS_g1_deepmimic_proj_heightfield \
#   --num_envs=1 \
#   --env.viser.enable=True \
#   --headless \
#   --env.deepmimic.use_human_videos=True \
#   --env.deepmimic.human_motion_source=resources/data_config/my_config.yaml \
#   --env.deepmimic.data_root=../data/my_custom_motions