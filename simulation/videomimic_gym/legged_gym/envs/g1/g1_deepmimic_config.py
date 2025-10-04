# 文件路径: /home/asus/VideoMimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py
"""
G1机器人深度模仿学习配置文件

此文件定义了VideoMimic预训练阶段1的所有配置参数,包括  
1. 深度模仿学习的核心参数
2. 机器人资产和控制参数  
3. 地形和环境配置
4. 奖励函数权重
5. 观察处理和策略网络结构
6. 训练算法参数

配置结构采用数据类(dataclass)组织,支持继承和覆盖
"""

from legged_gym.utils.deepmimic_terrain import DeepMimicTerrain
import numpy as np
from dataclasses import MISSING
import torch
from typing import Union, List

from legged_gym.utils.configclass import configclass

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotTerrainCfg,
    LeggedRobotInitStateCfg,
    LeggedRobotEnvCfg,
    LeggedRobotDomainRandCfg,
    LeggedRobotControlCfg,
    LeggedRobotAssetCfg,
    LeggedRobotRewardsCfg,
    LeggedRobotNormalizationCfg,
    LeggedRobotNoiseCfg,
    LeggedRobotSimCfg,
    LeggedRobotCommandsCfg,
    LeggedRobotCfg,
    LeggedRobotPolicyCfg,
    LeggedRobotAlgorithmCfg,
    LeggedRobotRunnerCfg,
    LeggedRobotCfgPPO,
    LeggedRobotSensorsCfg,
    DepthCameraCfg,
    HeightfieldCfg,
    MultiLinkHeightCfg,
)

@configclass
class LeggedRobotDeepMimicCfg:
    """
    深度模仿学习核心配置类
    
    定义了深度模仿学习的关键参数,控制数据加载、动作跟踪、
    成功率评估和自适应权重调整等核心功能
    """
    
    # === 初始化和重置策略 ===
    init_velocities = True              # 是否从参考动作初始化速度 ( 有助于更好的跟踪)
    randomize_start_offset = True       # 是否随机化起始位置偏移 ( 增加数据多样性)
    n_prepend = 0                       # 在动作序列前添加的冻结帧数
    n_append = 0                        # 在动作序列后添加的冻结帧数 ( 用于稳定结束姿态)
    respawn_z_offset = 0.0              # 重生时的垂直偏移 ( 如果脚部与地面相交可以调整)
    height_direct_offset = 0.0          # 高度直接偏移
    
    # === 终止条件 ===
    link_pos_error_threshold = 0.3      # 关节位置误差阈值(米),超过此值episode终止
    viz_replay = False                  # 是否可视化重放数据
    viz_replay_sync_robot = False       # 是否同步机器人与重放数据
    num_next_obs = 1                    # 下一个观察的数量
    
    # === 数据处理参数 ===
    truncate_rollout_length = -1        # 截断rollout长度,用于处理超长序列 ( -1表示不截断)
    upsample_data = True                # 是否上采样数据以匹配仿真频率
    default_data_fps = -1               # 默认数据帧率 ( -1表示自动检测)

    # === 接触检测配置 ===
    contact_names = ['left_foot', 'right_foot']  # 用于接触检测的身体部位名称

    # === 跟踪的身体部位 ===
    # 这些身体部位将被用于位置跟踪奖励计算
    tracked_body_names = [
        'pelvis',                    # 骨盆 ( 核心)
        
        # 左腿关键关节
        'left_hip_pitch_link',       # 左髋关节俯仰
        'left_knee_link',            # 左膝关节
        'left_ankle_roll_link',      # 左踝关节滚动
        
        # 右腿关键关节
        'right_hip_pitch_link',      # 右髋关节俯仰
        'right_knee_link',           # 右膝关节
        'right_ankle_roll_link',     # 右踝关节滚动
        
        # 上身关键关节
        'left_shoulder_pitch_link',  # 左肩关节俯仰
        'right_shoulder_pitch_link', # 右肩关节俯仰
    ]

    # === 地形随机化 ===
    randomize_terrain_offset = False    # 是否随机化地形偏移
    randomize_terrain_offset_range = 1.0 # 地形偏移随机化范围

    # === 片段权重策略 ===
    # 控制如何为不同动作片段分配训练权重
    clip_weighting_strategy: str = 'success_rate_adaptive'  # 成功率自适应权重
    # 可选值:
    # - 'uniform_step': 每个步骤等概率
    # - 'uniform_clip': 每个片段等概率
    # - 'success_rate_adaptive': 根据成功率反比例分配权重
    
    # === 自适应权重参数 ===
    min_success_rate_weight_factor: float = 1.0 / 3.0  # 最小成功率权重因子
    max_success_rate_weight_factor: float = 3.0        # 最大成功率权重因子
    adaptive_weight_update_interval: int = 5000        # 自适应权重更新间隔 ( 仿真步数)

    # === Episode内权重策略 ===
    weighting_strategy = 'uniform'      # episode内的权重策略 ( uniform或linear)

    num_tracked_links = len(tracked_body_names)  # 跟踪的关节数量

    # === 额外跟踪的身体部位 ===
    extra_link_names = [
        'torso_link'                # 躯干链接
    ]

    # === 初始化模式配置 ===
    init_default_frac = 0.0            # 使用默认位置初始化的比例 ( 0表示总是从重放数据初始化)

    # === AMASS数据配置 ===
    use_amass = True                   # 是否使用AMASS动作捕捉数据集
    amass_replay_data_path = 'lafan_walk_and_dance/*.pkl'  # AMASS数据文件路径模式

    # === 人类视频数据配置 ===
    # 定义人类视频数据的文件模式
    human_video_data_pattern = 'retarget_poses_g1.h5'     # 人类视频重定向数据文件模式
    human_video_terrain_pattern = 'background_mesh.obj'    # 对应的地形网格文件模式

    # === 教师模型配置 ===
    # AMASS数据使用的教师检查点
    amass_teacher_checkpoint_run_name: str = "20250410_063030_g1_deepmimic"
    amass_terrain_difficulty = 2       # AMASS地形难度等级 ( 1-5,1为平地,5为最难)

    # === 数据根目录配置 ===
    data_root = '../data/videomimic_captures'    # 主要数据目录
    alt_data_root= ''                            # 备用数据目录
    amass_data_root = '../data/unitree_lafan'    # AMASS数据目录

    # === 数据导入参数 ===
    is_csv_joint_only=False            # 是否只使用CSV关节数据
    default_joint_order_type="g1"      # 默认关节顺序类型
    cut_off_import_length=-1           # 导入动作的最大长度 ( -1表示不限制)

    # === 坐标系配置 ===
    zero_torso_xy = False              # 是否将躯干XY坐标置零
    zero_torso_yaw = False             # 是否将躯干偏航角置零

    # === 人类视频数据配置 ===
    use_human_videos = False           # 是否使用人类视频数据
    # 人类动作数据源  YAML文件路径或单个文件夹名称
    human_motion_source: str = 'resources/data_config/human_motion_list.yaml'
    human_video_oversample_factor = 1  # 人类视频过采样因子 ( 用于平衡数据集)

@configclass
class LeggedRobotDeepMimicTerrainCfg(LeggedRobotTerrainCfg):
    """
    深度模仿学习地形配置
    
    定义了地形生成和处理的参数,支持  
    1. 自定义地形类
    2. 地形噪声生成
    3. 网格到高度场转换
    """
    terrain_class = 'DeepMimicTerrain'  # 使用专门的深度模仿地形类
    mesh_type = 'trimesh'               # 网格类型  trimesh支持复杂几何形状
    n_rows = 1                          # 地形网格行数 ( 用于并行环境布局)

    # 地形噪声配置 ( 当前禁用)
    terrain_noise = None
    # 可选的噪声配置示例  
    # terrain_noise = {
    #     'base_frequency': 5.0,      # 基础频率
    #     'amplitude': 0.3,           # 振幅
    #     'octaves': 1,               # 倍频数
    #     'persistence': 0.5,         # 持续性
    #     'random_z_scaling_enable': False,  # 随机Z轴缩放
    #     'random_z_scaling_scale': 0.02     # Z轴缩放比例
    # }

    # 地形转换选项
    alternate_cast_to_heightfield = False  # 是否使用备用高度场转换
    cast_mesh_to_heightfield = False       # 是否将网格转换为高度场

@configclass
class G1DeepMimicRewardScalesCfg:
    """
    G1深度模仿学习奖励权重配置
    
    定义了所有奖励项的权重,控制训练过程中不同行为的重要性
    正值表示奖励,负值表示惩罚
    """
    
    # === 基础运动跟踪奖励 ( 在深度模仿中通常设为0) ===
    tracking_lin_vel = 0.0      # 线速度跟踪奖励
    tracking_ang_vel = 0.0      # 角速度跟踪奖励

    # === 稳定性奖励 ===
    lin_vel_z = 0.0             # Z轴线速度奖励 ( 防止跳跃)
    ang_vel_xy = 0.00           # XY平面角速度奖励 ( 防止翻滚)
    orientation = 0.0           # 姿态奖励
    base_height = 0.0           # 基座高度奖励
    
    # === 正则化惩罚项 ===
    dof_acc = 0.0               # 关节加速度惩罚 ( 平滑动作)
    dof_vel = 0.0               # 关节速度惩罚
    torques = 0.0               # 扭矩惩罚 ( 能效)
    energy = 0.0                # 能量消耗惩罚
    action_rate = -0.2          # 动作变化率惩罚 ( 重要  鼓励平滑动作)
    action_accel = 0.0          # 动作加速度惩罚

    ankle_action = 0.0          # 踝关节动作惩罚

    # === 行为约束 ===
    no_fly = 0.0                # 防飞行惩罚 ( 防止机器人离地)
    collision = -15.0           # 碰撞惩罚 ( 重要  避免不当接触)
    dof_pos_limits = -50.0      # 关节位置限制惩罚 ( 重要  防止过度伸展)
    alive = 0.0                 # 存活奖励
    hip_pos = 0.0               # 髋关节位置奖励

    # === 接触相关奖励 ===
    contact_no_vel = -100.0     # 接触时无速度惩罚 ( 重要  防止滑动)
    feet_swing_height = 0.0     # 脚部摆动高度奖励
    contact = 0.0               # 一般接触奖励
    feet_orientation = 0.0      # 脚部朝向奖励

    # === 深度模仿核心奖励 ===
    # 这些是深度模仿学习的核心奖励项,权重需要仔细调节
    root_vel_tracking = 0.0             # 根部速度跟踪
    root_ang_vel_tracking = 0.0         # 根部角速度跟踪
    joint_pos_tracking = 120.0          # 关节位置跟踪 ( 核心奖励)
    link_pos_tracking = 30.0            # 身体部位位置跟踪 ( 核心奖励)
    root_pos_tracking = 0.0             # 根部位置跟踪
    torso_pos_tracking = 15.0           # 躯干位置跟踪 ( 重要)
    root_orientation_tracking = 15.0    # 根部姿态跟踪 ( 重要)
    torso_orientation_tracking = 15.0   # 躯干姿态跟踪 ( 重要)
    link_vel_tracking = 5.0             # 身体部位速度跟踪
    joint_vel_tracking = 24.0           # 关节速度跟踪 ( 重要)

    # === 接触匹配奖励 ===
    feet_contact_matching = 1.0         # 脚部接触匹配 ( 重要  匹配参考接触模式)
    contact_smoothness = 0.0            # 接触平滑性

    feet_max_height_for_this_air = 0.0  # 脚部最大高度限制

    # === 终止奖励 ===
    termination= -500.0                 # 终止惩罚 ( 重要  严重惩罚失败)
    feet_air_time = 2000.0              # 脚部空中时间奖励 ( 鼓励自然步态)

@configclass
class G1DeepMimicRewardsCfg(LeggedRobotRewardsCfg):
    """
    G1深度模仿学习奖励配置
    
    包含奖励权重和计算参数
    """
    soft_dof_pos_limit = 0.98    # 软关节位置限制 ( 0.98表示98%的关节范围)
    base_height_target = 0.78    # 目标基座高度 ( 米)
   
    scales = G1DeepMimicRewardScalesCfg()  # 奖励权重

    # === 跟踪奖励计算参数 ===
    # 这些参数控制跟踪奖励的计算方式  reward = exp(-error * k)
    # k值越高,只有误差很小时才能获得高奖励
    joint_pos_tracking_k = 2.0          # 关节位置跟踪系数
    joint_vel_tracking_k = 0.01         # 关节速度跟踪系数
    torso_pos_tracking_k = 50.0         # 躯干位置跟踪系数
    torso_orientation_tracking_k = 3.0  # 躯干姿态跟踪系数
    link_pos_tracking_k = 5.0           # 身体部位位置跟踪系数
    link_vel_tracking_k = 0.1           # 身体部位速度跟踪系数

    # === 禁用的跟踪项 ===
    root_pos_tracking_k = 20.0          # 根部位置跟踪系数
    root_orientation_tracking_k = 3.0   # 根部姿态跟踪系数
    root_vel_tracking_k = 10.0          # 根部速度跟踪系数
    root_ang_vel_tracking_k = 0.01      # 根部角速度跟踪系数

    only_positive_rewards = False       # 是否只使用正奖励 ( False允许负奖励/惩罚)

@configclass
class G1DeepMimicNormalizationCfg(LeggedRobotNormalizationCfg):
    """
    G1深度模仿学习归一化配置
    
    定义观察值的缩放和裁剪参数,确保网络输入在合理范围内
    """
    @configclass
    class ObsScales:
        """观察值缩放参数"""
        lin_vel = 2.0                    # 线速度缩放因子
        ang_vel = 0.25                   # 角速度缩放因子
        dof_pos = 1.0                    # 关节位置缩放因子
        dof_vel = 0.05                   # 关节速度缩放因子
        height_measurements = 5.0        # 高度测量缩放因子
    
    obs_scales = ObsScales()
    clip_observations = 100.             # 观察值裁剪范围 ( 防止极值)
    clip_actions = 8.0                   # 动作裁剪范围 ( 防止过大动作)

@configclass
class G1DeepMimicNoiseCfg(LeggedRobotNoiseCfg):
    """
    G1深度模仿学习噪声配置
    
    定义训练中添加的噪声类型和强度,提高策略的鲁棒性
    """
    add_noise = True                     # 是否添加噪声
    noise_level = 1.0                    # 噪声总体缩放因子

    # === 非相关噪声 ( 每步独立采样) ===
    @configclass
    class NoiseScales:
        dof_pos = 0.01                   # 关节位置噪声
        dof_vel = 1.5                    # 关节速度噪声
        lin_vel = 0.1                    # 线速度噪声
        ang_vel = 0.2                    # 角速度噪声
        gravity = 0.05                   # 重力噪声
        rel_xy = 0.01                    # 相对XY位置噪声
        rel_yaw = 0.01                   # 相对偏航角噪声
    noise_scales = NoiseScales()

    # === 固定偏移 ( 相关噪声,episode期间保持不变) ===
    @configclass
    class OffsetScales:
        action = 0.0                     # 动作偏移
        dof_pos = 0.00                   # 关节位置偏移
        gravity = 0.0                    # 重力偏移
    offset_scales = OffsetScales()

    # === 初始化噪声 ( 重置时应用) ===
    @configclass
    class InitNoiseScales:
        dof_pos = 0.0                    # 关节位置初始噪声
        dof_vel = 0.0                    # 关节速度初始噪声
        root_xy = 0.0                    # 根部XY位置初始噪声
        root_z = 0.0                     # 根部Z位置初始噪声
        root_quat = 0.0                  # 根部四元数初始噪声
    init_noise_scales = InitNoiseScales()

    # === 回放噪声 ( 用于域适应) ===
    @configclass
    class PlayBackNoiseScales:
        freeze_env_prob = 0.0            # 冻结环境概率
        unfreeze_env_prob = 0.0          # 解冻环境概率
    playback_noise_scales = PlayBackNoiseScales()

# === G1机器人资产配置 ===

@configclass
class G1BaseAsset(LeggedRobotAssetCfg):
    """G1机器人基础资产配置"""
    file: str = MISSING                 # URDF文件路径 ( 将被子类覆盖)
    num_dofs: int = MISSING             # 自由度数量 ( 将被子类覆盖)
    
    foot_name = "ankle_roll"            # 脚部链接名称
    penalize_contacts_on = []           # 需要惩罚接触的身体部位
    terminate_after_contacts_on = []    # 接触后需要终止的身体部位
    
    # === 接触力终止配置 ===
    terminate_after_large_feet_contact_forces = False  # 是否在大接触力后终止
    large_feet_contact_force_threshold = 1000.0        # 大接触力阈值 ( 牛顿)
    
    # === 物理参数 ===
    self_collisions = 0                 # 自碰撞检测 ( 0启用,1禁用)
    flip_visual_attachments = False     # 是否翻转视觉附件
    collapse_fixed_joints = False       # 是否折叠固定关节
    armature = 0.001                    # 电枢参数 ( 影响数值稳定性)

@configclass
class G129Anneal23DofAsset(G1BaseAsset):
    """
    G1机器人29自由度退火至23自由度配置
    
    这是预训练使用的主要机器人配置,包含完整的身体控制
    """
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_anneal_23dof.urdf'
    
    # === 备用文件配置 ( 用于域随机化) ===
    use_alt_files = True                # 是否使用备用文件
    alt_files = [
        '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_anneal_23dof_spheres.urdf'
    ]
    
    num_dofs = 23                       # 实际控制的自由度数

    # === 碰撞组配置 ( 防止相邻关节碰撞) ===
    dont_collide_groups = {
        0: {
            "left_ankle_roll_link",
            "left_ankle_pitch_link", 
            "left_knee_link"
        },
        1: {
            "right_ankle_roll_link",
            "right_ankle_pitch_link",
            "right_knee_link"
        }
    }

    # === 跟踪的身体部位 ( 扩展版本) ===
    tracked_body_names = [
        'pelvis',                        # 骨盆
        
        # 左腿
        'left_hip_pitch_link',
        'left_knee_link', 
        'left_ankle_roll_link',
        
        # 右腿
        'right_hip_pitch_link',
        'right_knee_link',
        'right_ankle_roll_link',
        
        # 左臂
        'left_shoulder_pitch_link',
        'left_elbow_link',
        'left_wrist_yaw_link',
        
        # 右臂
        'right_shoulder_pitch_link',
        'right_elbow_link', 
        'right_wrist_yaw_link',
    ]

    num_tracked_links = len(tracked_body_names)

    # === 接触终止配置 ===
    # 目前禁用,但可以配置哪些身体部位接触后需要终止
    terminate_after_contacts_on = [
        # "left_elbow_link",
        # "right_elbow_link", 
        # ... 其他不应接触地面的身体部位
    ]

    # === 接触惩罚配置 ===
    penalize_contacts_on = [
        "left_rubber_hand",              # 左手橡胶部分
        "right_rubber_hand",             # 右手橡胶部分
    ]

    # === 上身关节名称 ( 用于特殊处理) ===
    upper_body_dof_names = [
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
    ]

# === 传感器配置 ===

# 跟踪的身体名称列表 ( 用于传感器配置)
tracked_body_names = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_pitch_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_pitch_link', 'right_elbow_link', 'right_wrist_yaw_link',
]

# 传感器配置列表
sensor_cfgs = [
    # === 地形高度场传感器 ( 清洁版本) ===
    HeightfieldCfg(
        name="terrain_height",
        body_name="torso_link",          # 传感器安装位置
        size=(1.0, 1.0),                 # 感知区域大小 ( 米)
        resolution=0.1,                  # 分辨率 ( 米)
        max_distance=5.0,                # 最大感知距离
        use_float=True,                  # 使用float32而不是uint8
        white_noise_scale=0.0,           # 白噪声缩放
        offset_noise_scale=0.0,          # 偏移噪声缩放
        roll_noise_scale=0.0,            # 横滚噪声缩放
        pitch_noise_scale=0.0,           # 俯仰噪声缩放
        yaw_noise_scale=0.0,             # 偏航噪声缩放
    ),
    
    # === 地形高度场传感器 ( 带噪声版本) ===
    HeightfieldCfg(
        name="terrain_height_noisy",
        body_name="torso_link",
        size=(1.0, 1.0),
        resolution=0.1,
        max_distance=5.0,
        use_float=True,
        white_noise_scale=0.02,          # 2%白噪声
        offset_noise_scale=0.02,         # 2%偏移噪声
        roll_noise_scale=0.04,           # 4%横滚噪声
        pitch_noise_scale=0.04,          # 4%俯仰噪声
        yaw_noise_scale=0.08,            # 8%偏航噪声
        max_delay=3,                     # 最大传感器延迟
        update_frequency_min=1,          # 最小更新频率
        update_frequency_max=5,          # 最大更新频率
        bad_distance_prob=0.01,          # 坏距离测量概率
    ),

    # === 根部高度传感器 ( 单点) ===
    HeightfieldCfg(
        name="root_height",
        body_name="pelvis",
        size=(0.0, 0.0),                 # 单点测量
        resolution=0.1,
        max_distance=5.0,
        use_float=True,
    ),
    
    # === 多关节高度传感器 ===
    MultiLinkHeightCfg(
        name="link_heights",
        body_name="pelvis",              # 参考身体
        max_distance=5.0,
        link_names=tracked_body_names,   # 所有跟踪的身体部位
        use_float=True,
    ),
    
    # === 深度相机配置 ( 可选) ===
    # DepthCameraCfg(
    #     name="depth_camera",
    #     body_name="d435_link",
    #     downsample_factor=2,           # 下采样因子
    #     width=320 // 2,                # 图像宽度
    #     height=240 // 2,               # 图像高度
    #     max_distance=5.0,              # 最大深度
    # ),
]

# === 控制参数配置 ===

# 低刚度控制配置 ( 适合动态运动)
low_stiffness_cfg = {
    'stiffness': {
        # 腿部关节刚度
        'hip_yaw': 75, 'hip_roll': 75, 'hip_pitch': 75,
        'knee': 75,
        'ankle_pitch': 20, 'ankle_roll': 20,
        
        # 腰部关节刚度
        'waist_yaw': 75, 'waist_roll': 75, 'waist_pitch': 75,
        
        # 臂部关节刚度
        'shoulder_pitch': 75, 'shoulder_roll': 75, 'shoulder_yaw': 75,
        'elbow': 75,
    },  # [N*m/rad]

    'damping': {
        # 腿部关节阻尼
        'hip_yaw': 2., 'hip_roll': 2., 'hip_pitch': 2.,
        'knee': 2.,
        'ankle_pitch': 0.2, 'ankle_roll': 0.1,
        
        # 腰部关节阻尼
        'waist_yaw': 2.0, 'waist_roll': 2.0, 'waist_pitch': 2.0,
        
        # 臂部关节阻尼
        'shoulder_pitch': 2.0, 'shoulder_roll': 2.0, 'shoulder_yaw': 2.0,
        'elbow': 2.0,
    },  # [N*m*s/rad]
}

    
@configclass
class G1DeepMimicCfg(LeggedRobotCfg):
    """
    G1深度模仿学习主配置类
    
    这是VideoMimic阶段1(MCPT)的完整配置
    整合了机器人资产、地形、传感器、初始状态、奖励、噪声等所有配置
    """

    # ========== 机器人资产配置 ==========
    asset = G129Anneal23DofAsset()
    # 使用29自由度退火到23自由度的G1模型
    # 包含完整的腿部、腰部、手臂控制
  
    # ========== 地形配置 ==========
    terrain = LeggedRobotDeepMimicTerrainCfg()
    # 使用DeepMimicTerrain类,支持从.obj文件加载地形网格

    # ========== 传感器配置 ==========
    sensors = LeggedRobotSensorsCfg(
        sensor_cfgs = sensor_cfgs,
        # 包含的传感器  
        # - terrain_height: 10x10清洁高度场
        # - terrain_height_noisy: 10x10带噪声高度场
        # - root_height: 根部单点高度
        # - link_heights: 13个身体部位的高度
    )

    # ========== 初始状态配置 ==========
    init_state = LeggedRobotInitStateCfg(
        pos = [0.0, 0.0, 0.78],  # 初始位置 x,y,z [米]
        # z=0.78米  机器人站立时的基座高度
        
        default_joint_angles = {  # 当action=0.0时的目标关节角度 [弧度]
           # === 左腿关节 ===
           'left_hip_yaw_joint' : 0.,            # 左髋偏航  0度 ( 中立)
           'left_hip_roll_joint' : 0,            # 左髋滚转  0度 ( 中立)
           'left_hip_pitch_joint' : -0.1,        # 左髋俯仰  -0.1弧度 ( 轻微后倾)
           'left_knee_joint' : 0.3,              # 左膝  0.3弧度 ( 约17度弯曲)
           'left_ankle_pitch_joint' : -0.2,      # 左踝俯仰  -0.2弧度 ( 平衡膝盖弯曲)
           'left_ankle_roll_joint' : 0,          # 左踝滚转  0度 ( 中立)
           
           # === 右腿关节 ( 对称配置)===
           'right_hip_yaw_joint' : 0.,           # 右髋偏航  0度
           'right_hip_roll_joint' : 0,           # 右髋滚转  0度
           'right_hip_pitch_joint' : -0.1,       # 右髋俯仰  -0.1弧度
           'right_knee_joint' : 0.3,             # 右膝  0.3弧度
           'right_ankle_pitch_joint': -0.2,      # 右踝俯仰  -0.2弧度
           'right_ankle_roll_joint' : 0,         # 右踝滚转  0度
           
           # === 躯干关节 ===
           'torso_joint' : 0.,                   # 躯干  0度 ( 直立)
           
           # === 腰部关节 ===
           'waist_yaw_joint' : 0.,               # 腰部偏航  0度
           'waist_pitch_joint' : 0.,             # 腰部俯仰  0度
           'waist_roll_joint' : 0.,              # 腰部滚转  0度
           
           # === 左臂关节 ===
           'left_shoulder_pitch_joint' : 0.,     # 左肩俯仰  0度 ( 手臂自然下垂)
           'left_shoulder_roll_joint' : 0.,      # 左肩滚转  0度
           'left_shoulder_yaw_joint' : 0.,       # 左肩偏航  0度
           'left_elbow_joint' : 0.,              # 左肘  0度 ( 伸直)
           'left_wrist_joint' : 0.,              # 左腕  0度
           
           # === 右臂关节 ( 对称配置)===
           'right_shoulder_pitch_joint' : 0.,    # 右肩俯仰  0度
           'right_shoulder_roll_joint' : 0.,     # 右肩滚转  0度
           'right_shoulder_yaw_joint' : 0.,      # 右肩偏航  0度
           'right_elbow_joint' : 0.,             # 右肘  0度
           'right_wrist_joint' : 0.,             # 右腕  0度
           
           # === 手腕额外自由度 ===
           'left_wrist_roll_joint' : 0.,         # 左腕滚转  0度
           'left_wrist_pitch_joint' : 0.,        # 左腕俯仰  0度
           'left_wrist_yaw_joint' : 0.,          # 左腕偏航  0度
           'right_wrist_roll_joint' : 0.,        # 右腕滚转  0度
           'right_wrist_pitch_joint' : 0.,       # 右腕俯仰  0度
           'right_wrist_yaw_joint' : 0.,         # 右腕偏航  0度
        }
        # 注意  这个姿态接近人类站立姿态
        # 膝盖轻微弯曲,重心稳定
    )

    # ========== 深度模仿学习配置 ==========
    deepmimic = LeggedRobotDeepMimicCfg(
        num_tracked_links = asset.num_tracked_links,      # 13个跟踪的身体部位
        tracked_body_names = asset.tracked_body_names,    # 身体部位名称列表
        # 继承LeggedRobotDeepMimicCfg中定义的所有参数  
        # - use_amass, amass_replay_data_path
        # - clip_weighting_strategy
        # - truncate_rollout_length
        # 等等
    )

    # ========== 动作空间配置 ==========
    num_actions = asset.num_dofs  # G1有23个可控自由度
    # 注意  虽然URDF有29个DOF,但实际控制23个

    # ========== 环境配置 ==========
    env = LeggedRobotEnvCfg(
        num_actions = num_actions,  # 23
        
        # === 观测列表  定义计算哪些观测 ===
        obs = [
            'torso',                    # 基础躯干观察 ( 包含线速度)
            'torso_real',               # 真实躯干观察 ( 不含线速度,用于sim2real)
            'deepmimic',                # 深度模仿参考数据 ( 完整动作信息)
            'teacher',                  # 教师观察 ( 用于蒸馏)
            'deepmimic_lin_ang_vel',    # 目标线速度+角速度 ( 根方向)
            'terrain_height',           # 地形高度场 ( 清洁版)
            'terrain_height_noisy',     # 地形高度场 ( 带噪声版)
            'root_height',              # 根部高度 ( 单点)
            'phase',                    # 动作相位 ( sin, cos编码)
            'torso_xy_rel',             # 躯干XY相对位置 ( 根方向)
            'torso_yaw_rel',            # 躯干偏航相对角度 ( 根方向)
            'torso_xy',                 # 躯干XY绝对位置
            'torso_yaw',                # 躯干偏航绝对角度
            'target_joints',            # 目标关节位置 ( 参考★)
            'target_root_roll',         # 目标根部滚转 ( 参考★)
            'target_root_pitch',        # 目标根部俯仰 ( 参考★)
            'target_root_yaw',          # 目标根部偏航
            'upper_body_joint_targets', # 上身关节目标
            'teacher_checkpoint_index', # 教师检查点索引 ( 多教师)
        ],
        
        # === 历史观测配置  定义哪些观测需要保存历史 ===
        obs_history = {
            'torso_real': 5,            # 本体感知保存5帧历史
            'torso_xy_rel': 5,          # 相对XY保存5帧历史
            'torso_yaw_rel': 5,         # 相对偏航保存5帧历史
            'torso_xy': 5,              # 绝对XY保存5帧历史
            'torso_yaw': 5,             # 绝对偏航保存5帧历史
            'deepmimic_lin_ang_vel': 5, # 目标速度保存5帧历史
        }
        # 历史长度=5意味着存储过去5个时间步的观测
        # 总历史时长 = 5步 × 0.02秒/步 = 0.1秒
    )

    # ========== 奖励配置 ==========
    rewards = G1DeepMimicRewardsCfg()
    # 包含所有奖励项的权重和计算参数
    # 详见G1DeepMimicRewardsCfg定义

    # ========== 归一化配置 ==========
    normalization = G1DeepMimicNormalizationCfg()
    # 包含观测值缩放和裁剪参数
    # 确保网络输入在合理范围内

    # ========== 噪声配置 ==========
    noise = G1DeepMimicNoiseCfg()
    # 包含各种噪声类型  
    # - 非相关噪声 ( 每步独立)
    # - 相关噪声 ( episode固定)
    # - 初始化噪声 ( 重置时)
    
    # ========== 域随机化配置 ==========
    domain_rand = LeggedRobotDomainRandCfg(
        # === 摩擦力随机化 ===
        randomize_friction = True,              # 启用摩擦力随机化
        friction_range = [0.1, 1.25],           # 摩擦系数范围  0.1 ( 冰面)到1.25 ( 橡胶)
        
        # === 质量随机化 ===
        randomize_base_mass = False,            # 禁用质量随机化 ( 训练脚本会覆盖)
        added_mass_range = [-1., 3.],           # 附加质量范围  -1到3公斤
        
        # === 推力干扰 ===
        push_robots = False,                    # 禁用随机推力 ( 训练脚本会覆盖)
        push_interval_s = 10,                   # 推力间隔  10秒
        max_push_vel_xy = 0.25,                 # 最大推力速度  0.25 m/s
        
        # === 扭矩随机化 ===
        torque_rfi_rand = False,                # 禁用扭矩随机故障注入
        torque_rfi_rand_scale = 0.04,           # 扭矩RFI缩放  4%
        
        # === PD增益随机化 ===
        p_gain_rand = False,                    # 禁用比例增益随机化 ( 训练脚本会覆盖)
        p_gain_rand_scale = 0.03,               # P增益随机化范围  ±3%
        d_gain_rand = False,                    # 禁用微分增益随机化 ( 训练脚本会覆盖)
        d_gain_rand_scale = 0.03,               # D增益随机化范围  ±3%
        
        # === 关节摩擦随机化 ===
        randomize_dof_friction = False,         # 禁用关节摩擦随机化
        max_dof_friction = 0.05,                # 最大关节摩擦  0.05
        dof_friction_buckets = 64               # 摩擦离散化桶数  64
        # 注释  对sim2sim至关重要
    )

    # ========== 控制参数配置 ==========
    control = LeggedRobotControlCfg(
        beta = 1.0,                             # 动作平滑系数  1.0 ( 无平滑)
        # action_t = beta * action_new + (1-beta) * action_old
        
        action_scale = 0.25,                    # 动作缩放因子  0.25
        # 实际关节角度变化 = action * 0.25
        # 限制动作幅度,提高稳定性
        
        decimation = 4,                         # 控制频率降采样  4
        # 仿真频率 = 200Hz
        # 控制频率 = 200Hz / 4 = 50Hz
        # 每个控制周期执行4次物理步
        
        control_type = 'P',                     # 控制类型  位置控制 ( P控制)
        # 选项  'P' ( 位置)、'V' ( 速度)、'T' ( 扭矩)、'DEEPMIMIC_DELTA'
        
        **low_stiffness_cfg,                    # 展开低刚度配置
        # 包含所有关节的刚度和阻尼参数
        # 低刚度适合动态运动 ( 如行走、跳跃)
    )

    # ========== 仿真参数配置 ==========
    sim = LeggedRobotSimCfg(
        # === 时间步长配置 ===
        # dt = 0.005                            # 备选  5ms
        # dt = 1 / 240.,                        # 备选  240Hz
        dt = 1 / 200.,                          # 使用  200Hz ( 5ms每步)
        # 更高频率 → 更精确但更慢
        
        # === 子步数配置 ===
        # substeps = 4                          # 备选  每步4个子步
        substeps = 1,                           # 使用  每步1个子步
        # 子步数越多,物理仿真越稳定但越慢
        
        # === 物理参数 ===
        gravity = [0., 0., -9.81],              # 重力加速度 [m/s²]
        # 标准地球重力,Z轴向下
        
        up_axis = 1,                            # 向上轴  1表示Z轴
        # 0=Y轴向上,1=Z轴向上

        # ========== PhysX求解器配置 ==========
        physx = LeggedRobotSimCfg.Physx(
            num_threads = 10,                   # CPU线程数  10
            # 更多线程 → 更快仿真 ( 如果CPU核心足够)
            
            solver_type = 1,                    # 求解器类型  1=TGS
            # 0=PGS ( 投影高斯-赛德尔)  快但不太稳定
            # 1=TGS ( 截断高斯-赛德尔)  慢但更稳定
            
            num_position_iterations = 4,        # 位置迭代次数  4
            # 更多迭代 → 更精确的约束求解
            
            num_velocity_iterations = 0,        # 速度迭代次数  0
            # 0表示只做位置约束
            
            contact_offset = 0.01,              # 接触偏移  1cm [米]
            # 物体在此距离内开始生成接触力
            
            rest_offset = 0.0,                  # 静止偏移  0 [米]
            # 物体静止时的接触距离
            
            bounce_threshold_velocity = 0.5,    # 弹跳阈值速度  0.5 [m/s]
            # 低于此速度的碰撞不会弹跳
            
            # === 穿透处理 ===
            # max_depenetration_velocity = 1.0  # 备选  较快
            max_depenetration_velocity = 0.1,   # 使用  0.1 m/s
            # 物体穿透时的最大修正速度
            # 较低值 → 更稳定但可能有轻微穿透
            
            # === GPU接触对数量 ===
            # max_gpu_contact_pairs = 2**23,    # 备选  8M对 ( 约8000环境)
            # max_gpu_contact_pairs = 2**24,    # 备选  16M对
            max_gpu_contact_pairs = 2**25,      # 使用  32M对
            # 需要足够大以支持大量并行环境
            # 4096环境需要大量接触对
            
            default_buffer_size_multiplier = 5, # 缓冲区大小倍数  5
            # 增大缓冲区避免内存溢出
            
            contact_collection = 2,             # 接触收集模式  2
            # 0: never ( 不收集)
            # 1: last sub-step ( 仅最后子步)
            # 2: all sub-steps ( 所有子步,默认)
            # 收集所有子步的接触 → 更准确的接触检测
        )
    )


# ========================================================================
# 特定场景配置 ( 继承并覆盖基础配置)
# ========================================================================

@configclass
class G1DeepMimicMocapCfg(G1DeepMimicCfg):
    """
    G1深度模仿MoCap配置
    
    专门用于纯MoCap数据预训练 ( 不使用人类视频)
    相比基础配置的修改  
    - 只使用AMASS数据
    - 调整终止条件
    - 修改奖励权重
    - 禁用噪声 ( 更容易收敛)
    """

    # ========== 资产配置覆盖 ==========
    asset = G129Anneal23DofAsset(
        terminate_after_large_feet_contact_forces=False
        # 禁用大接触力终止
        # MoCap训练时允许较大的脚部接触力
    )

    # ========== 深度模仿配置覆盖 ==========
    deepmimic = LeggedRobotDeepMimicCfg(
        # === 数据源配置 ===
        use_amass=True,                         # 启用AMASS动作捕捉数据
        use_human_videos=False,                 # 禁用人类视频数据
        
        # === 终止条件 ===
        link_pos_error_threshold=0.5,           # 关节位置误差阈值  0.5米
        # 相比默认0.3米更宽松,允许更大误差
        
        # === 地形难度 ===
        amass_terrain_difficulty=1,             # 地形难度  1 ( 仅平地)
        # 1=flat ( 平地)
        # 2=flat+rough_d1 ( 平地+难度1)
        # MoCap预训练先在简单地形上学习
        
        # === 数据处理 ===
        default_data_fps=30,                    # 数据帧率  30FPS ( Lafan数据集)
        # 正确设置帧率以匹配数据集
        
        cut_off_import_length=1000,             # 截断导入长度  1000帧
        # 每个动作序列最多导入1000帧 ( 约33秒@30FPS)
        # 设为-1则使用完整数据集
        
        # === 数据路径 ===
        amass_replay_data_path="lafan_walk_and_dance/*.pkl",
        # 使用Lafan数据集的行走和跳舞动作
        
        # === 跟踪配置 ===
        num_tracked_links = asset.num_tracked_links,      # 13个
        tracked_body_names = asset.tracked_body_names,    # 继承资产配置
    )

    # ========== 奖励配置覆盖 ==========
    rewards = G1DeepMimicRewardsCfg(
        scales = G1DeepMimicRewardScalesCfg(
            # === 调整奖励权重 ===
            contact_no_vel=5.0,                 # 接触无速度奖励  5.0
            # 增加权重,鼓励接触时减速 ( 防止滑动)
            
            feet_air_time=0.0,                  # 脚部空中时间奖励  0.0
            # 禁用,MoCap预训练不需要
            
            dof_pos_limits=-5.0,                # 关节限制惩罚  -5.0
            # 降低惩罚 ( 相比默认-50.0)
            # MoCap数据通常不会超限
            
            action_rate=-0.1,                   # 动作变化率惩罚  -0.1
            # 降低惩罚 ( 相比默认-0.2)
            # 允许更快的动作变化以跟踪MoCap
        )
    )

    # ========== 噪声配置覆盖 ==========
    noise = G1DeepMimicNoiseCfg(add_noise=False)
    # 禁用所有噪声
    # MoCap预训练先在干净环境学习
    # 后续阶段再添加噪声提高鲁棒性


# ========================================================================
# 策略网络配置
# ========================================================================

@configclass
class G1DeepMimicPolicyCfg(LeggedRobotPolicyCfg):
    """
    G1深度模仿策略网络配置 ( MCPT - Motion Capture Pre-Training)
    
    定义Actor和Critic网络的结构和输入
    这是阶段1的默认策略配置
    """
    init_noise_std = 0.8  # 策略初始化噪声标准差
    # 探索-利用平衡  较高值鼓励早期探索

    @configclass
    class ObsProcActor:
        """
        Actor网络观测处理配置
        
        定义哪些观测输入Actor网络,以及如何处理
        Actor网络用于生成动作,部署到真实机器人
        """
        # === 已禁用的观测 ( 注释掉)===
        # torso = { 'type': 'identity' }              # 不使用 ( 包含线速度,真机不可靠)
        # deepmimic = { 'type': 'identity' }          # 不使用 ( Critic专用)
        
        # === 本体感知 ( 历史)===
        history_torso_real = {'type': 'flatten'}
        # 输入  (4096, 5, 75) → 输出  (4096, 375)
        # 包含  5帧的角速度+重力+关节状态+动作
        # 作用  让策略了解自身运动趋势
        
        # === 根方向 ( 历史)===
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}  # 备选  速度历史
        
        history_torso_xy_rel = {'type': 'flatten'}
        # 输入  (4096, 5, 2) → 输出  (4096, 10)
        # 包含  5帧的相对XY位置
        # 作用  了解位置跟踪趋势
        
        history_torso_yaw_rel = {'type': 'flatten'}
        # 输入  (4096, 5, 1) → 输出  (4096, 5)
        # 包含  5帧的相对偏航角
        # 作用  了解朝向跟踪趋势

        # === 参考关节和姿态 ( 当前帧)- 核心！ ===
        target_joints = {'type': 'identity'}
        # 输入  (4096, 23) → 输出  (4096, 23)
        # 包含  目标关节角度
        # 作用  ★告诉每个关节应该在什么角度
        
        target_root_roll = {'type': 'identity'}
        # 输入  (4096, 1) → 输出  (4096, 1)
        # 包含  目标滚转角
        # 作用  ★告诉身体应该有多少侧倾
        
        target_root_pitch = {'type': 'identity'}
        # 输入  (4096, 1) → 输出  (4096, 1)
        # 包含  目标俯仰角
        # 作用  ★告诉身体应该前倾还是后倾
        
        # === 未使用的观测 ( 注释掉)===
        # target_root_yaw = {'type': 'flatten'}       # 不使用偏航目标
        # history_torso_xy = {'type': 'flatten'}      # 不使用绝对位置历史
        # history_torso_yaw = {'type': 'flatten'}     # 不使用绝对偏航历史
        # phase = {'type': 'identity'}                # 不使用相位信息
        # terrain_height = { ... }                    # 不使用地形 ( 阶段1)
        # front_camera = { ... }                      # 不使用相机
        # down_camera = { ... }                       # 不使用相机
    
    @configclass
    class ObsProcCritic:
        """
        Critic网络观测处理配置
        
        Critic用于价值估计,可以使用更多特权信息
        不需要部署到真机,因此可以用仿真器才有的信息
        """
        # === Critic特权信息 ===
        torso = {'type': 'identity'}
        # 完整躯干信息,包含线速度
        # Actor不能用 ( 真机IMU测线速度不准)
        # Critic可以用 ( 仿真器准确)
        
        deepmimic = {'type': 'identity'}
        # 完整参考动作信息 ( 129维)
        # 包含所有身体部位的位置、速度、姿态等
        # 帮助Critic更准确地估计价值

        # === 本体感知和根方向 ( 同Actor)===
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        # === 参考信息 ( 同Actor)===
        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # === 未使用的观测 ===
        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # terrain_height = { ... }  # Critic也不用地形 ( 阶段1)
    
    # === 应用配置 ===
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()


# ========================================================================
# 地形感知策略配置 ( 阶段2使用)
# ========================================================================

@configclass
class G1DeepmimicHeightFieldPolicyCfg(G1DeepMimicPolicyCfg):
    """
    地形高度场策略配置 ( 阶段2  地形上的跟踪)
    
    在MCPT基础上增加地形感知能力
    用于在复杂地形上执行参考动作
    """

    @configclass
    class ObsProcActor:
        """Actor观测配置 - 增加地形感知"""
        
        # === 同MCPT的观测 ( 本体+根方向+参考)===
        history_torso_real = {'type': 'flatten'}       # 本体感知历史
        history_torso_xy_rel = {'type': 'flatten'}     # 根方向历史
        history_torso_yaw_rel = {'type': 'flatten'}    # 根方向历史
        target_joints = {'type': 'identity'}           # 参考关节
        target_root_roll = {'type': 'identity'}        # 参考姿态
        target_root_pitch = {'type': 'identity'}       # 参考姿态

        # === 新增  地形感知 ===
        terrain_height = {
            'type': 'flatten_then_embed_with_attention',
            'output_dim': 415
        }
        # 处理流程  
        # 1. flatten: (4096, 10, 10) → (4096, 100)
        # 2. embed: (4096, 100) → (4096, hidden_dim)  嵌入到高维空间
        # 3. attention: 使用注意力机制聚焦重要区域
        # 4. 输出: (4096, 415)
        # 
        # 作用  
        # - 理解前方地形起伏
        # - 自适应调整步态
        # - 避开障碍物
    
    @configclass
    class ObsProcCritic:
        """Critic观测配置 - 同样增加地形"""
        torso = {'type': 'identity'}
        deepmimic = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}
        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        terrain_height = {
            'type': 'flatten_then_embed_with_attention',
            'output_dim': 623
        }
        # Critic使用不同的输出维度 ( 更大的嵌入)
        # 帮助更准确地估计价值
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()


@configclass
class G1DeepMimicCfgProjHeightfieldPolicyCfg(G1DeepMimicPolicyCfg):
    """
    投影地形高度场策略配置
    
    使用'to_hidden'模式的地形嵌入
    将地形信息直接嵌入到隐藏层
    """

    @configclass
    class ObsProcActor:
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}
        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        terrain_height = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
        # 'to_hidden'模式  
        # - 地形特征直接投影到网络隐藏层
        # - 不指定output_dim,由网络自适应
        # - 更灵活的特征学习
    
    @configclass
    class ObsProcCritic:
        torso = {'type': 'identity'}
        deepmimic = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}
        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        terrain_height = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()


@configclass
class G1DeepMimicCfgRootHeightfieldPolicyCfg(G1DeepMimicPolicyCfg):
    """
    仅根方向+地形策略配置 ( 蒸馏平面策略)
    
    关键特征  
    - 不使用参考关节 ( target_joints被注释)
    - 不使用参考姿态 ( target_root_roll/pitch被注释)
    - 仅使用根方向信息
    - 使用地形感知
    
    用途  
    - 对比实验  验证参考关节的重要性
    - 蒸馏目标  从MCPT教师蒸馏到简化学生
    """

    @configclass
    class ObsProcActor:
        """Actor观测 - 仅本体+根方向+地形"""
        
        # === 本体感知 ===
        history_torso_real = {'type': 'flatten'}       # 375维
        
        # === 根方向 ( 仅有的参考信息！)===
        history_torso_xy_rel = {'type': 'flatten'}     # 10维
        history_torso_yaw_rel = {'type': 'flatten'}    # 5维
        
        # === 注意  参考关节和姿态全部被注释！ ===
        # target_joints = {'type': 'identity'}          # ✗ 不使用
        # target_root_roll = {'type': 'identity'}       # ✗ 不使用
        # target_root_pitch = {'type': 'identity'}      # ✗ 不使用
        
        # 结果  策略只知道往哪走,不知道怎么走
        # 必须自己学习如何协调四肢运动

        # === 地形感知 ===
        terrain_height = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
        # 使用带噪声地形以提高鲁棒性

    @configclass
    class ObsProcCritic:
        """Critic观测 - 仍然使用完整信息"""
        torso = {'type': 'identity'}
        deepmimic = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}
        
        # === Critic仍然使用参考信息 ===
        target_joints = {'type': 'identity'}           # ✓ Critic可以用
        target_root_roll = {'type': 'identity'}        # ✓ 帮助价值估计
        target_root_pitch = {'type': 'identity'}       # ✓ 更准确的评估
        
        # Critic使用带噪声地形
        terrain_height_noisy = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()
    
    # === 网络结构  更深更宽 ===
    actor_hidden_dims = [1024, 512, 256, 128]
    # Actor隐藏层  4层,从1024降到128
    # 更大容量补偿缺失的参考信息
    
    critic_hidden_dims = [1024, 512, 256, 128]
    # Critic隐藏层  4层,匹配Actor


@configclass
class G1DeepMimicCfgRootHeightfieldNoHistoryPolicyCfg(G1DeepMimicPolicyCfg):
    """
    无历史根方向+地形策略配置
    
    进一步简化  
    - 不使用参考关节
    - 不使用历史 ( 仅当前帧)
    - 使用地形
    """

    @configclass
    class ObsProcActor:
        """Actor观测 - 最小化配置"""
        
        # === 本体感知 ( 仍保留历史)===
        history_torso_real = {'type': 'flatten'}
        
        # === 根方向 ( 无历史！仅当前帧)===
        torso_xy_rel = {'type': 'identity'}            # (4096, 2) 不展平
        torso_yaw_rel = {'type': 'identity'}           # (4096, 1) 不展平
        # 注意  使用identity而非flatten
        # 不使用history_torso_xy_rel
        
        # === 参考关节  禁用 ===
        # target_joints = {'type': 'identity'}          # ✗ 禁用
        # target_root_roll = {'type': 'identity'}       # ✗ 禁用
        # target_root_pitch = {'type': 'identity'}      # ✗ 禁用
        
        # === 地形感知 ( 带噪声)===
        terrain_height_noisy = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
        # 使用带噪声版本,提高sim2real迁移能力

    @configclass
    class ObsProcCritic:
        """Critic观测 - 完整信息"""
        torso = {'type': 'identity'}
        deepmimic = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}     # Critic用历史
        history_torso_yaw_rel = {'type': 'flatten'}    # Critic用历史
        target_joints = {'type': 'identity'}           # Critic用参考
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}
        
        terrain_height_noisy = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()
    
    # === 更大的网络 ===
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]
    # 补偿信息缺失


@configclass
class G1DeepMimicCfgRootPolicyCfg(G1DeepMimicPolicyCfg):
    """
    纯根方向策略配置 ( 无地形版)
    
    最简化配置  
    - 仅本体感知
    - 仅根方向
    - 无参考关节
    - 无地形感知
    
    用途  
    - 消融实验  测试最小化输入的性能
    - 对比基线  与MCPT对比
    """

    @configclass
    class ObsProcActor:
        """Actor观测 - 最小输入"""
        
        # === 本体感知 ===
        history_torso_real = {'type': 'flatten'}       # 375维
        
        # === 根方向 ( 仅有的参考)===
        history_torso_xy_rel = {'type': 'flatten'}     # 10维
        history_torso_yaw_rel = {'type': 'flatten'}    # 5维
        
        # === 所有参考信息  全部禁用 ===
        # target_joints = {'type': 'identity'}          # ✗
        # target_root_roll = {'type': 'identity'}       # ✗
        # target_root_pitch = {'type': 'identity'}      # ✗
        
        # === 地形感知  也禁用 ===
        # terrain_height = { ... }                      # ✗
        # terrain_height_noisy = { ... }                # ✗
        
        # 总输入  375 + 10 + 5 = 390维
        # 对比MCPT的415维,少了25维参考
        # 对比地形策略的830维,少了440维参考+地形

    @configclass
    class ObsProcCritic:
        """Critic观测 - 仍然用完整信息"""
        torso = {'type': 'identity'}
        deepmimic = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}
        
        # Critic用参考信息 ( 帮助训练)
        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}
        
        # Critic用带噪声地形
        terrain_height_noisy = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()
    
    # === 更大网络补偿信息不足 ===
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]


@configclass
class G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPolicyCfg(G1DeepMimicPolicyCfg):
    """
    无历史根方向+投影关节策略配置
    
    特殊配置  
    - 使用参考关节,但通过attention嵌入
    - 不使用根方向历史
    - 使用地形
    """

    @configclass
    class ObsProcActor:
        """Actor观测 - 投影关节版本"""
        
        # === 本体感知 ( 仍用历史)===
        history_torso_real = {'type': 'flatten'}
        
        # === 根方向 ( 无历史！)===
        torso_xy_rel = {'type': 'identity'}            # 当前帧XY
        torso_yaw_rel = {'type': 'identity'}           # 当前帧偏航
        # 不使用 history_torso_xy_rel
        # 不使用 history_torso_yaw_rel

        # === 参考关节 ( 使用attention嵌入)===
        target_joints = {'type': 'embed_with_attention_to_hidden'}
        # 不是直接输入,而是先通过attention处理
        # 可能学习到关节之间的依赖关系
        
        target_root_roll = {'type': 'embed_with_attention_to_hidden'}
        target_root_pitch = {'type': 'embed_with_attention_to_hidden'}

        # === 地形 ( 带噪声)===
        terrain_height_noisy = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }

    @configclass
    class ObsProcCritic:
        """Critic观测 - 完整信息"""
        torso = {'type': 'identity'}
        deepmimic = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}     # Critic用历史
        history_torso_yaw_rel = {'type': 'flatten'}
        
        # Critic用原始参考 ( 不用attention)
        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        terrain_height_noisy = {
            'type': 'flatten_then_embed_with_attention_to_hidden'
        }
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()
    
    # === 大网络 ===
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]


# ========================================================================
# 网络结构变体配置
# ========================================================================

@configclass
class G1DeepMimicCfgRecurrent(G1DeepMimicPolicyCfg):
    """
    循环神经网络策略配置
    
    使用LSTM处理时序信息
    适合需要长期记忆的任务
    """
    # === 前馈网络层 ===
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    # 较小的前馈层 ( 因为有RNN)
    
    activation = 'elu'  # 激活函数  ELU
    # ELU vs ReLU  更平滑,负值也有梯度

    # === LSTM配置 ===
    rnn_type = 'lstm'                    # RNN类型  LSTM
    rnn_hidden_size = 512                # LSTM隐藏层大小  512
    rnn_num_layers = 1                   # LSTM层数  1
    # LSTM可以记忆更长的历史信息

@configclass
class G1DeepMimicCfgBigDeep(G1DeepMimicPolicyCfg):
    """
    大型深度网络策略配置
    
    更大的网络容量
    适合复杂任务和大量数据
    """
    actor_hidden_dims = [1024, 512, 256, 128]
    # 4层,从1024降到128
    
    critic_hidden_dims = [1024, 512, 256, 128]
    # 4层,匹配Actor
    
    activation = 'elu'  # ELU激活函数

@configclass
class G1DeepMimicCfgBigDeepRecurrent(G1DeepMimicCfgBigDeep):
    """
    大型深度循环网络配置
    
    结合大网络和LSTM
    最强大但也最慢的配置
    """
    rnn_type = 'lstm'                    # LSTM
    rnn_hidden_size = 1024               # 更大的LSTM  1024
    rnn_num_layers = 2                   # 2层LSTM ( 更深)


# ========================================================================
# PPO算法配置
# ========================================================================

@configclass
class G1DeepMimicAlgorithmCfg(LeggedRobotAlgorithmCfg):
    """
    G1深度模仿PPO算法配置
    
    定义PPO训练的超参数
    """
    # === 损失函数权重 ===
    value_loss_coef = 1.0                    # 价值损失系数  1.0
    # critic损失的权重
    
    use_clipped_value_loss = True            # 使用裁剪价值损失
    # 防止价值函数更新过大
    
    # === PPO核心参数 ===
    clip_param = 0.2                         # PPO裁剪参数  0.2
    # 限制策略更新幅度  ratio ∈ [0.8, 1.2]
    
    entropy_coef = 0.0025                    # 熵系数  0.0025
    # 鼓励探索,防止过早收敛
    
    # === 训练参数 ===
    num_learning_epochs = 5                  # 每次更新的训练轮数  5
    # 每批数据重复训练5次
    
    num_mini_batches = 4                     # Mini-batch数量  4
    # 将经验缓冲区分成4份
    
    learning_rate = 1.e-3                    # 学习率  0.001
    # Adam优化器的学习率
    
    schedule = 'adaptive'                    # 学习率调度  自适应
    # 根据KL散度自动调整学习率
    
    # === 优势函数参数 ( GAE)===
    gamma = 0.99                             # 折扣因子  0.99
    # 未来奖励的折扣
    
    lam = 0.95                               # GAE lambda  0.95
    # 优势估计的偏差-方差权衡
    
    desired_kl = 0.02                        # 目标KL散度  0.02
    # 用于自适应学习率
    
    max_grad_norm = 1.0                      # 最大梯度范数  1.0
    # 梯度裁剪,防止梯度爆炸

    # === 额外损失项 ===
    bounds_loss_coef = 0.0005                # 边界损失系数  0.0005
    # 惩罚超出合理范围的动作
    
    clip_actions_threshold = 8.0             # 动作裁剪阈值  8.0
    # 用于边界损失计算和教师动作裁剪
    
    # === 蒸馏相关参数 ( 默认禁用)===
    bc_loss_coef = 0.0                       # 行为克隆损失系数  0.0
    # 蒸馏时会设置为1.0
    
    policy_to_clone: Union[str, List[str]] = None
    # 要克隆的策略路径 ( 教师策略)
    # 单个教师  './logs/jit/policy.pt'
    # 多个教师  ['./logs/teacher1.pt', './logs/teacher2.pt']
    
    clip_teacher_actions: bool = True        # 是否裁剪教师动作
    # 防止教师输出过大动作
    
    take_teacher_actions: bool = False       # 是否直接采用教师动作
    # False  学生策略自己推理
    # True  直接用教师动作 ( 调试用)

    # === 多教师配置 ===
    use_multi_teacher: bool = False          # 是否使用多教师
    # 不同动作片段用不同教师
    
    multi_teacher_select_obs_var: str = 'teacher_checkpoint_index'
    # 用于选择教师的观测变量名
    # 根据这个索引决定用哪个教师


# ========================================================================
# 完整训练配置组合
# ========================================================================

@configclass
class G1DeepMimicCfgPPO(LeggedRobotCfgPPO):
    """
    G1深度模仿完整PPO配置
    
    组合策略配置、算法配置、运行器配置
    这是训练时使用的顶层配置
    """
    policy = G1DeepMimicPolicyCfg()          # 策略网络配置
    algorithm = G1DeepMimicAlgorithmCfg()    # PPO算法配置
    
    runner = LeggedRobotRunnerCfg(
        max_iterations = 100000,             # 最大训练迭代  10万次
        # 每次迭代收集24步 × 4096环境 = 98,304步
        # 总训练步数 ≈ 98亿步
        
        experiment_name = 'g1_deepmimic',    # 实验名称
        # 日志保存在 logs/g1_deepmimic/
        
        save_interval = 500,                 # 保存间隔  每500次迭代
        # 约每500次迭代保存一次检查点
        
        # policy_class_name = 'ActorCriticRecurrent',  # 备选  RNN策略
        # 默认使用前馈网络 'ActorCritic'
    )


# ========================================================================
# 变体配置  不同策略的PPO配置
# ========================================================================

@configclass
class G1DeepmimicHeightFieldCfgPPO(G1DeepMimicCfgPPO):
    """地形高度场PPO配置 ( 阶段2)"""
    policy = G1DeepmimicHeightFieldPolicyCfg()
    # 使用地形感知策略

@configclass
class G1DeepMimicCfgProjHeightfieldPPO(G1DeepMimicCfgPPO):
    """投影地形高度场PPO配置"""
    policy = G1DeepMimicCfgProjHeightfieldPolicyCfg()
    # 使用to_hidden投影的地形策略

@configclass
class G1DeepMimicCfgRootHeightfieldPPO(G1DeepMimicCfgPPO):
    """根方向+地形PPO配置 ( 蒸馏平面策略)"""
    policy = G1DeepMimicCfgRootHeightfieldPolicyCfg()
    # 仅根方向+地形,无参考关节

@configclass
class G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPPO(G1DeepMimicCfgPPO):
    """无历史根方向+投影关节PPO配置"""
    policy = G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPolicyCfg()
    # 投影关节+无历史根方向+地形


# ========================================================================
# 蒸馏 ( Dagger)配置
# ========================================================================

@configclass
class G1DeepMimicCfgDagger(G1DeepMimicCfgPPO):
    """
    G1深度模仿蒸馏配置 ( Dagger - Dataset Aggregation)
    
    用途  
    - 从MCPT教师蒸馏到简化学生
    - 学生只用部分观测 ( 如仅根方向)
    - 通过模仿教师动作来学习
    """

    @configclass
    class ObsProcActor:
        """学生Actor观测 - 简化输入"""
        
        # === 本体感知 ===
        history_torso_real = {'type': 'flatten'}       # 375维
        
        # === 根方向 ===
        history_torso_xy_rel = {'type': 'flatten'}     # 10维
        history_torso_yaw_rel = {'type': 'flatten'}    # 5维

        # === 上身关节目标 ( 部分参考)===
        upper_body_joint_targets = {'type': 'identity'}
        # 只提供上身关节目标,不提供腿部
        # 让策略自己学习腿部协调

        # === 禁用完整参考 ===
        # target_joints = {'type': 'identity'}          # ✗ 禁用全身关节
        # target_root_roll = {'type': 'identity'}       # ✗ 禁用姿态
        # target_root_pitch = {'type': 'identity'}      # ✗ 禁用姿态
        
        # 总输入  375 + 10 + 5 + upper_body_dim ≈ 401维

    # Critic配置被注释掉,使用默认
    
    # === 策略网络配置 ===
    policy = G1DeepMimicPolicyCfg(obs_proc_actor = ObsProcActor())
    # 可选的备用配置  
    # policy = G1DeepMimicCfgRecurrent(...)      # RNN版本
    # policy = G1DeepMimicCfgBigDeep(...)        # 大网络版本
    # policy = G1DeepMimicCfgBigDeepRecurrent(...)  # 大RNN版本

    # === 蒸馏算法配置 ===
    algorithm = G1DeepMimicAlgorithmCfg(
        bc_loss_coef = 1.0,                    # 行为克隆损失系数  1.0
        # 主要损失项  模仿教师动作
        
        policy_to_clone = './logs/jit/policy.pt',
        # 教师策略路径 ( MCPT训练好的策略)
        
        learning_rate = 3.e-4,                 # 学习率  0.0003
        # 较低学习率,稳定蒸馏
        
        bounds_loss_coef = 0.0005,             # 边界损失系数
        # 防止学生策略输出过大动作
    )


# ========================================================================
# 蒸馏配置变体
# ========================================================================

@configclass
class G1DeepmimicRootHeightfieldDagger(G1DeepMimicCfgDagger):
    """根方向+地形蒸馏配置"""
    policy = G1DeepMimicCfgRootHeightfieldPolicyCfg()
    # 学生只用根方向+地形
    # 教师是MCPT ( 有参考关节)

@configclass
class G1DeepmimicRootDagger(G1DeepMimicCfgDagger):
    """纯根方向蒸馏配置 ( 无地形)"""
    policy = G1DeepMimicCfgRootPolicyCfg()
    # 学生只用根方向
    # 最简化的蒸馏配置

@configclass
class G1DeepmimicRootHeightfieldNoHistoryDagger(G1DeepMimicCfgDagger):
    """无历史根方向+地形蒸馏配置"""
    policy = G1DeepMimicCfgRootHeightfieldNoHistoryPolicyCfg()
    # 学生用无历史根方向+地形


# ========================================================================
# 任务注册 ( 重要！)
# ========================================================================

from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.g1.g1_deepmimic import G1DeepMimic

# 注册所有任务配置到全局注册表
# 格式  task_registry.register(任务名, 环境类, 环境配置, 训练配置)

# [1] 基础深度模仿任务
task_registry.register(
    "g1_deepmimic",                          # 任务名 ( 用于--task参数)
    G1DeepMimic,                             # 环境类
    G1DeepMimicCfg(),                        # 环境配置
    G1DeepMimicCfgPPO()                      # 训练配置 ( PPO)
)

# [2] MoCap预训练任务
task_registry.register(
    "g1_deepmimic_mocap",                    # MoCap专用配置
    G1DeepMimic,
    G1DeepMimicMocapCfg(),                   # 仅AMASS,无人类视频
    G1DeepMimicCfgPPO()
)

# [3] 蒸馏任务 ( 基础)
task_registry.register(
    "g1_deepmimic_dagger",                   # Dagger蒸馏
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepMimicCfgDagger()                   # 使用蒸馏配置
)

# [4] 地形高度场任务 ( 阶段2)
task_registry.register(
    "g1_deepmimic_heightfield",              # 地形感知训练
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepmimicHeightFieldCfgPPO()           # 使用地形策略
)

# [5] 投影地形任务
task_registry.register(
    "g1_deepmimic_proj_heightfield",         # 投影地形版本
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepMimicCfgProjHeightfieldPPO()
)

# [6] 根方向+地形任务 ( 蒸馏平面策略)
task_registry.register(
    "g1_deepmimic_root_heightfield",         # 仅根方向+地形
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepMimicCfgRootHeightfieldPPO()       # 无参考关节
)

# [7] 根方向+地形蒸馏
task_registry.register(
    "g1_deepmimic_root_heightfield_dagger",  # 根方向+地形蒸馏
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepmimicRootHeightfieldDagger()
)

# [8] 无历史根方向+地形蒸馏
task_registry.register(
    "g1_deepmimic_root_heightfield_no_history_dagger",
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepmimicRootHeightfieldNoHistoryDagger()
)

# [9] 无历史根方向+投影关节PPO
task_registry.register(
    "g1_deepmimic_root_heightfield_no_history_with_proj_joints_ppo",
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPPO()
)

# [10] 纯根方向蒸馏
task_registry.register(
    "g1_deepmimic_root_dagger",              # 纯根方向 ( 无地形)蒸馏
    G1DeepMimic,
    G1DeepMimicCfg(),
    G1DeepmimicRootDagger()
)

# ========================================================================
# 使用说明 - 10个任务配置完整列表
# ========================================================================
"""
===============================================================================
核心训练流程 (4个阶段使用的3个任务配置)
===============================================================================

✅ [1] g1_deepmimic - 阶段1使用
命令  bash videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh
说明  MoCap预训练 (MCPT - Motion Capture Pre-Training)
特点  ✓完整参考 ( 关节+姿态) ✗地形感知 ✓AMASS数据 ✓域随机化
观测  history_torso_real + history_torso_xy_rel + history_torso_yaw_rel 
      + target_joints + target_root_roll + target_root_pitch
脚本  通过命令行参数配置 (--env.deepmimic.use_amass=True)



✅ [5] g1_deepmimic_proj_heightfield - 阶段2使用
命令  bash videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh
说明  地形适应训练 ( Terrain Tracking)
特点  ✓完整参考 ( 关节+姿态) ✓地形感知 ✓人类视频数据
观测  同阶段1 + terrain_height ( 投影地形嵌入to_hidden模式)
从阶段1检查点加载继续训练



✅ [8] g1_deepmimic_root_heightfield_no_history_dagger - 阶段3、4使用
命令 ( 阶段3)  bash videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh
说明  蒸馏 ( Distillation)- 从教师策略学习简化策略
特点  ✗参考关节 ✗参考姿态 ✓地形感知 ✓蒸馏 ( bc_loss_coef=1.0)
观测  history_torso_real + torso_xy_rel + torso_yaw_rel + terrain_height_noisy
      ( 无历史根方向,无参考关节)
从阶段2检查点加载作为教师


命令 ( 阶段4)  bash videomimic_gym/legged_gym/scripts/train_stage_4_rl_finetune.sh
说明  RL微调 ( RL Fine-tuning)- 纯强化学习优化
特点  ✗参考关节 ✗参考姿态 ✓地形感知 ✗蒸馏 ( bc_loss_coef=0.0)
观测  同阶段3 ( 使用同一任务配置)
关键区别  关闭行为克隆损失,纯RL


===============================================================================
备选和实验配置 (7个任务,不在核心流程中)
===============================================================================

🔄 [2] g1_deepmimic_mocap - 阶段1备用配置
用途  预配置的MoCap训练 ( 无需命令行参数)
特点  固定配置 ( use_amass=True, add_noise=False, link_pos_error_threshold=0.5)
区别  阶段1脚本用[1]+命令行参数更灵活,这个是简化版
适用  快速测试、简单场景



🧪 [3] g1_deepmimic_dagger - 基础蒸馏模板 ( 渐进式蒸馏)
用途  实验性蒸馏配置,保留部分参考信息
特点  Actor包含upper_body_joint_targets ( 上身关节目标)
区别  阶段3完全移除参考,这个是中间步骤
适用  渐进式蒸馏研究 ( 腿部自学,上身跟随)



🔄 [4] g1_deepmimic_heightfield - 阶段2备选配置
用途  地形适应的备选方案
特点  使用固定维度地形嵌入 ( output_dim=415/623)
区别  阶段2用[5]的to_hidden模式 ( 自适应维度),这个是固定维度
适用  需要精确控制网络结构时

对比  
[4] terrain_height = {'type': '...', 'output_dim': 415}  # 固定维度
[5] terrain_height = {'type': '..._to_hidden'}           # 自适应维度




🧪 [6] g1_deepmimic_root_heightfield - 对比实验 ( 直接RL vs 蒸馏)
用途  测试"蒸馏是否必要？能否直接训练简化策略？"
特点  直接RL训练根方向策略 ( 无蒸馏,bc_loss_coef=0.0)
区别  阶段3先蒸馏后RL,这个跳过蒸馏直接RL
适用  消融研究、对比实验

实验设计  
- [6] 直接RL训练根方向策略 → 性能A
- [8] 先蒸馏再RL训练 → 性能B
- 对比A vs B,验证蒸馏的价值

示例  
python train.py --task=g1_deepmimic_root_heightfield \
    --load_run=stage2_run_name --resume


🔄 [7] g1_deepmimic_root_heightfield_dagger - 阶段3备选配置
用途  蒸馏的备选方案 ( 保留历史观测)
特点  使用历史根方向 ( history_torso_xy_rel, history_torso_yaw_rel)
区别  阶段3用[8]无历史版本 ( 仅当前帧),这个保留5帧历史
适用  需要更好性能但可接受更大观测空间时

对比  
[7] history_torso_xy_rel = {'type': 'flatten'}  # 5帧×2维=10维
[8] torso_xy_rel = {'type': 'identity'}         # 1帧×2维=2维

权衡  
- 有历史  更好的跟踪性能,更大的观测空间 ( 不利于真机部署)
- 无历史  更简单的部署,更适合真机 ( 阶段3/4选择)

示例  
python train.py --task=g1_deepmimic_root_heightfield_dagger \
    --train.algorithm.policy_to_clone=stage2_run_name


🧪 [9] g1_deepmimic_root_heightfield_no_history_with_proj_joints_ppo
用途  实验性配置,测试attention嵌入参考关节
特点  不移除参考关节,而是用embed_with_attention_to_hidden处理
区别  阶段1直接输入参考,阶段3移除参考,这个用attention投影参考
适用  研究attention机制是否能更好利用参考信息

研究问题  
- "attention投影 vs 直接输入,哪个更鲁棒？"
- "attention能否学习关节间依赖关系？"

对比  
[1] target_joints = {'type': 'identity'}                        # 直接输入
[8] # target_joints 被注释                                       # 完全移除
[9] target_joints = {'type': 'embed_with_attention_to_hidden'}  # attention投影

示例  
python train.py --task=g1_deepmimic_root_heightfield_no_history_with_proj_joints_ppo


🧪 [10] g1_deepmimic_root_dagger - 消融实验 ( 无地形)
用途  测试"地形感知是否必要？" ( 最简化配置)
特点  仅本体感知+根方向,无地形,无参考关节
区别  阶段3有地形感知,这个完全移除地形
适用  平地场景、消融研究

观测对比  
[8] 阶段3  history_torso_real + torso_xy_rel + torso_yaw_rel + terrain_height_noisy
[10] 这个   history_torso_real + history_torso_xy_rel + history_torso_yaw_rel
            ( 无terrain_height_noisy)

实验问题  
- "在简单平地上,地形感知是否必要？"
- "移除地形后性能下降多少？"

示例  
python train.py --task=g1_deepmimic_root_dagger \
    --train.algorithm.policy_to_clone=teacher_run_name


===============================================================================
快速查找表
===============================================================================

任务名称                                               | 阶段   | 用途     | 参考关节 | 地形 | 历史
-----------------------------------------------------|-------|---------|--------|-----|-----
g1_deepmimic                                         | 阶段1  | 核心     | ✓      | ✗   | ✓
g1_deepmimic_mocap                                   | -     | 备用1    | ✓      | ✗   | ✓
g1_deepmimic_dagger                                  | -     | 实验     | 部分   | ✗   | ✓
g1_deepmimic_heightfield                             | -     | 备用2    | ✓      | ✓   | ✓
g1_deepmimic_proj_heightfield                        | 阶段2  | 核心     | ✓      | ✓   | ✓
g1_deepmimic_root_heightfield                        | -     | 对比实验  | ✗      | ✓   | ✓
g1_deepmimic_root_heightfield_dagger                 | -     | 备用3    | ✗      | ✓   | ✓
g1_deepmimic_root_heightfield_no_history_dagger      | 阶段3/4| 核心     | ✗      | ✓   | ✗
g1_deepmimic_root_heightfield_no_history_with_proj.. | -     | 实验     | 投影   | ✓   | ✗
g1_deepmimic_root_dagger                             | -     | 消融实验  | ✗      | ✗   | ✓


===============================================================================
核心流程总结
===============================================================================

阶段1 (g1_deepmimic)
  ↓ 学习基础动作模仿,完整参考 + AMASS数据
  
阶段2 (g1_deepmimic_proj_heightfield)
  ↓ 增加地形感知,完整参考 + 人类视频 + 地形
  
阶段3 (g1_deepmimic_root_heightfield_no_history_dagger + bc_loss_coef=1.0)
  ↓ 蒸馏到简化策略,移除参考关节,保留根方向 + 地形
  
阶段4 (g1_deepmimic_root_heightfield_no_history_dagger + bc_loss_coef=0.0)
  ↓ RL微调,纯强化学习优化,可部署到真机

关键变化  
- 阶段1→2: 增加地形感知
- 阶段2→3: 移除参考关节,通过蒸馏学习 ( 教师有参考,学生无参考)
- 阶段3→4: 关闭蒸馏损失,纯强化学习微调

输入复杂度  
- 阶段1/2: 观测维度最大 ( 完整参考 + 历史 + 地形)
- 阶段3/4: 观测维度最小 ( 仅本体 + 根方向 + 地形,无历史无参考)

===============================================================================
"""