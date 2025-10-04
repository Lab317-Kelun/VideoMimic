from legged_gym.envs.base.robot_deepmimic import RobotDeepMimic
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_env import G1Robot
from typing import Tuple, List
import glob
import os
import shutil
import yaml
import torch
import time # For periodic logging

class G1DeepMimic(RobotDeepMimic, G1Robot):
    """
    G1机器人深度模仿学习环境类
    
    主要功能：
    1. 加载AMASS动作捕捉数据和人类视频数据
    2. 实现深度模仿学习的奖励机制
    3. 跟踪训练成功率和数据分布
    4. 支持多教师策略的自适应权重调整
    """
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """
        初始化G1深度模仿学习环境
        
        Args:
            cfg: 配置对象，包含所有训练参数
            sim_params: 仿真参数
            physics_engine: 物理引擎
            sim_device: 仿真设备
            headless: 是否无头模式
        """
        # 存储默认的数据文件模式，可能会被YAML配置覆盖
        self.default_human_video_data_pattern = cfg.deepmimic.human_video_data_pattern
        self.default_human_video_terrain_pattern = cfg.deepmimic.human_video_terrain_pattern
    
        # 初始化教师检查点映射列表
        # 用于支持多教师策略，每个动作片段可以对应不同的教师模型
        self.teacher_checkpoints: List[str] = []

        # 初始化父类RobotDeepMimic，这会创建self.replay_data_loader
        RobotDeepMimic.__init__(self, cfg, sim_params, physics_engine, sim_device, headless)

        # --- 成功率跟踪初始化 ---
        # 用于跟踪每个动作片段的训练成功率，支持自适应权重调整
        self.success_history_length = 1000  # 历史记录长度
        
        # 获取重放数据加载器使用的原始路径列表
        self.original_replay_data_paths = self.replay_data_loader.get_pkl_paths()
        self.num_clips = len(self.original_replay_data_paths)  # 总片段数（可能有重复）

        # 找到唯一路径并创建映射关系
        self.unique_clip_paths = sorted(list(dict.fromkeys(self.original_replay_data_paths)))
        self.num_unique_clips = len(self.unique_clip_paths)

        if self.num_unique_clips > 0:
            # 创建路径到唯一索引的映射
            self.clip_path_to_unique_idx = {path: i for i, path in enumerate(self.unique_clip_paths)}

            # 映射原始索引到唯一索引
            self.original_idx_to_unique_idx = torch.tensor(
                [self.clip_path_to_unique_idx[path] for path in self.original_replay_data_paths],
                dtype=torch.long,
                device=self.device
            )

            # 历史缓冲区：存储成功(1)或失败(0)记录
            self.clip_success_history = torch.full(
                (self.num_unique_clips, self.success_history_length),
                fill_value=-1,  # 使用-1表示还没有数据
                dtype=torch.long,
                device=self.device
            )
            # 循环缓冲区的插入指针
            self.clip_history_ptr = torch.zeros(self.num_unique_clips, dtype=torch.long, device=self.device)
            # 每个片段记录的rollout数量（最多到history_length）
            self.clip_rollout_count = torch.zeros(self.num_unique_clips, dtype=torch.long, device=self.device)
        else:
            print("Warning: No unique clips found for success rate tracking.")
            # 初始化空张量避免后续错误
            self.clip_path_to_unique_idx = {}
            self.original_idx_to_unique_idx = torch.empty((0,), dtype=torch.long, device=self.device)
            self.clip_success_history = torch.empty((0, self.success_history_length), dtype=torch.long, device=self.device)
            self.clip_history_ptr = torch.empty((0,), dtype=torch.long, device=self.device)
            self.clip_rollout_count = torch.empty((0,), dtype=torch.long, device=self.device)

        # 设置日志记录间隔
        self.log_success_rate_interval = 100  # 每100步记录一次成功率
        self.last_log_step = 0
        # 自适应权重更新间隔
        self.adaptive_weight_update_interval = cfg.deepmimic.adaptive_weight_update_interval
        self.last_weight_update_step = 0

        # --- 片段分布跟踪初始化 ---
        # 跟踪每个时间步在各个片段上的环境分布
        if self.num_unique_clips > 0:
            # 历史缓冲区：存储每个步骤中各个唯一片段的环境数量
            self.step_clip_distribution_hisget_observationstr = torch.zeros((), dtype=torch.long, device=self.device)
            # 分布历史中记录的步数
            self.step_dist_rollout_count = torch.zeros((), dtype=torch.long, device=self.device)
        else:
            # 如果没有片段则初始化空张量
            self.step_clip_distribution_history = torch.empty((self.success_history_length, 0), dtype=torch.long, device=self.device)
            self.step_dist_history_ptr = torch.zeros((), dtype=torch.long, device=self.device)
            self.step_dist_rollout_count = torch.zeros((), dtype=torch.long, device=self.device)

    def get_replay_terrain_path(self, cfg: LeggedRobotCfg) -> Tuple[List[str], List[str]]:
        """
        获取重放数据路径和地形路径
        
        处理以下数据源的加载：
        1. AMASS动作捕捉数据（配合随机地形）
        2. 单个人类视频文件夹
        3. YAML配置的人类视频列表
        
        同时填充教师检查点映射
        
        Returns:
            Tuple[List[str], List[str]]: (重放数据路径列表, 地形路径列表, FPS覆盖列表)
        """
        # 使用局部列表进行生成，最后赋值给实例变量
        replay_data_paths = []
        terrain_paths = []
        local_teacher_checkpoints = []  # 存储遇到的唯一教师检查点名称
        local_terrain_to_checkpoint_idx = []  # 存储路径到检查点的映射
        local_data_fps_override = []  # FPS覆盖设置

        def _get_checkpoint_idx(name):
            """获取或添加检查点索引的辅助函数"""
            if name and isinstance(name, str) and name.strip():
                name = name.strip()
                if name not in local_teacher_checkpoints:
                    local_teacher_checkpoints.append(name)
                return local_teacher_checkpoints.index(name)
            return -1  # 如果没有提供有效名称则默认为-1

        # 设置数据根目录
        data_root = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.deepmimic.data_root)
        alt_data_root = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.deepmimic.alt_data_root)
        amass_data_root = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.deepmimic.amass_data_root)

        # --- AMASS数据加载 ---
        if hasattr(cfg.deepmimic, 'use_amass') and cfg.deepmimic.use_amass:
            print("正在加载AMASS动作捕捉数据...")
            amass_teacher_name = cfg.deepmimic.amass_teacher_checkpoint_run_name
            amass_ckpt_idx = _get_checkpoint_idx(amass_teacher_name)

            search_path = os.path.join(amass_data_root, cfg.deepmimic.amass_replay_data_path)
            amass_replay_data_paths_init = glob.glob(search_path)
            
            if not amass_replay_data_paths_init:
                print(f"Warning: No AMASS replay data paths found at {search_path}")
            else:
                # 为每个AMASS数据文件生成不同难度的地形组合
                for replay_data_path in amass_replay_data_paths_init:
                    for difficulty in range(cfg.deepmimic.amass_terrain_difficulty):
                        terrain_name = "flat" if difficulty == 0 else f"rough_d{difficulty}"
                        terrain_path = f'{LEGGED_GYM_ROOT_DIR}/resources/motions/amass/ground_mesh_{terrain_name}.obj'
                        terrain_paths.append(terrain_path)
                        replay_data_paths.append(replay_data_path)
                        local_data_fps_override.append(None)
                        local_terrain_to_checkpoint_idx.append(amass_ckpt_idx)
        
        # --- 人类视频数据加载 ---
        if hasattr(cfg.deepmimic, 'use_human_videos') and cfg.deepmimic.use_human_videos:
            print("正在加载人类视频数据...")
            source = cfg.deepmimic.human_motion_source
            
            if source.lower().endswith('.yaml'):
                # 从YAML文件加载动作列表
                yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, source)
                try:
                    with open(yaml_path, 'r') as f:
                        motion_list = yaml.safe_load(f)
                    if not isinstance(motion_list, list):
                        raise ValueError(f"YAML file {yaml_path} does not contain a list.")

                    for item in motion_list:
                        folder_path_rel = item.get('folder_path')
                        # 处理遗留文件的路径分割
                        folder_path_rel = folder_path_rel.split('/')[-1]
                        
                        if not folder_path_rel:
                            print(f"Warning: Skipping item in {yaml_path} due to missing 'folder_path'. Item: {item}")
                            continue

                        # 获取该片段的特定教师检查点名称
                        human_teacher_name = item.get('teacher_checkpoint_run_name', None)
                        human_ckpt_idx = _get_checkpoint_idx(human_teacher_name)

                        # 使用YAML中的特定模式或配置中的默认模式
                        data_pattern = item.get('human_video_data_pattern', self.default_human_video_data_pattern)
                        terrain_pattern = item.get('human_video_terrain_pattern', self.default_human_video_terrain_pattern)

                        # 在data_root中查找匹配的文件夹（使用部分匹配）
                        folder_path_abs = None
                        for dirpath in os.listdir(data_root):
                            if folder_path_rel in dirpath:
                                folder_path_abs = os.path.join(data_root, dirpath)
                                break
                        if folder_path_abs is None:
                            # 在备用数据根目录中查找
                            for dirpath in os.listdir(alt_data_root):
                                if folder_path_rel in dirpath:
                                    folder_path_abs = os.path.join(alt_data_root, dirpath)
                                    break

                        # 如果找不到匹配文件夹，使用原始拼接路径
                        if folder_path_abs is None:
                            folder_path_abs = os.path.join(data_root, folder_path_rel)
                            print(f"Warning: No partial match found for '{folder_path_rel}' in {data_root}, using direct path.")

                        source_replay_path = os.path.join(folder_path_abs, data_pattern)
                        source_terrain_path = os.path.join(folder_path_abs, terrain_pattern)

                        # 根据过采样因子添加多个副本
                        for _ in range(cfg.deepmimic.human_video_oversample_factor):
                            replay_data_paths.append(source_replay_path)
                            terrain_paths.append(source_terrain_path)
                            local_data_fps_override.append(item.get('default_data_fps_override', None))
                            local_terrain_to_checkpoint_idx.append(human_ckpt_idx)

                except FileNotFoundError:
                    print(f"Error: Human motion YAML file not found at {yaml_path}")
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML file {yaml_path}: {e}")
                except Exception as e:
                    print(f"Error processing YAML file {yaml_path}: {e}")

            elif isinstance(source, str) and source:  # 处理单个文件夹名称
                folder_path_rel = source
                
                # 在data_root中查找匹配文件夹
                folder_path_abs = None
                for dirpath in os.listdir(data_root):
                    if folder_path_rel in dirpath:
                        folder_path_abs = os.path.join(data_root, dirpath)
                        break
                
                if folder_path_abs is None:
                    folder_path_abs = os.path.join(data_root, folder_path_rel)
                    print(f"Warning: No partial match found for '{folder_path_rel}' in {data_root}, using direct path.")

                # 使用配置中的默认模式
                data_pattern = self.default_human_video_data_pattern
                terrain_pattern = self.default_human_video_terrain_pattern

                # 单文件夹模式没有定义特定教师，使用默认-1
                single_folder_ckpt_idx = -1

                for _ in range(cfg.deepmimic.human_video_oversample_factor):
                    replay_data_paths.append(os.path.join(folder_path_abs, data_pattern))
                    terrain_paths.append(os.path.join(folder_path_abs, terrain_pattern))
                    local_terrain_to_checkpoint_idx.append(single_folder_ckpt_idx)
            else:
                print(f"Warning: Invalid human_motion_source format: {source}. Expected YAML path or folder name.")

        if not replay_data_paths:
             print("Warning: No replay data paths were loaded. Check AMASS and human video configurations.")

        # 将生成的列表赋值给实例变量
        self.teacher_checkpoints = local_teacher_checkpoints
        self.terrain_to_checkpoint_idx = torch.tensor(local_terrain_to_checkpoint_idx, device=self.device)

        # 确保映射长度与路径长度匹配
        if len(self.terrain_to_checkpoint_idx) != len(replay_data_paths):
             print(f"Warning: Mismatch in length between terrain_to_checkpoint_idx ({len(self.terrain_to_checkpoint_idx)}) and replay_data_paths ({len(replay_data_paths)}). This should not happen.")

        return replay_data_paths, terrain_paths, local_data_fps_override

    def get_available_episodes(self) -> List[str]:
        """
        返回可用于可视化的episode名称/路径列表
        根据使用AMASS还是人类视频数据进行适配
        """
        amass_data_root = os.path.join(LEGGED_GYM_ROOT_DIR, self.cfg.deepmimic.amass_data_root)
        available_episodes = []

        # --- AMASS Episodes ---
        if hasattr(self.cfg.deepmimic, 'use_amass') and self.cfg.deepmimic.use_amass:
            search_path = os.path.join(amass_data_root, self.cfg.deepmimic.amass_replay_data_path)
            amass_episodes = glob.glob(search_path)
            amass_episodes_base = [os.path.splitext(os.path.basename(ep))[0] for ep in amass_episodes]
            available_episodes.extend([f'{ep}_d{difficulty}' for ep in amass_episodes_base for difficulty in range(self.cfg.deepmimic.amass_terrain_difficulty)])

        # --- Human Video Episodes ---
        if hasattr(self.cfg.deepmimic, 'use_human_videos') and self.cfg.deepmimic.use_human_videos:
            source = self.cfg.deepmimic.human_motion_source
            if source.lower().endswith('.yaml'):
                yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, source)
                try:
                    with open(yaml_path, 'r') as f:
                        motion_list = yaml.safe_load(f)
                    if isinstance(motion_list, list):
                        for item in motion_list:
                            folder_path_rel = item.get('folder_path')
                            if folder_path_rel:
                                folder_path_rel = folder_path_rel.split('/')[-1]
                                available_episodes.append(folder_path_rel)
                except Exception as e:
                    print(f"Warning: Could not load or parse YAML {yaml_path} for available episodes: {e}")
            elif isinstance(source, str) and source:
                available_episodes.append(source)

        return available_episodes

    def _obs_teacher_checkpoint_index(self):
        """
        获取当前环境对应的教师检查点索引观察值
        用于多教师策略，让策略网络知道当前应该模仿哪个教师
        """
        row_indices = self.replay_data_loader.episode_indices
        checkpoint_indices = self.terrain_to_checkpoint_idx[row_indices]
        return checkpoint_indices

    def check_termination(self):
        """
        检查环境是否需要重置并更新成功率历史
        
        重写父类方法，添加成功率跟踪功能：
        - 记录每个片段的成功/失败情况
        - 更新循环缓冲区中的历史记录
        """
        # 首先调用父类方法确定reset_buf和time_out_buf
        super().check_termination()

        # --- 更新成功率历史 ---
        if self.num_unique_clips > 0:
            # 找到在此步骤中重置的环境
            reset_env_ids = torch.where(self.reset_buf)[0]

            if len(reset_env_ids) > 0:
                # 确定成功情况（1表示超时，0表示其他原因）
                # 注意：time_out_buf只有在超时是重置的主要原因时才为True
                # 我们只将超时视为成功
                is_success = self.time_out_buf[reset_env_ids].long()

                # 获取重置环境的原始片段索引
                # 需要在reset_idx可能改变之前获取索引
                original_clip_indices = self.replay_data_loader.episode_indices[reset_env_ids]

                # 映射到唯一片段索引
                unique_clip_indices = self.original_idx_to_unique_idx[original_clip_indices]

                # 获取这些唯一片段的当前历史指针
                history_pointers = self.clip_history_ptr[unique_clip_indices]

                # 在指针位置更新历史缓冲区
                self.clip_success_history[unique_clip_indices, history_pointers] = is_success

                # 增加指针（循环）
                self.clip_history_ptr[unique_clip_indices] = (history_pointers + 1) % self.success_history_length

                # 增加rollout计数（限制在history_length）
                current_counts = self.clip_rollout_count[unique_clip_indices]
                new_counts = torch.min(
                    current_counts + 1,
                    torch.tensor(self.success_history_length, device=self.device, dtype=torch.long)
                )
                self.clip_rollout_count[unique_clip_indices] = new_counts

    def _compute_and_log_success_rates(self):
        """
        计算并记录每个片段的成功率
        
        Returns:
            torch.Tensor: 每个唯一片段的当前成功率
        """
        if self.num_unique_clips == 0:
            # 如果没有片段则返回NaN成功率
            current_success_rates = torch.full((0,), float('nan'), device=self.device, dtype=torch.float32)
            return current_success_rates

        overall_rollout_count = 0
        overall_success_count = 0
        # 存储每个唯一片段成功率的张量
        current_success_rates = torch.full((self.num_unique_clips,), float('nan'), device=self.device, dtype=torch.float32)

        for i in range(self.num_unique_clips):
            count = self.clip_rollout_count[i].item()
            clip_path = "_".join(self.unique_clip_paths[i].split("/")[-2:])
            overall_rollout_count += count
            overall_success_count += torch.sum(self.clip_success_history[i, :count]).item()

            if count > 0:
                # 获取有效的历史条目（忽略-1占位符）
                history = self.clip_success_history[i, :count]
                valid_history = history[history != -1]  # 过滤掉初始-1值
                valid_count = len(valid_history)

                if valid_count > 0:
                    success_rate = torch.mean(valid_history.float()).item()
                    self.extras["episode"][f"success/clip_{clip_path}"] = success_rate
                    current_success_rates[i] = success_rate  # 存储用于自适应权重
                else:
                    self.extras["episode"][f"success/clip_{clip_path}"] = 0.0
            else:
                self.extras["episode"][f"success/clip_{clip_path}"] = 0.0

        # 计算总体成功率（仅基于有历史记录的片段）
        valid_overall_mask = ~torch.isnan(current_success_rates)
        if valid_overall_mask.any():
            # 基于每个有效片段的rollout计数的加权平均
            valid_counts = self.clip_rollout_count[valid_overall_mask]
            valid_successes = torch.sum(self.clip_success_history[valid_overall_mask, :], dim=1)
            # 进一步过滤成功次数，只考虑有效条目（-1）
            for idx, unique_idx in enumerate(torch.where(valid_overall_mask)[0]):
                history = self.clip_success_history[unique_idx, :self.clip_rollout_count[unique_idx]]
                valid_history = history[history != -1]
                valid_successes[idx] = valid_history.sum()
                valid_counts[idx] = len(valid_history)
            
            overall_success_count = valid_successes.sum().item()
            overall_rollout_count = valid_counts.sum().item()
            overall_success_rate = overall_success_count / (overall_rollout_count + 1e-6)
        else:
            overall_success_rate = 0.0
        
        self.extras["episode"]["success/overall"] = overall_success_rate

        return current_success_rates  # 返回每个唯一片段的成功率

    def compute_observations(self):
        """
        计算观察值
        
        重写父类方法，添加：
        1. 片段分布记录
        2. 成功率定期记录
        3. 自适应权重更新
        """
        # --- 记录片段分布 --- 
        if self.num_unique_clips > 0:
            # 获取所有环境的当前原始片段索引
            original_clip_indices = self.replay_data_loader.episode_indices

            # 映射到唯一片段索引
            unique_clip_indices = self.original_idx_to_unique_idx[original_clip_indices]

            # 计算每个唯一索引在所有环境中的出现次数
            current_step_distribution = torch.bincount(
                unique_clip_indices, 
                minlength=self.num_unique_clips
            ).long()  # 确保结果是long张量

            # 在历史缓冲区中存储分布
            ptr = self.step_dist_history_ptr.item()
            self.step_clip_distribution_history[ptr] = current_step_distribution

            # 增加指针（循环）
            self.step_dist_history_ptr = (self.step_dist_history_ptr + 1) % self.success_history_length

            # 增加步数计数（限制在history_length）
            self.step_dist_rollout_count = torch.min(
                self.step_dist_rollout_count + 1,
                torch.tensor(self.success_history_length, device=self.device, dtype=torch.long)
            )

        # 定期记录成功率
        current_step = self.gym.get_frame_count(self.sim)  # 使用帧计数作为步数的代理
        if current_step >= self.last_log_step + self.log_success_rate_interval:
            # 计算成功率（也用于潜在的权重更新）
            current_unique_success_rates = self._compute_and_log_success_rates()
            self._compute_and_log_clip_distribution()
            self.last_log_step = current_step

            # 如果策略处于活跃状态且间隔已过，则更新自适应权重
            if self.cfg.deepmimic.clip_weighting_strategy == 'success_rate_adaptive' and self.num_unique_clips > 0:
                 # 将唯一成功率映射回完整的原始片段列表
                success_rates_full = torch.full((self.num_clips,), float('nan'), device=self.device, dtype=torch.float32)
                # 使用映射：original_idx -> unique_idx -> success_rate
                valid_unique_mask = ~torch.isnan(current_unique_success_rates)
                valid_unique_indices = torch.where(valid_unique_mask)[0]
                
                # 为映射到有效唯一索引的原始索引创建掩码
                original_indices_with_valid_rates_mask = torch.zeros(self.num_clips, dtype=torch.bool, device=self.device)
                for unique_idx in valid_unique_indices:
                     original_indices_with_valid_rates_mask |= (self.original_idx_to_unique_idx == unique_idx)
                
                # 应用成功率
                valid_original_indices = torch.where(original_indices_with_valid_rates_mask)[0]
                if len(valid_original_indices) > 0:
                    unique_map_for_valid_originals = self.original_idx_to_unique_idx[valid_original_indices]
                    success_rates_full[valid_original_indices] = current_unique_success_rates[unique_map_for_valid_originals]

                # 在ReplayDataLoader中更新权重
                self.replay_data_loader.update_adaptive_weights(success_rates_full)
                self.last_weight_update_step = current_step  # 更新后重置计时器

        return super().compute_observations()

    def _compute_and_log_clip_distribution(self):
        """
        计算并记录历史中每个片段的步数分布
        用于监控训练过程中不同动作片段的使用情况
        """
        if self.num_unique_clips == 0 or self.step_dist_rollout_count == 0:
            return

        num_recorded_steps = self.step_dist_rollout_count.item()

        # 在记录的历史上求和分布
        # 如果循环缓冲区还没有填满，需要正确处理
        if num_recorded_steps < self.success_history_length:
            summed_dist = torch.sum(self.step_clip_distribution_history[:num_recorded_steps], dim=0)
        else:
            summed_dist = torch.sum(self.step_clip_distribution_history, dim=0)
        
        total_steps_in_history = summed_dist.sum().float()
        
        if total_steps_in_history > 0:
            clip_percentages = (summed_dist.float() / total_steps_in_history)
            for i in range(self.num_unique_clips):
                clip_path = "_".join(self.unique_clip_paths[i].split("/")[-2:])  # 使用与成功率相同的命名
                percentage = clip_percentages[i].item()
                self.extras["episode"][f"dist/clip_{clip_path}"] = percentage
        else:
            # 处理还没有记录步数的情况（如果count > 0不应该发生）
            for i in range(self.num_unique_clips):
                clip_path = "_".join(self.unique_clip_paths[i].split("/")[-2:])
                self.extras["episode"][f"dist/clip_{clip_path}"] = 0.0


