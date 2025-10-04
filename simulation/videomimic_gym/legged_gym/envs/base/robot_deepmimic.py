from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.tensor_utils.replay_data import ReplayDataLoader
from legged_gym.tensor_utils.torch_jit_utils import quat_mul, quat_conjugate, calc_heading_quat_inv, quat_rotate, calc_heading
from legged_gym.utils.deepmimic_terrain import DeepMimicTerrain
from isaacgym import gymapi, gymtorch, gymutil
from legged_gym.tensor_utils.torch_jit_utils import  *
import torch
import glob

# 导入抽象基类
from abc import ABC, abstractmethod
from typing import Tuple, List


class RobotDeepMimic(LeggedRobot):
    """
    深度模仿机器人基类
    继承自LeggedRobot，专门用于深度模仿学习任务
    支持动作捕捉数据回放和参考动作跟踪
    """

    @abstractmethod
    def get_replay_terrain_path(self, cfg: LeggedRobotCfg) -> Tuple[List[str], List[str]]:
        """
        抽象方法：获取回放数据路径和地形路径
        子类必须实现此方法来指定数据源
        """
        pass

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """
        初始化深度模仿机器人环境
        
        Args:
            cfg: 机器人配置
            sim_params: 仿真参数
            physics_engine: 物理引擎
            sim_device: 仿真设备
            headless: 是否无头模式
        """
        # 加载回放数据路径
        self.device = sim_device # TODO -- 应该在基础任务中处理
        replay_data_path, terrain_paths, data_fps_override = self.get_replay_terrain_path(cfg)

        self.terrain_paths = terrain_paths

        # 设置地形
        assert cfg.terrain.terrain_class == 'DeepMimicTerrain'
        # 连接地形等
        self.terrain = DeepMimicTerrain(cfg.terrain, cfg.env.num_envs, terrain_paths)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 设置回放数据加载器
        dof_names_in_file = [name.split('_joint')[0] for name in self.dof_names]

        # 设置跟踪的身体部位名称
        if hasattr(cfg.deepmimic, 'tracked_body_names'):
            self.tracked_body_names = cfg.deepmimic.tracked_body_names
        else:
            self.tracked_body_names = self.body_names

        self.tracked_body_indices = [self.body_names.index(name) for name in self.tracked_body_names]

        # 设置额外链接名称（如躯干）
        if hasattr(cfg.deepmimic, 'extra_link_names'):
            self.extra_link_names = cfg.deepmimic.extra_link_names
            self.torso_index = self.body_names.index(self.extra_link_names[0])
            self.extra_link_torso_index = self.extra_link_names.index('torso_link')
        else:
            self.extra_link_names = None

        # 初始化回放数据加载器
        self.replay_data_loader = ReplayDataLoader(
            replay_data_path,
            self.num_envs, self.device, self.dt,
            dof_names=dof_names_in_file, 
            motor_names=dof_names_in_file, 
            link_names=self.tracked_body_names, 
            contact_names=cfg.deepmimic.contact_names,
            data_quat_format='xyzw', 
            adjust_root_pos=False,
            start_offset=0, 
            height_direct_offset=cfg.deepmimic.height_direct_offset, 
            randomize_start_offset=cfg.deepmimic.randomize_start_offset, 
            n_prepend=cfg.deepmimic.n_prepend,
            n_append=cfg.deepmimic.n_append,
            extra_link_names=cfg.deepmimic.extra_link_names if hasattr(cfg.deepmimic, 'extra_link_names') else None,
            is_csv_joint_only=cfg.deepmimic.is_csv_joint_only,
            default_joint_order_type=cfg.deepmimic.default_joint_order_type,
            cut_off_import_length=cfg.deepmimic.cut_off_import_length,
            default_data_fps=cfg.deepmimic.default_data_fps if cfg.deepmimic.default_data_fps != -1 else 1/self.dt,
            data_fps_override=data_fps_override,
            upsample_data=cfg.deepmimic.upsample_data,
            weighting_strategy=cfg.deepmimic.weighting_strategy,
            inorder_envs=cfg.env.export_trajectory,
            clip_weighting_strategy=cfg.deepmimic.clip_weighting_strategy,
            min_weight_factor=cfg.deepmimic.min_success_rate_weight_factor,
            max_weight_factor=cfg.deepmimic.max_success_rate_weight_factor,
        )
        self.ep_lengths = self.replay_data_loader.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.update_replay_data()

        self.camera_set = False
        self.env_offsets = self.terrain.get_terrain_offset(self.replay_data_loader.episode_indices)

        # 从配置初始化可视化模式
        self.viz_replay_sync_robot = cfg.deepmimic.viz_replay_sync_robot if hasattr(cfg.deepmimic, 'viz_replay_sync_robot') else False

        if self.use_viser_viz:
            available_episodes = self.get_available_episodes()
            self.viser_viz.setup_clip_selection(available_episodes)
    
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        self.last_contact_state = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_target_contact = torch.zeros(self.num_envs, 2, device=self.device)
    
    def get_terrain_paths(self):
        """获取地形路径"""
        return self.terrain_paths
        
    
    """辅助函数：在环境坐标系和世界坐标系之间转换（因为地形偏移）"""

    @property
    def env_root_pos(self):
        """环境坐标系中的根部位置"""
        return self.root_states[:, 0:3] - self.env_offsets
    
    @property
    def env_rigid_body_pos(self):
        """环境坐标系中的刚体位置"""
        return self.rigid_body_pos - self.env_offsets.unsqueeze(1)
    
    def env_frame_to_world_frame(self, points, env_ids=None):
        """将环境坐标系转换为世界坐标系"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if points.shape[1] == 3:
            return points + self.env_offsets[env_ids]
        else:
            return points + self.env_offsets[env_ids].unsqueeze(1)
    
    def world_frame_to_env_frame(self, points, env_ids=None):
        """将世界坐标系转换为环境坐标系"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if points.shape[1] == 3:
            return points - self.env_offsets[env_ids]
        else:
            return points - self.env_offsets[env_ids].unsqueeze(1)

    def viz_replay_data(self, points=None, set_robot_pos=False):
        """
        可视化回放数据
        首先更新当前目标和当前关键点，其次如果使用运动学回放，将这些状态设置到机器人上
        """

        state = self.replay_data_loader.get_next_data()

        reset = self.episode_length_buf > self.ep_lengths   
        self.ep_lengths = self.replay_data_loader.reset(reset)
        self.episode_length_buf[reset] = 0

        if not self.camera_set:
            base_pos = state.root_pos[0, 0, :]
            camera_offset = torch.tensor([1.0, 0.0, 1.0], device=self.device)
            self.set_viewer_camera(position=base_pos + camera_offset, lookat=base_pos)
            self.camera_set = True
        
        env_id = 0
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.int32)

        # 链接目标的当前位置
        points = self.env_frame_to_world_frame(state.link_pos[env_id, 0, :,], env_ids)

        i_rb = [self.body_names.index(self.tracked_body_names[k]) for k in range(len(self.tracked_body_names))]
        # 回放链接目标的当前位置
        points_2 = self.rigid_body_pos[env_id, i_rb]

        # 目标链接速度（当前未可视化）
        points_3 = self.env_frame_to_world_frame(state.link_pos[env_id, 0, :,] + state.link_vels[env_id, 0, :,], env_ids)
        # 链接速度
        root_vels = self.rigid_body_states.view(self.num_envs, -1, 13)[env_id, i_rb, 7:10]
        points_4 = points + root_vels

        # 更新环境0的Viser可视化
        if env_id == 0:
            target_points = points.cpu().numpy()
            current_points = points_2.cpu().numpy()
            velocity_points = points_4.cpu().numpy()
            if hasattr(self, 'viser_viz'):
                self.viser_viz.update_keypoints(target_points, current_points, velocity_points)


        if set_robot_pos:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.root_states[env_ids, 0:3] = self.env_frame_to_world_frame(state.root_pos[env_ids, 0, :], env_ids)
            self.root_states[env_ids, 3:7] = state.root_quat[env_ids, 0, :]
            self.root_states[env_ids, 7:13] = 0

            self.dof_pos[env_ids] = state.dofs[env_ids, 0, :]
            self.dof_vel[env_ids] = 0.

            env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def set_visualization_episode(self, episode_idx: int, start_offset: int = 0):
        """
        设置要在第0个环境中可视化的episode
        
        Args:
            episode_idx: 要可视化的episode/clip索引
            start_offset: 从episode的哪个位置开始
        """
        print(f'设置episode {episode_idx}，起始偏移 {start_offset}')
        # 存储选定的episode索引用于循环
        self.selected_episode_idx = episode_idx
        self.selected_start_offset = start_offset
        # 重置环境以应用新数据
        self.reset_idx(torch.tensor([0], device=self.device), already_reset_replay_data=True)
        

    def reset_idx(self, env_ids, already_reset_replay_data=False):
        """
        按索引重置环境
        
        主要调用回放数据加载器的重置函数来获取新的clips进行播放
        """
        # 对于可视化环境（环境0），如果我们有选定的episode，保持循环它
        if 0 in env_ids and hasattr(self, 'selected_episode_idx'):
            # 不要为环境0重置回放数据，我们将单独处理
            other_env_ids = env_ids[env_ids != 0]
            if len(other_env_ids) > 0:
                env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                env_mask[other_env_ids] = True
                if not already_reset_replay_data:
                    reset_episode_length = self.replay_data_loader.reset(env_mask)
                    if self.cfg.deepmimic.truncate_rollout_length > 0:
                        self.max_episode_length = torch.min(self.cfg.deepmimic.truncate_rollout_length * torch.ones_like(reset_episode_length), reset_episode_length)
                    else:
                        self.max_episode_length = reset_episode_length
                
            
            # 对于环境0，明确将其设置回选定的episode
            self.replay_data_loader.set_env_data(0, self.selected_episode_idx, self.selected_start_offset)
        else:
            # 其他环境的正常重置行为
            env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_mask[env_ids] = True
            if not already_reset_replay_data:
                if self.cfg.deepmimic.truncate_rollout_length > 0:
                    reset_episode_length = self.replay_data_loader.reset(env_mask)
                    self.max_episode_length = torch.min(self.cfg.deepmimic.truncate_rollout_length * torch.ones_like(reset_episode_length), reset_episode_length)
                else:
                    self.max_episode_length = self.replay_data_loader.reset(env_mask)
        
        self.reset_start_state = self.replay_data_loader.get_current_data()

        # 地形偏移
        # 如果您在一个clip上训练以防止机器人之间的碰撞，这很有用
        # 但警告 -- 当您需要动作与地形对齐时不要使用
        if len(env_ids) > 0:
            self.env_offsets[env_ids] = self.terrain.get_terrain_offset(self.reset_start_state.clip_index[env_ids])
            if self.cfg.deepmimic.randomize_terrain_offset:
                self.env_offsets[env_ids, 0:2] += torch.randn_like(self.env_offsets[env_ids, 0:2]) * self.cfg.deepmimic.randomize_terrain_offset_range


        if not self.camera_set:
            base_pos = self.reset_start_state.root_pos[0, :]
            camera_offset = torch.tensor([1.0, 0.0, 1.0], device=self.device)
            self.set_viewer_camera(position=base_pos + camera_offset, lookat=base_pos)
            self.camera_set = True
        
        super().reset_idx(env_ids)

        # 确保动作正确以匹配轨迹，防止随机大幅抖动
        if len(env_ids) > 0:
            self.actions[env_ids] = (self.dof_pos[env_ids] - self.default_dof_pos) / self.cfg.control.action_scale
            self.last_actions[env_ids] = self.actions[env_ids]
            self.last_last_actions[env_ids] = self.last_actions[env_ids]
            self.last_dof_vel[env_ids] = self.dof_vel[env_ids]

    def _reset_dofs(self, env_ids):
        """重置自由度状态"""

        self.dof_pos[env_ids] = self.reset_start_state.dofs[env_ids]
        if hasattr(self.cfg.deepmimic, 'init_default_frac') and self.cfg.deepmimic.init_default_frac > 0:
            reset_default_frac = self.cfg.deepmimic.init_default_frac
            reset_default_mask = torch.rand(env_ids.shape, device=self.device) < reset_default_frac
            reset_default_env_ids = env_ids[reset_default_mask]
            self.dof_pos[reset_default_env_ids] = self.default_dof_pos
        else: 
            reset_default_env_ids = env_ids

        if self.cfg.deepmimic.init_velocities:
            self.dof_vel[env_ids] = self.reset_start_state.motor_vels[env_ids]
            self.dof_vel[reset_default_env_ids] = 0.
        else:
            self.dof_vel[env_ids] = 0.
        
        # 添加噪声
        self.dof_pos[env_ids] = self.dof_pos[env_ids] + self.cfg.noise.init_noise_scales.dof_pos * (torch.randn_like(self.dof_pos[env_ids]))
        self.dof_vel[env_ids] = self.dof_vel[env_ids] + self.cfg.noise.init_noise_scales.dof_vel * (torch.randn_like(self.dof_pos[env_ids]))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                          gymtorch.unwrap_tensor(self.dof_state),
                                          gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """重置根部状态"""
        self.root_states[env_ids, 0:3] = self.env_frame_to_world_frame(self.reset_start_state.root_pos[env_ids], env_ids)
        self.root_states[env_ids, 0:2] += self.cfg.noise.init_noise_scales.root_xy * (torch.randn_like(self.root_states[env_ids, 0:2]))
        self.root_states[env_ids, 2:3] += self.cfg.noise.init_noise_scales.root_z * (torch.randn_like(self.root_states[env_ids, 2:3]))
        self.root_states[env_ids, 2] += self.cfg.deepmimic.respawn_z_offset
        # self.root_states[env_ids, 3:7] = self.reset_start_state.root_quat[env_ids]
        random_quat = torch.randn_like(self.reset_start_state.root_quat[env_ids])
        self.root_states[env_ids, 3:7] = self.reset_start_state.root_quat[env_ids] + random_quat * self.cfg.noise.init_noise_scales.root_quat
        # 重新归一化
        self.root_states[env_ids, 3:7] = self.root_states[env_ids, 3:7] / torch.norm(self.root_states[env_ids, 3:7], dim=-1, keepdim=True)
        if self.cfg.deepmimic.init_velocities:
            self.root_states[env_ids, 7:10] = self.reset_start_state.root_vel[env_ids]
            self.root_states[env_ids, 10:13] = self.reset_start_state.root_ang_vel[env_ids]
        else:
            self.root_states[env_ids, 7:10] = 0.0
            self.root_states[env_ids, 10:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                 gymtorch.unwrap_tensor(self.root_states),
                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_replay_data(self):
        """更新回放数据"""
        state = self.replay_data_loader.get_current_data()
        self.target_root_pos = state.root_pos
        self.target_root_quat = state.root_quat
        self.target_root_vel = state.root_vel
        self.target_root_ang_vel = state.root_ang_vel
        self.target_dofs = state.dofs
        self.target_motors = state.motors
        self.target_motor_vels = state.motor_vels
        self.target_link_pos = state.link_pos
        self.target_link_quat = state.link_quat
        self.target_link_vel = state.link_vels
        self.target_contacts = state.contacts
        self.target_extra_link_pos = state.extra_link_pos
        self.target_extra_link_quat = state.extra_link_quat
        self.target_extra_link_vel = state.extra_link_vel
        self.target_extra_link_ang_vel = state.extra_link_ang_vels

    def _post_physics_step_callback(self):
        """物理步骤后回调"""

        freeze_env_prob = self.cfg.noise.playback_noise_scales.freeze_env_prob
        unfreeze_env_prob = self.cfg.noise.playback_noise_scales.unfreeze_env_prob

        if freeze_env_prob > 0:

            freeze_mask = torch.rand(self.num_envs, device=self.device) < freeze_env_prob
            unfreeze_mask = torch.rand(self.num_envs, device=self.device) < unfreeze_env_prob

            if not hasattr(self, 'frozen_env_mask'):
                self.frozen_env_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            
            self.frozen_env_mask[freeze_mask] = True
            self.frozen_env_mask[unfreeze_mask] = False


            self.replay_data_loader.increment_indices(env_ids=(~self.frozen_env_mask).nonzero(as_tuple=False).flatten())
            self.episode_length_buf[self.frozen_env_mask] -= 1
        else:
            self.replay_data_loader.increment_indices()

        self.update_replay_data()

        self.link_pos_error = self._compute_link_pos_error()
        self.link_vel_error = self._compute_link_vel_error()

        if self.cfg.deepmimic.viz_replay:
            self.viz_replay_data(set_robot_pos=self.viz_replay_sync_robot)
            if self.viz_replay_sync_robot:
                return

        super()._post_physics_step_callback()
    
    def get_current_replay_state(self, env_ids):
        """获取给定环境ID的当前回放状态"""
        state = self.replay_data_loader.get_current_data()
        for attr in state.__dict__:
            if isinstance(getattr(state, attr), torch.Tensor):
                setattr(state, attr, getattr(state, attr)[env_ids])
        
        return state
    
    def get_next_replay_state(self, env_ids, K=1, collapse_next_dim=True):
        """
        获取给定环境ID的下一个回放状态
        
        Args:
            env_ids: 要获取下一个回放状态的环境ID
            K: 要向前看的步数（即返回当前步骤之后的步骤1到K）
            collapse_next_dim: 是否折叠张量的内部维度（仅适用于K=1）
        """
        state = self.replay_data_loader.get_next_data(K=K)
        for attr in state.__dict__:
            if isinstance(getattr(state, attr), torch.Tensor):
                to_set = getattr(state, attr)
                # 折叠时间维度
                if K == 1 and collapse_next_dim and len(to_set.shape) > 2:
                    to_set = to_set.squeeze(1)
                setattr(state, attr, to_set[env_ids])
        
        return state

    
    def _compute_invalid_changes(self):
        """计算无效的接触状态变化"""

        current_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        target_contact = self.target_contacts
        
        # 检测接触状态变化
        contact_changed = current_contact != self.last_contact_state
        target_changed = target_contact != self.last_target_contact  # 与之前的目标比较
        current_is_wrong = current_contact != target_contact
        
        # 惩罚不匹配目标模式的变化
        invalid_changes = contact_changed & ~target_changed & current_is_wrong
        invalid_changes = (invalid_changes) & (self.episode_length_buf > 10).unsqueeze(1)

        self.invalid_changes = invalid_changes

        # 存储当前状态用于下一次迭代
        self.last_contact_state = current_contact
        self.last_target_contact = target_contact

    def check_termination(self):
        """检查终止条件"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        link_pos_error = torch.norm(self.link_pos_error, dim=-1)
        link_pos_error_threshold = self.cfg.deepmimic.link_pos_error_threshold
        self.reset_buf |= torch.any(link_pos_error > link_pos_error_threshold, dim=1) & (self.episode_length_buf >= 2)
        self._compute_invalid_changes()

        if self.cfg.asset.terminate_after_large_feet_contact_forces:
            self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.asset.large_feet_contact_force_threshold, dim=1)

        self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _compute_link_pos_error(self):
        """计算链接位置误差"""
        return self.env_rigid_body_pos[:, self.tracked_body_indices] - self.target_link_pos
    
    def _compute_link_vel_error(self):
        """计算链接速度误差"""
        return self.rigid_body_vel[:, self.tracked_body_indices] - self.target_link_vel

    def compute_reward(self):
        """计算奖励"""
        super().compute_reward()

    @torch.jit.export
    def _reward_joint_pos_tracking(self):
        """关节位置跟踪奖励"""
        motor_pos_error = self.dof_pos - self.target_motors
        k = self.cfg.rewards.joint_pos_tracking_k
        return (
            torch.exp(-torch.pow(motor_pos_error, 2).sum(dim=-1) * k)
        )
    
    @torch.jit.export
    def _reward_joint_vel_tracking(self):
        """关节速度跟踪奖励"""
        motor_vel_error = self.dof_vel - self.target_motor_vels
        k = self.cfg.rewards.joint_vel_tracking_k
        return (
            torch.exp(-torch.pow(motor_vel_error, 2).sum(dim=-1) * k)
        )

    @torch.jit.export
    def _reward_root_pos_tracking(self):
        """根部位置跟踪奖励"""
        root_pos_error = self.env_root_pos - self.target_root_pos
        k = self.cfg.rewards.root_pos_tracking_k
        return torch.exp(-torch.pow(root_pos_error, 2).sum(dim=-1) * k)

    @torch.jit.export
    def _reward_root_orientation_tracking(self):
        """根部方向跟踪奖励"""
        quat_diff = quat_mul(self.root_states[:, 3:7], quat_conjugate(self.target_root_quat))
        root_orientation_error = 2. * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
        k = self.cfg.rewards.root_orientation_tracking_k
        return torch.exp(-root_orientation_error * k)
    
    @torch.jit.export
    def _reward_torso_pos_tracking(self):
        """躯干位置跟踪奖励"""
        torso_pos_error = self.env_rigid_body_pos[:, self.torso_index] - self.target_extra_link_pos[:, self.extra_link_torso_index]
        k = self.cfg.rewards.torso_pos_tracking_k
        return torch.exp(-torch.pow(torso_pos_error, 2).sum(dim=-1) * k)

    @torch.jit.export
    def _reward_torso_orientation_tracking(self):
        """躯干方向跟踪奖励"""
        quat_diff = quat_mul(self.rigid_body_quat[:, self.torso_index], quat_conjugate(self.target_extra_link_quat[:, self.extra_link_torso_index]))
        torso_orientation_error = 2. * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
        k = self.cfg.rewards.torso_orientation_tracking_k
        return torch.exp(-torso_orientation_error * k)
    
    
    def _reward_link_pos_tracking(self):
        """链接位置跟踪奖励"""
        k = self.cfg.rewards.link_pos_tracking_k

        assert len(self.tracked_body_names) == 13

        return (
            torch.exp(-torch.pow(self.link_pos_error, 2).sum(dim=1).sum(dim=1) * k)
            # +torch.exp(-torch.pow(motor_pos_error_arms, 2).sum(dim=-1).sum(dim=-1) * k)
        )
    
    def _reward_feet_max_height_for_this_air(self):
        """脚部最大高度奖励（用于长步）"""
        # 需要过滤接触，因为PhysX在网格上的接触报告不可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        contact_filt = torch.logical_or(contact, self.last_contacts) 
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)

        self.last_contacts = contact
        self.last_contacts_filt = contact_filt

        self.feet_air_max_height = torch.max(self.feet_air_max_height, self.env_rigid_body_pos[:, self.feet_indices, 2])

        desired_feet_max_height_for_this_air = 0.2

        rew_feet_max_height = torch.sum((torch.clamp_min(desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact, dim=1) # 仅在首次接触地面时奖励
        self.feet_air_max_height *= ~contact_filt

        # print(f'rew_feet_max_height: {rew_feet_max_height[0]}')
        return rew_feet_max_height

    
    def _reward_link_vel_tracking(self):
        """链接速度跟踪奖励"""
        k = self.cfg.rewards.link_vel_tracking_k

        return (
            torch.exp(-torch.pow(self.link_vel_error, 2).sum(dim=1).sum(dim=1) * k)
        )

    def _reward_root_vel_tracking(self):
        """根部速度跟踪奖励"""
        root_vel_error = self.root_states[:, 7:10] - self.target_root_vel
        k = self.cfg.rewards.root_vel_tracking_k
        return torch.exp(-torch.norm(root_vel_error, dim=-1) * k)
    
    def _reward_root_ang_vel_tracking(self):
        """根部角速度跟踪奖励"""
        root_ang_vel_error = self.root_states[:, 10:13] - self.target_root_ang_vel
        k = self.cfg.rewards.root_ang_vel_tracking_k
        return torch.exp(-torch.norm(root_ang_vel_error, dim=-1) * k)

    def _reward_feet_contact_matching(self):
        """脚部接触匹配奖励"""
        if self.cfg.deepmimic.contact_names is None:
            raise ValueError('未设置接触名称')
        
        
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        desired_contact = self.target_contacts
        # print(f'contact: {contact[0]} desired_contact: {desired_contact[0]}')
        # print(f'desired_contact: {desired_contact[0]}')
        return torch.sum((contact == desired_contact).float(), dim=1)

    def _reward_contact_smoothness(self):
        """接触平滑性奖励：惩罚偏离目标模式的接触状态快速变化"""
        return self.invalid_changes.float().sum(dim=1)

    def _reward_no_fly(self):
        """无飞行奖励：惩罚不应该飞行时的飞行状态"""
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0

        fly = (~contacts[:, 0]) & (~contacts[:, 1])

        should_fly = (self.target_contacts[:, 0]).bool() & (self.target_contacts[:, 1]).bool()

        fly_penalty = (fly & ~should_fly).float()

        return fly_penalty

    def _reward_feet_swing_height(self):
        """脚部摆动高度奖励"""
        # TODO -- 这个公式有问题
        swing_height_target = 0.08
        if self.cfg.deepmimic.contact_names is None:
            contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
            pos_error = torch.square(self.feet_pos[:, :, 2] - swing_height_target) * (1-contact.float())
            pos_error += swing_height_target * contact.float()
        else:
            pos_error = torch.square(self.feet_pos[:, :, 2] - swing_height_target) * (1-self.target_contacts.float())
            pos_error += swing_height_target * self.target_contacts.float()

        return torch.sum(pos_error, dim=(1))

    def _reward_ankle_action(self):
        """踝关节动作奖励"""
        ankle_indices = [self.dof_names.index('right_ankle_pitch_joint'), self.dof_names.index('left_ankle_pitch_joint'),
                         self.dof_names.index('right_ankle_roll_joint'), self.dof_names.index('left_ankle_roll_joint')]
        ankle_action = self.actions[:, ankle_indices]
        return torch.pow(ankle_action, 2).sum(dim=-1)

    
    def _reward_feet_orientation(self):
        """脚部方向奖励"""
        feet_quat = self.rigid_body_quat[:, self.feet_indices]
        body_quat = self.root_states[:, 3:7]
        assert len(self.feet_indices) == 2

        feet_orientation_error = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for i in range(2):
            foot_heading = calc_heading(feet_quat[:, i])
            body_heading = calc_heading(body_quat)
            heading_error = torch.abs(foot_heading - body_heading)
            feet_orientation_error += heading_error

        k = 6.0

        return torch.exp(-feet_orientation_error * k)


    def _obs_torso(self):
        """躯干观察"""
        return torch.cat((  
            # 使用传感器中的根部高度（已经是浮点格式）
            self.get_sensor_data('root_height').squeeze(2),
            self.base_lin_vel  * self.obs_scales.lin_vel, # 3
            self.base_ang_vel  * self.obs_scales.ang_vel, # 3
            self.projected_gravity, # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # num_dof
            self.dof_vel * self.obs_scales.dof_vel, # num_dof
            self.actions, # num_actions
        ), dim=-1)

    def _init_randomisation_buffers(self):
        """初始化episode随机化缓冲区"""
        super()._init_randomisation_buffers()
        self.dof_pos_rand_seed = torch.randn_like(self.dof_pos)
        self.gravity_rand_seed = torch.randn_like(self.projected_gravity)
        if self.cfg.domain_rand.randomize_odom_update_frequency:
            self.odom_update_frequency = torch.randint(self.cfg.domain_rand.odom_update_steps_min, self.cfg.domain_rand.odom_update_steps_max, (self.num_envs, ), device=self.device)
            self.odom_update_frequency_offset = torch.randint(0, self.cfg.domain_rand.odom_update_steps_max, (self.num_envs, ), device=self.device)

    def _resample_episodic_randomisations(self, env_ids):
        """重新采样episode随机化"""
        super()._resample_episodic_randomisations(env_ids)
        self.dof_pos_rand_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.gravity_rand_seed[env_ids] = torch.randn_like(self.gravity_rand_seed[env_ids])
        if self.cfg.domain_rand.randomize_odom_update_frequency:
            self.odom_update_frequency[env_ids] = torch.randint(self.cfg.domain_rand.odom_update_steps_min, self.cfg.domain_rand.odom_update_steps_max, (len(env_ids), ), device=self.device)
            self.odom_update_frequency_offset[env_ids] = torch.randint(0, self.cfg.domain_rand.odom_update_steps_max, (len(env_ids), ), device=self.device)

    def _obs_torso_real(self):
        """真实躯干观察（带噪声）"""

        obs = torch.cat((  
            # 使用传感器中的根部高度（已经是浮点格式）
            # self.get_sensor_data('root_height').squeeze(2),
            # self.base_lin_vel  * self.obs_scales.lin_vel, # 3
            self.base_ang_vel  * self.obs_scales.ang_vel, # 3
            self.projected_gravity + self.gravity_rand_seed*self.cfg.noise.offset_scales.gravity, # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos + self.dof_pos_rand_seed*self.cfg.noise.offset_scales.dof_pos, # num_dof
            self.dof_vel * self.obs_scales.dof_vel, # num_dof
            self.actions, # num_actions
        ), dim=-1)

        if not hasattr(self, 'obs_torso_real_noise_scale'):
            self.obs_torso_real_noise_scale = self._get_noise_scale_vec_torso_real(obs)

        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.obs_torso_real_noise_scale

        return obs
    
    def _obs_phase(self):
        """相位观察"""
        # TODO -- 使其了解轨迹中的实际位置，这样我们就不需要从头开始初始化
        phase = self.replay_data_loader.get_episode_phase()
        if hasattr(self, 'phase_offset'):
            phase = phase + self.phase_offset / self.max_episode_length
        sin_phase = torch.sin(2 * torch.pi * phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase ).unsqueeze(1)
        return torch.cat((sin_phase, cos_phase), dim=-1)
    
    def _obs_torso_xy_rel(self):
        """躯干XY相对位置观察"""

        torso_pos = self.env_rigid_body_pos[:, self.torso_index]
        torso_pos_rel = self.target_extra_link_pos[:, self.extra_link_torso_index] - torso_pos

        heading = calc_heading_quat_inv(self.rigid_body_quat[:, self.torso_index])
        obs_torso_pos_rel = quat_rotate(heading, torso_pos_rel)

        # obs_root_pos = quat_rotate(heading.unsqueeze(1).repeat(1, K, 1).view(self.num_envs * K, 4), obs_root_pos.view(self.num_envs * K, 3)).view(self.num_envs, K, 3)
        # return torso_pos_rel[:, :2]

        rel_xy =  obs_torso_pos_rel[:, :2]

        if self.cfg.deepmimic.zero_torso_xy:
            rel_xy[:] = 0.0

        rel_xy += self.cfg.noise.noise_scales.rel_xy * torch.randn_like(rel_xy)
        if self.cfg.domain_rand.randomize_odom_update_frequency:
            if hasattr(self, 'last_rel_xy'):
                last_rel_xy = self.last_rel_xy
            else:
                last_rel_xy = rel_xy
            
            update = torch.remainder(self.episode_length_buf+self.odom_update_frequency_offset, self.odom_update_frequency) == 0
            # import pdb; pdb.set_trace()
            update = update & (self.episode_length_buf > 0)

            rel_xy[~update] = last_rel_xy[~update]

        # 检查是否在手动控制模式且启用了viser
        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            # 初始化目标速度
            manual_offset_target = torch.zeros((self.num_envs, 2), device=self.device)
            
            # 线性速度缩放（0.5 m/s）
            lin_vel_scale = 0.5
            # # 角速度缩放（1.0 rad/s）
            # ang_vel_scale = 1.0

            # 根据按钮状态设置线性速度
            if self.viser_viz.move_forward.value:
                manual_offset_target[:, 0] = lin_vel_scale  # +X
            if self.viser_viz.move_back.value:
                manual_offset_target[:, 0] = -lin_vel_scale  # -X
            if self.viser_viz.move_left.value:
                manual_offset_target[:, 1] = lin_vel_scale  # +Y
            if self.viser_viz.move_right.value:
                manual_offset_target[:, 1] = -lin_vel_scale  # -Y

            return manual_offset_target

        # if hasattr(self, 'rel_xy'):    
        #     update_mask = torch.rand_like(rel_xy) < 0.4
        #     self.rel_xy[update_mask] = rel_xy[update_mask]
        # else:
        #     self.rel_xy = rel_xy

        return rel_xy.view(-1, 2)
    

    def _obs_torso_yaw_rel(self):
        """躯干偏航相对角度观察"""
        torso_quat = self.rigid_body_quat[:, self.torso_index]
        target_quat = self.target_extra_link_quat[:, self.extra_link_torso_index]

        target_heading = calc_heading(target_quat)
        torso_heading = calc_heading(torso_quat)
        heading_error = target_heading - torso_heading

        heading_error = normalize_angle(heading_error)

        # TODO -- 包装偏航误差？


        if self.cfg.deepmimic.zero_torso_yaw:
            heading_error[:] = 0.0

        heading_error += self.cfg.noise.noise_scales.rel_yaw * torch.randn_like(heading_error)

        if self.cfg.domain_rand.randomize_odom_update_frequency:
            if hasattr(self, 'last_torso_yaw_rel'):
                last_torso_yaw_rel = self.last_torso_yaw_rel
            else:
                last_torso_yaw_rel = heading_error
            
            update = torch.remainder(self.episode_length_buf+self.odom_update_frequency_offset, self.odom_update_frequency) == 0
            update = update & (self.episode_length_buf > 0)

            heading_error[~update] = last_torso_yaw_rel[~update]


        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            # 初始化目标速度
            manual_ang_vel = torch.zeros((self.num_envs, 1), device=self.device)
            

            ang_vel_scale = 0.3

                        # 根据按钮状态设置角速度
            if self.viser_viz.rotate_left.value:
                manual_ang_vel[:, 0] = ang_vel_scale  # 正Z旋转
            if self.viser_viz.rotate_right.value:
                manual_ang_vel[:, 0] = -ang_vel_scale  # 负Z旋转
            


            return manual_ang_vel


        
        # if hasattr(self, 'heading_error'):
        #     update_mask = torch.rand_like(heading_error) < 0.4
        #     self.heading_error[update_mask] = heading_error[update_mask]
        # else:
        #     self.heading_error = heading_error

        return heading_error.view(-1, 1)
    
    def _obs_upper_body_joint_targets(self):
        
        upper_body_dof_names = self.cfg.asset.upper_body_dof_names
        
        upper_body_dof_indices = [self.dof_names.index(name) for name in upper_body_dof_names]
        upper_body_joint_targets = (
            self.target_motors[:, upper_body_dof_indices] - self.default_dof_pos[:, upper_body_dof_indices]
        ) * self.obs_scales.dof_pos
        return upper_body_joint_targets

    def _obs_torso_xy(self):
        """躯干XY观察"""
        torso_pos = self.env_rigid_body_pos[:, self.torso_index]
        return torso_pos[:, :2]
    
    def _obs_torso_yaw(self):
        """躯干偏航观察"""
        torso_quat = self.rigid_body_quat[:, self.torso_index]
        return calc_heading(torso_quat)
    

    def _obs_target_joints(self):
        """目标关节观察"""
        return (self.target_motors - self.default_dof_pos) * self.obs_scales.dof_pos
    
    def _obs_target_root_roll(self):
        """目标根部滚转观察"""
        return normalize_angle(get_roll(self.target_root_quat)).unsqueeze(1)
    
    def _obs_target_root_pitch(self):
        """目标根部俯仰观察"""
        return normalize_angle(get_pitch(self.target_root_quat)).unsqueeze(1)
    
    
    def _obs_target_root_yaw(self):
        """目标根部偏航观察"""
        return normalize_angle(get_yaw(self.target_root_quat)).unsqueeze(1)
    
    def _get_noise_scale_vec_torso_real(self, obs):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        noise_vec = torch.zeros_like(obs[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:6+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6+self.num_actions:6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6+2*self.num_actions:6+3*self.num_actions] = 0. # previous actions
        # noise_vec[6+3*self.num_actions:6+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _obs_deepmimic(self):
        """深度模仿观察"""
        K = self.cfg.deepmimic.num_next_obs

        state = self.replay_data_loader.get_next_data(K=K)
        target_root_pos = state.root_pos
        target_root_quat = state.root_quat
        target_dofs = state.dofs
        target_motors = state.motors
        target_motor_vels = state.motor_vels
        target_link_pos = state.link_pos
        target_link_quat = state.link_quat
        target_link_vel = state.link_vels
        target_contacts = state.contacts

        torso_rot = self.root_states[:, 3:7]
        heading = calc_heading_quat_inv(torso_rot)

        obs_root_pos = target_root_pos - self.env_root_pos.unsqueeze(1)
        obs_root_pos = quat_rotate(heading.unsqueeze(1).repeat(1, K, 1).view(self.num_envs * K, 4), obs_root_pos.view(self.num_envs * K, 3)).view(self.num_envs, K, 3)

        obs_root_quat = quat_mul(target_root_quat, quat_conjugate(torso_rot.unsqueeze(1).repeat(1, K, 1)))

        obs_joints = target_motors - self.dof_pos.unsqueeze(1)

        current_tracked_link_pos = self.env_rigid_body_pos[:, self.tracked_body_indices]
        target_link_pos_global = target_link_pos
        obs_link_pos = (target_link_pos_global - current_tracked_link_pos.unsqueeze(1))

        # 从多链接高度传感器获取所有跟踪链接的高度
        obs_tracked_link_heights = self.get_sensor_data('link_heights')

        num_tracked_links = self.cfg.deepmimic.num_tracked_links

        heading_expand = heading.unsqueeze(1).repeat(1, K * num_tracked_links, 1).view(self.num_envs * K * num_tracked_links, 4)
        obs_link_pos = quat_rotate(heading_expand, obs_link_pos.view(self.num_envs * K * num_tracked_links, 3)).view(self.num_envs, K, num_tracked_links, 3)

        # obs_torso = self.compute_obs_torso()


        heading_expand_vels = heading.unsqueeze(1).repeat(1, num_tracked_links, 1).view(self.num_envs * num_tracked_links, 4)
        link_vels_global = self.rigid_body_vel[:, self.tracked_body_indices]
        obs_link_vel = quat_rotate(heading_expand_vels, link_vels_global.view(self.num_envs * num_tracked_links, 3)).view(self.num_envs, num_tracked_links, 3)

        obs = torch.cat((
            # obs_torso.view(self.num_envs, -1),
            obs_tracked_link_heights.view(self.num_envs, -1),
            obs_root_quat.view(self.num_envs, -1),
            obs_root_pos.view(self.num_envs, -1),
            obs_joints.view(self.num_envs, -1),
            obs_link_pos.view(self.num_envs, -1),
            obs_link_vel.view(self.num_envs, -1),
        ), dim=-1)

        if self.cfg.deepmimic.contact_names is not None:
            # target_contact = target_contacts[:, 0] # only get the next contact
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.

            contact_changed = contact != self.last_contact_state
            target_changed = self.target_contacts != self.last_target_contact  # Compare with previous target
            current_is_wrong = contact != self.target_contacts

            invalid_changes = contact_changed & ~target_changed & current_is_wrong

            obs = torch.cat((
                obs,
                contact.float().view(self.num_envs, -1),
                target_contacts.float().view(self.num_envs, -1),
                # target_contact.float().view(self.num_envs, -1),
                self.last_contact_state.float().view(self.num_envs, -1),
                invalid_changes.float().view(self.num_envs, -1),
            ), dim=-1)
            
        return obs
    
    def _get_noise_scale_vec_lin_ang_vel(self, obs):
        # these are effectively commands but we want to noise them for robustness
        noise_vec = torch.zeros_like(obs[0])
        noise_vec[:3] = self.cfg.noise.noise_scales.lin_vel * self.cfg.noise.noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = self.cfg.noise.noise_scales.ang_vel * self.cfg.noise.noise_level * self.obs_scales.ang_vel
        return noise_vec
    
    def _obs_deepmimic_lin_ang_vel(self):
        """深度模仿线性和角速度观察"""
        K = 1
        state = self.replay_data_loader.get_next_data(K=K)
        target_root_vel = state.root_vel
        target_root_ang_vel = state.root_ang_vel

        torso_rot = self.root_states[:, 3:7]
        heading = calc_heading_quat_inv(torso_rot)

        # 检查是否在手动控制模式且启用了viser
        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            # 初始化目标速度
            manual_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
            manual_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
            
            # 线性速度缩放（0.5 m/s）
            lin_vel_scale = 0.5
            # 角速度缩放（1.0 rad/s）
            ang_vel_scale = 1.0

            # 根据按钮状态设置线性速度
            if self.viser_viz.move_forward.value:
                manual_lin_vel[:, 0] = lin_vel_scale  # +X
            if self.viser_viz.move_back.value:
                manual_lin_vel[:, 0] = -lin_vel_scale  # -X
            if self.viser_viz.move_left.value:
                manual_lin_vel[:, 1] = lin_vel_scale  # +Y
            if self.viser_viz.move_right.value:
                manual_lin_vel[:, 1] = -lin_vel_scale  # -Y
            if self.viser_viz.move_up.value:
                manual_lin_vel[:, 2] = lin_vel_scale  # +Z
            if self.viser_viz.move_down.value:
                manual_lin_vel[:, 2] = -lin_vel_scale  # -Z

            # 根据按钮状态设置角速度
            if self.viser_viz.rotate_left.value:
                manual_ang_vel[:, 2] = ang_vel_scale  # 正Z旋转
            if self.viser_viz.rotate_right.value:
                manual_ang_vel[:, 2] = -ang_vel_scale  # 负Z旋转
            
            obs_lin_vel = manual_lin_vel
            obs_ang_vel = manual_ang_vel

            obs = torch.cat((
                obs_lin_vel.view(self.num_envs, -1),
                obs_ang_vel.view(self.num_envs, -1),
            ), dim=-1)

        else:
            # 使用深度模仿值从回放数据
            obs_lin_vel = quat_rotate(heading, target_root_vel.squeeze(1))
            obs_ang_vel = quat_rotate(heading, target_root_ang_vel.squeeze(1))

            obs = torch.cat((
                obs_lin_vel.view(self.num_envs, -1),
                obs_ang_vel.view(self.num_envs, -1),
            ), dim=-1)

            if not hasattr(self, 'obs_lin_ang_vel_noise_scale'):
                self.obs_lin_ang_vel_noise_scale = self._get_noise_scale_vec_lin_ang_vel(obs)

            if self.add_noise:
                obs += (2 * torch.rand_like(obs) - 1) * self.obs_lin_ang_vel_noise_scale

        return obs

    def _manual_obs_teacher(self):
        # TODO -- replace with tacher having the input net saved :) 
        # TODO -- could history be too old here?
        return torch.cat((
            self.obs_dict['history_torso_real'].view(self.num_envs, -1),
            self.obs_dict['history_torso_xy_rel'].view(self.num_envs, -1),
            self.obs_dict['history_torso_yaw_rel'].view(self.num_envs, -1),
            self.obs_dict['target_joints'].view(self.num_envs, -1),
            self.obs_dict['target_root_roll'].view(self.num_envs, -1),
            self.obs_dict['target_root_pitch'].view(self.num_envs, -1),
        ), dim=-1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type in ['P', 'V', 'T', 'DIRECT']:
            return super()._compute_torques(actions)
        elif control_type == 'DEEPMIMIC_DELTA':
            torques = (actions_scaled + self.target_motors - self.dof_pos) * self.p_gains - self.d_gains * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # Check if pushing is enabled in viser (if viser is being used)
        if self.use_viser_viz and not self.viser_viz.enable_push_robots.value:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return

        # Get the base max velocity and apply the scale factor if viser is enabled
        max_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        max_vel_z = 0.1  # Default to small vertical push
        if self.use_viser_viz:
            max_vel_xy *= self.viser_viz.push_force_scale.value
            max_vel_z = self.viser_viz.push_force_z_scale.value

        # Apply random pushes in XY plane
        self.root_states[:, 7:9] += torch_rand_float(-max_vel_xy, max_vel_xy, (self.num_envs, 2), device=self.device)  # lin vel x/y
        # Apply random pushes in Z direction
        self.root_states[:, 9:10] += torch_rand_float(-max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device)  # lin vel z
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
