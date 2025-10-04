<div align="center">
  <h1 align="center">📹 <em>VideoMimic Gym</em> </h1>
</div>

<p align="center">
  <strong>这是一个用于从人类视频中学习地形运动的代码库。包含正常行走、DeepMimic和蒸馏的实现。还提供深度渲染、高度图等支持。</strong> 
</p>


## 📦 安装与配置

请参考 [setup.md](/doc/setup.md) 获取安装和配置步骤。

## 🔁 流程概述

使用强化学习实现运动控制的基本工作流程为：

`训练` → `播放` → `仿真2仿真` → `仿真2真实`

- **训练**：使用Gym仿真环境让机器人与环境交互，找到最大化设计奖励的策略。不建议在训练期间使用实时可视化，以避免降低效率。
- **播放**：使用播放命令验证训练好的策略，确保它符合预期。
- **仿真2仿真**：将Gym训练的策略部署到其他仿真器，确保它不会过度特化于Gym特性。
- **仿真2真实**：将策略部署到物理机器人上实现运动控制。

## 🛠️ 用户指南

### 1. 训练

运行以下命令开始训练：

```bash
python legged_gym/scripts/train.py --task=xxx
```

对于多GPU训练，使用torchrun：

```bash
torchrun --nproc-per-node <num_gpus> legged_gym/scripts/train.py --multi_gpu --task=xxx
```

#### ⚙️ 参数说明

##### 基础参数
- `--task`：必需参数；值可以是：
  - `g1_deepmimic`：用于普通强化学习训练
  - `g1_deepmimic_dagger`：用于策略克隆/蒸馏
- `--headless`：默认启动图形界面；设置为true进入无头模式（更高效率）
- `--resume`：从日志中的检查点恢复训练（从`load_run`的检查点恢复）
- `--experiment_name`：要运行/加载的实验名称
- `--run_name`：要执行/加载的运行名称
- `--load_run`：要加载的运行名称；默认为最新运行
- `--checkpoint`：要加载的检查点编号；默认为最新文件
- `--num_envs`：并行训练的环境数量
- `--seed`：随机种子
- `--max_iterations`：最大训练迭代次数
- `--sim_device`：仿真计算设备；指定CPU为`--sim_device=cpu`
- `--rl_device`：强化学习计算设备；指定CPU为`--rl_device=cpu`
- `--multi_gpu`：启用多GPU训练
- `--wandb_note`：向Weights & Biases日志添加注释（对于包含空格的字符串使用引号`""`）

我们也可以通过设置`--env.x=y`覆盖Python配置中的环境参数，或使用`--train.x=y`覆盖训练参数。以下详细说明了一些最重要的参数：

##### 环境参数 (--env.*)
- `deepmimic.use_amass`：使用AMASS动作捕捉数据（True/False）
- `deepmimic.amass_terrain_difficulty`：AMASS地形数据的难度级别。（AMASS数据与随机粗糙地形配对。1表示无粗糙地形，最高5表示在无地形和最难地形之间采样。）
- `deepmimic.use_human_videos`：使用人类视频数据（True/False）
- `deepmimic.human_video_oversample_factor`：人类视频数据的过采样因子。基本上如果设置它将创建多个人类地形。在将少数人类视频与更大的amass数据集混合时很有用。
- `deepmimic.amass_replay_data_path`：AMASS数据文件路径。可以包含通配符（例如 ACCAD_export_retargeted_vnp6/*.pkl）
- `deepmimic.human_video_folders`：包含人类视频数据的文件夹列表
- `deepmimic.init_velocities`：从参考运动初始化速度（重置时。）
- `deepmimic.randomize_start_offset`：随机化起始位置偏移（重置时，否则总是初始化到运动的开始。）
- `deepmimic.n_append`：要追加到运动的冻结帧数。用于强制模型在运动结束时保持稳定。
- `deepmimic.link_pos_error_threshold`：连接位置误差阈值。如果任何关节的笛卡尔误差超过此值，episode将被终止。
- `deepmimic.is_csv_joint_only`：是否只使用CSV关节数据。（仅用于将Unitree LaFan数据重新导出为pkl格式）。
- `deepmimic.cut_off_import_length`：导入运动的最大长度（在导入超长运动时很有用。）
- `deepmimic.respawn_z_offset`：重生的垂直偏移。如果您的运动有脚与地形相交，并想提高根部以防止这种情况，这很有用。
- `deepmimic.weighting_strategy`：用于在episode内对起始位置进行采样的权重策略。选项为"uniform"或"linear"。
- `terrain.n_rows`：地形行数。用于效率目的（见下面数据加载说明部分）
- `asset.terminate_after_large_feet_contact_forces`：是否在大接触力后终止episode。用于限制机器人不要过猛地撞击地面。
- `asset.large_feet_contact_force_threshold`：大接触力阈值
- `asset.use_alt_files`：使用替代机器人模型文件。如果您想在不同GPU上稍微随机化机器人几何形状，这很有用（例如，我们一直在实验使用球体碰撞几何形状。）
- `rewards.scales.xx`：xx奖励的权重（参见 [g1_deepmimic_config.py](/legged_gym/envs/g1/g1_deepmimic_config.py) 了解可能的值）
- `rewards.only_positive_rewards`：只使用正奖励。对于普通非deepmimic环境设置为`True`，但即使它会在训练开始时崩溃性能，建议设置为`False`，否则它会忽略惩罚。
- `rewards.joint_pos_tracking_k`：关节位置跟踪系数。基本上跟踪关节位置的奖励是exp(- <关节位置误差总和> * k) -- 所以k值越高意味着只有在更接近参考时才获得奖励。但是如果k太高，它可能会学会忽略奖励。
- `rewards.joint_vel_tracking_k`：关节速度跟踪系数。同上
- `rewards.link_pos_tracking_k`：连接位置跟踪系数。同上。
- `rewards.collision`：碰撞惩罚权重。
- `rewards.feet_contact_matching`：脚部接触匹配权重
- `normalization.clip_actions`：允许的最大动作值。推荐值：G1约为10。
- `normalization.clip_observations`：允许的最大观察值。推荐值：G1约为100
- `control.beta`：控制对动作输出应用多少EMA的参数（较低值=更多平均，1.0=无ema）
- `domain_rand.randomize_base_mass`：是否随机化机器人基础质量
- `domain_rand.push_robots`：是否对机器人施加随机推力
- `domain_rand.max_push_vel_xy`：xy平面最大推力速度
- `domain_rand.max_push_vel_interval`：推力之间的最大间隔
- `domain_rand.torque_rfi_rand`：是否随机化扭矩RFI
- `domain_rand.p_gain_rand`：是否随机化P增益
- `domain_rand.p_gain_rand_scale`：P增益随机化比例
- `domain_rand.d_gain_rand`：是否随机化D增益
- `domain_rand.d_gain_rand_scale`：D增益随机化比例
- `domain_rand.control_delays`：是否添加控制延迟
- `domain_rand.control_delay_min`：最小控制延迟
- `domain_rand.control_delay_max`：最大控制延迟

##### 训练参数 (--train.*)
- `policy.re_init_std`：用噪声重新初始化策略
- `policy.init_noise_std`：策略初始化噪声的标准差
- `algorithm.learning_rate`：训练学习率
- `algorithm.bc_loss_coef`：行为克隆损失系数（用于dagger）
- `algorithm.policy_to_clone`：要克隆的策略路径（用于dagger）
- `algorithm.bounds_loss_coef`：边界损失系数。这基本上防止策略平均动作超出`clip_actions`指定的范围（见上文）。推荐值约0.0005。
- `algorithm.entropy_coef`：熵正则化系数。较高的值将支持策略std在episode后期继续鼓励探索。
- `algorithm.schedule`：学习率调度类型。'fixed'表示固定LR，`adaptive`表示基于kl散度的。
- `algorithm.desired_kl`：目标KL散度
- `runner.save_interval`：模型保存间隔

### 数据加载说明

目前，我们有2种类型的数据：
* AMASS / 其他没有地形的动作捕捉数据
* 带有地形的VideoMimic(TM)数据

加载在 [replay_data.py](/legged_gym/utils/replay_data.py) 中完成。这个类接受一个pickle文件列表。然后我们使用成员方法从中采样。运动片段被导入为pkl文件。它们预计从一个名为`retargeted_data`的文件中获取，该文件被克隆到与videomimic_gym存储库相同的文件夹中。您可以从Arthur的仓库[这里](https://github.com/ArthurAllshire/retargeted_data)获取一些示例数据。

IsaacGym（和其他仿真器）通常喜欢通过让不同环境共享地形网格来批量使用地形。这使事情高效，但是当我们想要为环境使用不同地形时很烦人。我们实现的解决方案是将不同地形的网格连接为一个，并有一个全局env_offsets变量（见 [robot_deepmimic.py](/legged_gym/envs/base/robot_deepmimic.py)），它被添加到片段的起始位置以将它们与地形对齐。

我们发现的另一个问题是，如果机器人在仿真器中重叠，Isaac Gym会注册它们之间的碰撞（虽然不应用它们——物理是正确的，但不知何故变得超级慢）。如果您有许多机器人同时在同一地形上执行相同运动，这会有问题，因为它会爆炸内存使用。因此`n_rows`变量，它将创建多行。这将有效地扩展地形数量并减少重叠机器人的数量。

地形网格的连接由 [DeepMimicTerrain](/legged_gym/utils/deepmimic_terrain.py) 完成。然后根据片段索引计算偏移。

我们在存储库中支持两种运动片段加载。它们在 [G1 Deepmimic class](/legged_gym/envs/g1/g1_deepmimic.py) 中被获取。第一种是普通的amass运动片段。用`amass_replay_data_path`指定这种运动的文件夹，用use_amass启用/禁用。我们将这些与随机地形配对。第二种是人类视频数据。因为这需要地形信息，我们将这些作为文件夹中的文件获取，同时包含pkl和网格信息。用use_human_videos标志切换（见上述参数文档），并可以用human_video_folders=[ /retargeted data文件夹内视频的路径列表/ ]指定人类视频列表。

**默认训练结果目录**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

#### 示例命令

1. 多GPU强化学习训练：
```bash
torchrun --nproc-per-node 2 legged_gym/scripts/train.py --multi_gpu --task=g1_deepmimic \
  --headless --wandb_note "new_ft_old_terrains" \
  --env.deepmimic.use_amass=False \
  --load_run 20250225_132031_g1_deepmimic --resume \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.human_video_oversample_factor=10 \
  --env.terrain.n_rows=6 \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --env.deepmimic.n_append=50 \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --env.deepmimic.init_velocities=True \
  --env.deepmimic.randomize_start_offset=True \
  --env.asset.terminate_after_large_feet_contact_forces=False \
  --env.asset.use_alt_files=False
```

2. 策略克隆（DAgger）：
```bash
torchrun --nproc_per_node 2 legged_gym/scripts/train.py --task=g1_deepmimic_dagger \
  --multi_gpu --headless --wandb_note "distill" \
  --env.deepmimic.use_amass=False \
  --env.terrain.n_rows=10 \
  --env.deepmimic.amass_terrain_difficulty=1 \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.init_velocities=True \
  --env.deepmimic.randomize_start_offset=True \
  --env.rewards.scales.feet_orientation=0.0 \
  --env.control.beta=1.0 \
  --train.runner.save_interval=50 \
  --train.algorithm.policy_to_clone_jitted=False \
  --train.algorithm.policy_to_clone=logs/g1_deepmimic/20250317_152046_g1_deepmimic \
  --train.algorithm.bc_loss_coef=1.0 \
  --train.algorithm.learning_rate=1e-4 \
  --env.deepmimic.n_append=50 \
  --env.asset.terminate_after_large_feet_contact_forces=False \
  --num_envs 2048
```

训练舞蹈（假设您按照[设置](./doc/setup_en.md)中指定的方式克隆了retargeted_data）：

```bash
torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
  --multi_gpu \
  --task=g1_deepmimic_mocap \
  --headless \
  --env.terrain.n_rows=4096 \
  --env.deepmimic.amass_replay_data_path=lafan_replay_data/env_11_dance1_subject2.pkl \
  --env.deepmimic.cut_off_import_length=1600
```

（如果您没有多个GPU，删除`multi_gpu`参数，只用`python legged_gym/scripts/train.py`。）

检查点将保存在`logs/g1_deepmimic/TAG`中，其中标签取决于日期和时间。如果您配置了WandB，您应该看到带有此标签的运行也出现在那里。

---

### 2. 播放

要在Gym中可视化训练结果，运行以下命令：

```bash
python legged_gym/scripts/play.py --task=xxx
```

#### 基础播放参数
- `--num_envs`：要可视化的环境数量（默认：1）
- `--load_run`：要加载的运行名称；默认为最新运行
- `--checkpoint`：要加载的检查点编号；默认为最新文件
- `--headless`：无GUI运行（用于录制很有用）

#### 可视化选项

##### 1. 标准Isaac Gym可视化
默认可视化使用Isaac Gym的内置查看器。这提供基本的可视化功能，但可能交互性较差。

##### 2. Viser可视化（推荐）
Viser提供增强的可视化体验，具有更多交互功能。您也可以通过网络使用它。要使用Viser：

```bash
python legged_gym/scripts/play.py --task=xxx --env.viser.enable=True
```

Viser特定参数：
- `env.viser.enable`：启用Viser可视化
- `env.control.decimation`：控制更新率（较高值=较慢可视化）
- `env.control.beta`：动作平滑因子（较低值=更平滑运动）

#### 示例命令

1. 使用最新模型的基本可视化：
```bash
python legged_gym/scripts/play.py --task=g1_deepmimic --num_envs 1
```

2. 使用特定模型和DeepMimic设置的Viser可视化（例如重播舞蹈）：
```bash
python legged_gym/scripts/play.py \\
  --task=g1_deepmimic_mocap \
  --env.viser.enable=True \
  --load_run TAG \
  --num_envs 1 \
  --env.deepmimic.amass_replay_data_path=lafan_replay_data/env_11_dance1_subject2.pkl \
  --headless
```

#### 💾 导出网络

可以从Viser UI轻松导出网络。

---

### 3. 仿真2真实（物理部署）

代码目前未发布，但我们使用了[Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)（用于Python初始测试）、[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)（用于Jetson上的真实部署）和[Humanoid Elevation Mapping](https://github.com/smoggy-P/elevation_mapping_humanoid)包。

---

## 🎉 致谢

此存储库建立在以下开源项目的支持和贡献之上。特别感谢：

- [legged_gym](https://github.com/leggedrobotics/legged_gym)：训练和运行代码的基础。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)：强化学习算法实现。
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python.git)：物理部署的硬件通信接口。
- [Unitree rl gym](https://github.com/unitreerobotics/unitree_rl_gym)：Unitree机器人的Gym。

---