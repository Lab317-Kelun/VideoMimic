# VideoMimic 仿真

## 安装设置

请参考[安装指南](setup.md)了解如何安装 `videomimic_gym` 和 `videomimic_rl`。

### 下载数据

我们可以通过以下命令下载检查点和视频数据集：

```
cd data
bash download_videomimic_data.sh
```

### 运行推理

在以下所有推理脚本中，运行命令后，viser UI 将在 localhost:8080 上可用，用于查看机器人运行情况。如果您在远程服务器上，可以通过 SSH 隧道、VS Code 的端口接口或持久化 cloudflare 隧道来转发。

确保您已激活 `videomimic` conda 环境，然后运行：

*地形策略*

```bash
bash videomimic_gym/legged_gym/scripts/play_terrain_policy.sh
```

（注意这会加载一个动作，您也可以传入 `resources/data_config/human_motion_list_123_motions.yaml` 来运行所有动作，只是注意 viser 加载所有网格可能需要更长时间）。


*蒸馏平面策略（用于比较）-- 采用根方向但不采用参考。*

```bash 
bash videomimic_gym/legged_gym/scripts/play_flat_policy.sh
```


*MCPT（阶段1）策略推理 -- 采用参考关节和根方向。* 
```bash
bash videomimic_gym/legged_gym/scripts/play_mcpt_policy.sh
```

注意：上述所有检查点都已在真实的 unitree G1 上测试过。

### 运行训练 

再次确保您已为以下所有操作激活 videomimic conda 环境。

#### MoCap 预训练

```bash
bash videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh
```

#### 阶段2（地形跟踪）

```bash
bash videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh
```

#### 阶段3（蒸馏）

```bash
LOAD_RUN=stage_2_run_name # 替换为阶段2的输出
bash videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh ${LOAD_RUN}
```

#### 阶段4（强化学习微调）

```bash
LOAD_RUN=stage_3_run_name # 替换为阶段3的输出
bash videomimic_gym/legged_gym/scripts/train_stage_4_rl_finetune.sh ${LOAD_RUN}
```


#### 其他注意事项

根据您可用的 GPU 数量调整 nproc-per-node。如果需要，我们也原生支持多节点训练。

注意：并非所有上述阶段都在发布前进行了端到端测试。结果也可能因数据而异。如果您遇到任何具体问题，请告诉我（Arthur），我很乐意帮助。

有关更改训练脚本参数的更多详细信息，请参阅 [videomimic_gym 的相应 readme](videomimic_gym/README.md)。
