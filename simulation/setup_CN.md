# 安装指南

## 系统要求

- **操作系统**: 推荐 Ubuntu 18.04 或更高版本  
- **GPU**: Nvidia GPU  
- **驱动版本**: 推荐版本 525 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序。推荐使用 Conda 创建虚拟环境。如果您的系统已安装 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适合创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n rlgpu python=3.8
```

### 1.3 激活虚拟环境

```bash
conda activate rlgpu
```

---

## 2. 安装依赖项

### 2.1 安装 PyTorch

PyTorch 是用于模型训练和推理的神经网络计算框架。使用以下命令安装：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 安装 Isaac Gym

Isaac Gym 是 Nvidia 提供的刚体仿真和训练框架。

#### 2.2.1 下载

从 Nvidia 官网下载 [Isaac Gym](https://developer.nvidia.com/isaac-gym)。

#### 2.2.2 安装

解压包后，导航到 `isaacgym/python` 文件夹，使用以下命令安装：

```bash
cd isaacgym/python
pip install -e .
```


#### 2.2.3 验证安装

运行以下命令。如果打开一个窗口显示 1080 个球下落，则安装成功：

```bash
cd examples
python 1080_balls_of_solitude.py
```

如果遇到任何问题，请参考 `isaacgym/docs/index.html` 中的官方文档。

### 2.3 安装 VideoMimic RL

VideoMimic RL 基于 `videomimic_rl`。

```bash
cd videomimic_rl
pip install -e .
cd ..
```

### 2.4 安装 videomimic_gym

导航到目录并安装：

```bash
cd videomimic_gym
pip install -e .
cd ..
```

---

## 总结

完成上述步骤后，您就可以在虚拟环境中运行相关程序了。如果遇到任何问题，请参考各组件的官方文档或检查依赖项是否正确安装。

