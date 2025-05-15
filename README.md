# 基站-电池能量管理DecisonTransformerTransformer项目

## 项目简介

本项目旨在研究和实现基于强化学习（DQN、SAC、PPO、DP）与Decision Transformer的基站-电池能量管理策略。项目包含了环境建模、强化学习训练、轨迹收集、DT训练与评估等完整流程。

## 目录结构

```
.
├── Code/                           # 主要代码目录
│   ├── BS_EV_Environment_Base.py   # 环境基类
│   ├── BS_EV_Environment_DQN.py    # DQN环境与训练
│   ├── BS_EV_Environment_SAC.py    # SAC环境与训练
│   ├── BS_EV_Environment_PPO.py    # PPO环境与训练
│   ├── BS_EV_Environment_DP.py     # DP环境与轨迹收集
│   ├── DecisionTransformer.py      # Decision Transformer模型与训练
│   ├── CollectTrajectories.py      # 轨迹收集脚本（命令行）
│   ├── train.py                    # Decision Transformer训练脚本（命令行）
│   └── main.ipynb                  # 交互式实验notebook
├── Data/                           # 数据文件
│   ├── charge                      # 充电需求数据
│   ├── traffic                     # 通信流量数据
│   ├── weather.csv                 # 天气数据
│   └── RTP.csv                     # 实时电价数据
├── Models/                         # 保存训练好的DT模型
├── Trajectories/                   # 保存收集到的轨迹
├── Log/                            # 日志文件
├── tmp/                            # 临时文件
└── README.md                       # 项目说明文档
```

## 数据准备

请将原始数据文件（`charge`, `traffic`, `weather.csv`, `RTP.csv`）放入`Data/`目录下。数据格式需与环境代码一致。

## 强化学习模型训练

每种RL算法（DQN、SAC、PPO）都可单独训练，训练时会自动保存模型和学习曲线。

- **DQN训练**  
  ```bash
  python Code/BS_EV_Environment_DQN.py
  ```

- **SAC训练**  
  ```bash
  python Code/BS_EV_Environment_SAC.py
  ```

- **PPO训练**  
  ```bash
  python Code/BS_EV_Environment_PPO.py
  ```

## 依赖环境

- Python 3.10
- 推荐使用requirements.txt一键安装依赖：

```bash
pip install -r requirements.txt
```

如需手动安装，核心依赖如下：
- numpy, pandas, torch, matplotlib, tqdm, scipy
