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
  - 模型和曲线保存在`figure/learning_curve_DQN.png`等。

- **SAC训练**  
  ```bash
  python Code/BS_EV_Environment_SAC.py
  ```
  - 模型和曲线保存在`figure/learning_curve_SAC.png`等。

- **PPO训练**  
  ```bash
  python Code/BS_EV_Environment_PPO.py
  ```
  - 模型和曲线保存在`figure/learning_curve_PPO.png`等。

## 轨迹收集

使用命令行脚本`CollectTrajectories.py`，可指定算法类型和episode数量，自动收集轨迹并保存到`Trajectories/`目录。

```bash
python Code/CollectTrajectories.py --model dqn --episodes 20
python Code/CollectTrajectories.py --model dp --episodes 10
python Code/CollectTrajectories.py --model ppo --episodes 15
python Code/CollectTrajectories.py --model sac --episodes 10
```
- 轨迹文件将保存为`Trajectories/trajectories_{model}.pkl`。

## Decision Transformer训练与评估

使用`train.py`脚本，指定轨迹类型（dp/dqn/sac/ppo）进行DT训练，模型自动保存到`Models/`目录。

```bash
python Code/train.py --type dqn --epochs 50 --batch_size 8 --lr 5e-5
python Code/train.py --type ppo
```
- 训练完成后模型保存在`Models/dt_model_{type}.pth`。
- 评估时会优先加载`Models/dt_model_best_{type}.pth`（如有），否则加载刚刚训练的模型，并输出评估统计。

## 流程

1. **训练RL模型**（任选DQN/SAC/PPO）
2. **收集轨迹**  
   `python Code/CollectTrajectories.py --model dqn --episodes 20`
3. **训练Decision Transformer**  
   `python Code/train.py --type dqn`
4. **查看模型与评估结果**  
   - 模型在`Models/`，评估结果在终端输出

## 依赖环境

- Python 3.10
- 推荐使用requirements.txt一键安装依赖：

```bash
pip install -r requirements.txt
```

如需手动安装，核心依赖如下：
- numpy, pandas, torch, matplotlib, tqdm, scipy
