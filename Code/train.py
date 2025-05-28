import torch
import pickle
import argparse
import os
from BS_EV_Environment_Base import BS_EV_Base
from DecisionTransformer import DecisionTransformer, train_decision_transformer, evaluate_on_trajectory
import json
import logging
import numpy as np

# 确保日志配置正确
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train.log')

# 配置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 检查是否已有处理器，避免重复添加
if not logger.handlers:
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    logging.info("=" * 50)
    logging.info(f"开始新的训练任务: {args.file}")
    logging.info(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # 环境初始化
    env = BS_EV_Base(n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, train_flag=True, pro_traces=None)
    
    # 加载轨迹数据
    traj_file = args.file
    try:
        with open(args.file, 'rb') as f:
            trajectories = pickle.load(f)
        logging.info(f"加载了 {len(trajectories)} 条轨迹: {traj_file}")
    except FileNotFoundError:
        logging.error(f"未找到轨迹文件 {traj_file}，请先运行轨迹收集")
        return

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")

    # 80% 用于训练, 20% 用于测试
    num_total = len(trajectories)
    num_train = int(0.8 * num_total)
    train_trajectories = trajectories[:num_train]
    test_trajectories = trajectories[num_train:]

    logging.info(f"训练集: {len(train_trajectories)} 条, 测试集: {len(test_trajectories)} 条")

    target_rtg = 44000.0  # 默认值，可根据环境实际情况调整
    logging.info(f"目标回报设置为: {target_rtg:.2f}")

    # -----------------------------------
    # 从 config.json 读取 Decision Transformer 超参数
    # -----------------------------------
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r', encoding='utf-8') as cf:
        config = json.load(cf)
    dt_config = config.get('decision_transformer', {})

    # 模型初始化
    model = DecisionTransformer(
        state_dim=env.n_states,
        action_dim=env.n_actions,
        config=dt_config
    )
    
    logging.info("开始训练模型...")
    train_decision_transformer(
        model=model,
        trajectories=train_trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    os.makedirs('../Models', exist_ok=True)
    model_name = os.path.splitext(os.path.basename(traj_file))[0]
    model_path = f'../Models/dt_model_{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已保存到 {model_path}")

if __name__ == "__main__":
    main()