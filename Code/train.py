import torch
import pickle
import numpy as np
import argparse
import os
import json
from Code.BS_EV_Environment_PPO import BS_EV
from DecisionTransformer import (
    DecisionTransformer, 
    train_model,
    evaluate_on_trajectories,
    analyze_test_trajectories
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='轨迹文件路径')
    parser.add_argument('--model_dir', type=str, default='../Models', help='模型保存路径')
    args = parser.parse_args()

    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 加载轨迹数据
    try:
        with open(args.file, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"加载了 {len(trajectories)} 条轨迹")
    except FileNotFoundError:
        print(f"未找到轨迹文件 {args.file}")
        return

    # 分割数据集：训练集(70%)、验证集(10%)、测试集(20%)
    n_trajectories = len(trajectories)
    train_size = int(n_trajectories * 0.7)
    val_size = int(n_trajectories * 0.1)
    
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:train_size + val_size]
    test_trajectories = trajectories[train_size + val_size:]
    
    print(f"训练集大小: {len(train_trajectories)}")
    print(f"验证集大小: {len(val_trajectories)}")
    print(f"测试集大小: {len(test_trajectories)}")

    # 计算目标回报
    all_rewards = []
    for traj in train_trajectories:
        all_rewards.extend(traj['rewards'])
    target_rtg = np.mean(all_rewards) * len(train_trajectories[0]['rewards'])
    print(f"目标回报设置为: {target_rtg:.2f}")

    # 模型初始化
    model = DecisionTransformer(
        state_dim=config['DT_params']['state_dim'],
        action_dim=config['DT_params']['action_dim'],
        hidden_dim=config['DT_params']['hidden_dim'],
        n_layers=config['DT_params']['n_layers'],
        n_heads=config['DT_params']['n_heads'],
        max_len=config['DT_params']['max_len']
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 训练模型
    best_model_path, _ = train_model(
        model=model,
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        config=config,
        target_rtg=target_rtg,
        model_dir=args.model_dir,
        device=device
    )

    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(best_model_path))
    test_mean_reward, test_std_reward = evaluate_on_trajectories(
        model, test_trajectories, target_rtg, device
    )
    print(f"测试集平均奖励: {test_mean_reward:.2f} ± {test_std_reward:.2f}")

    # 分析测试轨迹
    analyze_test_trajectories(model, test_trajectories, target_rtg, interval=100, device=device)

if __name__ == "__main__":
    main()