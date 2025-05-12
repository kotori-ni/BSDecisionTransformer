import torch
import pickle
import numpy as np
import argparse
import os
from Code.BS_EV_Environment_PPO import BS_EV
from DecisionTransformer import DecisionTransformer, train_decision_transformer, evaluate_decision_transformer

def main():
    parser = argparse.ArgumentParser(description='DecisionTransformer训练脚本')
    parser.add_argument('--type', type=str, required=True, choices=['dp', 'dqn', 'sac', 'ppo'], help='选择用于训练的轨迹类型')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    args = parser.parse_args()

    # 环境初始化
    env = BS_EV(n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, train_flag=True)
    
    # 加载轨迹数据
    traj_file = f'Trajectory/trajectories_{args.traj_type}.pkl'
    try:
        with open(traj_file, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"加载了 {len(trajectories)} 条{args.traj_type.upper()}轨迹")
    except FileNotFoundError:
        print(f"未找到轨迹文件 {traj_file}，请先运行轨迹收集")
        return

    # 计算目标回报
    all_rewards = []
    for traj in trajectories:
        all_rewards.extend(traj['rewards'])
    target_rtg = 40000
    print(f"目标回报设置为: {target_rtg:.2f}")

    # 模型初始化
    model = DecisionTransformer(
        state_dim=env.n_states,
        action_dim=env.n_actions,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        max_len=30
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("开始训练模型...")
    train_decision_transformer(
        model=model,
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    os.makedirs('Models', exist_ok=True)
    model_path = f'Models/dt_model_{args.traj_type}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # 评估模型
    print("\n开始评估模型...")
    # 若有best模型可加载best，否则加载刚刚训练的
    best_model_path = f'Models/dt_model_best_{args.traj_type}.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"加载最优模型: {best_model_path}")
    else:
        model.load_state_dict(torch.load(model_path))
        print(f"加载刚刚训练的模型: {model_path}")
    eval_rewards = []
    for _ in range(30):
        reward = evaluate_decision_transformer(model, env, target_rtg, max_len=30, device=device)
        eval_rewards.append(reward)
        print(f"评估回合奖励: {reward:.2f}")
    
    print(f"\n评估结果:")
    print(f"平均奖励: {np.mean(eval_rewards):.2f}")
    print(f"标准差: {np.std(eval_rewards):.2f}")
    print(f"最大奖励: {np.max(eval_rewards):.2f}")
    print(f"最小奖励: {np.min(eval_rewards):.2f}")

if __name__ == "__main__":
    main()