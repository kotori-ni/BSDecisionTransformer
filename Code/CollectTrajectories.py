import argparse
import os
import sys
import pickle
from tqdm import tqdm  # 导入 tqdm 用于进度条

# 动态导入
sys.path.append(os.path.dirname(__file__))

MODEL_MAP = {
    'dp': 'BS_EV_Environment_DP',
    'dqn': 'BS_EV_Environment_DQN',
    'sac': 'BS_EV_Environment_SAC',
    'ppo': 'BS_EV_Environment_PPO',
}

CLASS_MAP = {
    'dp': 'BS_EV_DP',
    'dqn': 'BS_EV_DQN',
    'sac': 'BS_EV_SAC',
    'ppo': 'BS_EV',
}

def main():
    parser = argparse.ArgumentParser(description='收集不同模型的轨迹数据')
    parser.add_argument('--model', type=str, required=True, choices=['dp', 'dqn', 'sac', 'ppo'], help='选择模型: dp, dqn, sac, ppo')
    parser.add_argument('--episodes', type=int, default=10, help='收集的episode数量')
    args = parser.parse_args()

    model_key = args.model.lower()
    module_name = MODEL_MAP[model_key]
    class_name = CLASS_MAP[model_key]
    num_episodes = args.episodes

    # 导入模块
    module = __import__(module_name)
    env_class = getattr(module, class_name)
    env = env_class()

    print(f'正在使用 {class_name} 收集 {num_episodes} 条轨迹...')

    # 初始化轨迹列表
    trajectories = []

    # 使用 tqdm 包装 episode 循环，显示进度条
    for episode in tqdm(range(num_episodes), desc="Collecting trajectories", unit="episode"):
        if model_key == 'dp':
            # DP 模型使用 collect_optimal_trajectories
            episode_trajectories = env.collect_optimal_trajectories(1)  # 收集单条轨迹
            trajectories.extend(episode_trajectories)
        elif model_key == 'ppo':
            # PPO 需要 policy 参数
            from BS_EV_Environment_PPO import Agent
            policy = Agent(n_actions=env.n_actions, input_dims=env.n_states)
            episode_trajectories = env.collect_trajectories(1, policy)  # 收集单条轨迹
            trajectories.extend(episode_trajectories)
        else:
            # DQN 和 SAC 使用 collect_trajectories
            episode_trajectories = env.collect_trajectories(1)  # 收集单条轨迹
            trajectories.extend(episode_trajectories)

    # 保存轨迹
    os.makedirs('Trajectories', exist_ok=True)
    save_path = f'Trajectories/trajectories_{model_key}_{args.episodes}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f'轨迹已保存到: {save_path}')

if __name__ == '__main__':
    main()