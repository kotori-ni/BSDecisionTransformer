import argparse
import os
import sys
import pickle

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

    print(f'正在使用{class_name}收集{num_episodes}条轨迹...')

    if model_key == 'dp':
        trajectories = env.collect_optimal_trajectories(num_episodes)
    elif model_key == 'ppo':
        # PPO需要policy参数
        from BS_EV_Environment_PPO import Agent
        policy = Agent(n_actions=env.n_actions, input_dims=env.n_states)
        trajectories = env.collect_trajectories(num_episodes, policy)
    else:
        trajectories = env.collect_trajectories(num_episodes)

    os.makedirs('Trajectories', exist_ok=True)
    save_path = f'Trajectories/trajectories_{model_key}_{args.episodes}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f'轨迹已保存到: {save_path}')

if __name__ == '__main__':
    main() 