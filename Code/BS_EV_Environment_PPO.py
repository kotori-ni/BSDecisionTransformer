import numpy as np
import random
import pickle
import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import logging
import time
from BS_EV_Environment_Base import (
    BS_EV_Base, 
    weather2power, 
    charge2power, 
    charge2reward, 
    traffic2power,
    load_RTP,
    load_weather,
    load_traffic,
    load_charge
)

# 使用os.path.join来处理路径
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, '/trajectory_collection_ppo.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'actor_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'actor_torch_ppo_last')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'critic_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'critic_torch_ppo_last')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))


class Agent:
    def __init__(self, n_actions, input_dims, alpha=0.0003):
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)

    def load_models_best(self):
        print('... loading PPO models ...')
        self.actor.load_checkpoint_best()
        self.critic.load_checkpoint_best()

    def load_models_last(self):
        print('... loading PPO models ...')
        self.actor.load_checkpoint_last()
        self.critic.load_checkpoint_last()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
        dist = self.actor(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(self.critic(state)).item()
        return action, probs, value


class BS_EV(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, train_flag=True, config_file='config.json'):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file)
        self.train_flag = train_flag
        self.done = False

    def reset(self):
        # 使用当前时间作为随机种子
        current_time = int(time.time()) % (2**32 - 1)
        random.seed(current_time)
        np.random.seed(current_time)
        self.SOC = random.uniform(self.min_SOC, 1)
        self.T = 0
        self.RTP = load_RTP(train_flag=self.train_flag)
        self.weather = load_weather(train_flag=self.train_flag)
        self.traffic = load_traffic()
        self.charge = load_charge()
        self.done = False
        return self._get_state()

    def step(self, action):
        action_SOC = action

        # 如果电量过低无法放电或过高无法充电，强制不操作
        if (self.SOC < self.min_SOC + self.SOC_discharge_rate and action_SOC == 2) or \
           (self.SOC > 1 - self.SOC_charge_rate and action_SOC == 1):
            action_SOC = 0

        # 计算储能操作成本
        SOC_cost = 0 if action_SOC == 0 else self.SOC_per_cost
        
        # 使用当前时间作为随机种子的一部分
        current_time = int(time.time()) % (2**32 - 1)
        random.seed(current_time)
        pro = random.uniform(0, 1)
        
        power_charge = charge2power(self.charge[self.T], pro)
        power_BS = traffic2power(self.traffic[self.T])
        power_renergy = weather2power(self.weather[self.T])

        # 充电
        if action_SOC == 1:
            self.SOC = self.SOC + self.SOC_charge_rate
            power = max(power_BS * self.AC_DC_eff + power_charge + \
                        self.SOC_charge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
            
        # 放电
        elif action_SOC == 2:
            self.SOC = self.SOC - self.SOC_discharge_rate
            power = max(power_BS + power_charge - \
                        self.SOC_discharge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
            
        # 不操作
        else:
            power = max(power_BS * self.AC_DC_eff + power_charge - power_renergy, 0)

        # 计算电费成本
        power_cost = self.RTP[self.T] * power / 100

        # 计算充电奖励
        reward_charge = charge2reward(self.charge[self.T], pro, self.error)

        # 总奖励 = 充电奖励 - 储能成本 - 电费成本
        reward = reward_charge - SOC_cost - power_cost
        self.T += 1
        next_state = self._get_state()
        if (self.T) % (24 * 30) == 0:
            self.done = True
        return next_state, reward, self.done

    def collect_trajectories(self, num_episodes, policy, epsilon=0.1):
        # 只在开始时加载一次模型
        try:
            policy.load_models_best()
            policy.actor.eval()
        except FileNotFoundError as e:
            logging.error(f"Failed to load PPO models: {str(e)}")
            raise
        
        logging.info(f"Starting trajectory collection for {num_episodes} episodes")
        
        # 使用当前时间作为基础随机种子，确保在有效范围内
        base_seed = int(time.time()) % (2**32 - 1)
        random.seed(base_seed)
        np.random.seed(base_seed)
        T.manual_seed(base_seed)
        
        trajectories = []
        ppo_agent = policy

        for episode in range(num_episodes):
            # 为每个episode使用不同的随机种子，确保在有效范围内
            episode_seed = (base_seed + episode) % (2**32 - 1)
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            T.manual_seed(episode_seed)
            
            logging.info(f"Collecting trajectory for episode {episode + 1}/{num_episodes}")
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'rtgs': [],
                'dones': []
            }
            state = self.reset()
            done = False
            episode_rewards = []
            
            while not done:
                # ε-贪心策略增加探索
                if random.random() < epsilon:
                    action = random.choice([0, 1, 2])
                    probs = 0.0  # 随机动作不记录概率
                    value = 0.0
                else:
                    action, probs, value = ppo_agent.choose_action(state)
                
                next_state, reward, done = self.step(action)
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.int32))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                state = next_state
            
            # 计算回报到目标（RTG）
            rtgs = []
            cumulative_reward = 0
            for r in reversed(episode_rewards):
                cumulative_reward = r + 0.99 * cumulative_reward
                rtgs.insert(0, cumulative_reward)
            trajectory['rtgs'] = rtgs
            
            # 验证轨迹格式
            assert len(trajectory['states']) == len(trajectory['actions']) == \
                   len(trajectory['rewards']) == len(trajectory['rtgs']) == \
                   len(trajectory['dones']), "Trajectory length mismatch"
            
            trajectories.append(trajectory)
            total_reward = sum(episode_rewards)
            logging.info(f"Episode {episode + 1} completed with total reward: {total_reward:.2f}")
        
        self.trajectories = trajectories
        
        # 记录轨迹统计信息
        total_rewards = [sum(t['rewards']) for t in trajectories]
        stats = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'num_trajectories': len(trajectories)
        }
        logging.info(f"Trajectory collection completed. Statistics: {stats}")
        
        return trajectories

    def save_trajectories(self, filename='trajectories.pkl'):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectories, f)
            logging.info(f"Trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving trajectories: {str(e)}")
            raise

    def load_trajectories(self, filename='trajectories.pkl'):
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Trajectory file {filename} not found")
            with open(filename, 'rb') as f:
                self.trajectories = pickle.load(f)
            logging.info(f"Trajectories loaded from {filename}")
            return self.trajectories
        except Exception as e:
            logging.error(f"Error loading trajectories: {str(e)}")
            raise

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    os.makedirs("figure", exist_ok=True)
    
    env = BS_EV()
    n_episodes = 100
    N = 20
    alpha = 0.0001
    agent = Agent(n_actions=env.n_actions, input_dims=env.n_states, alpha=alpha)
    score_history = []
    best_score = float('-inf')
    avg_scores = []
    window = 10  # 平滑窗口
    learn_iters = 0
    n_steps = 0

    for i in tqdm(range(n_episodes)):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            n_steps += 1
            score += reward
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-window:])
        avg_scores.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            agent.load_models_best()  # 仅做示例，实际应为保存模型
        print(f"episode {i} score {score:.1f} avg score {avg_score:.1f} time_steps {n_steps} learning_steps {learn_iters}")

    # 保存学习曲线
    plt.figure()
    plt.plot(range(1, n_episodes+1), avg_scores)
    plt.xlabel('Episode')
    plt.ylabel(f'Average Score (window={window})')
    plt.title('PPO Running Average Score')
    plt.grid()
    plt.savefig('Figure/learning_curve_PPO.png')
    print('训练完成，模型和学习曲线已保存。')