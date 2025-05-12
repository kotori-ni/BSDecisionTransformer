import numpy as np
import pandas as pd
import pickle
import os
import logging
import time
from BS_EV_Environment_Base import BS_EV_Base


# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'trajectory_collection_dp.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

class BS_EV_DP(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, error=1.0):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, error)
        
        # 动态规划相关参数
        self.SOC_states = np.linspace(self.min_SOC, 1.0, 50)  # 将SOC离散化为50个状态
        self.gamma = 0.99  # 折扣因子
        self.max_iterations = 1000  # 最大迭代次数
        self.convergence_threshold = 1e-6  # 收敛阈值

    def _find_nearest_SOC_state(self, soc):
        return np.abs(self.SOC_states - soc).argmin()

    def value_iteration(self):
        logging.info("Starting value iteration...")
        
        # 初始化价值函数和策略
        V = np.zeros((len(self.SOC_states), 24*30))
        policy = np.zeros((len(self.SOC_states), 24*30), dtype=int)
        
        for iteration in range(self.max_iterations):
            delta = 0
            for t in range(24*30-1, -1, -1):
                self.T = t
                for s_idx, soc in enumerate(self.SOC_states):
                    self.SOC = soc
                    v = V[s_idx, t]
                    
                    # 计算每个动作的价值
                    action_values = []
                    for action in range(3):
                        if (soc < self.min_SOC + self.SOC_discharge_rate and action == 2) or \
                           (soc > 1 - self.SOC_charge_rate and action == 1):
                            action_values.append(float('-inf'))
                            continue
                            
                        next_soc = self._get_next_SOC(action)
                        next_s_idx = self._find_nearest_SOC_state(next_soc)
                        
                        # 使用当前时间作为随机种子
                        current_time = int(time.time()) % (2**32 - 1)
                        np.random.seed(current_time)
                        pro = np.random.uniform(0, 1)
                        
                        reward = self._get_reward(action, pro)
                        if t < 24*30-1:
                            action_value = reward + self.gamma * V[next_s_idx, t+1]
                        else:
                            action_value = reward
                            
                        action_values.append(action_value)
                    
                    # 更新价值函数和策略
                    if action_values:
                        V[s_idx, t] = max(action_values)
                        policy[s_idx, t] = np.argmax(action_values)
                    
                    delta = max(delta, abs(v - V[s_idx, t]))
            
            logging.info(f"Iteration {iteration + 1}, delta: {delta}")
            if delta < self.convergence_threshold:
                logging.info("Value iteration converged!")
                break
        
        return V, policy

    def collect_optimal_trajectories(self, num_episodes):
        logging.info(f"Starting optimal trajectory collection for {num_episodes} episodes")
        
        # 运行价值迭代获取最优策略
        V, policy = self.value_iteration()
        
        trajectories = []
        for episode in range(num_episodes):
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
                # 获取当前SOC对应的状态索引
                soc_idx = self._find_nearest_SOC_state(self.SOC)
                
                # 使用最优策略选择动作
                action = policy[soc_idx, self.T]
                
                # 执行动作
                current_time = int(time.time()) % (2**32 - 1)
                np.random.seed(current_time)
                pro = np.random.uniform(0, 1)
                
                reward = self._get_reward(action, pro)
                self.SOC = self._get_next_SOC(action)
                self.T += 1
                
                # 记录轨迹
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.int32))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                
                state = self._get_state()
                done = (self.T >= 24*30)
            
            # 计算回报到目标（RTG）
            rtgs = []
            cumulative_reward = 0
            for r in reversed(episode_rewards):
                cumulative_reward = r + self.gamma * cumulative_reward
                rtgs.insert(0, cumulative_reward)
            trajectory['rtgs'] = rtgs
            
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
        logging.info(f"Optimal trajectory collection completed. Statistics: {stats}")
        
        return trajectories

    def save_trajectories(self, filename='optimal_trajectories.pkl'):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectories, f)
            logging.info(f"Optimal trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving optimal trajectories: {str(e)}")
            raise

    def load_trajectories(self, filename='optimal_trajectories.pkl'):
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Optimal trajectory file {filename} not found")
            with open(filename, 'rb') as f:
                self.trajectories = pickle.load(f)
            logging.info(f"Optimal trajectories loaded from {filename}")
            return self.trajectories
        except Exception as e:
            logging.error(f"Error loading optimal trajectories: {str(e)}")
            raise 