import numpy as np
import pickle
import os
import logging
import time
from BS_EV_Environment_Base import BS_EV_Base


# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
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
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json'):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file)
        
        # 从配置文件加载DP相关参数
        dp_config = self.config['dp']
        self.SOC_states = np.linspace(self.min_SOC, 1.0, dp_config['n_SOC_states'])
        self.gamma = dp_config['gamma']
        self.max_iterations = dp_config['max_iterations']
        self.convergence_threshold = dp_config['convergence_threshold']
        
        # 初始化随机数生成器
        self.rng = np.random.RandomState(int(time.time()))

    def _find_nearest_SOC_state(self, soc):
        return np.abs(self.SOC_states - soc).argmin()

    def _is_action_feasible(self, soc, action):
        if action == 1:  # 充电
            return soc <= 1 - self.SOC_charge_rate
        elif action == 2:  # 放电
            return soc >= self.min_SOC + self.SOC_discharge_rate
        return True

    def value_iteration(self):
        logging.info("Starting value iteration...")
        
        # 初始化价值函数和策略
        V = np.zeros((len(self.SOC_states), 24*30))
        policy = np.zeros((len(self.SOC_states), 24*30), dtype=int)
        
        # 记录迭代过程中的统计信息
        iteration_stats = []
        
        for iteration in range(self.max_iterations):
            delta = 0
            action_counts = {0: 0, 1: 0, 2: 0}  # 记录每个动作的选择次数
            
            for t in range(24*30-1, -1, -1):
                self.T = t
                for s_idx, soc in enumerate(self.SOC_states):
                    self.SOC = soc
                    v = V[s_idx, t]
                    
                    # 计算每个动作的价值
                    action_values = []
                    feasible_actions = []
                    
                    for action in range(3):
                        if not self._is_action_feasible(soc, action):
                            continue
                            
                        next_soc = self._get_next_SOC(action)
                        next_s_idx = self._find_nearest_SOC_state(next_soc)
                        
                        # 使用随机数生成器生成随机数
                        pro = self.rng.uniform(0, 1)
                        
                        reward = self._get_reward(action, pro)
                        if t < 24*30-1:
                            action_value = reward + self.gamma * V[next_s_idx, t+1]
                        else:
                            action_value = reward
                            
                        action_values.append(action_value)
                        feasible_actions.append(action)
                    
                    # 更新价值函数和策略
                    if action_values:
                        best_action_idx = np.argmax(action_values)
                        V[s_idx, t] = action_values[best_action_idx]
                        policy[s_idx, t] = feasible_actions[best_action_idx]
                        action_counts[feasible_actions[best_action_idx]] += 1
                    else:
                        # 如果没有可行动作，默认选择不操作
                        V[s_idx, t] = 0
                        policy[s_idx, t] = 0
                        action_counts[0] += 1
                    
                    delta = max(delta, abs(v - V[s_idx, t]))
            
            # 记录每次迭代的统计信息
            iteration_stats.append({
                'iteration': iteration + 1,
                'delta': delta,
                'action_distribution': action_counts
            })
            
            logging.info(f"Iteration {iteration + 1}, delta: {delta}")
            logging.info(f"Action distribution: {action_counts}")
            
            if delta < self.convergence_threshold:
                logging.info("Value iteration converged!")
                break
        
        # 记录最终的策略统计信息
        logging.info("Final policy statistics:")
        for action, count in action_counts.items():
            logging.info(f"Action {action}: {count} times")
        
        return V, policy

    def collect_optimal_trajectories(self, num_episodes):
        logging.info(f"Starting optimal trajectory collection for {num_episodes} episodes")
        
        # 运行价值迭代获取最优策略
        V, policy = self.value_iteration()
        
        trajectories = []
        episode_stats = []  # 记录每个episode的统计信息
        
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
            
            # 记录episode的统计信息
            episode_stat = {
                'soc_values': [],
                'action_counts': {0: 0, 1: 0, 2: 0},
                'rewards': []
            }
            
            while not done:
                # 获取当前SOC对应的状态索引
                soc_idx = self._find_nearest_SOC_state(self.SOC)
                
                # 使用最优策略选择动作
                action = policy[soc_idx, self.T]
                
                # 执行动作
                pro = self.rng.uniform(0, 1)
                
                reward = self._get_reward(action, pro)
                self.SOC = self._get_next_SOC(action)
                self.T += 1
                
                # 记录轨迹
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.int32))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                
                # 更新统计信息
                episode_stat['soc_values'].append(self.SOC)
                episode_stat['action_counts'][action] += 1
                episode_stat['rewards'].append(reward)
                
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
            episode_stats.append(episode_stat)
            
            # 记录episode的统计信息
            total_reward = sum(episode_rewards)
            mean_soc = np.mean(episode_stat['soc_values'])
            std_soc = np.std(episode_stat['soc_values'])
            
            logging.info(f"Episode {episode + 1} completed:")
            logging.info(f"Total reward: {total_reward:.2f}")
            logging.info(f"Mean SOC: {mean_soc:.3f}, Std SOC: {std_soc:.3f}")
            logging.info(f"Action distribution: {episode_stat['action_counts']}")
        
        self.trajectories = trajectories
        
        # 记录整体统计信息
        total_rewards = [sum(t['rewards']) for t in trajectories]
        all_socs = [soc for stat in episode_stats for soc in stat['soc_values']]
        all_actions = [action for stat in episode_stats for action, count in stat['action_counts'].items() for _ in range(count)]
        
        stats = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'num_trajectories': len(trajectories),
            'mean_soc': np.mean(all_socs),
            'std_soc': np.std(all_socs),
            'action_distribution': {
                action: all_actions.count(action) for action in range(3)
            }
        }
        logging.info(f"Trajectory collection completed. Statistics: {stats}")
        
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