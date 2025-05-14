import numpy as np
import pickle
import os
import logging
from BS_EV_Environment_Base import BS_EV_Base, load_RTP, load_weather, load_traffic, load_charge

log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'dp.log')

logger = logging.getLogger()
logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

class BS_EV_DP(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', trace_idx=0):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file, trace_idx)
        self.set_mode('test')  # 默认设置为测试模式
        
        # 从配置文件加载DP相关参数
        dp_config = self.config['dp']
        self.SOC_states = np.linspace(self.min_SOC, 1.0, dp_config['n_SOC_states'])
        self.gamma = dp_config['gamma']
        self.max_iterations = dp_config['max_iterations']
        self.convergence_threshold = dp_config['convergence_threshold']

    def _find_nearest_SOC_state(self, soc):
        return np.abs(self.SOC_states - soc).argmin()

    def _is_action_feasible(self, soc, action):
        if action == 1:  # 充电
            return soc <= 1 - self.SOC_charge_rate
        elif action == 2:  # 放电
            return soc >= self.min_SOC + self.SOC_discharge_rate
        return True

    def _get_reward(self, action):
        # 计算奖励，基于充电奖励、储能操作成本和电费成本
        action_SOC = action
        
        # 防止电池过充或过放
        if (self.SOC < self.min_SOC + self.SOC_discharge_rate and action_SOC == 2) or \
           (self.SOC > 1 - self.SOC_charge_rate and action_SOC == 1):
            action_SOC = 0

        # 计算储能操作成本
        SOC_cost = 0 if action_SOC == 0 else self.SOC_per_cost
        
        # 使用当前时间步的 pro 值
        pro = self.current_pro_trace[self.T]
        power_charge = self.charge2power(self.charge[self.T], pro)  # 电动车充电功率
        power_BS = self.traffic2power(self.traffic[self.T])  # 基站功率需求
        power_renergy = self.weather2power(self.weather[self.T])  # 可再生能源功率

        # 计算净用电量（从电网购买的电量）
        if action_SOC == 1:  # 储能系统充电
            power = max(power_BS * self.AC_DC_eff + power_charge + \
                        self.SOC_charge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
        elif action_SOC == 2:  # 储能系统放电
            power = max(power_BS + power_charge - \
                        self.SOC_discharge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
        else:  # 不操作
            power = max(power_BS * self.AC_DC_eff + power_charge - power_renergy, 0)

        # 计算电费成本
        power_cost = self.RTP[self.T] * power / 100

        # 计算充电奖励
        reward_charge = self.charge2reward(self.charge[self.T], pro, self.error)

        # 总奖励 = 充电奖励 - 储能成本 - 电费成本
        return reward_charge - SOC_cost - power_cost

    def value_iteration(self, trace_idx):
        logging.info(f"Trace {trace_idx}: Starting value iteration...")

        # 初始化数据
        self.RTP = load_RTP(train_flag=False, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
        self.weather = load_weather(train_flag=False, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
        self.traffic = load_traffic(config=self.config)
        self.charge = load_charge(config=self.config)
        self.current_pro_trace = self.pro_traces[trace_idx]["pro_trace"]
        
        # 初始化价值函数和策略
        V = np.zeros((len(self.SOC_states), 24*30))
        policy = np.zeros((len(self.SOC_states), 24*30), dtype=int)
        
        # 记录动作分布
        action_counts = {0: 0, 1: 0, 2: 0}
        
        # 单次逆序遍历
        for t in range(24*30-1, -1, -1):
            self.T = t
            for s_idx, soc in enumerate(self.SOC_states):
                self.SOC = soc
                
                # 计算每个动作的价值
                action_values = []
                feasible_actions = []
                
                for action in range(3):
                    if not self._is_action_feasible(soc, action):
                        continue
                        
                    next_soc = self._get_next_SOC(action)
                    next_s_idx = self._find_nearest_SOC_state(next_soc)
                    
                    reward = self._get_reward(action)
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
        
        # 记录策略统计信息
        logging.info(f"Trace {trace_idx}: Value iteration completed")
        logging.info(f"Trace {trace_idx}: Action distribution: {action_counts}")
        logging.info(f"Trace {trace_idx}: Final policy statistics:")
        for action, count in action_counts.items():
            logging.info(f"Trace {trace_idx}: Action {action}: {count} times")
        
        return V, policy, trace_idx

    def collect_optimal_trajectories(self):
        """在测试模式下收集最优轨迹"""
        logging.info(f"Starting optimal trajectory collection for {len(self.pro_traces)} traces")
        
        trajectories = []
        trace_stats = []
        
        # 确保在测试模式下运行
        self.set_mode('test')
        
        for trace_idx in range(len(self.pro_traces)):
            logging.info(f"Collecting trajectory for trace {trace_idx}/{len(self.pro_traces)-1}")
            
            # 运行价值迭代获取最优策略
            V, policy, _ = self.value_iteration(trace_idx)
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'rtgs': [],
                'dones': [],
                'trace_idx': trace_idx
            }
            
            # 重置环境，使用对应的 pro_trace
            self.trace_idx = trace_idx
            state = self.reset(trace_idx=trace_idx)  # 使用测试集的trace_idx
            done = False
            episode_rewards = []
            
            # 记录trace的统计信息
            trace_stat = {
                'trace_idx': trace_idx,
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
                reward = self._get_reward(action)
                self.SOC = self._get_next_SOC(action)
                self.T += 1
                
                # 记录轨迹
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.int32))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                
                # 更新统计信息
                trace_stat['soc_values'].append(self.SOC)
                trace_stat['action_counts'][action] += 1
                trace_stat['rewards'].append(reward)
                
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
            trace_stats.append(trace_stat)
            
            # 记录trace的统计信息
            total_reward = sum(episode_rewards)
            mean_soc = np.mean(trace_stat['soc_values'])
            std_soc = np.std(trace_stat['soc_values'])
            
            logging.info(f"Trace {trace_idx}: Completed")
            logging.info(f"Trace {trace_idx}: Total reward: {total_reward:.2f}")
            logging.info(f"Trace {trace_idx}: Mean SOC: {mean_soc:.3f}, Std SOC: {std_soc:.3f}")
            logging.info(f"Trace {trace_idx}: Action distribution: {trace_stat['action_counts']}")
        
        self.trajectories = trajectories
        
        # 记录整体统计信息
        total_rewards = [sum(t['rewards']) for t in trajectories]
        all_socs = [soc for stat in trace_stats for soc in stat['soc_values']]
        all_actions = [action for stat in trace_stats for action, count in stat['action_counts'].items() for _ in range(count)]
        
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

    def save_trajectories(self, filename='../Trajectories/optimal_trajectories_dp.pkl'):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectories, f)
            logging.info(f"Optimal trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving optimal trajectories: {str(e)}")
            raise

    def load_trajectories(self, filename='../Trajectories/optimal_trajectories_dp.pkl'):
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

if __name__ == "__main__":
    env = BS_EV_DP()
    trajectories = env.collect_optimal_trajectories()
    env.save_trajectories()