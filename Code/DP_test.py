import numpy as np
import pickle
import os
import logging
import matplotlib.pyplot as plt
from BS_EV_Environment_Base import BS_EV_Base

log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'DP.log')

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
    def __init__(self, traces_file=None):
        super().__init__(traces_file=traces_file)
        
        # 从配置文件加载DP相关参数
        dp_config = self.config['dp']
        self.SOC_states = np.linspace(0.00, 1.01, dp_config['n_SOC_states'])
        self.gamma = dp_config['gamma']
        self.max_iterations = dp_config['max_iterations']
        self.convergence_threshold = dp_config['convergence_threshold']
        self.soc_step = self.SOC_states[1] - self.SOC_states[0]

    def _find_nearest_SOC_state(self, soc):
        # 优化：直接用数学方法
        idx = int(round((soc - self.SOC_states[0]) / self.soc_step))
        return min(max(idx, 0), len(self.SOC_states) - 1)

    def _is_action_feasible(self, soc, action):
        if action == 1:  # 充电
            return soc <= 1 - self.SOC_charge_rate
        elif action == 2:  # 放电
            return soc >= self.min_SOC + self.SOC_discharge_rate
        return True

    def _get_next_SOC_static(self, soc, action):
        if action == 1:  # 充电
            return min(1.0, soc + self.SOC_charge_rate)
        elif action == 2:  # 放电
            return max(self.min_SOC, soc - self.SOC_discharge_rate)
        else:  # 不操作
            return soc

    def _get_reward_static(self, soc, t, action, trace):
        # 复制 _get_reward 逻辑，但用参数而不是self属性
        # 只取用必要的环境参数
        action_SOC = action
        if (soc < self.min_SOC + self.SOC_discharge_rate and action_SOC == 2) or \
           (soc > 1 - self.SOC_charge_rate and action_SOC == 1):
            action_SOC = 0

        SOC_cost = 0 if action_SOC == 0 else self.SOC_per_cost
        pro = trace["pro_trace"][t]
        power_charge = self.charge2power(self.charge[t], pro)
        power_BS = self.traffic2power(self.traffic[t])
        power_renergy = self.weather2power(self.weather[t])

        if action_SOC == 1:
            power = max(power_BS * self.AC_DC_eff + power_charge +
                        self.SOC_charge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
        elif action_SOC == 2:
            power = max(power_BS + power_charge -
                        self.SOC_discharge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
        else:
            power = max(power_BS * self.AC_DC_eff + power_charge - power_renergy, 0)

        power_cost = self.RTP[t] * power / 100
        reward_charge = self.charge2reward(self.charge[t], pro, self.error, self.RTP[t], self.RTP, t)
        return reward_charge - SOC_cost - power_cost

    def value_iteration(self, trace):
        self.reset(trace)
        V = np.zeros((len(self.SOC_states), 24*30))
        policy = np.zeros((len(self.SOC_states), 24*30), dtype=int)
        action_counts = {0: 0, 1: 0, 2: 0}

        # 预先计算reward表
        reward_table = np.zeros((len(self.SOC_states), 3, 24*30))
        for t in range(24*30):
            for s_idx, soc in enumerate(self.SOC_states):
                for action in range(3):
                    reward_table[s_idx, action, t] = self._get_reward_static(soc, t, action, trace)

        for t in range(24*30-1, -1, -1):
            for s_idx, soc in enumerate(self.SOC_states):
                action_values = []
                feasible_actions = []
                for action in range(3):
                    if not self._is_action_feasible(soc, action):
                        continue
                    next_soc = self._get_next_SOC_static(soc, action)
                    next_s_idx = self._find_nearest_SOC_state(next_soc)
                    reward = reward_table[s_idx, action, t]
                    if t < 24*30-1:
                        action_value = reward + self.gamma * V[next_s_idx, t+1]
                    else:
                        action_value = reward
                    action_values.append(action_value)
                    feasible_actions.append(action)
                if action_values:
                    best_action_idx = np.argmax(action_values)
                    V[s_idx, t] = action_values[best_action_idx]
                    policy[s_idx, t] = feasible_actions[best_action_idx]
                    action_counts[feasible_actions[best_action_idx]] += 1
                else:
                    V[s_idx, t] = 0
                    policy[s_idx, t] = 0
                    action_counts[0] += 1
        return V, policy

    def collect_optimal_trajectories(self):
        logging.info(f"Starting optimal trajectory collection for {len(self.traces)} traces")
        
        trajectories = []
        trace_stats = []
        
        # 创建图形保存目录
        figure_dir = os.path.join(os.path.dirname(__file__), '..', 'Figure')
        os.makedirs(figure_dir, exist_ok=True)

        trace_rewards = []
        trace_actions = []
        
        for trace_idx in range(len(self.traces)):
            logging.info(f"Collecting trajectory for trace {trace_idx}/{len(self.traces)-1}")

            trace_action = []
            # 运行价值迭代获取最优策略
            V, policy = self.value_iteration(self.traces[trace_idx])

            # 重置环境，使用对应的 pro_trace
            state = self.reset(self.traces[trace_idx])
            done = False
            episode_rewards = []
            
            trajectory = {
                'states': [],
                'actions': [],
                'rtgs': [],
                'trace_idx': trace_idx
            }
            
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
                trace_action.append(action)
                
                # 执行动作
                reward = self._get_reward_static(self.SOC, self.T, action, self.traces[trace_idx])
                self.SOC = self._get_next_SOC_static(self.SOC, action)
                self.T += 1
                
                # 记录轨迹
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.int32))
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
            
            logging.info(f"Trace {trace_idx}: Total reward: {total_reward:.2f}")

            trace_rewards.append(total_reward)
            trace_actions.append(trace_action)
        
        self.trajectories = trajectories

        # 保存信息
        with open("../Data/pro_traces.pkl", "rb") as f:
            pro_traces = pickle.load(f)
            
        for idx in range(len(trace_rewards)):
            pro_traces[idx]['DP_reward'] = trace_rewards[idx]
            pro_traces[idx]['DP_action'] = trace_actions[idx]

        with open("../Data/pro_traces.pkl", "wb") as f:
            pickle.dump(pro_traces, f)
        
        # 记录整体统计信息
        total_rewards = [t['rtgs'][0] for t in trajectories]
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
        
        # 创建reward分布图
        plt.figure(figsize=(12, 6))
        
        # 计算bin的边界（以100为单位）
        min_reward = np.min(total_rewards)
        max_reward = np.max(total_rewards)
        bin_start = np.floor(min_reward / 100) * 100
        bin_end = np.ceil(max_reward / 100) * 100
        bins = np.arange(bin_start, bin_end + 100, 100)
        
        # 绘制柱状图
        plt.hist(total_rewards, bins=bins, edgecolor='black', alpha=0.7)
        
        # 添加统计信息
        plt.axvline(stats['mean_reward'], color='red', linestyle='--', label=f'Mean: {stats["mean_reward"]:.2f}')
        plt.axvline(stats['mean_reward'] + stats['std_reward'], color='green', linestyle=':', label=f'Mean+Std: {stats["mean_reward"] + stats["std_reward"]:.2f}')
        plt.axvline(stats['mean_reward'] - stats['std_reward'], color='green', linestyle=':', label=f'Mean-Std: {stats["mean_reward"] - stats["std_reward"]:.2f}')
        
        # 设置图形属性
        plt.title('Distribution of Total Rewards')
        plt.xlabel('Total Reward')
        plt.ylabel('Number of Traces')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存图形
        plt.savefig(os.path.join(figure_dir, 'dp_reward_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return trajectories

    def save_trajectories(self, filename='../Trajectories/optimal_dp.pkl'):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectories, f)
            logging.info(f"Optimal trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving optimal trajectories: {str(e)}")
            raise

if __name__ == "__main__":
    env = BS_EV_DP(traces_file="../Data/pro_traces.pkl")
    trajectories = env.collect_optimal_trajectories()
    env.save_trajectories()
