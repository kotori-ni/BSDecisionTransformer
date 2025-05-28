import numpy as np
import random
import os
import logging
from BS_EV_Environment_Base import BS_EV_Base, load_pro_traces, generate_pro_traces

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'DP.log')
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

def dynamic_programming_max_reward(T, trace_idx=0):
    # 确保pro_traces文件存在
    pro_traces_file = "../Data/pro_traces.pkl"
    if not os.path.exists(pro_traces_file):
        generate_pro_traces(sum_traces=10, T=T, output_file=pro_traces_file)
    
    # 加载pro_traces
    pro_traces = load_pro_traces(pro_traces_file)
    
    # 创建环境实例，传递必需的pro_traces参数
    env = BS_EV_Base(pro_traces=pro_traces)
    
    # 重置环境并获取数据
    env.reset(trace_idx=trace_idx)
    
    # 获取环境数据
    pro = env.trace["pro_trace"]

    RTP = env.RTP
    traffic = env.traffic
    weather = env.weather
    charge = env.charge

    # 环境参数
    min_SOC = env.min_SOC
    SOC_charge_rate = env.SOC_charge_rate
    SOC_discharge_rate = env.SOC_discharge_rate
    SOC_eff = env.SOC_eff
    AC_DC_eff = env.AC_DC_eff
    SOC_per_cost = env.SOC_per_cost
    ESS_cap = env.ESS_cap
    error = env.error
    T_hours = 24 * T  # 转换为小时数

    # 离散化SOC
    SOC_values = np.arange(0.00, 1.01, 0.01)
    SOC_values = np.round(SOC_values, 2)
    n_SOC = len(SOC_values)
    SOC_to_idx = {soc: idx for idx, soc in enumerate(SOC_values)}

    # 价值函数和策略
    V = np.full((T_hours + 1, n_SOC), -np.inf)
    policy = np.zeros((T_hours, n_SOC), dtype=int)
    V[T_hours, :] = 0  # 边界条件

    # 动态规划
    for t in range(T_hours - 1, -1, -1):
        for soc_idx, soc in enumerate(SOC_values):
            max_reward = -np.inf
            best_action = 0
            possible_actions = [0]
            if soc <= 1.0 - SOC_charge_rate:
                possible_actions.append(1)
            if soc >= min_SOC + SOC_discharge_rate:
                possible_actions.append(2)

            for action in possible_actions:
                SOC_cost = SOC_per_cost if action in [1, 2] else 0
                power_charge = env.charge2power(charge[t], pro[t])
                power_BS = env.traffic2power(traffic[t])
                power_renergy = env.weather2power(weather[t])

                if action == 1:
                    power = max(power_BS * AC_DC_eff + power_charge + SOC_charge_rate * ESS_cap * SOC_eff - power_renergy, 0)
                    next_SOC = min(soc + SOC_charge_rate, 1.0)
                elif action == 2:
                    power = max(power_BS + power_charge - SOC_discharge_rate * ESS_cap * SOC_eff - power_renergy, 0)
                    next_SOC = max(soc - SOC_discharge_rate, min_SOC)
                else:
                    power = max(power_BS * AC_DC_eff + power_charge - power_renergy, 0)
                    next_SOC = soc

                power_cost = RTP[t] * power / 100
                reward_charge = env.charge2reward(charge[t], pro[t], error, RTP[t], RTP, t)
                reward = reward_charge - SOC_cost - power_cost

                next_SOC = round(next_SOC, 2)
                if next_SOC in SOC_to_idx:
                    next_soc_idx = SOC_to_idx[next_SOC]
                    total_reward = reward + V[t + 1, next_soc_idx]
                    if total_reward > max_reward:
                        max_reward = total_reward
                        best_action = action

            V[t, soc_idx] = max_reward
            policy[t, soc_idx] = best_action

    # 回溯最优策略
    optimal_actions = []
    soc = 0.5  # 初始SOC
    soc = min(SOC_values, key=lambda x: abs(x - soc))
    soc_idx = SOC_to_idx[soc]
    total_reward = 0
    soc_sequence = [soc]  # 记录SOC序列

    # 无动作策略
    no_action_soc = 0.5
    no_action_soc = min(SOC_values, key=lambda x: abs(x - no_action_soc))
    no_action_soc_idx = SOC_to_idx[no_action_soc]
    no_action_total_reward = 0
    no_action_soc_sequence = [no_action_soc]

    for t in range(T_hours):
        # DP策略
        action = policy[t, soc_idx]
        optimal_actions.append(action)

        SOC_cost = SOC_per_cost if action in [1, 2] else 0
        power_charge = env.charge2power(charge[t], pro[t])
        power_BS = env.traffic2power(traffic[t])
        power_renergy = env.weather2power(weather[t])

        if action == 1:
            power = max(power_BS * AC_DC_eff + power_charge + SOC_charge_rate * ESS_cap * SOC_eff - power_renergy, 0)
            next_SOC = min(soc + SOC_charge_rate, 1.0)
        elif action == 2:
            power = max(power_BS + power_charge - SOC_discharge_rate * ESS_cap * SOC_eff - power_renergy, 0)
            next_SOC = max(soc - SOC_discharge_rate, min_SOC)
        else:
            power = max(power_BS * AC_DC_eff + power_charge - power_renergy, 0)
            next_SOC = soc

        power_cost = RTP[t] * power / 100
        reward_charge = env.charge2reward(charge[t], pro[t], error, RTP[t], RTP, t)
        reward = reward_charge - SOC_cost - power_cost
        total_reward += reward

        # 无动作策略
        no_action_power = max(power_BS * AC_DC_eff + power_charge - power_renergy, 0)
        no_action_power_cost = RTP[t] * no_action_power / 100
        no_action_reward_charge = env.charge2reward(charge[t], pro[t], error, RTP[t], RTP, t)
        no_action_reward = no_action_reward_charge - no_action_power_cost
        no_action_total_reward += no_action_reward
        no_action_next_SOC = no_action_soc

        soc = round(next_SOC, 2)
        soc_idx = SOC_to_idx[soc]
        soc_sequence.append(soc)

        no_action_soc = round(no_action_next_SOC, 2)
        no_action_soc_idx = SOC_to_idx[no_action_soc]
        no_action_soc_sequence.append(no_action_soc)

    # 计算动作分布
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in optimal_actions:
        action_counts[action] += 1
    action_ratios = {k: v / T_hours for k, v in action_counts.items()}

    # 计算SOC平均值和标准差
    soc_mean = np.mean(soc_sequence)
    soc_std = np.std(soc_sequence)
    no_action_soc_mean = np.mean(no_action_soc_sequence)
    no_action_soc_std = np.std(no_action_soc_sequence)

    return (total_reward, optimal_actions, action_counts, action_ratios, soc_mean, soc_std,
            no_action_total_reward, no_action_soc_mean, no_action_soc_std)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    T = 30  # 在main函数中指定天数
    # 修正函数调用参数
    (total_reward, optimal_actions, action_counts, action_ratios, soc_mean, soc_std, no_action_total_reward, no_action_soc_mean, no_action_soc_std) = dynamic_programming_max_reward(T=T, trace_idx=0)
    logging.info(f"DP总奖励: {total_reward:.2f}")
    logging.info(f"无操作总奖励: {no_action_total_reward:.2f}")
    logging.info(f"Action Distribution (Counts): {action_counts}")
    logging.info(f"DP SOC Mean: {soc_mean:.4f}")
    logging.info(f"DP SOC Standard Deviation: {soc_std:.4f}")