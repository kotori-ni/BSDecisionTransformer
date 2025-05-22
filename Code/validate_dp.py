import numpy as np
import pandas as pd
import pickle
import random
import os
import json
import logging

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'BS_EV_DP.log')
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

def load_config(config_file='config.json'):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        logging.error(f"文件 {file_path} 不存在")
        raise FileNotFoundError(f"文件 {file_path} 不存在")

def load_RTP(train_flag, T=31, trace_idx=None, pro_traces=None, config=None):
    if config is None:
        config = load_config()
    rtp_file = config['data']['RTP_file']
    check_file_exists(rtp_file)
    try:
        if pro_traces is None:
            with open(config['data']['pro_traces_file'], 'rb') as file:
                pro_traces = pickle.load(file)
        skiprows = 24 * pro_traces[trace_idx]["start_idx"] if trace_idx is not None else 24 * random.randint(0, 570)
        df = pd.read_table(rtp_file, sep=",", nrows=24*T, skiprows=skiprows)
        RTP = [float(str(row.iloc[1])[str(row.iloc[1]).find("$")+1:]) for _, row in df.iterrows()]
        return RTP
    except Exception as e:
        logging.error(f"读取电价数据失败: {str(e)}")
        raise

def load_weather(train_flag, T=31, trace_idx=None, pro_traces=None, config=None):
    if config is None:
        config = load_config()
    weather_file = config['data']['weather_file']
    check_file_exists(weather_file)
    try:
        if pro_traces is None:
            with open(config['data']['pro_traces_file'], 'rb') as file:
                pro_traces = pickle.load(file)
        skiprows = 24 * pro_traces[trace_idx]["start_idx"] if trace_idx is not None else 24 * random.randint(0, 570)
        df = pd.read_table(weather_file, sep=",", nrows=24*T, skiprows=skiprows)
        weather = [[float(data[-2]), float(data[-4])] for _, row in df.iterrows() for data in [str(row.iloc[1]).split(",")]]
        return weather
    except Exception as e:
        logging.error(f"读取天气数据失败: {str(e)}")
        raise

def load_traffic(T=31, train_flag=True, config=None):
    if config is None:
        config = load_config()
    traffic_file = config['data']['traffic_file']
    check_file_exists(traffic_file)
    try:
        with open(traffic_file, "rb") as file:
            bytes_list = pickle.load(file)
            bytes_list = np.r_[bytes_list, bytes_list, bytes_list, bytes_list]
            return list(bytes_list)[:24 * T]
    except Exception as e:
        logging.error(f"读取通信流量数据失败: {str(e)}")
        raise

def load_charge(T=31, train_flag=True, config=None):
    if config is None:
        config = load_config()
    charge_file = config['data']['charge_file']
    check_file_exists(charge_file)
    try:
        with open(charge_file, "rb") as file:
            charge = pickle.load(file)
            return charge.tolist() * 31
    except Exception as e:
        logging.error(f"读取充电需求数据失败: {str(e)}")
        raise

def charge2power(charge, pro):
    return 0 if pro > (1 - charge[0] - charge[1]) else 50

def traffic2power(traffic, traffic_max=150):
    return 2 * traffic / traffic_max + 2

def weather2power(weather, power_WT=1000, power_PV=1):
    return (power_WT * weather[0] + power_PV * weather[1]) / 100

def charge2reward(charge, pro, error):
    if pro > (1 - charge[0] - charge[1]):
        return 0
    elif pro < charge[0] * error:
        return 60
    else:
        return 100

def dynamic_programming_max_reward(config_file='config.json', trace_idx=0):
    # 加载配置和数据
    config = load_config(config_file)
    pro_traces_file = config['data']['pro_traces_file']
    check_file_exists(pro_traces_file)
    with open(pro_traces_file, 'rb') as f:
        pro_traces = pickle.load(f)
    pro = pro_traces[trace_idx]["pro_trace"]
    RTP = load_RTP(train_flag=False, T=31, trace_idx=trace_idx, pro_traces=pro_traces, config=config)
    traffic = load_traffic(T=31, train_flag=False, config=config)
    weather = load_weather(train_flag=False, T=31, trace_idx=trace_idx, pro_traces=pro_traces, config=config)
    charge = load_charge(T=31, train_flag=False, config=config)

    # 环境参数
    min_SOC = config['environment']['min_SOC']
    SOC_charge_rate = config['environment']['SOC_charge_rate']
    SOC_discharge_rate = config['environment']['SOC_discharge_rate']
    SOC_eff = config['environment']['SOC_eff']
    AC_DC_eff = config['environment']['AC_DC_eff']
    SOC_per_cost = config['environment']['SOC_per_cost']
    ESS_cap = config['environment']['ESS_cap']
    error = config['environment']['error']
    T = 720  # 744 (31天×24小时)

    # 离散化SOC
    SOC_values = np.arange(0.00, 1.01, 0.01)
    SOC_values = np.round(SOC_values, 2)
    n_SOC = len(SOC_values)
    SOC_to_idx = {soc: idx for idx, soc in enumerate(SOC_values)}

    # 价值函数和策略
    V = np.full((T + 1, n_SOC), -np.inf)
    policy = np.zeros((T, n_SOC), dtype=int)
    V[T, :] = 0  # 边界条件

    # 动态规划
    for t in range(T - 1, -1, -1):
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
                power_charge = charge2power(charge[t], pro[t])
                power_BS = traffic2power(traffic[t])
                power_renergy = weather2power(weather[t])

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
                reward_charge = charge2reward(charge[t], pro[t], error)
                reward = reward_charge - SOC_cost - power_cost

                next_SOC = round(next_SOC, 1)
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

    for t in range(T):
        action = policy[t, soc_idx]
        optimal_actions.append(action)

        SOC_cost = SOC_per_cost if action in [1, 2] else 0
        power_charge = charge2power(charge[t], pro[t])
        power_BS = traffic2power(traffic[t])
        power_renergy = weather2power(weather[t])

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
        reward_charge = charge2reward(charge[t], pro[t], error)
        reward = reward_charge - SOC_cost - power_cost
        total_reward += reward

        soc = round(next_SOC, 1)
        soc_idx = SOC_to_idx[soc]
        soc_sequence.append(soc)

    # 计算动作分布
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in optimal_actions:
        action_counts[action] += 1
    action_ratios = {k: v / T for k, v in action_counts.items()}

    # 计算SOC平均值和标准差
    soc_mean = np.mean(soc_sequence)
    soc_std = np.std(soc_sequence)

    return total_reward, optimal_actions, action_counts, action_ratios, soc_mean, soc_std

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    total_reward, optimal_actions, action_counts, action_ratios, soc_mean, soc_std = dynamic_programming_max_reward(config_file='config.json', trace_idx=0)
    logging.info(f"Total Reward: {total_reward:.2f}")
    logging.info(f"Optimal Actions (first 10): {optimal_actions[:10]}")
    logging.info(f"Total Actions Length: {len(optimal_actions)}")
    logging.info(f"Action Distribution (Counts): {action_counts}")
    logging.info(f"SOC Mean: {soc_mean:.4f}")
    logging.info(f"SOC Standard Deviation: {soc_std:.4f}")