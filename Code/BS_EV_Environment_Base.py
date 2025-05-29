import pickle
import os
import numpy as np
import pandas as pd
import random
import logging
import json

def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def generate_pro_traces(sum_traces, T, output_file, seed):
    try:
        random.seed(seed)
        pro_traces = []
        for _ in range(sum_traces):
            start_idx = random.randint(0, 699)
            pro_trace = [random.uniform(0, 1) for _ in range(24 * T)]
            pro_traces.append({
                    "pro_trace": pro_trace, # 充电概率序列 
                    "start_idx": start_idx, # 电价及新能源数据起始index
                    "Nop_reward": None,
                    "DP_reward": None,
                    "DP_action": None,
                    "DT_reward": None,
                    "DT_action": None,
                    "PPO_reward": None,
                    "PPO_action": None,
                    "SAC_reward": None,
                    "SAC_action": None,
                    "DQN_reward": None,
                    "DQN_action": None
            })
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(pro_traces, f)
        logging.info(f"成功生成 {sum_traces} 条数据，保存到 {output_file}")
    
    except Exception as e:
        logging.error(f"生成数据序列失败: {str(e)}")
        raise

def load_RTP(train_flag, T, pro_trace):
    if pro_trace:
        start_idx = pro_trace["start_idx"]
        df = pd.read_table('../Data/RTP.csv', sep=",", nrows=24*T, skiprows=start_idx)
    elif train_flag:
        df = pd.read_table('../Data/RTP.csv', sep=",", nrows=24*T, skiprows=24*random.randint(0, 570))
    else:
        df = pd.read_table('../Data/RTP.csv', sep=",", nrows=24*T, skiprows=24*random.randint(600, 699))

    df = df.to_numpy()
    RTP = []
    for data in df:        
        RTP.append(float(data[1][data[1].find("$")+1:]))
    return RTP

def load_weather(train_flag, T, pro_trace):
    if pro_trace:
        start_idx = pro_trace["start_idx"]
        df = pd.read_table('../Data/weather.csv', sep=",", nrows=24*T, skiprows=start_idx)
    elif train_flag:
        df = pd.read_table('../Data/weather.csv', sep=",", nrows=24*T, skiprows=24*random.randint(0, 570))
    else:
        df = pd.read_table('../Data/weather.csv', sep=",", nrows=24*T, skiprows=24*random.randint(600, 699))

    df = df.to_numpy()
    weather = []
    for data in df:        
        data = data[1].split(",")
        weather.append([float(data[-2]), float(data[-4])])
    return weather

def load_traffic():
    file = open("../Data/traffic", "rb")
    traffic_list = pickle.load(file)
    traffic_list = np.r_[traffic_list, traffic_list, traffic_list, traffic_list]
    return list(traffic_list)

def load_charge():
    file = open("../Data/charge", "rb")
    charge = pickle.load(file)
    return charge.tolist()*31

def load_traces(trace_file):
    file = open(trace_file, "rb")
    trace_list = pickle.load(file)
    return trace_list

class BS_EV_Base:
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', train_flag=False, traces_file=None):
        self.config = load_config(config_file)
        self.n_charge = n_charge
        self.n_traffic = n_traffic
        self.n_RTP = n_RTP
        self.n_weather = n_weather
        self.done = False # flag to illustrate the end of an episode
        
        self.n_states = n_RTP + n_weather * 2 + n_traffic + n_charge * 2 + 1
        self.n_actions = 3  # 0:不动作, 1:充电, 2:放电
        self.SOC = 0.5
        self.T = 0 # time index
    
        self.min_SOC = self.config.get('environment', {}).get('min_SOC', 0.19)
        self.SOC_charge_rate = self.config.get('environment', {}).get('SOC_charge_rate', 0.1)
        self.SOC_discharge_rate = self.config.get('environment', {}).get('SOC_discharge_rate', 0.1)
        self.SOC_per_cost = self.config.get('environment', {}).get('SOC_per_cost', 0.01)
        self.SOC_eff = self.config.get('environment', {}).get('SOC_eff', 1.1)
        self.AC_DC_eff = self.config.get('environment', {}).get('AC_DC_eff', 1.1)
        self.ESS_cap = self.config.get('environment', {}).get('ESS_cap', 500)
        self.error = self.config.get('environment', {}).get('error', 1.00)
        self.train_flag = train_flag # train or test

        self.traffic = load_traffic()
        self.charge = load_charge()
        self.trace = None
        self.traces = load_traces(traces_file) if traces_file else None

    def reset(self, trace):
        self.SOC = 0.5
        self.T = 0

        self.trace = trace
        self.RTP = load_RTP(train_flag=True, T=31, pro_trace=self.trace)
        self.weather = load_weather(train_flag=True, T=31, pro_trace=self.trace)
        
        return self._get_state()

    def charge2power(self, charge, pro):
        # 计算电动车充电功率
        if pro > (1 - charge[0] - charge[1]):
            power_charge = 0
        else:
            power_charge = 50
        return power_charge

    def traffic2power(self, traffic, traffic_max=150):
        # 计算基站功率需求
        power_BS = 2 * traffic / traffic_max + 2
        return power_BS

    def weather2power(self, weather, power_WT=1000, power_PV=1):
        # 计算可再生能源（风能和光伏）的总功率
        power_renergy = power_WT * weather[0] + power_PV * weather[1]
        return power_renergy / 100

    def charge2reward(self, charge, pro, error, current_rtp, rtp_list, t):
        # 如果充电概率超出阈值，不进行充电one
        if pro > (1 - charge[0] - charge[1]):
            return 0
        
        # 获取前一小时和后一小时的电价
        prev_rtp = rtp_list[t-1] if t > 0 else rtp_list[t]
        next_rtp = rtp_list[t+1] if t < len(rtp_list)-1 else rtp_list[t]
        
        # 取相邻时段电价的差值最大值
        max_diff = max(current_rtp - prev_rtp, current_rtp - next_rtp)
        
        if current_rtp > 0:
            diff_ratio = max(0, max_diff / current_rtp)
        else:
            diff_ratio = 0

        reward = 100 * diff_ratio
        
        # 如果是局部极大值，额外增加收益
        if current_rtp > prev_rtp and current_rtp > next_rtp:
            reward = reward * 1.5

        if pro < charge[0] * error:
            reward = reward * 0.6
        
        return reward

    def _get_next_SOC(self, action):
        if action == 1:  # 充电
            return min(1.0, self.SOC + self.SOC_charge_rate)
        elif action == 2:  # 放电
            return max(self.min_SOC, self.SOC - self.SOC_discharge_rate)
        else:  # 不操作
            return self.SOC

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
        pro = self.trace["pro_trace"][self.T]
        power_charge = self.charge2power(self.charge[self.T], pro)
        power_BS = self.traffic2power(self.traffic[self.T])
        power_renergy = self.weather2power(self.weather[self.T])
        
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
        reward_charge = self.charge2reward(self.charge[self.T], pro, self.error, self.RTP[self.T], self.RTP, self.T)

        # 总奖励 = 充电奖励 - 储能成本 - 电费成本
        return reward_charge - SOC_cost - power_cost

    def _get_state(self):
        observation = [
            self.RTP[self.T:self.T+self.n_RTP],
            self.weather[self.T:self.T+self.n_weather],
            self.traffic[self.T:self.T+self.n_traffic],
            self.charge[self.T:self.T+self.n_charge],
            [self.SOC]
        ]
        observation[1] = list(np.concatenate(observation[1]).flat)
        observation[3] = list(np.concatenate(observation[3]).flat)
        return list(np.concatenate(observation).flat)

    def step(self, action):
        if (self.SOC < self.min_SOC+self.SOC_discharge_rate and action == 2) or (self.SOC > 1-self.SOC_charge_rate and action == 1):
            action = 0

        reward = self._get_reward(action)
        self.SOC = self._get_next_SOC(action)
        self.T += 1
        done = False
        if (self.T) % (24 * 30) == 0:  # 30 天后结束 episode
            done = True
        next_state = self._get_state()
        logging.debug(f"Reward: {reward}, SOC: {self.SOC}, Action: {action}")
        return next_state, reward, done, action


def simulate_actions(action_type, trace_file):
        
    assert action_type in ["DP", "PPO", "SAC", "DQN", "Nop"]

    traces = load_traces(trace_file)
    env = BS_EV_Base()

    for idx in range(len(traces)):
        env.reset(traces[idx])
    
        total_reward = 0.0
        reward_name = f"{action_type}_reward"
        action_name = f"{action_type}_action"
        action_list = traces[idx][action_name] if action_name != "Nop_action" else [0] * 720

        for i, action in enumerate(action_list): 
            _, reward, done = env.step(action)
            total_reward += reward

        traces[idx][reward_name] = total_reward
        print(f"trace_idx: {idx}, action: {action_name}, Total reward: {total_reward}")

    with open(trace_file, "wb") as f:
        pickle.dump(traces, f)

if __name__ == '__main__':
    traces_file = "../Data/pro_traces.pkl"
    # if not os.path.exists(traces_file):
    #     generate_pro_traces(sum_traces=10, T=30, output_file=traces_file, seed=42)
    #     pro_traces = load_traces(traces_file)

    env = BS_EV_Base()
    simulate_actions(action_type="Nop", trace_file=traces_file)
