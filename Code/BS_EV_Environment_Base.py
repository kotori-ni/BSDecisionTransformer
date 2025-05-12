import numpy as np
import pandas as pd
import pickle
import os
import logging
import random

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'environment.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

# 根据天气条件（风速和光照强度）计算可再生能源（风能和光伏）的总功率
def weather2power(weather, power_WT = 1000, power_PV = 1):
    power_renergy = power_WT*weather[0] + power_PV*weather[1]
    return power_renergy / 100

# 根据充电需求和随机比例决定充电功率
def charge2power(charge, pro):
    # charge = [当前电池电量, 已安排的充电功率或充电需求]
    if pro > (1-charge[0]-charge[1]):
        power_charge = 0
    else: 
        power_charge = 50
    return power_charge

# 根据充电需求和误差计算奖励
def charge2reward(charge, pro, error):
    # error：误差因子，调整奖励计算
    if pro > (1-charge[0]-charge[1]):
        reward = 0
    elif pro < charge[0]*error:
        reward = 60
    else:
        reward = 100
    return reward

# 根据通信流量计算基站的功率需求
def traffic2power(traffic, traffic_max = 150):
    # 空载最低功率消耗为2
    power_BS = 2 * traffic / traffic_max + 2
    return power_BS

# 读取电价数据
def load_RTP(T=31, train_flag=True, start_idx=None):
    if start_idx is not None:
        skiprows = 24 * start_idx
    else:
        skiprows = 24 * random.randint(0, 570) if train_flag else 24 * random.randint(600, 699)
    df = pd.read_table('../Data/RTP.csv', sep=",", nrows=24*T, skiprows=skiprows)
    RTP = []
    for _, row in df.iterrows():
        price_str = str(row.iloc[1])
        price = float(price_str[price_str.find("$")+1:])
        RTP.append(price)
    return RTP

# 读取天气数据
def load_weather(T=31, train_flag=True):
    if train_flag:
        df = pd.read_table('../Data/weather.csv', sep=",", nrows=24*T, skiprows=24*random.randint(0, 570))
    else:
        df = pd.read_table('../Data/weather.csv', sep=",", nrows=24*T, skiprows=24*random.randint(600, 699))
    
    weather = []
    for _, row in df.iterrows():
        data = str(row.iloc[1]).split(",")
        weather.append([float(data[-2]), float(data[-4])])
    return weather

# 读取通信流量数据
def load_traffic(T=31, train_flag=True):
    file = open("../Data/traffic", "rb")
    bytes_list = pickle.load(file)
    bytes_list = np.r_[bytes_list, bytes_list, bytes_list, bytes_list]
    return list(bytes_list)

# 读取充电需求数据
def load_charge(T=31, train_flag=True):
    file = open("../Data/charge", "rb")
    charge = pickle.load(file)
    return charge.tolist()*31

class BS_EV_Base:
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, error=1.0):
        self.n_states = n_RTP + 2 * n_weather + n_traffic + 2 * n_charge + 1
        self.n_traffic = n_traffic
        self.n_actions = 3
        self.RTP = []
        self.weather = []
        self.traffic = []
        self.charge = []
        self.SOC = 0
        self.n_RTP = n_RTP
        self.n_weather = n_weather
        self.T = 0
        self.min_SOC = 0.2
        self.SOC_charge_rate = 0.1
        self.SOC_discharge_rate = 0.1
        self.SOC_per_cost = 0.01
        self.SOC_eff = 1.1
        self.AC_DC_eff = 1.1
        self.ESS_cap = 500
        self.n_charge = n_charge
        self.error = error
        self.trajectories = []

    def reset(self):
        self.SOC = np.random.uniform(self.min_SOC, 1)
        self.T = 0
        self.RTP = load_RTP(train_flag=False)
        self.weather = load_weather(train_flag=False)
        self.traffic = load_traffic()
        self.charge = load_charge()
        return self._get_state()

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

    def _get_reward(self, action, pro):
        action_SOC = action
        
        # 如果电量过低无法放电或过高无法充电，强制不操作
        if (self.SOC < self.min_SOC + self.SOC_discharge_rate and action_SOC == 2) or \
           (self.SOC > 1 - self.SOC_charge_rate and action_SOC == 1):
            action_SOC = 0

        # 计算储能操作成本
        SOC_cost = 0 if action_SOC == 0 else self.SOC_per_cost
        
        power_charge = charge2power(self.charge[self.T], pro)
        power_BS = traffic2power(self.traffic[self.T])
        power_renergy = weather2power(self.weather[self.T])

        # 充电
        if action_SOC == 1:
            power = max(power_BS * self.AC_DC_eff + power_charge + \
                        self.SOC_charge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
            
        # 放电
        elif action_SOC == 2:
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
        return reward_charge - SOC_cost - power_cost

    def _get_next_SOC(self, action):
        if action == 1:  # 充电
            return min(1.0, self.SOC + self.SOC_charge_rate)
        elif action == 2:  # 放电
            return max(self.min_SOC, self.SOC - self.SOC_discharge_rate)
        else:  # 不操作
            return self.SOC

    def save_trajectories(self, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectories, f)
            logging.info(f"Trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving trajectories: {str(e)}")
            raise

    def load_trajectories(self, filename):
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