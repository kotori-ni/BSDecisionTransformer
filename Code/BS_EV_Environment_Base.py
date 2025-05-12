import numpy as np
import pandas as pd
import pickle
import os
import logging
import random
import json

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
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

def load_config(config_file='config.json'):
    """加载配置文件"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"配置文件 {config_file} 不存在")
        raise
    except json.JSONDecodeError:
        logging.error(f"配置文件 {config_file} 格式错误")
        raise

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

def check_file_exists(file_path):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

# 读取电价数据
def load_RTP(T=31, train_flag=True, start_idx=None, config=None):
    if config is None:
        config = load_config()
    
    rtp_file = config['data']['RTP_file']
    check_file_exists(rtp_file)
    
    if start_idx is not None:
        skiprows = 24 * start_idx
    else:
        skiprows = 24 * random.randint(0, 570) if train_flag else 24 * random.randint(600, 699)
    
    try:
        df = pd.read_table(rtp_file, sep=",", nrows=24*T, skiprows=skiprows)
        RTP = []
        for _, row in df.iterrows():
            price_str = str(row.iloc[1])
            price = float(price_str[price_str.find("$")+1:])
            RTP.append(price)
        return RTP
    except Exception as e:
        logging.error(f"读取电价数据失败: {str(e)}")
        raise

# 读取天气数据
def load_weather(T=31, train_flag=True, config=None):
    if config is None:
        config = load_config()
    
    weather_file = config['data']['weather_file']
    check_file_exists(weather_file)
    
    try:
        if train_flag:
            df = pd.read_table(weather_file, sep=",", nrows=24*T, skiprows=24*random.randint(0, 570))
        else:
            df = pd.read_table(weather_file, sep=",", nrows=24*T, skiprows=24*random.randint(600, 699))
        
        weather = []
        for _, row in df.iterrows():
            data = str(row.iloc[1]).split(",")
            weather.append([float(data[-2]), float(data[-4])])
        return weather
    except Exception as e:
        logging.error(f"读取天气数据失败: {str(e)}")
        raise

# 读取通信流量数据
def load_traffic(T=31, train_flag=True, config=None):
    if config is None:
        config = load_config()
    
    traffic_file = config['data']['traffic_file']
    check_file_exists(traffic_file)
    
    try:
        with open(traffic_file, "rb") as file:
            bytes_list = pickle.load(file)
            bytes_list = np.r_[bytes_list, bytes_list, bytes_list, bytes_list]
            return list(bytes_list)
    except Exception as e:
        logging.error(f"读取通信流量数据失败: {str(e)}")
        raise

# 读取充电需求数据
def load_charge(T=31, train_flag=True, config=None):
    if config is None:
        config = load_config()
    
    charge_file = config['data']['charge_file']
    check_file_exists(charge_file)
    
    try:
        with open(charge_file, "rb") as file:
            charge = pickle.load(file)
            return charge.tolist()*31
    except Exception as e:
        logging.error(f"读取充电需求数据失败: {str(e)}")
        raise

class BS_EV_Base:
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json'):
        # 加载配置
        self.config = load_config(config_file)
        env_config = self.config['environment']
        
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
        
        # 从配置文件加载参数
        self.min_SOC = env_config['min_SOC']
        self.SOC_charge_rate = env_config['SOC_charge_rate']
        self.SOC_discharge_rate = env_config['SOC_discharge_rate']
        self.SOC_per_cost = env_config['SOC_per_cost']
        self.SOC_eff = env_config['SOC_eff']
        self.AC_DC_eff = env_config['AC_DC_eff']
        self.ESS_cap = env_config['ESS_cap']
        self.error = env_config['error']
        
        self.n_charge = n_charge
        self.trajectories = []

    def reset(self):
        self.SOC = np.random.uniform(self.min_SOC, 1)
        self.T = 0
        self.RTP = load_RTP(train_flag=False, config=self.config)
        self.weather = load_weather(train_flag=False, config=self.config)
        self.traffic = load_traffic(config=self.config)
        self.charge = load_charge(config=self.config)
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