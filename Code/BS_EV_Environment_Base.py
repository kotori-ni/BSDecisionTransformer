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
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"配置文件 {config_file} 不存在")
        raise
    except json.JSONDecodeError:
        logging.error(f"配置文件 {config_file} 格式错误")
        raise

def generate_pro_traces(n_traces=10, T=31, train_flag=False, output_file="../Data/pro_traces.pkl"):
    # 生成并保存随机 pro 序列及其对应的 start_idx，用于不同算法的性能比较
    try:
        pro_traces = []
        for _ in range(n_traces):
            # 生成一条长度为 24*T 的 pro 序列
            pro_trace = [random.uniform(0, 1) for _ in range(24 * T)]
            # 生成对应的 start_idx
            start_idx = random.randint(0, 699)
            pro_traces.append({"pro_trace": pro_trace, "start_idx": start_idx})
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(pro_traces, f)
        logging.info(f"成功生成 {n_traces} 条 pro 序列，保存到 {output_file}")

        return pro_traces
    
    except Exception as e:
        logging.error(f"生成 pro 序列失败: {str(e)}")
        raise

# 计算可再生能源（风能和光伏）的总功率
def weather2power(weather, power_WT=1000, power_PV=1):
    power_renergy = power_WT * weather[0] + power_PV * weather[1]
    return power_renergy / 100

# 根据电动车充电需求和随机比例决定充电功率
def charge2power(charge, pro):
    # charge: [当前电池电量 (0-1), 已安排的充电需求]
    # pro: 随机比例 (0-1)，表示充电决策的倾向
    if pro > (1 - charge[0] - charge[1]):
        power_charge = 0
    else:
        power_charge = 50
    return power_charge

# 根据电动车充电需求和随机比例计算充电奖励
def charge2reward(charge, pro, error):
    if pro > (1 - charge[0] - charge[1]):
        reward = 0
    elif pro < charge[0] * error:
        reward = 60
    else:
        reward = 100
    return reward

# 根据通信流量计算基站的功率需求
def traffic2power(traffic, traffic_max=150):
    # traffic: 当前通信流量
    # traffic_max: 最大通信流量，默认为 150
    power_BS = 2 * traffic / traffic_max + 2
    return power_BS

def check_file_exists(file_path):
    """检查指定文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

# 读取实时电价数据
def load_RTP(T=31, trace_idx=0, pro_traces=None, config=None):
    if config is None:
        config = load_config()
    
    rtp_file = config['data']['RTP_file']
    check_file_exists(rtp_file)
    
    if pro_traces is None or trace_idx >= len(pro_traces):
        raise ValueError(f"无效的 pro_traces 或 trace_idx {trace_idx}")
    
    # 使用与 pro 序列关联的 start_idx
    skiprows = 24 * pro_traces[trace_idx]["start_idx"]
    
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
def load_weather(T=31, trace_idx=0, pro_traces=None, config=None):
    if config is None:
        config = load_config()
    
    weather_file = config['data']['weather_file']
    check_file_exists(weather_file)
    
    if pro_traces is None or trace_idx >= len(pro_traces):
        raise ValueError(f"无效的 pro_traces 或 trace_idx {trace_idx}")
    
    # 使用与 pro 序列关联的 start_idx
    skiprows = 24 * pro_traces[trace_idx]["start_idx"]
    
    try:
        df = pd.read_table(weather_file, sep=",", nrows=24*T, skiprows=skiprows)
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

# 读取电动车充电需求数据
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

class BS_EV_Base:
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', trace_idx=0):
        self.config = load_config(config_file)
        env_config = self.config['environment']
        
        # 状态维度：电价 + 天气（风速和光照） + 通信流量 + 充电需求（电量和需求） + 电池电量（SOC）
        self.n_states = n_RTP + 2 * n_weather + n_traffic + 2 * n_charge + 1
        self.n_traffic = n_traffic
        self.n_actions = 3  # 动作：0-不操作，1-充电，2-放电
        self.RTP = []  # 实时电价列表
        self.weather = []  # 天气条件（风速和光照强度）
        self.traffic = []  # 通信流量
        self.charge = []  # 电动车充电需求
        self.SOC = 0  # 储能系统当前电量（0-1）
        self.n_RTP = n_RTP
        self.n_weather = n_weather
        self.T = 0  # 当前时间步
        self.trace_idx = trace_idx  # 当前使用的 pro 序列和 start_idx 索引
        self.pro_traces = []  # 包含 pro 序列和 start_idx 的列表
        self.current_pro_trace = []  # 当前 episode 使用的 pro 序列
        
        # 从配置文件加载储能系统和环境参数
        self.min_SOC = env_config['min_SOC']  # 最小电池电量
        self.SOC_charge_rate = env_config['SOC_charge_rate']  # 充电速率
        self.SOC_discharge_rate = env_config['SOC_discharge_rate']  # 放电速率
        self.SOC_per_cost = env_config['SOC_per_cost']  # 每次充放电的固定成本
        self.SOC_eff = env_config['SOC_eff']  # 储能系统充放电效率
        self.AC_DC_eff = env_config['AC_DC_eff']  # AC/DC 转换效率
        self.ESS_cap = env_config['ESS_cap']  # 储能系统容量（kWh）
        self.error = env_config['error']  # 充电奖励的误差因子
        
        self.n_charge = n_charge
        self.trajectories = []  # 存储轨迹数据

        # 加载 pro 序列和 start_idx
        try:
            pro_traces_file = self.config['data']['pro_traces_file']
            check_file_exists(pro_traces_file)
            with open(pro_traces_file, 'rb') as f:
                self.pro_traces = pickle.load(f)
            logging.info(f"成功加载第{self.trace_idx}条 pro 序列")
        except Exception as e:
            logging.error(f"加载 pro 序列失败: {str(e)}")
            raise

    def reset(self):
        # 重置环境，初始化电池电量和数据
        self.SOC = np.random.uniform(self.min_SOC, 1)
        self.T = 0
        # 使用与 pro 序列关联的 start_idx 加载数据
        self.RTP = load_RTP(trace_idx=self.trace_idx, pro_traces=self.pro_traces, config=self.config)
        self.weather = load_weather(trace_idx=self.trace_idx, pro_traces=self.pro_traces, config=self.config)
        self.traffic = load_traffic(config=self.config)
        self.charge = load_charge(config=self.config)
        # 选择当前 episode 使用的 pro 序列
        if self.trace_idx < len(self.pro_traces):
            self.current_pro_trace = self.pro_traces[self.trace_idx]["pro_trace"]
        else:
            raise ValueError(f"trace_idx {self.trace_idx} 超出 pro 序列数量 {len(self.pro_traces)}")
        return self._get_state()

    def _get_state(self):
        # 获取当前状态，包括电价、天气、流量、充电需求和电池电量
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
        power_charge = charge2power(self.charge[self.T], pro)  # 电动车充电功率
        power_BS = traffic2power(self.traffic[self.T])  # 基站功率需求
        power_renergy = weather2power(self.weather[self.T])  # 可再生能源功率

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
        reward_charge = charge2reward(self.charge[self.T], pro, self.error)

        # 总奖励 = 充电奖励 - 储能成本 - 电费成本
        return reward_charge - SOC_cost - power_cost

    def _get_next_SOC(self, action):
        # 根据动作更新电池电量（SOC）
        if action == 1:  # 充电
            return min(1.0, self.SOC + self.SOC_charge_rate)
        elif action == 2:  # 放电
            return max(self.min_SOC, self.SOC - self.SOC_discharge_rate)
        else:  # 不操作
            return self.SOC

    def step(self, action):
        # 执行一步动作，更新环境状态并返回下一状态、奖励和是否结束
        reward = self._get_reward(action)
        self.SOC = self._get_next_SOC(action)
        self.T += 1
        
        done = False
        if (self.T) % (24 * 30) == 0:  # 30 天后结束 episode
            done = True
        
        next_state = self._get_state()
        return next_state, reward, done

    def save_trajectories(self, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectories, f)
            logging.info(f"轨迹数据已保存到 {filename}")
        except Exception as e:
            logging.error(f"保存轨迹数据失败: {str(e)}")
            raise

    def load_trajectories(self, filename):
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"轨迹文件 {filename} 不存在")
            with open(filename, 'rb') as f:
                self.trajectories = pickle.load(f)
            logging.info(f"轨迹数据已从 {filename} 加载")
            return self.trajectories
        except Exception as e:
            logging.error(f"加载轨迹数据失败: {str(e)}")
            raise

if __name__ == '__main__':
    config = load_config()
    pro_traces_file = config['data']['pro_traces_file']
    pro_traces = generate_pro_traces(n_traces=1000, T=31, train_flag=False, output_file=pro_traces_file)
    print(pro_traces[0])