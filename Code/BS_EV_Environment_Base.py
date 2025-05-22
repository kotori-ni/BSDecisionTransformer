import pickle
import os
import numpy as np
import pandas as pd
import random
import logging
import json

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'BS_EV_Environment.log')

# 检查是否已经配置过日志
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

def generate_pro_traces(n_traces=1000, T=31, train_flag=False, output_file="../Data/pro_traces.pkl"):
    # 生成并保存随机 pro 序列及其对应的 start_idx，用于不同算法的性能比较
    try:
        random.seed(42)
        pro_traces = []
        for _ in range(n_traces):
            start_idx = random.randint(0, 699)
            pro_trace = [random.uniform(0, 1) for _ in range(24 * T)]
            pro_traces.append({"pro_trace": pro_trace, "start_idx": start_idx})
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(pro_traces, f)
        logging.info(f"成功生成 {n_traces} 条 pro 序列，保存到 {output_file}")
        return pro_traces
    except Exception as e:
        logging.error(f"生成 pro 序列失败: {str(e)}")
        raise

def load_RTP(train_flag, T=31, trace_idx=None, pro_traces=None, config=None,):
    if config is None:
        config = load_config()
    rtp_file = config['data']['RTP_file']
    check_file_exists(rtp_file)
    try:
        if pro_traces is None:
            with open(config['data']['pro_traces_file'], 'rb') as file:
                pro_traces = pickle.load(file)
        
        # 确定skiprows
        if trace_idx is not None:
            skiprows = 24 * pro_traces[trace_idx]["start_idx"]
        else:
            if train_flag:
                skiprows = 24 * np.random.randint(0, 570)  # 训练集范围
            else:
                skiprows = 24 * 600
                # skiprows = 24 * np.random.randint(600, 699)  # 测试集范围
                
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

def load_weather(train_flag, T=31, trace_idx=None, pro_traces=None, config=None):
    if config is None:
        config = load_config()
    weather_file = config['data']['weather_file']
    check_file_exists(weather_file)
    try:
        if pro_traces is None:
            with open(config['data']['pro_traces_file'], 'rb') as file:
                pro_traces = pickle.load(file)
                
        # 确定skiprows
        if trace_idx is not None:
            skiprows = 24 * pro_traces[trace_idx]["start_idx"]
        else:
            if train_flag:
                skiprows = 24 * np.random.randint(0, 570)  # 训练集范围
            else:
                skiprows = 24 * 600
                # skiprows = 24 * np.random.randint(600, 699)  # 测试集范围
                
        df = pd.read_table(weather_file, sep=",", nrows=24*T, skiprows=skiprows)
        weather = []
        for _, row in df.iterrows():
            data = str(row.iloc[1]).split(",")
            weather.append([float(data[-2]), float(data[-4])])
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
            return list(bytes_list)[:24 * T]  # 截断到 24 * T
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

class BS_EV_Base:
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', trace_idx=0):
        self.config = load_config(config_file)
        self.n_charge = n_charge
        self.n_traffic = n_traffic
        self.n_RTP = n_RTP
        self.n_weather = n_weather
        self.trace_idx = trace_idx
        self.mode = 'train'  # 新增：用于区分训练/验证/测试模式
        self.pro_traces = None  # 初始化为None
        
        # 状态空间和动作空间维度
        self.n_states = n_RTP + n_weather * 2 + n_traffic + n_charge * 2 + 1  # RTP + weather(2维) + traffic + charge(2维) + SOC
        self.n_actions = 3  # 0:不动作, 1:充电, 2:放电
        self.soc_index = n_RTP + n_weather * 2 + n_traffic + n_charge * 2  # SOC在状态向量中的位置
        
        # 环境参数
        self.min_SOC = self.config['environment']['min_SOC']
        self.SOC_charge_rate = self.config['environment']['SOC_charge_rate']
        self.SOC_discharge_rate = self.config['environment']['SOC_discharge_rate']
        self.SOC_eff = self.config['environment']['SOC_eff']
        self.AC_DC_eff = self.config['environment']['AC_DC_eff']
        self.SOC_per_cost = self.config['environment']['SOC_per_cost']
        self.ESS_cap = self.config['environment']['ESS_cap']
        self.error = self.config['environment']['error']

    def set_mode(self, mode):
        """设置环境模式：'train', 'validation', 或 'test'"""
        assert mode in ['train', 'validation', 'test'], f"Invalid mode: {mode}"
        self.mode = mode
        
        # 只在测试模式下加载pro序列
        if mode == 'test' and self.pro_traces is None:
            try:
                pro_traces_file = self.config['data']['pro_traces_file']
                check_file_exists(pro_traces_file)
                with open(pro_traces_file, 'rb') as f:
                    self.pro_traces = pickle.load(f)
                logging.info(f"成功加载第{self.trace_idx}条 pro 序列")
            except Exception as e:
                logging.error(f"加载 pro 序列失败: {str(e)}")
                raise

    def reset(self, trace_idx=None, pro_trace=None):
        """
        重置环境
        Args:
            trace_idx: 测试集pro trace的索引，仅用于测试
            pro_trace: 验证用的固定pro trace，仅用于验证
        """
        self.SOC = 0.5
        self.T = 0
        
        # 加载RTP和天气数据（保持随机性）
        self.RTP = load_RTP(train_flag=True, T=31, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
        self.weather = load_weather(train_flag=True, T=31, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
        self.traffic = load_traffic(config=self.config)
        self.charge = load_charge(config=self.config)
        
        # 根据场景选择pro trace
        if self.mode == 'test' and trace_idx is not None:
            # 测试场景：使用测试集pro trace
            self.current_pro_trace = self.pro_traces[trace_idx]["pro_trace"]
        elif self.mode == 'validation' and pro_trace is not None:
            # 验证场景：使用传入的固定pro trace
            self.current_pro_trace = pro_trace
        else:
            # 训练场景：随机生成pro trace
            self.current_pro_trace = [random.uniform(0, 1) for _ in range(24 * 31)]
        
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

    def charge2reward(self, charge, pro, error):
        # 计算充电奖励
        if pro > (1 - charge[0] - charge[1]):
            reward = 0
        elif pro < charge[0] * error:
            reward = 60
        else:
            reward = 100
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
        pro = self.current_pro_trace[self.T]
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
        reward_charge = self.charge2reward(self.charge[self.T], pro, self.error)

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
        return next_state, reward, done

if __name__ == '__main__':
    pro_traces_file = "../Data/pro_traces.pkl"
    if not os.path.exists(pro_traces_file):
        pro_traces = generate_pro_traces(n_traces=1000, T=31, train_flag=False, output_file=pro_traces_file)
        print(f"First trace start_idx: {pro_traces[0]['start_idx']}, pro_trace (first 10): {pro_traces[0]['pro_trace'][:10]}")