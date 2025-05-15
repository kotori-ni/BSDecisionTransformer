import numpy as np
import random
import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from BS_EV_Environment_Base import (
    BS_EV_Base, 
    load_RTP,
    load_weather,
    load_traffic,
    load_charge,
    load_config
)

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'dqn.log')

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

class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

class QNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(QNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'qnetwork_dqn_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'qnetwork_dqn_last')
        self.q = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        q_values = self.q(state)
        return q_values

    def save_checkpoint_best(self):
        T.save(self.state_dict(), self.checkpoint_file_best)

    def save_checkpoint_last(self):
        T.save(self.state_dict(), self.checkpoint_file_last)

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, epsilon=1.0, 
                 eps_min=0.01, eps_dec=5e-6, batch_size=64, buffer_size=1000000, 
                 target_update=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.target_update = target_update
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)
        self.q_eval = QNetwork(input_dims, n_actions, alpha)
        self.q_target = QNetwork(input_dims, n_actions, alpha)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

    def choose_action(self, observation, evaluate=False):
        state = T.tensor([observation], dtype=T.float32).to(self.device)
        if evaluate or random.random() > self.epsilon:
            with T.no_grad():
                q_values = self.q_eval(state)
                action = T.argmax(q_values).item()
        else:
            action = random.randint(0, self.n_actions - 1)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.long).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        states_ = T.tensor(states_, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.device)

        q_pred = self.q_eval(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with T.no_grad():
            q_next = self.q_target(states_).max(dim=1)[0]
            q_next[dones] = 0.0
            target = rewards + self.gamma * q_next

        loss = F.mse_loss(q_pred, target.detach())

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

    def save_models_best(self):
        logging.info('... saving best models ...')
        self.q_eval.save_checkpoint_best()
        self.q_target.save_checkpoint_best()

    def save_models_last(self):
        logging.info('... saving last models ...')
        self.q_eval.save_checkpoint_last()
        self.q_target.save_checkpoint_last()

    def load_models_best(self):
        logging.info('... loading best models ...')
        self.q_eval.load_checkpoint_best()
        self.q_target.load_checkpoint_best()

    def load_models_last(self):
        logging.info('... loading last models ...')
        self.q_eval.load_checkpoint_last()
        self.q_target.load_checkpoint_last()

class BS_EV_DQN(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', train_flag=True):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file, trace_idx=0)
        self.n_actions = 3  # 充电、放电、不操作
        self.n_states = n_RTP + n_weather * 2 + n_traffic + n_charge * 2 + 1  # 状态维度
        self.gamma = self.config.get('dqn', {}).get('gamma', 0.99)
        self.train_flag = train_flag
        self.set_mode('train' if train_flag else 'test')

    def reset(self, trace_idx=None, pro_trace=None):
        self.SOC = 0.5
        self.T = 0
        
        if self.mode == 'test':
            self.RTP = load_RTP(train_flag=False, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
            self.weather = load_weather(train_flag=False, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
        elif self.mode == 'validation':
            self.RTP = load_RTP(train_flag=False, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
            self.weather = load_weather(train_flag=False, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
        else:
            self.RTP = load_RTP(train_flag=True, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
            self.weather = load_weather(train_flag=True, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
        
        self.traffic = load_traffic(config=self.config)
        self.charge = load_charge(config=self.config)
        
        if self.mode == 'test' and trace_idx is not None:
            self.current_pro_trace = self.pro_traces[trace_idx]["pro_trace"]
        elif self.mode == 'validation' and pro_trace is not None:
            self.current_pro_trace = pro_trace
        else:
            self.current_pro_trace = [random.uniform(0, 1) for _ in range(24 * 31)]
        
        return self._get_state()

    def evaluate_on_fixed_pro_traces(self, agent, fixed_pro_traces):
        agent.q_eval.eval()
        total_rewards = []
        
        for idx, pro_trace in enumerate(fixed_pro_traces):
            self.set_mode('validation')
            state = self.reset(trace_idx=None, pro_trace=pro_trace)
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.choose_action(state, evaluate=True)
                next_state, reward, done = self.step(action)
                episode_reward += reward
                state = next_state
                
            total_rewards.append(episode_reward)
        
        self.set_mode('train')
        avg_reward = np.mean(total_rewards)
        return avg_reward

    def collect_optimal_trajectories(self, agent, filename='../Trajectories/optimal_trajectories_dqn.pkl'):
        logging.info("Starting optimal trajectory collection")
        agent.load_models_best()
        agent.q_eval.eval()
        
        self.set_mode('test')
        
        trajectories = []
        trace_stats = []

        for trace_idx in range(len(self.pro_traces)):
            logging.info(f"Collecting trajectory for test trace {trace_idx}/{len(self.pro_traces)-1}")
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'rtgs': [],
                'dones': [],
                'trace_idx': trace_idx
            }
            
            state = self.reset(trace_idx=trace_idx)
            done = False
            episode_rewards = []
            soc_values = []
            action_counts = {0: 0, 1: 0, 2: 0}
            
            while not done:
                action = agent.choose_action(state, evaluate=True)
                next_state, reward, done = self.step(action)
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.int32))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                soc_values.append(self.SOC)
                action_counts[action] += 1
                state = next_state
            
            rtgs = []
            cumulative_reward = 0
            for r in reversed(episode_rewards):
                cumulative_reward = r + self.gamma * cumulative_reward
                rtgs.insert(0, cumulative_reward)
            trajectory['rtgs'] = rtgs
            
            trajectories.append(trajectory)
            trace_stats.append({
                'trace_idx': trace_idx,
                'total_reward': sum(episode_rewards),
                'mean_soc': np.mean(soc_values),
                'std_soc': np.std(soc_values),
                'action_counts': action_counts
            })
            
            logging.info(f"Trace {trace_idx}: Total reward: {sum(episode_rewards):.2f}")
            logging.info(f"Trace {trace_idx}: Mean SOC: {np.mean(soc_values):.3f}, Std SOC: {np.std(soc_values):.3f}")
            logging.info(f"Trace {trace_idx}: Action distribution: {action_counts}")
        
        self.trajectories = trajectories
        
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(trajectories, f)
            logging.info(f"Optimal trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving optimal trajectories: {str(e)}")
            raise
            
        return trajectories

def plot_learning_curve(x, train_scores, val_scores, figure_file):
    # 确保Figure目录存在
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, train_scores, 'b-', label='Training Score')
    plt.plot(x, val_scores, 'r-', label='Validation Score')
    plt.title('Learning Curves')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_file)
    plt.close()

if __name__ == "__main__":
    # 初始化必要的目录
    directories = [
        '../Log',
        '../Models',
        '../Trajectories',
        '../Figure'
    ]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logging.error(f"创建目录失败 {directory}: {str(e)}")
            raise

    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    random.seed(seed)

    # 加载配置
    config = load_config()
    dqn_config = config['dqn']

    # 初始化环境（训练模式）
    env = BS_EV_DQN(train_flag=True)
    n_fixed_pro_traces = dqn_config['n_fixed_pro_traces']  # 固定验证pro trace数量
    n_games = dqn_config['n_games']  # 训练episode数量
    batch_size = dqn_config['batch_size']
    alpha = dqn_config['alpha']
    N = dqn_config['learn_interval']  # 每N步学习一次

    # 生成固定验证pro trace（独立于测试集）
    fixed_pro_traces = []
    for _ in range(n_fixed_pro_traces):
        pro_trace = [random.uniform(0, 1) for _ in range(24 * 31)]
        fixed_pro_traces.append(pro_trace)
    logging.info(f"Generated {len(fixed_pro_traces)} fixed pro traces for validation")

    # 初始化DQN代理
    agent = Agent(n_actions=env.n_actions,
                 input_dims=env.n_states,
                 gamma=dqn_config['gamma'],
                 alpha=alpha,
                 epsilon=dqn_config['epsilon'],
                 eps_min=dqn_config['epsilon_min'],
                 eps_dec=dqn_config['epsilon_dec'],
                 batch_size=batch_size,
                 buffer_size=dqn_config['mem_size'])

    # 训练DQN模型
    best_score = float('-inf')
    train_score_history = []  # 新增：记录训练分数
    val_score_history = []    # 重命名：验证分数
    n_steps = 0
    learn_iters = 0
    figure_file = 'Figure/learning_curve_dqn.png'

    for i in tqdm(range(n_games), desc="Training DQN"):
        # 训练阶段：使用随机生成的pro trace
        observation = env.reset()  # 不传trace_idx和pro_trace，将随机生成
        done = False
        score = 0
        agent.q_eval.train()
        agent.q_target.train()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        
        train_score_history.append(score)  # 新增：记录训练分数
        
        # 验证阶段：使用固定的pro trace
        avg_score = env.evaluate_on_fixed_pro_traces(agent, fixed_pro_traces)
        val_score_history.append(avg_score)  # 重命名：验证分数
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models_best()
        
        logging.info(f"Episode {i}: Train score {score:.1f}, Avg validation score {avg_score:.1f}, "
                     f"Time steps {n_steps}, Learning steps {learn_iters}")
    
    agent.save_models_last()
    logging.info(f"Training completed. Best validation score: {best_score:.1f}")

    # 绘制学习曲线
    x = [i+1 for i in range(len(val_score_history))]
    plot_learning_curve(x, train_score_history, val_score_history, figure_file)
    logging.info(f"Learning curve saved to {figure_file}")