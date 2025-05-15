import numpy as np
import random
import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
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
log_file = os.path.join(log_dir, 'sac.log')

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

class ValueNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(ValueNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'value_sac_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'value_sac_last')
        self.value = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.value(state)
        return value

    def save_checkpoint_best(self):
        T.save(self.state_dict(), self.checkpoint_file_best)

    def save_checkpoint_last(self):
        T.save(self.state_dict(), self.checkpoint_file_last)

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))

class QNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(QNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, self.__class__.__name__ + '_sac_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, self.__class__.__name__ + '_sac_last')
        self.q = nn.Sequential(
            nn.Linear(input_dims + n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        sa = T.cat([state, action], dim=-1)
        q = self.q(sa)
        return q

    def save_checkpoint_best(self):
        T.save(self.state_dict(), self.checkpoint_file_best)

    def save_checkpoint_last(self):
        T.save(self.state_dict(), self.checkpoint_file_last)

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))

class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(PolicyNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'policy_sac_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'policy_sac_last')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = T.softmax(z, dim=-1)
        log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def save_checkpoint_best(self):
        T.save(self.state_dict(), self.checkpoint_file_best)

    def save_checkpoint_last(self):
        T.save(self.state_dict(), self.checkpoint_file_last)

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))

class Agent:
    def __init__(self, n_actions, input_dims, max_action=1, gamma=0.99, alpha=0.0003, 
                 tau=0.005, batch_size=64, reward_scale=1.0, target_entropy=-3.0, 
                 buffer_size=1000000):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.n_actions = n_actions
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)
        
        self.value = ValueNetwork(input_dims, alpha)
        self.target_value = ValueNetwork(input_dims, alpha)
        self.q1 = QNetwork(input_dims, n_actions, alpha)
        self.q2 = QNetwork(input_dims, n_actions, alpha)
        self.policy = PolicyNetwork(input_dims, n_actions, alpha)
        
        self.target_entropy = T.tensor(target_entropy, dtype=T.float32).to(self.device)
        self.log_alpha = T.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)
        
        self.update_target_networks(tau=1.0)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models_best(self):
        logging.info('... saving best models ...')
        self.value.save_checkpoint_best()
        self.q1.save_checkpoint_best()
        self.q2.save_checkpoint_best()
        self.policy.save_checkpoint_best()

    def save_models_last(self):
        logging.info('... saving last models ...')
        self.value.save_checkpoint_last()
        self.q1.save_checkpoint_last()
        self.q2.save_checkpoint_last()
        self.policy.save_checkpoint_last()

    def load_models_best(self):
        logging.info('... loading best models ...')
        self.value.load_checkpoint_best()
        self.q1.load_checkpoint_best()
        self.q2.load_checkpoint_best()
        self.policy.load_checkpoint_best()

    def load_models_last(self):
        logging.info('... loading last models ...')
        self.value.load_checkpoint_last()
        self.q1.load_checkpoint_last()
        self.q2.load_checkpoint_last()
        self.policy.load_checkpoint_last()

    def choose_action(self, observation, evaluate=False):
        state = T.tensor([observation], dtype=T.float32).to(self.device)
        if evaluate:
            self.policy.eval()
            with T.no_grad():
                mu, _ = self.policy.forward(state)
                action = T.softmax(mu, dim=-1)
                action_idx = T.argmax(action, dim=-1).item()
            self.policy.train()
            return action_idx, None, None
        else:
            action, log_prob = self.policy.sample(state)
            action_idx = T.argmax(action, dim=-1).item()
            log_prob = log_prob.item()
            return action_idx, action, log_prob

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.long).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        states_ = T.tensor(states_, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.device)

        # Compute value and target value
        value = self.value(states).view(-1)
        value_ = self.target_value(states_).view(-1)
        value_[dones] = 0.0

        # Compute Q values for current actions
        actions_one_hot = T.zeros(self.batch_size, self.n_actions, device=self.device)
        actions_one_hot.scatter_(1, actions.unsqueeze(1), 1.0)

        # Compute target for Q losses
        with T.no_grad():
            actions_next, log_probs_next = self.policy.sample(states_)
            q1_next = self.q1(states_, actions_next).view(-1)
            q2_next = self.q2(states_, actions_next).view(-1)
            q_next = T.min(q1_next, q2_next)
            target = rewards + self.gamma * (value_ - T.exp(self.log_alpha) * log_probs_next.view(-1))

        # Value loss: recompute q1, q2, and log_probs_pi to ensure gradients
        q1 = self.q1(states, actions_one_hot).view(-1)
        q2 = self.q2(states, actions_one_hot).view(-1)
        actions_pi, log_probs_pi = self.policy.sample(states)  # Recompute for value_loss
        value_loss = ((value - T.min(q1, q2) + T.exp(self.log_alpha) * log_probs_pi.view(-1)) ** 2).mean()

        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        # Q1 loss: recompute q1
        q1 = self.q1(states, actions_one_hot).view(-1)
        q1_loss = ((q1 - target.detach()) ** 2).mean()

        self.q1.optimizer.zero_grad()
        q1_loss.backward()
        self.q1.optimizer.step()

        # Q2 loss: recompute q2
        q2 = self.q2(states, actions_one_hot).view(-1)
        q2_loss = ((q2 - target.detach()) ** 2).mean()

        self.q2.optimizer.zero_grad()
        q2_loss.backward()
        self.q2.optimizer.step()

        # Policy loss
        actions_pi, log_probs_pi = self.policy.sample(states)
        q1_pi = self.q1(states, actions_pi).view(-1)
        q2_pi = self.q2(states, actions_pi).view(-1)
        q_pi = T.min(q1_pi, q2_pi)
        policy_loss = (T.exp(self.log_alpha) * log_probs_pi.view(-1) - q_pi).mean()

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs_pi.view(-1) + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.update_target_networks()

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

class BS_EV_SAC(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', train_flag=True):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file, trace_idx=0)
        self.n_actions = 3  # 充电、放电、不操作
        self.n_states = n_RTP + n_weather * 2 + n_traffic + n_charge * 2 + 1  # 状态维度
        self.gamma = self.config.get('sac', {}).get('gamma', 0.99)
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
        agent.policy.eval()
        total_rewards = []
        
        for idx, pro_trace in enumerate(fixed_pro_traces):
            self.set_mode('validation')
            state = self.reset(trace_idx=None, pro_trace=pro_trace)
            done = False
            episode_reward = 0
            
            while not done:
                action, _, _ = agent.choose_action(state, evaluate=True)
                next_state, reward, done = self.step(action)
                episode_reward += reward
                state = next_state
                
            total_rewards.append(episode_reward)
        
        self.set_mode('train')
        avg_reward = np.mean(total_rewards)
        return avg_reward

    def collect_optimal_trajectories(self, agent, filename='../Trajectories/optimal_trajectories_sac.pkl'):
        logging.info("Starting optimal trajectory collection")
        agent.load_models_best()
        agent.policy.eval()
        
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
                action, _, _ = agent.choose_action(state, evaluate=True)
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
    sac_config = config['sac']

    # 初始化环境（训练模式）
    env = BS_EV_SAC(train_flag=True)
    n_fixed_pro_traces = sac_config['n_fixed_pro_traces']  # 固定验证pro trace数量
    n_games = sac_config['n_games']  # 训练episode数量
    batch_size = sac_config['batch_size']
    alpha = sac_config['alpha']
    N = sac_config['learn_interval']  # 每N步学习一次

    # 生成固定验证pro trace（独立于测试集）
    fixed_pro_traces = []
    for _ in range(n_fixed_pro_traces):
        pro_trace = [random.uniform(0, 1) for _ in range(24 * 31)]
        fixed_pro_traces.append(pro_trace)
    logging.info(f"Generated {len(fixed_pro_traces)} fixed pro traces for validation")

    # 初始化SAC代理
    agent = Agent(n_actions=env.n_actions,
                 input_dims=env.n_states,
                 gamma=sac_config['gamma'],
                 alpha=alpha,
                 tau=sac_config['tau'],
                 batch_size=batch_size,
                 reward_scale=sac_config['reward_scale'],
                 buffer_size=sac_config['mem_size'])

    # 训练SAC模型
    best_score = float('-inf')
    train_score_history = []  # 新增：记录训练分数
    val_score_history = []    # 重命名：验证分数
    n_steps = 0
    learn_iters = 0
    figure_file = 'Figure/learning_curve_sac.png'

    for i in tqdm(range(n_games), desc="Training SAC"):
        # 训练阶段：使用随机生成的pro trace
        observation = env.reset()  # 不传trace_idx和pro_trace，将随机生成
        done = False
        score = 0
        agent.value.train()
        agent.target_value.train()
        agent.q1.train()
        agent.q2.train()
        agent.policy.train()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            n_steps += 1
            score += reward
            agent.memory.store_transition(observation, action, reward, observation_, done)
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