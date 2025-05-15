import numpy as np
import random
import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
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

log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ppo.log')

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

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(ActorNetwork, self).__init__()
        # 确保模型目录存在
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'actor_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'actor_torch_ppo_last')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint_best(self):
        T.save(self.state_dict(), self.checkpoint_file_best)

    def save_checkpoint_last(self):
        T.save(self.state_dict(), self.checkpoint_file_last)

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='../Models/'):
        super(CriticNetwork, self).__init__()
        # 确保模型目录存在
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'critic_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'critic_torch_ppo_last')
        self.critic = nn.Sequential(
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
        value = self.critic(state)
        return value

    def save_checkpoint_best(self):
        T.save(self.state_dict(), self.checkpoint_file_best)

    def save_checkpoint_last(self):
        T.save(self.state_dict(), self.checkpoint_file_last)

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file_best, map_location=self.device))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file_last, map_location=self.device))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models_best(self):
        logging.info('... saving best models ...')
        self.actor.save_checkpoint_best()
        self.critic.save_checkpoint_best()

    def save_models_last(self):
        logging.info('... saving last models ...')
        self.actor.save_checkpoint_last()
        self.critic.save_checkpoint_last()

    def load_models_best(self):
        logging.info('... loading best models ...')
        self.actor.load_checkpoint_best()
        self.critic.load_checkpoint_best()

    def load_models_last(self):
        logging.info('... loading last models ...')
        self.actor.load_checkpoint_last()
        self.critic.load_checkpoint_last()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * \
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage, dtype=T.float32).to(self.actor.device)
            values = T.tensor(values, dtype=T.float32).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

class BS_EV_PPO(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json', train_flag=True):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file, trace_idx=0)
        self.n_actions = 3  # 充电、放电、不操作
        self.n_states = n_RTP + n_weather * 2 + n_traffic + n_charge * 2 + 1  # 状态维度
        self.gamma = self.config.get('ppo', {}).get('gamma', 0.99)
        self.epsilon = self.config.get('ppo', {}).get('epsilon', 0.1)
        self.train_flag = train_flag  # 控制训练/测试数据加载
        self.set_mode('train' if train_flag else 'test')  # 根据train_flag设置初始模式

    def reset(self, trace_idx=None, pro_trace=None):
        """
        重置环境
        Args:
            trace_idx: 测试集pro trace的索引，仅用于测试
            pro_trace: 验证用的固定pro trace，仅用于验证
        """
        self.SOC = 0.5
        self.T = 0
        
        # 根据模式加载数据
        if self.mode == 'test':
            # 测试模式：使用测试集数据
            self.RTP = load_RTP(train_flag=False, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
            self.weather = load_weather(train_flag=False, trace_idx=trace_idx, pro_traces=self.pro_traces, config=self.config)
        elif self.mode == 'validation':
            # 验证模式：使用训练集数据
            self.RTP = load_RTP(train_flag=False, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
            self.weather = load_weather(train_flag=False, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
        else:  # train mode
            # 训练模式：使用训练集数据
            self.RTP = load_RTP(train_flag=True, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
            self.weather = load_weather(train_flag=True, trace_idx=None, pro_traces=self.pro_traces, config=self.config)
        
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

    def evaluate_on_fixed_pro_traces(self, agent, fixed_pro_traces):
        """在固定pro trace上评估代理，返回平均奖励"""
        agent.actor.eval()
        agent.critic.eval()
        total_rewards = []
        
        for idx, pro_trace in enumerate(fixed_pro_traces):
            # 设置验证模式
            self.set_mode('validation')
            state = self.reset(trace_idx=None, pro_trace=pro_trace)
            done = False
            episode_reward = 0
            
            while not done:
                action, _, _ = agent.choose_action(state)
                next_state, reward, done = self.step(action)
                episode_reward += reward
                state = next_state
                
            total_rewards.append(episode_reward)
        
        # 恢复训练模式
        self.set_mode('train')
        avg_reward = np.mean(total_rewards)
        return avg_reward

    def collect_optimal_trajectories(self, agent, filename='../Trajectories/optimal_trajectories_ppo.pkl'):
        """使用最佳模型在pro_traces.pkl上收集轨迹"""
        logging.info("Starting optimal trajectory collection")
        agent.load_models_best()
        agent.actor.eval()
        agent.critic.eval()
        
        # 确保在测试模式下运行
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
            
            # 使用测试集数据
            state = self.reset(trace_idx=trace_idx)
            done = False
            episode_rewards = []
            soc_values = []
            action_counts = {0: 0, 1: 0, 2: 0}
            
            while not done:
                action, _, _ = agent.choose_action(state)
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
        
        # 保存轨迹
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(trajectories, f)
            logging.info(f"Optimal trajectories saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving optimal trajectories: {str(e)}")
            raise
            
        return trajectories

def plot_learning_curve(x, scores, figure_file):
    # 确保Figure目录存在
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.figure(figsize=(8, 6))
    plt.plot(x, running_avg)
    plt.title('Running Average of Previous 100 Validation Scores')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
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
    ppo_config = config['ppo']

    # 初始化环境（训练模式）
    env = BS_EV_PPO(train_flag=True)
    n_fixed_pro_traces = ppo_config['n_fixed_pro_traces']  # 固定验证pro trace数量
    n_games = ppo_config['n_games']  # 训练episode数量
    batch_size = ppo_config['batch_size']
    n_epochs = ppo_config['n_epochs']
    alpha = ppo_config['alpha']
    N = ppo_config['learn_interval']  # 每N步学习一次

    # 生成固定验证pro trace（独立于测试集）
    fixed_pro_traces = []
    for _ in range(n_fixed_pro_traces):
        pro_trace = [random.uniform(0, 1) for _ in range(24 * 31)]
        fixed_pro_traces.append(pro_trace)
    logging.info(f"Generated {len(fixed_pro_traces)} fixed pro traces for validation")

    # 初始化PPO代理
    agent = Agent(n_actions=env.n_actions, 
                 input_dims=env.n_states,
                 gamma=ppo_config['gamma'],
                 alpha=alpha,
                 gae_lambda=ppo_config['gae_lambda'],
                 policy_clip=ppo_config['policy_clip'],
                 batch_size=batch_size,
                 n_epochs=n_epochs)

    # 训练PPO模型
    best_score = float('-inf')
    score_history = []
    n_steps = 0
    learn_iters = 0
    figure_file = 'Figure/learning_curve_ppo.png'

    for i in tqdm(range(n_games), desc="Training PPO"):
        # 训练阶段：使用随机生成的pro trace
        observation = env.reset()  # 不传trace_idx和pro_trace，将随机生成
        done = False
        score = 0
        agent.actor.train()
        agent.critic.train()
        
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        
        # 验证阶段：使用固定的pro trace
        avg_score = env.evaluate_on_fixed_pro_traces(agent, fixed_pro_traces)
        score_history.append(avg_score)
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models_best()
        
        logging.info(f"Episode {i}: Train score {score:.1f}, Avg validation score {avg_score:.1f}, "
                     f"Time steps {n_steps}, Learning steps {learn_iters}")
    
    agent.save_models_last()
    logging.info(f"Training completed. Best validation score: {best_score:.1f}")

    # 绘制学习曲线
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    logging.info(f"Learning curve saved to {figure_file}")