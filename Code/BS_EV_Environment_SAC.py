import os
import logging

log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'trajectory_collection_sac.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from BS_EV_Environment_Base import BS_EV_Base

class ReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims))
        self.new_state_memory = np.zeros((self.mem_size, input_dims))
        self.action_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

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
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='../tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='../tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='actor', chkpt_dir='../tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class SACAgent:
    def __init__(self, input_dims, n_actions, alpha=0.0003, beta=0.0003, reward_scale=2):
        self.gamma = 0.99
        self.tau = 0.005
        self.memory = ReplayBuffer(1000000, input_dims)
        self.batch_size = 256
        self.n_actions = n_actions

        self.actor = ActorNetwork(input_dims, n_actions)
        self.critic_1 = CriticNetwork(input_dims, n_actions)
        self.critic_2 = CriticNetwork(input_dims, n_actions)
        self.value = ValueNetwork(input_dims)
        self.target_value = ValueNetwork(input_dims)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=beta)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=beta)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=beta)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = T.nn.Parameter(
                tau * value_state_dict[name].clone() + \
                (1-tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

class BS_EV_SAC(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json'):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file)
        self.agent = SACAgent(input_dims=self.n_states, n_actions=self.n_actions)
        self.training_steps = 1000000
        self.eval_interval = 1000
        self.best_reward = float('-inf')
        
        # 添加缺失的属性
        self.charge_power = 0.1  # 充电功率
        self.discharge_power = 0.1  # 放电功率
        self.SOC_max = 1.0  # 最大SOC
        self.SOC_min = 0.0  # 最小SOC
        self.current_step = 0  # 当前步数
        self.max_steps = 720  # 最大步数（24小时 * 30天）
        self.SOC = 0.5  # 初始SOC

    def _calculate_reward(self, action):
        """
        计算奖励值
        Args:
            action: 动作值
        Returns:
            reward: 奖励值
        """
        # 充电奖励
        charge_reward = 0
        if action == 1:  # 充电
            charge_reward = 1.0
        
        # 存储成本
        storage_cost = -0.1 * self.SOC
        
        # 电力成本
        power_cost = -0.2 * self.RTP[self.current_step % self.n_RTP]
        
        # 总奖励
        reward = charge_reward + storage_cost + power_cost
        
        return reward

    def step(self, action):
        """
        执行一步环境交互
        Args:
            action: 动作值
        Returns:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
        """
        # 将连续动作转换为离散动作
        action = np.clip(action, -1, 1)
        discrete_action = int((action + 1) * (self.n_actions - 1) / 2)
        
        # 更新SOC
        if discrete_action == 1:  # 充电
            self.SOC = min(self.SOC + self.charge_power, self.SOC_max)
        elif discrete_action == 2:  # 放电
            self.SOC = max(self.SOC - self.discharge_power, self.SOC_min)
        
        # 计算奖励
        reward = self._calculate_reward(discrete_action)
        
        # 更新时间和状态
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 获取新状态
        next_state = self._get_state()
        
        return next_state, reward, done

    def train(self):
        logging.info("Starting SAC training...")
        step = 0
        best_reward = float('-inf')
        
        while step < self.training_steps:
            state = self.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step += 1

                if step % self.eval_interval == 0:
                    eval_reward = self.evaluate()
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        self.agent.save_models()
                        logging.info(f"New best model saved with reward: {best_reward:.2f}")

            logging.info(f"Episode reward: {episode_reward:.2f}")

    def evaluate(self, num_episodes=5):
        total_reward = 0
        for _ in range(num_episodes):
            state = self.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / num_episodes

    def collect_trajectories(self, num_episodes):
        logging.info(f"Starting trajectory collection for {num_episodes} episodes")
        
        # 加载训练好的模型
        try:
            self.agent.load_models()
            logging.info("Successfully loaded SAC models")
        except Exception as e:
            logging.error(f"Failed to load SAC models: {str(e)}")
            raise

        trajectories = []
        for episode in range(num_episodes):
            logging.info(f"Collecting trajectory for episode {episode + 1}/{num_episodes}")
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'rtgs': [],
                'dones': []
            }
            
            state = self.reset()
            done = False
            episode_rewards = []
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.step(action)
                
                # 记录轨迹
                trajectory['states'].append(np.array(state, dtype=np.float32))
                trajectory['actions'].append(np.array(action, dtype=np.float32))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                
                state = next_state
            
            # 计算回报到目标（RTG）
            rtgs = []
            cumulative_reward = 0
            for r in reversed(episode_rewards):
                cumulative_reward = r + 0.99 * cumulative_reward
                rtgs.insert(0, cumulative_reward)
            trajectory['rtgs'] = rtgs
            
            trajectories.append(trajectory)
            total_reward = sum(episode_rewards)
            logging.info(f"Episode {episode + 1} completed with total reward: {total_reward:.2f}")
        
        self.trajectories = trajectories
        
        # 记录轨迹统计信息
        total_rewards = [sum(t['rewards']) for t in trajectories]
        stats = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'num_trajectories': len(trajectories)
        }
        logging.info(f"Trajectory collection completed. Statistics: {stats}")
        
        return trajectories 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    os.makedirs("figure", exist_ok=True)
    
    env = BS_EV_SAC()
    n_episodes = 50  # 可根据需要调整
    score_history = []
    best_score = float('-inf')
    avg_scores = []
    window = 10  # 平滑窗口

    for i in tqdm(range(n_episodes)):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = env.agent.choose_action(state)
            next_state, reward, done = env.step(action)
            env.agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
        score_history.append(score)
        avg_score = np.mean(score_history[-window:])
        avg_scores.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            env.agent.save_models()
        print(f"episode {i} score {score:.1f} avg score {avg_score:.1f}")

    # 保存学习曲线
    plt.figure()
    plt.plot(range(1, n_episodes+1), avg_scores)
    plt.xlabel('Episode')
    plt.ylabel(f'Average Score (window={window})')
    plt.title('SAC Running Average Score')
    plt.grid()
    plt.savefig('Figure/learning_curve_SAC.png')
    print('训练完成，模型和学习曲线已保存。') 