import os
import logging

log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'trajectory_collection_dqn.log')

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
import random
import time
from BS_EV_Environment_Base import BS_EV_Base, charge2power, traffic2power, weather2power, charge2reward

class ReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims))
        self.new_state_memory = np.zeros((self.mem_size, input_dims))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
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

class DQNNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='dqn', chkpt_dir='../Models/dqn'):
        super(DQNNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.q(x)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class DQNAgent:
    def __init__(self, input_dims, n_actions, gamma=0.99, epsilon=1.0, lr=0.0001,
                 mem_size=100000, batch_size=64, epsilon_min=0.01, epsilon_dec=5e-4,
                 replace=1000, chkpt_dir='../Models/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DQNNetwork(input_dims, n_actions, name='q_eval', chkpt_dir=chkpt_dir)
        self.q_next = DQNNetwork(input_dims, n_actions, name='q_next', chkpt_dir=chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float32).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

class BS_EV_DQN(BS_EV_Base):
    def __init__(self, n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, config_file='config.json'):
        super().__init__(n_charge, n_traffic, n_RTP, n_weather, config_file)
        self.agent = DQNAgent(input_dims=self.n_states, n_actions=self.n_actions)
        self.training_steps = 1000000
        self.eval_interval = 1000
        self.best_reward = float('-inf')
        self.done = False

    def step(self, action):
        action_SOC = action

        # 如果电量过低无法放电或过高无法充电，强制不操作
        if (self.SOC < self.min_SOC + self.SOC_discharge_rate and action_SOC == 2) or \
           (self.SOC > 1 - self.SOC_charge_rate and action_SOC == 1):
            action_SOC = 0

        # 计算储能操作成本
        SOC_cost = 0 if action_SOC == 0 else self.SOC_per_cost
        
        # 使用当前时间作为随机种子的一部分
        current_time = int(time.time()) % (2**32 - 1)
        random.seed(current_time)
        pro = random.uniform(0, 1)
        
        power_charge = charge2power(self.charge[self.T], pro)
        power_BS = traffic2power(self.traffic[self.T])
        power_renergy = weather2power(self.weather[self.T])

        # 充电
        if action_SOC == 1:
            self.SOC = self.SOC + self.SOC_charge_rate
            power = max(power_BS * self.AC_DC_eff + power_charge + \
                        self.SOC_charge_rate * self.ESS_cap * self.SOC_eff - power_renergy, 0)
            
        # 放电
        elif action_SOC == 2:
            self.SOC = self.SOC - self.SOC_discharge_rate
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
        reward = reward_charge - SOC_cost - power_cost
        self.T += 1
        next_state = self._get_state()
        if (self.T) % (24 * 30) == 0:
            self.done = True
        return next_state, reward, self.done

    def train(self):
        logging.info("Starting DQN training...")
        step = 0
        best_reward = float('-inf')
        
        while step < self.training_steps:
            state = self.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.learn()
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
            logging.info("Successfully loaded DQN models")
        except Exception as e:
            logging.error(f"Failed to load DQN models: {str(e)}")
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
                trajectory['actions'].append(np.array(action, dtype=np.int64))
                trajectory['rewards'].append(np.array(reward, dtype=np.float32))
                trajectory['dones'].append(np.array(done, dtype=bool))
                episode_rewards.append(reward)
                
                state = next_state
            
            # 计算回报到目标（RTG）
            rtgs = []
            cumulative_reward = 0
            for r in reversed(episode_rewards):
                cumulative_reward = r + self.agent.gamma * cumulative_reward
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
    os.makedirs("figure", exist_ok=True)
    
    env = BS_EV_DQN()
    n_episodes = 50  # 可根据需要调整
    score_history = []
    best_score = float('-inf')
    avg_scores = []
    window = 10  # 平滑窗口

    for i in range(n_episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = env.agent.choose_action(state)
            next_state, reward, done = env.step(action)
            env.agent.store_transition(state, action, reward, next_state, done)
            env.agent.learn()
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
    plt.title('DQN Running Average Score')
    plt.grid()
    plt.savefig('Figure/learning_curve_DQN.png')
    print('训练完成，模型和学习曲线已保存。') 