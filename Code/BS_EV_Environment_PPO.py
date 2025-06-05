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
    load_config,
    load_traces
)

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'PPO.log')

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

# 分割训练和测试 traces 的函数
def split_traces(traces, train_ratio=0.8, split_seed=42):
    """固定分割训练和测试 traces"""
    np.random.seed(split_seed)
    indices = np.arange(len(traces))
    np.random.shuffle(indices)
    split_idx = int(len(traces) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_traces = [traces[i] for i in train_indices]
    test_traces = [traces[i] for i in test_indices]
    return train_traces, test_traces

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
    def __init__(self, n_actions, input_dims, alpha, use_lstm=False, fc1_dims=256, fc2_dims=128, fc3_dims=64, chkpt_dir='../Models/'):
        super(ActorNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'actor_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'actor_torch_ppo_last')
        self.use_lstm = use_lstm

        if use_lstm:
            # LSTM 架构，适合时间序列数据
            self.lstm = nn.LSTM(input_dims, fc1_dims, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.LayerNorm(fc2_dims),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
            )
        else:
            # 简化全连接网络
            self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.LayerNorm(fc1_dims),
                nn.Dropout(0.2),  # 提高 Dropout 率
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.LayerNorm(fc2_dims),
                nn.Linear(fc2_dims, fc3_dims),
                nn.ReLU(),
                nn.LayerNorm(fc3_dims),
                nn.Linear(fc3_dims, n_actions),
                nn.Softmax(dim=-1)
            )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, soc=None, min_soc=0.19, soc_charge_rate=0.1, soc_discharge_rate=0.1):
        if self.use_lstm:
            # LSTM 输入需要增加时间步维度
            state = state.unsqueeze(0)  # [batch, seq_len=1, input_dims]
            lstm_out, _ = self.lstm(state)
            dist = self.fc(lstm_out.squeeze(0))
        else:
            dist = self.actor(state)
        
        # 动作掩码：根据 SOC 屏蔽无效动作
        if soc is not None:
            mask = T.ones_like(dist)
            if soc < min_soc + soc_discharge_rate:  # 禁止放电 (action=2)
                mask[:, 2] = 0
            if soc > 1 - soc_charge_rate:  # 禁止充电 (action=1)
                mask[:, 1] = 0
            dist = dist * mask
            dist = dist / (dist.sum(dim=-1, keepdim=True) + 1e-10)  # 重新归一化
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
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=128, fc3_dims=64, chkpt_dir='../Models/'):
        super(CriticNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'critic_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'critic_torch_ppo_last')
        
        # 简化 Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.LayerNorm(fc1_dims),
            nn.Dropout(0.2),  # 提高 Dropout 率
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.LayerNorm(fc2_dims),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.LayerNorm(fc3_dims),
            nn.Linear(fc3_dims, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)
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
                 policy_clip=0.2, batch_size=64, n_epochs=10, use_lstm=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha, use_lstm=use_lstm)
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

    def choose_action(self, observation, soc, min_soc, soc_charge_rate, soc_discharge_rate):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
        dist = self.actor(state, soc=soc, min_soc=min_soc, 
                         soc_charge_rate=soc_charge_rate, soc_discharge_rate=soc_discharge_rate)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        logging.debug(f"Action: {action}, Prob: {probs}, Value: {value}")
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
                dist = self.actor(states)  # 不需要 SOC 掩码，因为动作已合法
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
                logging.debug(f"Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

def predict_actions_for_traces(model_path_best, env, traces, output_file):
    logging.info("开始使用训练好的模型进行预测...")
    
    # 创建临时 agent 用于加载模型
    agent = Agent(n_actions=env.n_actions, 
                 input_dims=env.n_states,
                 gamma=1.00,
                 alpha=0.0001,
                 use_lstm=True)  # 默认使用 FC 网络
    
    # 加载最佳模型
    try:
        actor_model_path = model_path_best.replace('_best', '') + '_best'
        agent.actor.checkpoint_file_best = actor_model_path
        agent.critic.checkpoint_file_best = actor_model_path.replace('actor', 'critic')
        agent.load_models_best()
        logging.info(f"成功加载模型: {actor_model_path}")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise
    
    # 设置为评估模式
    agent.actor.eval()
    agent.critic.eval()
    
    # 统计动作分布
    action_counts = {0: 0, 1: 0, 2: 0}
    state_stats = {'mean': [], 'std': []}
    
    # 对每个 trace 进行预测
    for trace_idx, trace in enumerate(traces):
        logging.info(f"正在预测 trace {trace_idx}/{len(traces)-1}")
        
        # 重置环境
        state = env.reset(trace)
        done = False
        action_sequence = []
        rewards = []
        
        # 运行一个完整 episode 并记录动作
        with T.no_grad():
            while not done:
                # 使用相同的动作约束逻辑
                action, _, _ = agent.choose_action(
                    state, env.SOC, env.min_SOC, env.SOC_charge_rate, env.SOC_discharge_rate)
                action_sequence.append(action)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                state = next_state
                action_counts[action] += 1
                state_stats['mean'].append(np.mean(state))
                state_stats['std'].append(np.std(state))
        
        # 将预测的动作序列写入 PPO_action 属性
        trace['PPO_action'] = action_sequence
        
        # 记录动作分布和奖励统计
        logging.info(f"Trace {trace_idx}: 预测了 {len(action_sequence)} 步动作")
        logging.info(f"Trace {trace_idx}: 动作分布 - 不动作: {action_sequence.count(0)}, "
                     f"充电: {action_sequence.count(1)}, 放电: {action_sequence.count(2)}")
        logging.info(f"Trace {trace_idx}: 奖励统计 - 均值: {np.mean(rewards):.2f}, "
                     f"标准差: {np.std(rewards):.2f}")
    
    # 汇总统计信息
    total_actions = sum(action_counts.values())
    action_dist = {k: v/total_actions for k, v in action_counts.items()}
    logging.info(f"所有 traces 的动作分布: {action_dist}")
    logging.info(f"状态均值: {np.mean(state_stats['mean']):.2f}, "
                 f"状态标准差: {np.mean(state_stats['std']):.2f}")
    
    # 保存更新后的 traces
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(traces, f)
        logging.info(f"预测结果已保存到: {output_file}")
    except Exception as e:
        logging.error(f"保存预测结果失败: {str(e)}")
        raise
    
    # 可视化动作分布
    plt.figure(figsize=(8, 6))
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title('Action Distribution Across All Traces')
    plt.xlabel('Action (0: No-op, 1: Charge, 2: Discharge)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('../Figure/action_distribution_ppo.png')
    plt.close()
    logging.info("动作分布图已保存到 ../Figure/action_distribution_ppo.png")
    
    logging.info("预测完成！")

def run_ppo_prediction(model_path, traces_file, output_file, config_file='config.json'):
    logging.info(f"开始 PPO 预测: 模型={model_path}, 输入={traces_file}, 输出={output_file}")
    
    # 初始化环境
    env = BS_EV_Base(n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, 
                     config_file=config_file, train_flag=False)
    
    # 加载 traces
    traces = load_traces(traces_file)
    logging.info(f"加载了 {len(traces)} 条 traces")
    
    # 进行预测
    predict_actions_for_traces(model_path, env, traces, output_file)
    logging.info("PPO 预测完成！")

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

    # 设置随机种子，确保可重复性
    seed = 42
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    random.seed(seed)

    # 加载配置
    config = load_config()
    ppo_config = config['ppo']

    # 初始化环境
    env = BS_EV_Base(train_flag=True)
    
    # 加载训练 traces
    train_traces_file = '../Data/pro_traces_train.pkl'
    
    all_traces = load_traces(train_traces_file)
    # 固定分割训练和测试 traces
    train_traces, test_traces = split_traces(all_traces, train_ratio=0.8, split_seed=42)
    logging.info(f"加载了 {len(train_traces)} 条训练 traces 和 {len(test_traces)} 条测试 traces")
    
    n_games = ppo_config['n_games']
    batch_size = ppo_config['batch_size']
    n_epochs = ppo_config['n_epochs']
    alpha = ppo_config['alpha']
    N = ppo_config['learn_interval']

    # 初始化 PPO 代理
    agent = Agent(n_actions=env.n_actions, 
                 input_dims=env.n_states,
                 gamma=ppo_config['gamma'],
                 alpha=alpha,
                 gae_lambda=ppo_config['gae_lambda'],
                 policy_clip=ppo_config['policy_clip'],
                 batch_size=batch_size,
                 n_epochs=n_epochs,
                 use_lstm=True)  # 切换到 LSTM 网络

    # 训练 PPO 模型
    best_score = float('-inf')
    train_score_history = []
    n_steps = 0
    learn_iters = 0
    figure_file = '../Figure/learning_curve_ppo.png'
    action_counts = {0: 0, 1: 0, 2: 0}
    state_stats = {'mean': [], 'std': []}
    reward_components = {'charge': [], 'soc_cost': [], 'power_cost': []}

    for i in tqdm(range(n_games), desc="Training PPO"):
        trace = random.choice(train_traces)
        state = env.reset(trace)
        done = False
        score = 0
        agent.actor.train()
        agent.critic.train()
        print(env.T)
        
        while not done:
            action, prob, val = agent.choose_action(
                state, env.SOC, env.min_SOC, env.SOC_charge_rate, env.SOC_discharge_rate)
            next_state, reward, done, action = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(state, action, prob, val, reward, done)
            action_counts[action] += 1
            state_stats['mean'].append(np.mean(state))
            state_stats['std'].append(np.std(state))
            
            # 记录奖励分解（需从环境中获取）
            if not done:
                reward_components['charge'].append(env.charge2reward(
                    env.charge[env.T], trace['pro_trace'][env.T], env.error, env.RTP[env.T], env.RTP, env.T))
                reward_components['soc_cost'].append(0 if action == 0 else env.SOC_per_cost)
                reward_components['power_cost'].append(env.RTP[env.T] * max(
                    env.traffic2power(env.traffic[env.T]) * env.AC_DC_eff + 
                    env.charge2power(env.charge[env.T], trace['pro_trace'][env.T]) - 
                    env.weather2power(env.weather[env.T]), 0) / 100)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            state = next_state
        
        train_score_history.append(score)
        
        # 保存最佳模型
        if score > best_score:
            best_score = score
            agent.save_models_best()
        
        # 记录训练统计信息
        logging.info(f"Episode {i}: Train score {score:.1f}, Best score {best_score:.1f}, "
                     f"Time steps {n_steps}, Learning steps {learn_iters}")
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_dist = {k: v/total_actions for k, v in action_counts.items()}
            logging.info(f"Episode {i}: 动作分布 - {action_dist}")
        logging.info(f"Episode {i}: 状态均值: {np.mean(state_stats['mean']):.2f}, "
                     f"状态标准差: {np.mean(state_stats['std']):.2f}")
        logging.info(f"Episode {i}: 奖励分解 - charge: {np.mean(reward_components['charge']):.2f}, "
                     f"soc_cost: {np.mean(reward_components['soc_cost']):.2f}, "
                     f"power_cost: {np.mean(reward_components['power_cost']):.2f}")
    
    agent.save_models_last()
    logging.info(f"Training completed. Best training score: {best_score:.1f}")

    # 绘制训练曲线
    x = [i+1 for i in range(len(train_score_history))]
    plt.figure(figsize=(10, 6))
    plt.plot(x, train_score_history, 'b-', label='Training Score')
    plt.title('PPO Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_file)
    plt.close()
    logging.info(f"Training curve saved to {figure_file}")
    
    # 绘制奖励分解图
    plt.figure(figsize=(10, 6))
    plt.plot(reward_components['charge'], label='Charge Reward')
    plt.plot(reward_components['soc_cost'], label='SOC Cost')
    plt.plot(reward_components['power_cost'], label='Power Cost')
    plt.title('Reward Components During Training')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Figure/reward_components_ppo.png')
    plt.close()
    logging.info("奖励分解图已保存到 ../Figure/reward_components_ppo.png")
    
    # 使用训练好的模型进行预测
    logging.info("开始对测试 traces 进行预测...")
    model_path = '../Models/actor_torch_ppo_best'
    output_file = '../Data/pro_traces_with_ppo_predictions_improved.pkl'
    predict_actions_for_traces(model_path, env, test_traces, output_file)