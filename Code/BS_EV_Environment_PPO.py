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
    load_config,
    load_traces
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
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=512, fc2_dims=512, fc3_dims=256, fc4_dims=128, chkpt_dir='../Models/'):
        super(ActorNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'actor_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'actor_torch_ppo_last')
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.LayerNorm(fc1_dims),
            nn.Dropout(0.1),
            
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.LayerNorm(fc2_dims),
            nn.Dropout(0.1),
            
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.LayerNorm(fc3_dims),
            nn.Dropout(0.1),
            
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.LayerNorm(fc4_dims),
            
            nn.Linear(fc4_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)
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
    def __init__(self, input_dims, alpha, fc1_dims=512, fc2_dims=512, fc3_dims=256, fc4_dims=128, chkpt_dir='../Models/'):
        super(CriticNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file_best = os.path.join(chkpt_dir, 'critic_torch_ppo_best')
        self.checkpoint_file_last = os.path.join(chkpt_dir, 'critic_torch_ppo_last')
        
        # 更深的网络结构，增加LayerNorm和Dropout
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.LayerNorm(fc1_dims),
            nn.Dropout(0.1),
            
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.LayerNorm(fc2_dims),
            nn.Dropout(0.1),
            
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.LayerNorm(fc3_dims),
            nn.Dropout(0.1),
            
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.LayerNorm(fc4_dims),
            
            nn.Linear(fc4_dims, 1)
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
                logging.debug(f"Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

# 预测函数：使用训练好的模型对traces进行预测
def predict_actions_for_traces(model_path_best, env, traces, output_file):
    """
    使用训练好的PPO模型对traces进行预测，并将动作序列写入PPO_action属性
    
    Args:
        model_path_best: 最佳模型的路径前缀（不包含文件扩展名）
        env: 环境实例
        traces: 包含pro_trace的traces列表
        output_file: 输出文件路径
    """
    logging.info("开始使用训练好的模型进行预测...")
    
    # 创建临时agent用于加载模型
    agent = Agent(n_actions=env.n_actions, 
                 input_dims=env.n_states,
                 gamma=0.99,  # 默认值
                 alpha=0.0001)  # 学习率在推理时不重要
    
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
    
    # 对每个trace进行预测
    for trace_idx, trace in enumerate(traces):
        logging.info(f"正在预测trace {trace_idx}/{len(traces)-1}")
        
        # 重置环境
        state = env.reset(trace)
        done = False
        action_sequence = []
        
        # 运行一个完整episode并记录动作
        with T.no_grad():  # 推理时不需要梯度
            while not done:
                action, _, _ = agent.choose_action(state)
                action_sequence.append(action)
                next_state, _, done = env.step(action)
                state = next_state
        
        # 将预测的动作序列写入PPO_action属性
        trace['PPO_action'] = action_sequence
        
        logging.info(f"Trace {trace_idx}: 预测了 {len(action_sequence)} 步动作")
        logging.info(f"Trace {trace_idx}: 动作分布 - 不动作: {action_sequence.count(0)}, "
                     f"充电: {action_sequence.count(1)}, 放电: {action_sequence.count(2)}")
    
    # 保存更新后的traces
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(traces, f)
        logging.info(f"预测结果已保存到: {output_file}")
    except Exception as e:
        logging.error(f"保存预测结果失败: {str(e)}")
        raise
    
    logging.info("预测完成！")

def run_ppo_prediction(model_path, traces_file, output_file, config_file='config.json'):
    """
    独立的预测函数，可以单独调用进行PPO模型预测
    
    Args:
        model_path: 训练好的模型路径（actor模型的完整路径）
        traces_file: 包含traces的输入文件
        output_file: 预测结果输出文件
        config_file: 配置文件路径
    
    使用示例:
        run_ppo_prediction('../Models/actor_torch_ppo_best', 
                          '../Data/pro_traces.pkl', 
                          '../Data/pro_traces_with_ppo_predictions.pkl')
    """
    logging.info(f"开始PPO预测: 模型={model_path}, 输入={traces_file}, 输出={output_file}")
    
    # 初始化环境
    env = BS_EV_Base(n_charge=24, n_traffic=24, n_RTP=24, n_weather=24, 
                     config_file=config_file, train_flag=False)
    
    # 加载traces
    traces = load_traces(traces_file)
    logging.info(f"加载了 {len(traces)} 条traces")
    
    # 进行预测
    predict_actions_for_traces(model_path, env, traces, output_file)
    logging.info("PPO预测完成！")

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

    # 初始化环境（使用BS_EV_Base）
    env = BS_EV_Base(train_flag=True)
    
    # 加载训练traces
    train_traces_file = '../Data/pro_traces.pkl'
    if not os.path.exists(train_traces_file):
        logging.error(f"训练traces文件不存在: {train_traces_file}")
        raise FileNotFoundError(f"请先生成训练traces文件: {train_traces_file}")
    
    all_traces = load_traces(train_traces_file)
    # 分割训练集和测试集（假设前80%为训练，后20%为测试）
    split_idx = int(len(all_traces) * 0.8)
    train_traces = all_traces[:split_idx]
    test_traces = all_traces[split_idx:]
    
    logging.info(f"加载了 {len(train_traces)} 条训练traces和 {len(test_traces)} 条测试traces")
    
    n_games = ppo_config['n_games']  # 训练episode数量
    batch_size = ppo_config['batch_size']
    n_epochs = ppo_config['n_epochs']
    alpha = ppo_config['alpha']
    N = ppo_config['learn_interval']  # 每N步学习一次

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
    train_score_history = []  # 记录训练分数
    n_steps = 0
    learn_iters = 0
    figure_file = '../Figure/learning_curve_ppo.png'

    for i in tqdm(range(n_games), desc="Training PPO"):
        # 训练阶段：从训练traces中随机选择一个trace
        trace = random.choice(train_traces)
        state = env.reset(trace)
        done = False
        score = 0
        agent.actor.train()
        agent.critic.train()
        
        while not done:
            action, prob, val = agent.choose_action(state)
            next_state, reward, done, action = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(state, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            state = next_state
        
        train_score_history.append(score)
        
        # 保存最佳模型（基于训练分数）
        if score > best_score:
            best_score = score
            agent.save_models_best()
        
        logging.info(f"Episode {i}: Train score {score:.1f}, Best score {best_score:.1f}, "
                     f"Time steps {n_steps}, Learning steps {learn_iters}")
    
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
    
    # 使用训练好的模型进行预测
    logging.info("开始对测试traces进行预测...")
    model_path = '../Models/actor_torch_ppo_best'
    output_file = '../Data/pro_traces_with_ppo_predictions.pkl'
    predict_actions_for_traces(model_path, env, test_traces, output_file)

    # 如果要单独运行预测，可以使用以下代码：
    # if __name__ == "__main__":
    #     run_ppo_prediction('../Models/actor_torch_ppo_best', 
    #                       '../Data/pro_traces.pkl', 
    #                       '../Data/pro_traces_with_ppo_predictions.pkl')