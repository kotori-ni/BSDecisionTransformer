import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import os

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3, n_heads=4, max_len=30):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_len, hidden_dim)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, states, actions, rtgs, timesteps):
        batch_size, seq_len = states.size(0), states.size(1)
        
        # 确保timesteps在嵌入范围内
        timesteps = torch.clamp(timesteps, 0, self.max_len - 1)
        
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        rtg_emb = self.rtg_embedding(rtgs.unsqueeze(-1))
        timestep_emb = self.timestep_embedding(timesteps)
        
        inputs = torch.stack((rtg_emb, state_emb, action_emb), dim=2).reshape(batch_size, seq_len * 3, self.hidden_dim)
        inputs = inputs + timestep_emb.repeat(1, 3, 1)
        
        mask = self._generate_square_subsequent_mask(seq_len * 3).to(states.device)
        output = self.transformer(inputs.transpose(0, 1), mask=mask).transpose(0, 1)
        
        action_token_states = output[:, 1::3]
        action_pred = self.action_head(action_token_states)
        
        return action_pred

    def _generate_square_subsequent_mask(self, sz):
        """生成方形后续掩码（下三角矩阵）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_len=30):
        self.trajectories = trajectories
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = np.array(traj['states'], dtype=np.float32)
        actions = np.array(traj['actions'], dtype=np.int64)
        rtgs = np.array(traj['rtgs'], dtype=np.float32)
        timesteps = np.arange(len(states), dtype=np.int64)

        # 截断或填充到 max_len
        if len(states) > self.max_len:
            start_idx = np.random.randint(0, len(states) - self.max_len)
            states = states[start_idx:start_idx + self.max_len]
            actions = actions[start_idx:start_idx + self.max_len]
            rtgs = rtgs[start_idx:start_idx + self.max_len]
            timesteps = timesteps[start_idx:start_idx + self.max_len]
        else:
            pad_len = self.max_len - len(states)
            states = np.pad(states, ((0, pad_len), (0, 0)), mode='constant')
            actions = np.pad(actions, (0, pad_len), mode='constant')
            rtgs = np.pad(rtgs, (0, pad_len), mode='constant')
            timesteps = np.pad(timesteps, (0, pad_len), mode='constant', constant_values=self.max_len - 1)

        timesteps = np.clip(timesteps, 0, self.max_len - 1)

        return {
            'states': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtgs': torch.tensor(rtgs, dtype=torch.float32),
            'timesteps': torch.tensor(timesteps, dtype=torch.long)
        }

def analyze_trajectory(traj, traj_idx, model_reward=None):
    """分析轨迹的动作分布和SOC状态"""
    actions = traj['actions']
    states = traj['states']
    
    # 确保actions是numpy数组
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    else:
        actions = np.array(actions)
    
    # 计算动作分布
    action_counts = np.bincount(actions, minlength=5)  # 假设至少有5个动作
    action_dist = action_counts / len(actions)
    
    # 提取SOC状态（假设SOC是状态向量的第一个元素）
    # 确保states是numpy数组
    if isinstance(states[0], torch.Tensor):
        soc_values = np.array([state[0].cpu().item() for state in states])
    else:
        soc_values = np.array([state[0] for state in states])
    
    print(f"\n轨迹 {traj_idx} 分析:")
    print(f"动作分布: {action_dist}")
    print(f"SOC范围: [{np.min(soc_values):.2f}, {np.max(soc_values):.2f}]")
    print(f"平均SOC: {np.mean(soc_values):.2f}")
    print(f"轨迹长度: {len(actions)}")
    
    # 计算原始奖励
    if 'rewards' in traj:
        rewards = traj['rewards']
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        else:
            rewards = np.array(rewards)
        print(f"原始奖励: {np.sum(rewards):.2f}")
    
    if model_reward is not None:
        print(f"模型预测奖励: {model_reward:.2f}")

def evaluate_on_trajectories_batch(model, trajectories, target_rtg, device, batch_size=32):
    """批量评估多条轨迹"""
    model.eval()
    total_rewards = []
    state_dim = model.state_dim
    max_len = model.max_len
    batch_rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
    batch_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    batch_t = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # 按批次处理轨迹
    for batch_idx in range(0, len(trajectories), batch_size):
        batch_trajectories = trajectories[batch_idx:batch_idx + batch_size]
        curr_batch_size = len(batch_trajectories)
        
        # 预分配张量
        states = torch.zeros((curr_batch_size, max_len, state_dim), dtype=torch.float32, device=device)
        actions = torch.zeros((curr_batch_size, max_len), dtype=torch.long, device=device)
        rtgs = torch.full((curr_batch_size, max_len), fill_value=target_rtg, dtype=torch.float32, device=device)
        timesteps = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(curr_batch_size, -1)
        
        # 初始化状态和轨迹长度
        batch_states = [traj['states'] for traj in batch_trajectories]
        batch_rewards.fill_(0)
        batch_done.fill_(False)
        batch_t.fill_(0)
        
        # 获取最大轨迹长度
        max_traj_len = max(len(traj['states']) for traj in batch_trajectories)
        
        with torch.no_grad():
            while not batch_done.all() and batch_t.max() < max_traj_len:
                # 更新当前状态
                for i in range(curr_batch_size):
                    if not batch_done[i]:
                        t = batch_t[i].item()
                        states[i, t % max_len] = torch.tensor(batch_states[i][min(t, len(batch_states[i])-1)], 
                                                             dtype=torch.float32, device=device)
                
                # 预测动作
                curr_len = min(batch_t.max().item() + 1, max_len)
                curr_states = states[:, :curr_len]
                curr_actions = actions[:, :curr_len]
                curr_rtgs = rtgs[:, :curr_len]
                curr_timesteps = timesteps[:, :curr_len]
                
                action_pred = model(curr_states, curr_actions, curr_rtgs, curr_timesteps)
                
                # 获取动作并更新
                for i in range(curr_batch_size):
                    if not batch_done[i]:
                        t = batch_t[i].item()
                        if t % max_len < action_pred.size(1):
                            action = torch.argmax(action_pred[i, t % max_len], dim=-1)
                        else:
                            action = torch.argmax(action_pred[i, -1], dim=-1)
                        actions[i, t % max_len] = action
                        
                        # 获取奖励和下一状态
                        if t < len(batch_trajectories[i]['rewards']):
                            reward = batch_trajectories[i]['rewards'][t]
                        else:
                            reward = 0
                        batch_rewards[i] += reward
                        rtgs[i, t % max_len] = rtgs[i, t % max_len] - reward  # Return-to-go 递减更新
                        
                        # 检查是否结束
                        if t >= len(batch_trajectories[i]['states']) - 1:
                            batch_done[i] = True
                        
                        batch_t[i] += 1
                
                # 滑动窗口
                if batch_t.max().item() % max_len == 0 and not batch_done.all():
                    shift = max_len // 2
                    for i in range(curr_batch_size):
                        if not batch_done[i]:
                            states[i] = torch.cat([states[i, shift:], 
                                                 torch.zeros((shift, state_dim), device=device)], dim=0)
                            actions[i] = torch.cat([actions[i, shift:], 
                                                  torch.zeros(shift, dtype=torch.long, device=device)], dim=0)
                            rtgs[i] = torch.cat([rtgs[i, shift:], rtgs[i, -shift:]], dim=0)
        
        total_rewards.extend(batch_rewards[:curr_batch_size].cpu().numpy())
    
    return total_rewards

def evaluate_on_trajectories(model, trajectories, target_rtg, device):
    """在多条轨迹上评估模型性能"""
    total_rewards = []
    
    # 使用tqdm显示进度
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(trajectories), 32), desc="评估轨迹批次")
    except ImportError:
        print("评估轨迹中...")
        iterator = range(0, len(trajectories), 32)
    
    for batch_idx in iterator:
        batch_trajectories = trajectories[batch_idx:batch_idx + 32]
        try:
            # 批量评估
            batch_rewards = evaluate_on_trajectories_batch(model, batch_trajectories, target_rtg, device, batch_size=32)
            total_rewards.extend(batch_rewards)
        except Exception as e:
            print(f"评估轨迹批次 {batch_idx} 时出错: {e}")
            continue
    
    if not total_rewards:
        print("警告: 没有成功评估任何轨迹")
        return 0.0, 0.0
    
    return np.mean(total_rewards), np.std(total_rewards)