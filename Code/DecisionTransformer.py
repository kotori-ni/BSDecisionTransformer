import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3, n_heads=4, max_len=30):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Embedding 层
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_len, hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, states, actions, rtgs, timesteps, mask=None):
        batch_size, seq_len = states.shape[0], states.shape[1]

        # 嵌入 token
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        rtg_emb = self.rtg_embedding(rtgs.unsqueeze(-1))

        # 添加 timestep 嵌入
        time_embed = self.timestep_embedding(timesteps)
        state_emb += time_embed
        action_emb += time_embed
        rtg_emb += time_embed

        # Interleave: [R1, s1, a1, R2, s2, a2, ..., Rt, st, at]
        inputs = torch.stack((rtg_emb, state_emb, action_emb), dim=2).reshape(batch_size, seq_len * 3, self.hidden_dim)

        # 掩码
        if mask is None:
            mask = self._generate_square_subsequent_mask(seq_len * 3).to(inputs.device)

        # Transformer 编码
        output = self.transformer(inputs, mask=mask)

        # 只使用 state token 的 hidden 来预测下一个动作
        action_token_states = output[:, 1::3]
        action_pred = self.action_head(action_token_states)

        return action_pred

    def _generate_square_subsequent_mask(self, sz):
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
    
    # 计算动作分布
    action_counts = np.bincount(actions, minlength=5)
    action_dist = action_counts / len(actions)
    
    # 提取SOC状态（假设SOC是状态向量的第一个元素）
    soc_values = states[:, 0]
    
    print(f"\n轨迹 {traj_idx} 分析:")
    print(f"动作分布: {action_dist}")
    print(f"SOC范围: [{np.min(soc_values):.2f}, {np.max(soc_values):.2f}]")
    print(f"平均SOC: {np.mean(soc_values):.2f}")
    print(f"轨迹长度: {len(actions)}")
    print(f"原始奖励: {sum(traj['rewards']):.2f}")
    if model_reward is not None:
        print(f"模型预测奖励: {model_reward:.2f}")

def evaluate_on_trajectory(model, traj, target_rtg, device):
    """在单条轨迹上评估模型性能"""
    model.eval()
    state = traj['states'][0]
    done = False
    total_reward = 0
    t = 0
    
    states = torch.zeros((1, model.max_len, model.state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((1, model.max_len), dtype=torch.long, device=device)
    rtgs = torch.full((1, model.max_len), fill_value=target_rtg, dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, model.max_len), dtype=torch.long, device=device)
    
    while not done and t < len(traj['states']):
        states[0, t] = torch.tensor(state, dtype=torch.float32, device=device)
        timesteps[0, t] = t
        
        with torch.no_grad():
            action_pred = model(states, actions, rtgs, timesteps)
        action = torch.argmax(action_pred[0, t], dim=-1).item()
        
        next_state = traj['states'][t]
        reward = traj['rewards'][t]
        done = (t == len(traj['states']) - 1)
        
        total_reward += reward
        actions[0, t] = action
        state = next_state
        t += 1
        
        if t >= model.max_len:
            states = torch.cat((states[:, 1:], torch.zeros((1, 1, model.state_dim), device=device)), dim=1)
            actions = torch.cat((actions[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
            rtgs = torch.cat((rtgs[:, 1:], torch.full((1, 1), fill_value=target_rtg, device=device)), dim=1)
            timesteps = torch.cat((timesteps[:, 1:], torch.tensor([[t]], dtype=torch.long, device=device)), dim=1)
            t -= 1
    
    return total_reward

def evaluate_on_trajectories(model, trajectories, target_rtg, device):
    """在多条轨迹上评估模型性能"""
    total_rewards = []
    
    for traj in trajectories:
        reward = evaluate_on_trajectory(model, traj, target_rtg, device)
        total_rewards.append(reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

def train_model(model, train_trajectories, val_trajectories, config, target_rtg, 
                model_dir='../Models', device='cuda'):
    """训练决策Transformer模型并保存最佳模型"""
    # 创建数据加载器
    train_dataset = TrajectoryDataset(train_trajectories, max_len=model.max_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['DT_training_params']['batch_size'],
        shuffle=True
    )

    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['DT_training_params']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)
    best_val_reward = float('-inf')
    best_model_path = f'{model_dir}/dt_model_best.pth'
    final_model_path = f'{model_dir}/dt_model_final.pth'

    print("\n开始训练模型...")
    for epoch in range(config['DT_training_params']['epochs']):
        # 训练阶段
        model.train()
        total_loss = 0
        correct_actions = 0
        total_actions = 0
        
        for batch in train_loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)

            # 前向传播
            action_pred = model(states, actions, rtgs, timesteps)
            loss = criterion(action_pred.view(-1, model.action_dim), actions.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            
            # 计算动作准确率
            pred_actions = torch.argmax(action_pred, dim=-1)
            correct_actions += (pred_actions == actions).sum().item()
            total_actions += actions.numel()

        # 计算训练指标
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_actions / total_actions
        
        # 在验证集上评估
        val_mean_reward, val_std_reward = evaluate_on_trajectories(
            model, val_trajectories, target_rtg, device
        )
        
        # 更新学习率
        scheduler.step(val_mean_reward)
        
        print(f"\nEpoch {epoch + 1}/{config['DT_training_params']['epochs']}")
        print(f"训练损失: {avg_loss:.4f}")
        print(f"训练准确率: {accuracy:.4f}")
        print(f"验证集平均奖励: {val_mean_reward:.2f} ± {val_std_reward:.2f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_mean_reward > best_val_reward:
            best_val_reward = val_mean_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"保存新的最佳模型，验证集奖励: {best_val_reward:.2f}")

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    print(f"\n训练完成！")
    print(f"最佳模型已保存到: {best_model_path}")
    print(f"最终模型已保存到: {final_model_path}")
    
    return best_model_path, final_model_path

def analyze_test_trajectories(model, test_trajectories, target_rtg, 
                              interval=100, device='cuda'):
    """分析测试轨迹，每interval条输出一次结果"""
    print("\n分析测试轨迹:")
    for i, traj in enumerate(test_trajectories):
        if (i + 1) % interval == 0:  # 每interval条轨迹分析一次
            reward = evaluate_on_trajectory(model, traj, target_rtg, device)
            analyze_trajectory(traj, i + 1, reward)
