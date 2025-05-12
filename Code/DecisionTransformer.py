import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3, n_heads=4, max_len=30):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Embedding 层
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)  # 使用embedding代替one-hot+linear
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_len, hidden_dim)   # 时间步嵌入

        # Transformer 编码器（带 causal mask）
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
        action_emb = self.action_embedding(actions)  # 假设actions为整数索引 [batch, seq_len]
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

        # 只使用 state token（在位置 1, 4, 7...）的 hidden 来预测下一个动作
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

        # 安全限制 timestep 不超过 embedding 上限
        timesteps = np.clip(timesteps, 0, self.max_len - 1)

        return {
            'states': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtgs': torch.tensor(rtgs, dtype=torch.float32),
            'timesteps': torch.tensor(timesteps, dtype=torch.long)
        }

# 训练 Decision Transformer 模型，使用轨迹数据优化动作预测
def train_decision_transformer(model, trajectories, epochs=20, batch_size=4, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    dataset = TrajectoryDataset(trajectories, max_len=model.max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_actions = 0
        total_actions = 0
        
        for batch in dataloader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            optimizer.step()
            total_loss += loss.item()
            
            # 计算动作准确率
            pred_actions = torch.argmax(action_pred, dim=-1)
            correct_actions += (pred_actions == actions).sum().item()
            total_actions += actions.numel()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_actions / total_actions
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), '../tmp/dt_model_best.pth')
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Action Accuracy: {accuracy:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

def evaluate_decision_transformer(model, env, target_rtg, max_len=30, device='cuda'):
    model.eval()
    state = env.reset()
    done = False
    total_reward = 0

    states = torch.zeros((1, max_len, env.n_states), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_len), dtype=torch.long, device=device)
    rtgs = torch.full((1, max_len), fill_value=target_rtg, dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, max_len), dtype=torch.long, device=device)
    t = 0

    while not done:
        states[0, t] = torch.tensor(state, dtype=torch.float32, device=device)
        timesteps[0, t] = t

        with torch.no_grad():
            action_pred = model(states, actions, rtgs, timesteps)
        action = torch.argmax(action_pred[0, t], dim=-1).item()

        next_state, reward, done = env.step(action)
        total_reward += reward

        actions[0, t] = action
        state = next_state
        t += 1

        if t >= max_len:
            # 滑动窗口
            states = torch.cat((states[:, 1:], torch.zeros((1, 1, env.n_states), device=device)), dim=1)
            actions = torch.cat((actions[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
            rtgs = torch.cat((rtgs[:, 1:], torch.full((1, 1), fill_value=target_rtg, device=device)), dim=1)
            timesteps = torch.cat((timesteps[:, 1:], torch.tensor([[t]], dtype=torch.long, device=device)), dim=1)
            t -= 1

    return total_reward
