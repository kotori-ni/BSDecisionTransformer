import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import logging
import argparse
import pickle
import json

# 设置日志配置
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'DT.log')

# 只有当没有配置过日志时才配置
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, config=None, hidden_dim=128, n_layers=3, n_heads=4, max_len=30, dropout=0.1):
        super(DecisionTransformer, self).__init__()

        # 允许通过 config.json 统一管理超参数
        if config is not None:
            hidden_dim = config.get("hidden_dim", hidden_dim)
            n_layers = config.get("n_layers", n_layers)
            n_heads = config.get("n_heads", n_heads)
            max_len = config.get("max_len", max_len)
            dropout = config.get("dropout", dropout)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.dropout = dropout

        # Embedding 层
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.state_layernorm = nn.LayerNorm(hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.action_layernorm = nn.LayerNorm(hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.rtg_layernorm = nn.LayerNorm(hidden_dim)
        self.timestep_embedding = nn.Embedding(max_len, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer 编码器（带 causal mask）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, states, actions, rtgs, timesteps, mask=None):
        batch_size, seq_len = states.shape[0], states.shape[1]

        # 嵌入 token
        state_emb = self.state_embedding(states)
        state_emb = self.state_layernorm(state_emb)
        action_emb = self.action_embedding(actions)
        action_emb = self.action_layernorm(action_emb)
        rtg_emb = self.rtg_embedding(rtgs.unsqueeze(-1))
        rtg_emb = self.rtg_layernorm(rtg_emb)

        # 添加 timestep 嵌入
        time_embed = self.timestep_embedding(timesteps)
        state_emb += time_embed
        action_emb += time_embed
        rtg_emb += time_embed

        # Interleave: [R1, s1, a1, R2, s2, a2, ..., Rt, st, at]
        inputs = torch.stack((rtg_emb, state_emb, action_emb), dim=2).reshape(batch_size, seq_len * 3, self.hidden_dim)
        inputs = self.embed_dropout(inputs)

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
    def __init__(self, trajectories, max_len=30, device='cuda'):
        self.trajectories = trajectories
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # 转换states
        try:
            if isinstance(traj['states'][0], np.ndarray):
                # 如果是numpy数组，先转换为列表
                states_list = [[float(x) for x in state] for state in traj['states']]
                states = torch.tensor(states_list, dtype=torch.float32)
            else:
                states = torch.tensor(traj['states'], dtype=torch.float32)
        except Exception as e:
            print(f"警告: 转换states失败: {e}")
        
        # 安全转换actions
        try:
            if isinstance(traj['actions'][0], np.ndarray):
                # 如果是numpy数组，需要提取值
                actions_list = [int(action.item() if hasattr(action, 'item') else action) for action in traj['actions']]
                actions = torch.tensor(actions_list, dtype=torch.long)
            else:
                actions = torch.tensor(traj['actions'], dtype=torch.long)
        except Exception as e:
            print(f"警告: 转换actions失败: {e}")
        
        # 安全转换rtgs
        try:
            if isinstance(traj['rtgs'][0], np.ndarray):
                # 如果是numpy数组，需要提取值
                rtgs_list = [float(rtg.item() if hasattr(rtg, 'item') else rtg) for rtg in traj['rtgs']]
                rtgs = torch.tensor(rtgs_list, dtype=torch.float32)
            else:
                rtgs = torch.tensor(traj['rtgs'], dtype=torch.float32)
        except Exception as e:
            print(f"警告: 转换rtgs失败: {e}")
        
        # 创建timesteps
        timesteps = torch.arange(len(states), dtype=torch.long)

        # 截断或填充到 max_len
        if len(states) > self.max_len:
            start_idx = torch.randint(0, len(states) - self.max_len, (1,)).item()
            states = states[start_idx:start_idx + self.max_len]
            actions = actions[start_idx:start_idx + self.max_len]
            rtgs = rtgs[start_idx:start_idx + self.max_len]
            timesteps = timesteps[start_idx:start_idx + self.max_len]
        else:
            pad_len = self.max_len - len(states)
            if pad_len > 0:
                states = torch.cat([states, torch.zeros((pad_len, states.shape[1]), dtype=torch.float32)], dim=0)
                actions = torch.cat([actions, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                rtgs = torch.cat([rtgs, torch.zeros(pad_len, dtype=torch.float32)], dim=0)
                timesteps = torch.cat([timesteps, torch.full((pad_len,), self.max_len - 1, dtype=torch.long)], dim=0)

        # 安全限制 timestep 不超过 embedding 上限
        timesteps = torch.clamp(timesteps, 0, self.max_len - 1)

        # 直接返回设备上的张量
        return {
            'states': states.to(self.device),
            'actions': actions.to(self.device),
            'rtgs': rtgs.to(self.device),
            'timesteps': timesteps.to(self.device)
        }

# 训练 Decision Transformer 模型，使用轨迹数据优化动作预测
def train_decision_transformer(model, trajectories, epochs=20, batch_size=4, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    dataset = TrajectoryDataset(trajectories, max_len=model.max_len, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logging.info(f"开始训练，共{epochs}个epochs，数据集大小：{len(dataset)}，批次大小：{batch_size}")

    best_loss = float('inf')
    best_accuracy = 0.0  # 跟踪最佳准确率
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_actions = 0
        total_actions = 0
        
        # 记录批次进度
        batch_count = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            states = batch['states']
            actions = batch['actions']
            rtgs = batch['rtgs']
            timesteps = batch['timesteps']

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
            
            # 在控制台显示进度（不写入日志）
            if (batch_idx + 1) % 10 == 0:
                batch_acc = (pred_actions == actions).float().mean().item()
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{batch_count}] "
                      f"损失: {loss.item():.4f}, 准确率: {batch_acc:.4f}, "
                      f"进度: {(batch_idx+1)/batch_count*100:.1f}%", end="\r")

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_actions / total_actions
        
        # 清除进度行
        print(" " * 100, end="\r")
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 根据准确率保存最佳模型
        model_improved = False
        if accuracy > best_accuracy:
            # 准确率更高，保存模型
            best_accuracy = accuracy
            best_loss = avg_loss  # 重置最佳损失
            model_improved = True
            logging.info(f"保存最佳模型，准确率提高: {accuracy:.4f}, 损失: {avg_loss:.4f}")
        elif accuracy == best_accuracy and avg_loss < best_loss:
            # 准确率相同但损失更低，也保存模型
            best_loss = avg_loss
            model_improved = True
            logging.info(f"保存最佳模型，相同准确率下损失降低: {accuracy:.4f}, 损失: {avg_loss:.4f}")
            
        if model_improved:
            os.makedirs('../Models', exist_ok=True)
            torch.save(model.state_dict(), '../Models/dt_model_best.pth')
        
        # 只记录每个epoch的总结信息
        logging.info(f"Epoch {epoch+1}/{epochs} - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 最佳准确率: {best_accuracy:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    logging.info("=" * 50)
    logging.info(f"开始新的训练任务: {args.file}")
    logging.info(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # 加载轨迹数据
    traj_file = args.file
    try:
        with open(traj_file, 'rb') as f:
            trajectories = pickle.load(f)
        logging.info(f"加载了 {len(trajectories)} 条轨迹: {traj_file}")
    except FileNotFoundError:
        logging.error(f"未找到轨迹文件 {traj_file}，请先运行轨迹收集")
        exit()

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")

    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r', encoding='utf-8') as cf:
        config = json.load(cf)
    dt_config = config.get('decision_transformer', {})

    # 模型初始化
    model = DecisionTransformer(
        state_dim=145,
        action_dim=3,
        config=dt_config
    )
    
    logging.info("开始训练模型...")
    train_decision_transformer(
        model=model,
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    os.makedirs('../Models', exist_ok=True)
    model_name = os.path.splitext(os.path.basename(traj_file))[0]
    model_path = f'../Models/dt_model_{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已保存到 {model_path}")