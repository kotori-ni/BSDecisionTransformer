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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, states, actions, rtgs, timesteps, mask=None):
        batch_size, seq_len = states.size(0), states.size(1)
        
        # 确保timesteps在嵌入范围内
        timesteps = torch.clamp(timesteps, 0, self.max_len - 1)
        
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
            mask = self._generate_square_subsequent_mask(seq_len * 3, device=inputs.device)

        # Transformer 编码
        output = self.transformer(inputs, mask=mask)

        # 只使用 state token（在位置 1, 4, 7...）的 hidden 来预测下一个动作
        action_token_states = output[:, 1::3]
        action_pred = self.action_head(action_token_states)
        
        return action_pred

    def _generate_square_subsequent_mask(self, sz, device=None):
        """生成方形后续掩码（下三角矩阵）

        参数:
            sz (int): 掩码的方形大小 (seq_len)
            device (torch.device, optional): 在哪个设备上创建掩码；默认为与模型参数相同的设备。
        """
        if device is None:
            device = next(self.parameters()).device

        # 直接在目标设备上创建，避免之后再调用 .to(device)
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
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

def train_decision_transformer(model, train_trajectories, epochs=20, batch_size=4, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_dataset = TrajectoryDataset(train_trajectories, max_len=model.max_len, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 记录训练过程中的指标
    train_losses_history = []
    train_accs_history = []
    epochs_history = []

    # 确保模型保存目录存在
    os.makedirs('../Models', exist_ok=True)

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss, correct, total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=True)
        for batch in train_pbar:
            states, actions, rtgs, timesteps = batch['states'], batch['actions'], batch['rtgs'], batch['timesteps']
            action_pred = model(states, actions, rtgs, timesteps)
            loss = criterion(action_pred.view(-1, model.action_dim), actions.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(action_pred, dim=-1)
            correct += (pred == actions).sum().item()
            total += actions.numel()
            
            # 更新进度条显示
            current_acc = correct / total if total > 0 else 0
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.4f}'})
        
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        
        # 记录训练指标
        epochs_history.append(epoch + 1)
        train_losses_history.append(train_loss)
        train_accs_history.append(train_acc)

        scheduler.step(train_loss)
        logging.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}')
        
        # 保存每个epoch的模型
        model_path = f'../Models/dt_model_{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        logging.info(f'模型已保存到 {model_path}')
        
    # 训练结束后绘制学习曲线
    plot_learning_curves(epochs_history, train_losses_history, train_accs_history, 
                         None, None, has_validation=False)

def plot_learning_curves(epochs, train_losses, train_accs, val_losses, val_accs, has_validation=True):
    """
    绘制训练和验证的学习曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制Loss曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    if has_validation and val_losses[0] is not None:
        # 过滤掉None值
        val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        val_loss_filtered = [v for v in val_losses if v is not None]
        ax1.plot(val_epochs, val_loss_filtered, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(epochs))
    
    # 绘制Accuracy曲线
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    if has_validation and val_accs[0] is not None:
        # 过滤掉None值
        val_epochs = [e for e, v in zip(epochs, val_accs) if v is not None]
        val_acc_filtered = [v for v in val_accs if v is not None]
        ax2.plot(val_epochs, val_acc_filtered, 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, max(epochs))
    ax2.set_ylim(0, 1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    figure_dir = '../Figure'
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, 'learning_curve_dt.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    logging.info(f"学习曲线已保存到: {figure_path}")
    
    # 打印训练摘要
    logging.info("=" * 50)
    logging.info("训练摘要")
    logging.info("=" * 50)
    logging.info(f"总epoch数: {len(epochs)}")
    logging.info(f"最终训练损失: {train_losses[-1]:.4f}")
    logging.info(f"最终训练准确率: {train_accs[-1]:.4f}")
    logging.info(f"最佳训练准确率: {max(train_accs):.4f}")
    logging.info(f"最低训练损失: {min(train_losses):.4f}")
    
    if has_validation and val_accs[0] is not None:
        val_loss_filtered = [v for v in val_losses if v is not None]
        val_acc_filtered = [v for v in val_accs if v is not None]
        logging.info(f"最终验证损失: {val_loss_filtered[-1]:.4f}")
        logging.info(f"最终验证准确率: {val_acc_filtered[-1]:.4f}")
        logging.info(f"最佳验证准确率: {max(val_acc_filtered):.4f}")
        logging.info(f"最低验证损失: {min(val_loss_filtered):.4f}")
    
    plt.close()  # 关闭图形以释放内存

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
        train_trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    logging.info("训练完成")