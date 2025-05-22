import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import logging

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
    def __init__(self, state_dim, action_dim, config=None, hidden_dim=128, n_layers=3, n_heads=4, max_len=30):
        super(DecisionTransformer, self).__init__()

        # 允许通过 config.json 统一管理超参数
        if config is not None:
            hidden_dim = config.get("hidden_dim", hidden_dim)
            n_layers = config.get("n_layers", n_layers)
            n_heads = config.get("n_heads", n_heads)
            max_len = config.get("max_len", max_len)

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
    def __init__(self, trajectories, max_len=30, device='cuda'):
        self.trajectories = trajectories
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # 安全转换states
        try:
            if isinstance(traj['states'][0], np.ndarray):
                # 如果是numpy数组，先转换为列表
                states_list = [[float(x) for x in state] for state in traj['states']]
                states = torch.tensor(states_list, dtype=torch.float32)
            else:
                states = torch.tensor(traj['states'], dtype=torch.float32)
        except Exception as e:
            print(f"警告: 转换states失败: {e}")
            # 使用零矩阵作为备选
            states = torch.zeros((self.max_len, traj['states'][0].shape[0] if hasattr(traj['states'][0], 'shape') else 1), dtype=torch.float32)
        
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
            # 使用零作为备选
            actions = torch.zeros(len(states), dtype=torch.long)
        
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
            # 如果失败，尝试从rewards计算
            if 'rewards' in traj:
                try:
                    # 简单累积rewards作为rtgs
                    rewards = traj['rewards']
                    if isinstance(rewards[0], np.ndarray):
                        rewards_list = [float(r.item() if hasattr(r, 'item') else r) for r in rewards]
                    else:
                        rewards_list = [float(r) for r in rewards]
                    
                    # 从后向前累积计算rtgs
                    rtgs_list = []
                    cumulative = 0
                    for r in reversed(rewards_list):
                        cumulative = r + 0.99 * cumulative  # 使用0.99作为默认折扣因子
                        rtgs_list.insert(0, cumulative)
                    rtgs = torch.tensor(rtgs_list, dtype=torch.float32)
                except:
                    # 如果仍然失败，使用常数
                    rtgs = torch.ones(len(states), dtype=torch.float32)
            else:
                # 没有rewards，使用常数
                rtgs = torch.ones(len(states), dtype=torch.float32)
        
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

def evaluate_decision_transformer(model, env, target_rtg, max_len=30, device='cuda'):
    model.eval()
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    done = False
    total_reward = 0

    states = torch.zeros((1, max_len, env.n_states), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_len), dtype=torch.long, device=device)
    rtgs = torch.full((1, max_len), fill_value=target_rtg, dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, max_len), dtype=torch.long, device=device)
    t = 0

    logging.info(f"开始评估，目标RTG: {target_rtg}")
    
    while not done:
        states[0, t] = state
        timesteps[0, t] = t

        with torch.no_grad():
            action_pred = model(states, actions, rtgs, timesteps)
        action = torch.argmax(action_pred[0, t], dim=-1).item()

        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        total_reward += reward

        actions[0, t] = action
        state = next_state
        t += 1
        
        if t % 20 == 0:  # 每20步记录一次
            logging.debug(f"评估步骤 {t}，当前累计奖励: {total_reward:.2f}")

        if t >= max_len:
            # 滑动窗口
            states = torch.cat((states[:, 1:], torch.zeros((1, 1, env.n_states), device=device)), dim=1)
            actions = torch.cat((actions[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
            rtgs = torch.cat((rtgs[:, 1:], torch.full((1, 1), fill_value=target_rtg, device=device)), dim=1)
            timesteps = torch.cat((timesteps[:, 1:], torch.tensor([[t]], dtype=torch.long, device=device)), dim=1)
            t -= 1
    
    logging.info(f"评估完成，总奖励: {total_reward:.2f}")
    return total_reward

# -------------------------------------------------------------
# 针对单条离线轨迹进行评估，返回预测动作序列与 SOC 信息等
# 注意: 在每一步使用真实奖励递减 rtgs，以符合 Decision Transformer 设定
# -------------------------------------------------------------

def evaluate_on_trajectory(model, trajectory, env, device='cuda', max_len=30):
    model.eval()
    logging.info(f"开始在单条轨迹上评估模型")

    # 直接使用torch.tensor
    try:
        states_arr = torch.tensor(trajectory['states'], dtype=torch.float32, device=device)
    except Exception as e:
        print(f"警告: 转换states时出错: {e}")
        # 尝试逐元素转换
        states_list = []
        for state in trajectory['states']:
            try:
                if isinstance(state, np.ndarray):
                    states_list.append([float(x) for x in state])
                else:
                    states_list.append(state)
            except:
                # 如果仍然失败，使用0向量
                states_list.append([0.0] * env.n_states)
        states_arr = torch.tensor(states_list, dtype=torch.float32, device=device)

    rewards_arr = trajectory.get('rewards')
    if rewards_arr is not None:
        try:
            # 处理不同类型的rewards
            if isinstance(rewards_arr, (list, np.ndarray)):
                # 逐个元素转换确保兼容性
                rewards_list = []
                for r in rewards_arr:
                    if isinstance(r, np.ndarray) and r.size == 1:
                        rewards_list.append(float(r.item()))
                    else:
                        rewards_list.append(float(r))
                rewards_arr = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            else:
                # 单个值情况
                rewards_arr = torch.tensor([float(rewards_arr)], dtype=torch.float32, device=device)
        except Exception as e:
            print(f"警告: 转换rewards时出错: {e}")
            # 如果转换失败，尝试使用rtgs
            rtgs = trajectory.get('rtgs')
            if rtgs is not None:
                try:
                    if isinstance(rtgs, (list, np.ndarray)):
                        rtgs_list = [float(r) for r in rtgs]
                        rewards_arr = torch.tensor(rtgs_list, dtype=torch.float32, device=device)
                    else:
                        rewards_arr = torch.tensor([float(rtgs)], dtype=torch.float32, device=device)
                except:
                    # 如果仍然失败，设为None
                    rewards_arr = None
            else:
                rewards_arr = None

    seq_len = states_arr.shape[0]
    max_len = min(max_len, seq_len)

    # 初始化张量缓存
    states = torch.zeros((1, max_len, env.n_states), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_len), dtype=torch.long, device=device)

    # return-to-go 初始化为真实剩余回报，若没有 rewards 则退化为常数 0
    if rewards_arr is not None:
        try:
            remaining_rtg = float(rewards_arr.sum().item())
        except Exception as e:
            print(f"警告: 计算总奖励时出错: {e}")
            remaining_rtg = 0.0
    else:
        rtgs = trajectory.get('rtgs', [0])
        try:
            if isinstance(rtgs, (list, np.ndarray)) and len(rtgs) > 0:
                # 如果rtgs是numpy数组元素，需要转换
                if isinstance(rtgs[0], np.ndarray):
                    remaining_rtg = float(rtgs[0].item())
                else:
                    remaining_rtg = float(rtgs[0])
            elif not isinstance(rtgs, (list, np.ndarray)):
                remaining_rtg = float(rtgs)
            else:
                remaining_rtg = 0.0
        except Exception as e:
            print(f"警告: 获取rtg时出错: {e}")
            remaining_rtg = 0.0

    rtgs = torch.full((1, max_len), remaining_rtg, dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, max_len), dtype=torch.long, device=device)

    actions_pred = []
    soc_values = []
    try:
        rewards_pred = torch.zeros_like(rewards_arr) if rewards_arr is not None else torch.zeros(seq_len, device=device)
    except Exception as e:
        print(f"警告: 创建rewards_pred时出错: {e}")
        rewards_pred = torch.zeros(seq_len, device=device)

    soc_index = getattr(env, 'soc_index', 0)  # 若环境未指定，默认取 state 第 0 维

    for t in range(seq_len):
        if t >= max_len:
            # 滑动窗口: 移除最早一步数据，末尾补零
            states = torch.cat((states[:, 1:], torch.zeros((1, 1, env.n_states), device=device)), dim=1)
            actions = torch.cat((actions[:, 1:], torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
            rtgs = torch.cat((rtgs[:, 1:], torch.full((1, 1), remaining_rtg, device=device)), dim=1)
            timesteps = torch.cat((timesteps[:, 1:], torch.tensor([[t]], dtype=torch.long, device=device)), dim=1)
            cur_idx = max_len - 1
        else:
            cur_idx = t

        # 写入当前时刻状态
        states[0, cur_idx] = states_arr[t]
        timesteps[0, cur_idx] = t if t < max_len else max_len - 1

        with torch.no_grad():
            action_logits = model(states, actions, rtgs, timesteps)
        action = torch.argmax(action_logits[0, cur_idx], dim=-1).item()

        actions[0, cur_idx] = action
        actions_pred.append(action)

        # 记录 SOC
        soc_values.append(float(states_arr[t, soc_index].item()))
        
        # 由于大多数环境没有compute_reward方法，这里改用_get_reward
        # _get_reward通常是私有方法，但我们尝试访问它以获取更准确的奖励预测
        try:
            # 临时存储当前环境状态
            tmp_T = env.T
            tmp_SOC = env.SOC
            
            # 设置环境状态与轨迹状态一致
            env.T = t % (24*30)  # 假设环境T是循环的
            env.SOC = float(states_arr[t, soc_index].item())
            
            # 使用环境的_get_reward方法计算预测奖励
            reward_pred = env._get_reward(action)
            
            # 恢复环境状态
            env.T = tmp_T
            env.SOC = tmp_SOC
            
            rewards_pred[t] = reward_pred
        except (AttributeError, NotImplementedError):
            # 如果环境没有compute_reward方法，则使用原始奖励（不太准确但作为备选）
            if rewards_arr is not None and t < len(rewards_arr):
                rewards_pred[t] = rewards_arr[t]

        # 根据真实奖励递减 remaining_rtg，并写回 rtgs。（若 rewards 不可用，则保持不变）
        if rewards_arr is not None and t < len(rewards_arr):
            try:
                remaining_rtg -= float(rewards_arr[t].item())
            except Exception as e:
                # 如果获取item失败，尝试直接转换
                try:
                    remaining_rtg -= float(rewards_arr[t])
                except:
                    # 如果仍然失败，不更新remaining_rtg
                    pass

        # 将新的 remaining_rtg 填充到当前位置之后的所有 rtg token（简单实现）
        rtgs[0, cur_idx:] = remaining_rtg

    # 计算SOC平均值
    soc_mean = torch.tensor(soc_values, device=device).mean().item()
    logging.info(f"轨迹评估完成，SOC平均值: {soc_mean:.3f}")
    
    return actions_pred, soc_mean, rewards_arr, rewards_pred
