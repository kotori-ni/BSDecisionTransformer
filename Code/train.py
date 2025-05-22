import torch
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from DecisionTransformer import DecisionTransformer, TrajectoryDataset
import argparse
from tqdm import tqdm
import logging

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'DT.log')

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

def train_model(model, train_loader, val_loader, config, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['DT_training_params']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    
    best_val_reward = float('-inf')
    best_model_path = 'best_model.pth'
    
    for epoch in range(config['DT_training_params']['epochs']):
        # 添加 tqdm 进度条
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["DT_training_params"]["epochs"]}')
        for batch in train_loader_tqdm:
            states, actions, rtgs, timesteps = [b.to(device) for b in batch]
            optimizer.zero_grad()
            action_pred = model(states, actions, rtgs, timesteps)
            loss = criterion(action_pred.view(-1, model.action_dim), actions.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            # 更新进度条显示损失
            train_loader_tqdm.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        total_val_rewards = []
        with torch.no_grad():
            for traj in val_loader:
                states, actions, rtgs, timesteps = [b.to(device) for b in traj]
                total_reward = evaluate_on_trajectory(model, states[0], config['DT_training_params']['target_rtg'], device)
                total_val_rewards.append(total_reward)
        val_mean_reward = np.mean(total_val_rewards)
        val_std_reward = np.std(total_val_rewards)
        
        # 记录训练和验证信息
        logger.info(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Reward: {val_mean_reward:.2f} ± {val_std_reward:.2f}')
        
        scheduler.step(val_mean_reward)
        if val_mean_reward > best_val_reward:
            best_val_reward = val_mean_reward
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'Saved best model with validation reward: {best_val_reward:.2f}')
        
        model.train()
    
    return model

def evaluate_on_trajectory(model, init_state, target_rtg, device):
    model.eval()
    states = torch.zeros((1, model.max_len, model.state_dim), dtype=torch.float32).to(device)
    actions = torch.zeros((1, model.max_len), dtype=torch.long).to(device)
    rtgs = torch.full((1, model.max_len), fill_value=target_rtg, dtype=torch.float32).to(device)
    timesteps = torch.arange(model.max_len).unsqueeze(0).to(device)
    states[0, 0] = init_state
    
    total_reward = 0
    with torch.no_grad():
        for t in range(model.max_len - 1):
            action_pred = model(states[:, :t+1], actions[:, :t+1], rtgs[:, :t+1], timesteps[:, :t+1])
            action = torch.argmax(action_pred[0, t], dim=-1).item()
            actions[0, t] = action
            
            # 模拟环境（假设有环境对象 env）
            next_state, reward, done = env.step(action)  # 需要替换为实际环境
            total_reward += reward
            rtgs[0, t+1] = rtgs[0, t] - reward  # Return-to-go 递减更新
            states[0, t+1] = torch.tensor(next_state, dtype=torch.float32).to(device)
            if done:
                break
    
    return total_reward

def evaluate_on_trajectories(model, trajectories, target_rtg, device):
    total_rewards = []
    for traj in trajectories:
        total_reward = evaluate_on_trajectory(model, traj['states'][0], target_rtg, device)
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)

def analyze_test_trajectories(model, trajectories, target_rtg, device):
    actions_list = []
    soc_list = []
    total_rewards = []
    
    for traj in trajectories:
        total_reward = evaluate_on_trajectory(model, traj['states'][0], target_rtg, device)
        total_rewards.append(total_reward)
        # 假设 SOC 是状态的第一个元素
        soc_list.append(traj['states'][0][0].item())
        actions_list.extend(traj['actions'].tolist())
    
    return {
        'action_distribution': np.bincount(actions_list) / len(actions_list),
        'soc': np.mean(soc_list),
        'reward_mean': np.mean(total_rewards),
        'reward_std': np.std(total_rewards)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.file, 'rb') as f:
        trajectories = pickle.load(f)
    
    all_rewards = [sum(traj['rewards']) for traj in trajectories]
    config = {
        'state_dim': trajectories[0]['states'].shape[-1],
        'action_dim': max([max(traj['actions']) for traj in trajectories]) + 1,
        'DT_training_params': {
            'batch_size': 64,
            'learning_rate': 1e-4,
            'epochs': 100,
            'target_rtg': np.mean(all_rewards) * 2 * len(trajectories[0]['rewards'])  # 初始 return-to-go 为平均奖励的 2 倍
        }
    }
    
    n_trajectories = len(trajectories)
    train_size = int(n_trajectories * 0.7)
    val_size = int(n_trajectories * 0.1)
    test_size = n_trajectories - train_size - val_size
    
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:train_size + val_size]
    test_trajectories = trajectories[train_size + val_size:]
    
    train_dataset = TrajectoryDataset(train_trajectories, max_len=30)
    val_dataset = TrajectoryDataset(val_trajectories, max_len=30)
    test_dataset = TrajectoryDataset(test_trajectories, max_len=30)
    
    train_loader = DataLoader(train_dataset, batch_size=config['DT_training_params']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DecisionTransformer(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        max_len=30
    ).to(device)
    
    model = train_model(model, train_loader, val_loader, config, device)
    
    test_mean_reward, test_std_reward = evaluate_on_trajectories(model, test_trajectories, config['DT_training_params']['target_rtg'], device)
    logger.info(f'Test Reward: {test_mean_reward:.2f} ± {test_std_reward:.2f}')
    
    analysis = analyze_test_trajectories(model, test_trajectories, config['DT_training_params']['target_rtg'], device)
    logger.info(f'Test Analysis: {analysis}')