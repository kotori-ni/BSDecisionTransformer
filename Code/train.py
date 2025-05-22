import torch
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from DecisionTransformer import DecisionTransformer, TrajectoryDataset, evaluate_on_trajectory, evaluate_on_trajectories, analyze_trajectory
import argparse
from tqdm import tqdm
import logging

# 设置日志
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'DT.log')

logger = logging.getLogger('DT')
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger('DT')

def train_model(model, train_loader, val_trajectories, config, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['DT_training_params']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    
    best_val_reward = float('-inf')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'Models')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'dt_model_best.pth')
    
    for epoch in range(config['DT_training_params']['epochs']):
        # 添加 tqdm 进度条
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["DT_training_params"]["epochs"]}')
        for batch in train_loader_tqdm:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            
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
        val_mean_reward, val_std_reward = evaluate_on_trajectories(
            model, val_trajectories, config['DT_training_params']['target_rtg'], device
        )
        
        # 记录训练和验证信息
        logger.info(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Reward: {val_mean_reward:.2f} ± {val_std_reward:.2f}')
        
        scheduler.step(val_mean_reward)
        if val_mean_reward > best_val_reward:
            best_val_reward = val_mean_reward
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'Saved best model with validation reward: {best_val_reward:.2f}')
        
        model.train()
    
    return model, best_model_path

def analyze_test_trajectories(model, test_trajectories, target_rtg, device, interval=100):
    """分析测试轨迹，按指定间隔输出详细信息"""
    logger.info("\n分析测试轨迹:")
    analyzed_count = 0
    
    for i, traj in enumerate(test_trajectories):
        if (i + 1) % interval == 0:  # 每interval条轨迹分析一次
            reward = evaluate_on_trajectory(model, traj, target_rtg, device)
            analyze_trajectory(traj, i + 1, reward)
            analyzed_count += 1
    
    if analyzed_count == 0 and len(test_trajectories) > 0:
        # 如果没有分析任何轨迹（测试集可能小于interval），至少分析一条
        reward = evaluate_on_trajectory(model, test_trajectories[0], target_rtg, device)
        analyze_trajectory(test_trajectories[0], 1, reward)
    
    # 计算总体统计信息
    actions_list = []
    soc_list = []
    rewards_list = []
    
    for traj in tqdm(test_trajectories, desc="计算测试集统计信息"):
        reward = evaluate_on_trajectory(model, traj, target_rtg, device)
        rewards_list.append(reward)
        # 假设 SOC 是状态的第一个元素
        soc_values = [state[0] for state in traj['states']]
        soc_list.extend(soc_values)
        actions_list.extend(traj['actions'])
    
    action_dist = np.bincount(actions_list, minlength=model.action_dim) / len(actions_list)
    
    logger.info(f"测试集统计:")
    logger.info(f"动作分布: {action_dist}")
    logger.info(f"SOC范围: [{np.min(soc_list):.2f}, {np.max(soc_list):.2f}]")
    logger.info(f"平均SOC: {np.mean(soc_list):.2f}")
    logger.info(f"平均奖励: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    logger.info(f"最大奖励: {np.max(rewards_list):.2f}")
    logger.info(f"最小奖励: {np.min(rewards_list):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='轨迹文件路径')
    parser.add_argument('--interval', type=int, default=100, help='分析轨迹的间隔')
    args = parser.parse_args()
    
    # 加载轨迹数据
    try:
        with open(args.file, 'rb') as f:
            trajectories = pickle.load(f)
        logger.info(f"加载了 {len(trajectories)} 条轨迹")
    except FileNotFoundError:
        logger.error(f"未找到轨迹文件 {args.file}")
        exit(1)
    
    # 检测状态和动作维度
    sample_state = trajectories[0]['states'][0]
    actual_state_dim = len(sample_state)
    actual_action_dim = max([max(traj['actions']) for traj in trajectories]) + 1
    
    # 计算目标回报
    all_rewards = [sum(traj['rewards']) for traj in trajectories]
    target_rtg = np.mean(all_rewards) * 2
    
    # 创建配置
    config = {
        'DT_params': {
            'hidden_dim': 128,
            'n_layers': 3,
            'n_heads': 4,
            'max_len': 30
        },
        'DT_training_params': {
            'batch_size': 64,
            'learning_rate': 1e-4,
            'epochs': 100,
            'target_rtg': target_rtg
        }
    }
    
    # 分割数据集
    n_trajectories = len(trajectories)
    train_size = int(n_trajectories * 0.7)
    val_size = int(n_trajectories * 0.1)
    
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:train_size + val_size]
    test_trajectories = trajectories[train_size + val_size:]
    
    logger.info(f"训练集大小: {len(train_trajectories)}")
    logger.info(f"验证集大小: {len(val_trajectories)}")
    logger.info(f"测试集大小: {len(test_trajectories)}")
    logger.info(f"目标回报设置为: {target_rtg:.2f}")
    
    # 创建数据集和数据加载器
    train_dataset = TrajectoryDataset(train_trajectories, max_len=config['DT_params']['max_len'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['DT_training_params']['batch_size'],
        shuffle=True
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    model = DecisionTransformer(
        state_dim=actual_state_dim,
        action_dim=actual_action_dim,
        hidden_dim=config['DT_params']['hidden_dim'],
        n_layers=config['DT_params']['n_layers'],
        n_heads=config['DT_params']['n_heads'],
        max_len=config['DT_params']['max_len']
    ).to(device)
    
    # 训练模型
    logger.info("开始训练模型...")
    model, best_model_path = train_model(model, train_loader, val_trajectories, config, device)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 在测试集上评估
    logger.info("\n在测试集上评估最佳模型...")
    test_mean_reward, test_std_reward = evaluate_on_trajectories(
        model, test_trajectories, config['DT_training_params']['target_rtg'], device
    )
    logger.info(f'测试集平均奖励: {test_mean_reward:.2f} ± {test_std_reward:.2f}')
    
    # 分析测试轨迹
    analyze_test_trajectories(
        model, 
        test_trajectories, 
        config['DT_training_params']['target_rtg'],
        device,
        args.interval
    )