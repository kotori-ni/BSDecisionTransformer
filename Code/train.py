import torch
import pickle
import argparse
import os
from BS_EV_Environment_Base import BS_EV_Base
from DecisionTransformer import DecisionTransformer, train_decision_transformer, evaluate_decision_transformer, evaluate_on_trajectory
import json
import logging
import numpy as np

# 确保日志配置正确
log_dir = os.path.join(os.path.dirname(__file__), '..', 'Log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train.log')

# 配置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 检查是否已有处理器，避免重复添加
if not logger.handlers:
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    logging.info("=" * 50)
    logging.info(f"开始新的训练任务: {args.file}")
    logging.info(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # 环境初始化
    env = BS_EV_Base(n_charge=24, n_traffic=24, n_RTP=24, n_weather=24)
    
    # 加载轨迹数据
    traj_file = args.file
    try:
        with open(traj_file, 'rb') as f:
            trajectories = pickle.load(f)
        logging.info(f"加载了 {len(trajectories)} 条轨迹: {traj_file}")
    except FileNotFoundError:
        logging.error(f"未找到轨迹文件 {traj_file}，请先运行轨迹收集")
        return

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")

    # 80% 用于训练, 20% 用于测试
    num_total = len(trajectories)
    num_train = int(0.8 * num_total)
    train_trajectories = trajectories[:num_train]
    test_trajectories = trajectories[num_train:]

    logging.info(f"训练集: {len(train_trajectories)} 条, 测试集: {len(test_trajectories)} 条")

    target_rtg = 44000.0  # 默认值，可根据环境实际情况调整
    logging.info(f"目标回报设置为: {target_rtg:.2f}")

    # -----------------------------------
    # 从 config.json 读取 Decision Transformer 超参数
    # -----------------------------------
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r', encoding='utf-8') as cf:
        config = json.load(cf)
    dt_config = config.get('decision_transformer', {})

    # 模型初始化
    model = DecisionTransformer(
        state_dim=env.n_states,
        action_dim=env.n_actions,
        config=dt_config
    )
    
    logging.info("开始训练模型...")
    train_decision_transformer(
        model=model,
        trajectories=train_trajectories,
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

    # 评估模型
    logging.info("\n开始评估模型...")
    # 若有best模型可加载best，否则加载刚刚训练的
    best_model_path = '../Models/dt_model_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"加载最优模型: {best_model_path}")
    else:
        model.load_state_dict(torch.load(model_path))
        logging.info(f"加载刚刚训练的模型: {model_path}")
    eval_rewards = torch.tensor([], device=device)
    for i in range(30):
        reward = evaluate_decision_transformer(model, env, target_rtg, max_len=dt_config.get('max_len', 30), device=device)
        eval_rewards = torch.cat([eval_rewards, torch.tensor([reward], device=device)])
        logging.info(f"评估回合 {i+1}/30 奖励: {reward:.2f}")
    
    logging.info(f"\n评估结果:")
    logging.info(f"平均奖励: {eval_rewards.mean().item():.2f}")
    logging.info(f"标准差: {eval_rewards.std().item():.2f}")
    logging.info(f"最大奖励: {eval_rewards.max().item():.2f}")
    logging.info(f"最小奖励: {eval_rewards.min().item():.2f}")

    # -------------------------------------------------
    # 针对测试集轨迹输出信息（每 100 条）
    # -------------------------------------------------
    logging.info("\n开始轨迹评估...")
    
    # 初始化测试集统计数据收集变量
    all_action_accuracy = []
    all_true_rewards = []
    all_pred_rewards = []
    all_reward_errors = []
    all_soc_values = []
    
    for idx, traj in enumerate(test_trajectories):
        actions_pred, soc_mean, rewards_true, rewards_pred = evaluate_on_trajectory(
            model, traj, env, device=device, max_len=dt_config.get('max_len', 30)
        )
        
        # 收集测试数据统计信息
        if 'actions' in traj:
            try:
                # 获取真实动作
                true_actions = []
                for action in traj['actions']:
                    if isinstance(action, np.ndarray) and action.size == 1:
                        true_actions.append(int(action.item()))
                    else:
                        true_actions.append(int(action))
                
                # 计算动作准确率
                min_len = min(len(true_actions), len(actions_pred))
                if min_len > 0:
                    matches = sum(true_actions[i] == actions_pred[i] for i in range(min_len))
                    accuracy = matches / min_len
                    all_action_accuracy.append(accuracy)
            except Exception as e:
                logging.warning(f"计算动作准确率出错: {e}")
        
        # 收集奖励统计信息
        if rewards_true is not None and rewards_pred is not None:
            try:
                # 汇总奖励
                all_true_rewards.append(rewards_true.sum().item())
                all_pred_rewards.append(rewards_pred.sum().item())
                
                # 奖励误差
                reward_errors = (rewards_pred - rewards_true).cpu().numpy()
                all_reward_errors.extend(reward_errors)
            except Exception as e:
                logging.warning(f"计算奖励统计出错: {e}")
        
        # 收集SOC值
        all_soc_values.append(soc_mean)

        # 每 100 条轨迹打印一次信息
        if (idx + 1) % 100 == 0:
            # 将actions_pred转换为张量，使用torch进行唯一值计数
            actions_tensor = torch.tensor(actions_pred, device=device)
            unique = actions_tensor.unique()
            action_dist = {}
            for u in unique:
                action_dist[int(u.item())] = int((actions_tensor == u).sum().item())

            logging.info("-" * 50)
            logging.info(f"测试集第 {idx + 1}/{len(test_trajectories)} 条轨迹:")
            logging.info(f"动作分布: {action_dist}")
            logging.info(f"SOC 平均值: {soc_mean:.3f}")
            
            # 计算奖励对比
            if rewards_true is not None and rewards_pred is not None:
                # 统计奖励差异
                reward_diff = rewards_pred - rewards_true
                reward_diff_abs = torch.abs(reward_diff)
                
                # 奖励汇总信息
                logging.info(f"真实奖励总和: {rewards_true.sum().item():.2f}")
                logging.info(f"预测奖励总和: {rewards_pred.sum().item():.2f}")
                logging.info(f"奖励差异: 平均: {reward_diff.mean().item():.2f}, 绝对平均: {reward_diff_abs.mean().item():.2f}")
                logging.info(f"奖励差异比例: {(rewards_pred.sum() / rewards_true.sum() - 1).item() * 100:.2f}%")
                
                # 可选：输出前10步的详细对比
                logging.info("\n前10步的奖励对比:")
                for step in range(min(10, len(rewards_true))):
                    logging.info(f"步骤 {step}: 真实={rewards_true[step].item():.2f}, 预测={rewards_pred[step].item():.2f}, 差异={reward_diff[step].item():.2f}")
    
    # -------------------------------------------------
    # 测试集整体性能评估
    # -------------------------------------------------
    logging.info("\n" + "=" * 50)
    logging.info("测试集整体性能评估结果:")
    
    # 1. 动作预测准确率分析 - 仅保留平均准确率
    if all_action_accuracy:
        accuracy_array = np.array(all_action_accuracy)
        logging.info(f"\n1. 动作预测准确率: {accuracy_array.mean():.4f}")
    else:
        logging.info("\n1. 动作预测准确率: 无法计算（测试数据中可能不包含真实动作）")
    
    # 2. 回报对比分析 - 仅保留平均回报误差与误差率
    if all_true_rewards and all_pred_rewards:
        true_rewards_array = np.array(all_true_rewards)
        pred_rewards_array = np.array(all_pred_rewards)
        
        logging.info(f"\n2. 回报对比分析:")
        logging.info(f"   平均回报误差: {(pred_rewards_array - true_rewards_array).mean():.2f}")
        logging.info(f"   平均回报误差率: {((pred_rewards_array / true_rewards_array) - 1).mean() * 100:.2f}%")
    else:
        logging.info("\n2. 回报对比分析: 无法计算（测试数据中可能不包含奖励信息）")
    
    # 3. 奖励预测误差分析 - 仅保留平均误差、绝对误差均值、最大正/负误差
    if all_reward_errors:
        reward_errors_array = np.array(all_reward_errors)
        abs_errors = np.abs(reward_errors_array)
        
        logging.info(f"\n3. 奖励预测误差分析:")
        logging.info(f"   平均误差: {reward_errors_array.mean():.4f}")
        logging.info(f"   绝对误差均值: {abs_errors.mean():.4f}")
        logging.info(f"   最大正误差: {reward_errors_array.max():.4f}")
        logging.info(f"   最大负误差: {reward_errors_array.min():.4f}")
    else:
        logging.info("\n3. 奖励预测误差分析: 无法计算（测试数据中可能不包含奖励信息）")
    
    logging.info("=" * 50)

if __name__ == "__main__":
    main()