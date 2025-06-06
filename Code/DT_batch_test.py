import pickle
import torch
import json
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from DecisionTransformer import DecisionTransformer
from BS_EV_Environment_Base import simulate_actions, load_traces, BS_EV_Base

def load_all_dt_models(models_dir="../Models/DT/"):
    """
    加载所有DT模型文件
    
    Returns:
        list: 包含(model_path, epoch)的元组列表
    """
    model_files = glob.glob(os.path.join(models_dir, "dt_model_*.pth"))
    model_info = []
    
    for model_file in model_files:
        # 从文件名中提取epoch信息
        filename = os.path.basename(model_file)
        if filename.startswith("dt_model_") and filename.endswith(".pth"):
            epoch_str = filename[9:-4]  # 去掉 "dt_model_" 和 ".pth"
            try:
                epoch = int(epoch_str)
                model_info.append((model_file, epoch))
            except ValueError:
                print(f"警告: 无法解析epoch信息: {filename}")
    
    # 按epoch排序
    model_info.sort(key=lambda x: x[1])
    return model_info

def generate_actions_for_model(model, traces, device):
    """
    为指定模型生成所有trace的动作序列
    
    Args:
        model: DT模型
        traces: 测试数据
        device: 计算设备
    
    Returns:
        list: 包含所有trace的动作序列
    """
    all_actions = []
    model.eval()
    
    for idx, trace in enumerate(tqdm(traces, desc="生成动作序列", leave=False)):
        env = BS_EV_Base()
        state = env.reset(trace)
        done = False
        action_seq = []
        max_len = model.max_len if hasattr(model, 'max_len') else 30
        
        # 设定目标RTG
        target_return = 2000.0
        rtgs = [target_return]
        states = [state]
        actions = [0]  # dummy action
        timesteps = [0]
        t = 0
        
        while not done and t < len(trace["pro_trace"]):
            # 构造模型输入（只取最近max_len步）
            states_input = torch.tensor([states[-max_len:]], dtype=torch.float32, device=device)
            actions_input = torch.tensor([actions[-max_len:]], dtype=torch.long, device=device)
            rtgs_input = torch.tensor([rtgs[-max_len:]], dtype=torch.float32, device=device)
            timesteps_input = torch.tensor([list(range(len(states_input[0])))], dtype=torch.long, device=device)
            
            # 推理
            with torch.no_grad():
                action_logits = model(states_input, actions_input, rtgs_input, timesteps_input)
                action = torch.argmax(action_logits[0, -1]).item()
            
            action_seq.append(action)
            
            # 环境交互
            next_state, reward, done, _ = env.step(action)
            
            # 更新序列
            states.append(next_state)
            actions.append(action)
            rtgs.append(rtgs[-1] - reward)
            t += 1
        
        all_actions.append(action_seq)
    
    return all_actions

def calculate_action_distribution(action_sequences):
    """
    计算动作序列的分布
    
    Args:
        action_sequences: 所有trace的动作序列列表
    
    Returns:
        dict: 动作分布字典
    """
    # 统计所有动作
    all_actions = []
    for action_seq in action_sequences:
        all_actions.extend(action_seq)
    
    # 计算分布
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    
    action_distribution = {}
    for action in range(3):  # 假设有3个动作 (0, 1, 2)
        count = action_counts.get(action, 0)
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        action_distribution[f"action_{action}"] = {
            "count": count,
            "percentage": percentage
        }
    
    action_distribution["total_actions"] = total_actions
    return action_distribution

def print_action_distribution(epoch, action_distribution):
    """
    打印动作分布信息
    
    Args:
        epoch: 模型epoch
        action_distribution: 动作分布字典
    """
    print(f"\nEpoch {epoch} 动作分布:")
    print("-" * 40)
    for action in range(3):
        action_key = f"action_{action}"
        if action_key in action_distribution:
            count = action_distribution[action_key]["count"]
            percentage = action_distribution[action_key]["percentage"]
            print(f"  动作 {action}: {count:5d} 次 ({percentage:5.1f}%)")
    print(f"  总动作数: {action_distribution['total_actions']}")

def batch_test_dt_models():
    """
    批量测试所有DT模型
    """
    # 配置
    models_dir = "../Models/DT/"
    test_file = "../Data/pro_traces_test.pkl"
    config_path = "config.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_file = "../Results/dt_batch_results.json"
    
    print(f"使用设备: {device}")
    
    # 确保结果目录存在
    os.makedirs("../Results", exist_ok=True)
    
    # 读取配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    dt_config = config.get("decision_transformer", {})
    
    # 加载测试数据
    try:
        with open(test_file, "rb") as f:
            traces = pickle.load(f)
        print(f"加载了 {len(traces)} 条测试trace")
        
        # 如果超过100条，只取前100条
        if len(traces) > 100:
            traces = traces[:100]
            print(f"使用前100条trace进行测试")
            
    except FileNotFoundError:
        print(f"未找到测试文件: {test_file}")
        return
    
    # 加载所有模型
    model_info = load_all_dt_models(models_dir)
    if not model_info:
        print(f"在 {models_dir} 中未找到任何DT模型文件")
        return
    
    print(f"找到 {len(model_info)} 个模型文件")
    
    # 模型初始化参数
    state_dim = 145
    action_dim = 3
    
    # 存储所有结果
    all_results = {}
    action_distributions = {}  # 存储动作分布
    
    # 测试每个模型
    for model_path, epoch in tqdm(model_info, desc="测试模型"):
        print(f"\n测试模型: Epoch {epoch} ({os.path.basename(model_path)})")
        
        try:
            # 加载模型
            model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, config=dt_config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            # 生成动作序列
            action_sequences = generate_actions_for_model(model, traces, device)
            
            # 计算动作分布
            action_distribution = calculate_action_distribution(action_sequences)
            action_distributions[f"epoch_{epoch}"] = action_distribution
            
            # 打印动作分布
            print_action_distribution(epoch, action_distribution)
            
            # 将动作序列写入traces
            for i, action_seq in enumerate(action_sequences):
                traces[i]["DT_action"] = action_seq
            
            # 保存带有动作序列的traces（临时文件）
            temp_file = f"../Data/temp_test_epoch_{epoch}.pkl"
            with open(temp_file, "wb") as f:
                pickle.dump(traces, f)
            
            # 计算收益并获取统计信息
            model_results = simulate_actions_with_stats("DT", temp_file, model_name=f"DT_epoch_{epoch}")
            
            # 删除临时文件
            os.remove(temp_file)
            
            # 将动作分布添加到结果中
            model_results["action_distribution"] = action_distribution
            
            # 存储结果
            all_results[f"epoch_{epoch}"] = model_results
            
            print(f"Epoch {epoch} - 平均收益: {model_results['avg_reward']:.2f}, "
                  f"标准差: {model_results['std_reward']:.2f}")
            
        except Exception as e:
            print(f"处理模型 {model_path} 时出错: {e}")
            all_results[f"epoch_{epoch}"] = {"error": str(e)}
    
    # 保存所有结果
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n所有结果已保存到: {results_file}")
    
    # 保存动作分布结果
    action_dist_file = "../Results/dt_action_distributions.json"
    with open(action_dist_file, "w", encoding="utf-8") as f:
        json.dump(action_distributions, f, indent=2, ensure_ascii=False)
    
    print(f"动作分布结果已保存到: {action_dist_file}")
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    
    if valid_results:
        best_epoch = max(valid_results.keys(), key=lambda k: valid_results[k]['avg_reward'])
        best_reward = valid_results[best_epoch]['avg_reward']
        
        print(f"最佳模型: {best_epoch}")
        print(f"最佳平均收益: {best_reward:.2f}")
        print(f"最佳模型标准差: {valid_results[best_epoch]['std_reward']:.2f}")
        
        print(f"\n各epoch性能对比:")
        for epoch_key in sorted(valid_results.keys(), key=lambda x: int(x.split('_')[1])):
            result = valid_results[epoch_key]
            print(f"{epoch_key}: 平均收益={result['avg_reward']:.2f}, 标准差={result['std_reward']:.2f}")
    
    # 打印动作分布汇总
    print("\n" + "=" * 60)
    print("动作分布汇总")
    print("=" * 60)
    for epoch_key in sorted(action_distributions.keys(), key=lambda x: int(x.split('_')[1])):
        epoch_num = epoch_key.split('_')[1]
        action_dist = action_distributions[epoch_key]
        print(f"\nEpoch {epoch_num}:")
        for action in range(3):
            action_key = f"action_{action}"
            if action_key in action_dist:
                count = action_dist[action_key]["count"]
                percentage = action_dist[action_key]["percentage"]
                print(f"  动作 {action}: {count:5d} 次 ({percentage:5.1f}%)")
        print(f"  总动作数: {action_dist['total_actions']}")

def simulate_actions_with_stats(action_type, trace_file, model_name=None):
    """
    计算收益并返回统计信息的增强版simulate_actions
    
    Args:
        action_type: 动作类型
        trace_file: trace文件路径
        model_name: 模型名称（用于日志）
    
    Returns:
        dict: 包含统计信息的字典
    """
    assert action_type in ["DP", "PPO", "SAC", "DQN", "Nop", "DT"]
    
    traces = load_traces(trace_file)
    env = BS_EV_Base()
    
    all_rewards = []
    reward_name = f"{action_type}_reward"
    action_name = f"{action_type}_action"
    
    for idx in range(len(traces)):
        env.reset(traces[idx])
        
        total_reward = 0.0
        action_list = traces[idx][action_name] if action_name != "Nop_action" else [0] * 720
        
        for i, action in enumerate(action_list):
            _, reward, done, _ = env.step(action)
            total_reward += reward
        
        traces[idx][reward_name] = total_reward
        all_rewards.append(total_reward)
        
        if model_name:
            print(f"{model_name} - trace_idx: {idx}, Total reward: {total_reward:.2f}")
        else:
            print(f"trace_idx: {idx}, action: {action_name}, Total reward: {total_reward:.2f}")
    
    # 保存更新后的traces
    with open(trace_file, "wb") as f:
        pickle.dump(traces, f)
    
    # 计算统计信息
    stats = {
        "avg_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "min_reward": np.min(all_rewards),
        "max_reward": np.max(all_rewards),
        "median_reward": np.median(all_rewards),
        "total_traces": len(all_rewards)
    }
    
    return stats

if __name__ == "__main__":
    batch_test_dt_models() 