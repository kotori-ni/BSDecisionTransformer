import numpy as np
import pandas as pd
import pickle
import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_traces(trace_file):
    """加载traces文件"""
    try:
        with open(trace_file, 'rb') as f:
            traces = pickle.load(f)
        logging.info(f"成功加载 {len(traces)} 条traces从 {trace_file}")
        return traces
    except Exception as e:
        logging.error(f"加载traces文件失败: {str(e)}")
        raise

def extract_rewards(traces):
    """
    从traces中提取各类reward数据
    
    Args:
        traces: traces列表
        
    Returns:
        dict: 包含各算法reward列表的字典
    """
    reward_types = ['DP_reward', 'DT_reward', 'Nop_reward', 'PPO_reward', 'SAC_reward', 'DQN_reward']
    reward_data = {reward_type: [] for reward_type in reward_types}
    
    for trace_idx, trace in enumerate(traces):
        for reward_type in reward_types:
            reward_value = trace.get(reward_type)
            if reward_value is not None:
                reward_data[reward_type].append(reward_value)
            else:
                logging.debug(f"Trace {trace_idx}: {reward_type} 为 None，跳过")
    
    # 记录每种算法的有效数据数量
    for reward_type in reward_types:
        count = len(reward_data[reward_type])
        logging.info(f"{reward_type}: {count} 条有效数据")
    
    return reward_data

def calculate_statistics(reward_data):
    """
    计算各算法reward的统计信息
    
    Args:
        reward_data: 包含各算法reward列表的字典
        
    Returns:
        pd.DataFrame: 统计结果DataFrame
    """
    statistics = []
    
    for algorithm, rewards in reward_data.items():
        if len(rewards) > 0:
            stats = {
                'Algorithm': algorithm.replace('_reward', ''),
                'Count': len(rewards),
                'Mean': np.mean(rewards),
                'Std': np.std(rewards),
                'Min': np.min(rewards),
                'Max': np.max(rewards),
                'Median': np.median(rewards)
            }
            statistics.append(stats)
        else:
            logging.warning(f"{algorithm}: 没有有效数据")
    
    df_stats = pd.DataFrame(statistics)
    return df_stats

def print_summary_table(df_stats):
    """打印汇总统计表格"""
    print("\n" + "="*80)
    print("各算法Reward统计汇总")
    print("="*80)
    
    # 格式化输出
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df_stats.to_string(index=False))
    
    print("\n" + "="*80)
    
    # 找出最佳算法
    if len(df_stats) > 0:
        best_mean_idx = df_stats['Mean'].idxmax()
        best_algorithm = df_stats.loc[best_mean_idx]
        
        print(f"最佳平均性能算法: {best_algorithm['Algorithm']}")
        print(f"平均Reward: {best_algorithm['Mean']:.2f}")
        print(f"标准差: {best_algorithm['Std']:.2f}")
        print(f"样本数量: {best_algorithm['Count']}")

def extract_actions(traces):
    """
    从traces中提取各类action数据
    
    Args:
        traces: traces列表
        
    Returns:
        dict: 包含各算法action列表的字典
    """
    action_types = ['DP_action', 'DT_action', 'PPO_action', 'SAC_action', 'DQN_action']
    action_data = {action_type: [] for action_type in action_types}
    
    for trace_idx, trace in enumerate(traces):
        for action_type in action_types:
            action_sequence = trace.get(action_type)
            if action_sequence is not None:
                action_data[action_type].append(action_sequence)
            else:
                logging.debug(f"Trace {trace_idx}: {action_type} 为 None，跳过")
    
    # 记录每种算法的有效数据数量
    for action_type in action_types:
        count = len(action_data[action_type])
        logging.info(f"{action_type}: {count} 条有效动作序列")
    
    return action_data

def calculate_action_accuracy(action_data):
    """
    计算各算法相对于DP的动作准确度
    
    Args:
        action_data: 包含各算法action列表的字典
        
    Returns:
        pd.DataFrame: 动作准确度统计结果
    """
    if 'DP_action' not in action_data or len(action_data['DP_action']) == 0:
        logging.warning("没有找到DP_action数据，无法计算动作准确度")
        return pd.DataFrame()
    
    dp_actions = action_data['DP_action']
    accuracy_stats = []
    
    # 要比较的算法（排除Nop和DP自身）
    compare_algorithms = ['DT_action', 'PPO_action', 'SAC_action', 'DQN_action']
    
    for algorithm in compare_algorithms:
        if algorithm in action_data and len(action_data[algorithm]) > 0:
            algorithm_actions = action_data[algorithm]
            
            # 计算每个trace的准确度
            trace_accuracies = []
            valid_comparisons = 0
            total_actions = 0
            correct_actions = 0
            
            # 动作分布统计
            action_distribution = {0: 0, 1: 0, 2: 0}  # 不动作、充电、放电
            dp_action_distribution = {0: 0, 1: 0, 2: 0}
            # 新增：分动作统计正确数量和DP中各动作总数
            correct_actions_per_type = {0: 0, 1: 0, 2: 0}
            dp_actions_count_per_type = {0: 0, 1: 0, 2: 0}
            
            for i in range(min(len(dp_actions), len(algorithm_actions))):
                dp_sequence = dp_actions[i]
                alg_sequence = algorithm_actions[i]
                
                if len(dp_sequence) == len(alg_sequence):
                    # 计算当前trace的准确度
                    correct = 0
                    for dp_act, alg_act in zip(dp_sequence, alg_sequence):
                        dp_actions_count_per_type[dp_act] += 1
                        if dp_act == alg_act:
                            correct += 1
                            correct_actions_per_type[dp_act] += 1
                    
                    accuracy = correct / len(dp_sequence) if len(dp_sequence) > 0 else 0
                    trace_accuracies.append(accuracy)
                    
                    # 累计统计
                    total_actions += len(dp_sequence)
                    correct_actions += correct
                    valid_comparisons += 1
                    
                    # 统计动作分布
                    for action in alg_sequence:
                        action_distribution[action] += 1
                    for action in dp_sequence:
                        dp_action_distribution[action] += 1
                else:
                    logging.warning(f"算法 {algorithm} trace {i} 的动作序列长度与DP不匹配")
            
            if valid_comparisons > 0:
                overall_accuracy = correct_actions / total_actions
                mean_trace_accuracy = np.mean(trace_accuracies)
                std_trace_accuracy = np.std(trace_accuracies)
                
                # 计算动作分布百分比
                total_alg_actions = sum(action_distribution.values())
                total_dp_actions = sum(dp_action_distribution.values())
                
                action_dist_pct = {k: v/total_alg_actions*100 if total_alg_actions > 0 else 0 
                                 for k, v in action_distribution.items()}
                dp_action_dist_pct = {k: v/total_dp_actions*100 if total_dp_actions > 0 else 0 
                                    for k, v in dp_action_distribution.items()}
                
                # 计算各动作类型的准确度
                accuracy_action_0 = correct_actions_per_type[0] / dp_actions_count_per_type[0] if dp_actions_count_per_type[0] > 0 else 0
                accuracy_action_1 = correct_actions_per_type[1] / dp_actions_count_per_type[1] if dp_actions_count_per_type[1] > 0 else 0
                accuracy_action_2 = correct_actions_per_type[2] / dp_actions_count_per_type[2] if dp_actions_count_per_type[2] > 0 else 0

                stats = {
                    'Algorithm': algorithm.replace('_action', ''),
                    'Overall_Accuracy': overall_accuracy,
                    'Mean_Trace_Accuracy': mean_trace_accuracy,
                    'Action0_Pct': action_dist_pct[0],  # 不动作百分比
                    'Action1_Pct': action_dist_pct[1],  # 充电百分比
                    'Action2_Pct': action_dist_pct[2],  # 放电百分比
                    'DP_Action0_Pct': dp_action_dist_pct[0],  # DP不动作百分比
                    'DP_Action1_Pct': dp_action_dist_pct[1],  # DP充电百分比
                    'DP_Action2_Pct': dp_action_dist_pct[2],   # DP放电百分比
                    'Accuracy_Action0': accuracy_action_0,
                    'Accuracy_Action1': accuracy_action_1,
                    'Accuracy_Action2': accuracy_action_2
                }
                accuracy_stats.append(stats)
            else:
                logging.warning(f"算法 {algorithm}: 没有有效的比较数据")
    
    df_accuracy = pd.DataFrame(accuracy_stats)
    return df_accuracy

def print_action_accuracy_table(df_accuracy):
    """打印动作准确度统计表格"""
    if len(df_accuracy) == 0:
        print("\n没有可用的动作准确度数据")
        return
    
    print("\n" + "="*120)
    print("各算法相对于DP的动作准确度分析")
    print("="*120)
    
    # 格式化输出 - 分两个表格显示
    # 表格1: 准确度统计
    accuracy_columns = ['Algorithm', 'Overall_Accuracy', 'Mean_Trace_Accuracy', 
                       'Accuracy_Action0', 'Accuracy_Action1', 'Accuracy_Action2']
    df_acc_display = df_accuracy[accuracy_columns].copy()
    
    # 格式化百分比
    df_acc_display['Overall_Accuracy'] = df_acc_display['Overall_Accuracy'].apply(lambda x: f"{x:.4f}")
    df_acc_display['Mean_Trace_Accuracy'] = df_acc_display['Mean_Trace_Accuracy'].apply(lambda x: f"{x:.4f}")
    df_acc_display['Std_Trace_Accuracy'] = df_acc_display['Std_Trace_Accuracy'].apply(lambda x: f"{x:.4f}")
    df_acc_display['Accuracy_Action0'] = df_acc_display['Accuracy_Action0'].apply(lambda x: f"{x:.4f}")
    df_acc_display['Accuracy_Action1'] = df_acc_display['Accuracy_Action1'].apply(lambda x: f"{x:.4f}")
    df_acc_display['Accuracy_Action2'] = df_acc_display['Accuracy_Action2'].apply(lambda x: f"{x:.4f}")
    
    print("准确度统计:")
    print(df_acc_display.to_string(index=False))
    
    # 表格2: 动作分布对比
    print("\n" + "-"*120)
    print("动作分布对比 (%):")
    distribution_columns = ['Algorithm', 'Action0_Pct', 'Action1_Pct', 'Action2_Pct', 
                           'DP_Action0_Pct', 'DP_Action1_Pct', 'DP_Action2_Pct']
    df_dist_display = df_accuracy[distribution_columns].copy()
    
    # 格式化百分比
    for col in ['Action0_Pct', 'Action1_Pct', 'Action2_Pct', 
                'DP_Action0_Pct', 'DP_Action1_Pct', 'DP_Action2_Pct']:
        df_dist_display[col] = df_dist_display[col].apply(lambda x: f"{x:.2f}")
    
    print(df_dist_display.to_string(index=False))
    
    print("\n" + "="*120)
    
    # 找出最佳准确度算法
    if len(df_accuracy) > 0:
        best_accuracy_idx = df_accuracy['Overall_Accuracy'].idxmax()
        best_algorithm = df_accuracy.loc[best_accuracy_idx]
        
        print(f"最高动作准确度算法: {best_algorithm['Algorithm']}")
        print(f"整体准确度: {best_algorithm['Overall_Accuracy']:.4f} ({best_algorithm['Overall_Accuracy']*100:.2f}%)")
        print(f"平均Trace准确度: {best_algorithm['Mean_Trace_Accuracy']:.4f} ± {best_algorithm['Std_Trace_Accuracy']:.4f}")
        print(f"有效比较Traces: {best_algorithm['Valid_Traces']}")
        print(f"总动作数: {best_algorithm['Total_Actions']}, 正确动作数: {best_algorithm['Correct_Actions']}")

def analyze_traces_with_actions(traces_file, output_dir='../Figure'):
    """
    综合分析函数，同时分析reward和动作准确度
    
    Args:
        traces_file: traces文件路径
        output_dir: 输出目录
    """
    logging.info(f"开始综合分析traces文件: {traces_file}")
    
    # 1. 加载数据
    traces = load_traces(traces_file)
    
    # 2. 提取reward数据并分析
    reward_data = extract_rewards(traces)
    df_stats = calculate_statistics(reward_data)
    print_summary_table(df_stats)
    
    # 3. 提取action数据并分析准确度
    action_data = extract_actions(traces)
    df_accuracy = calculate_action_accuracy(action_data)
    print_action_accuracy_table(df_accuracy)
    
    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存reward统计结果
    stats_output_path = os.path.join(output_dir, 'reward_statistics.csv')
    df_stats.to_csv(stats_output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Reward统计结果已保存到: {stats_output_path}")
    
    # 保存动作准确度结果
    if len(df_accuracy) > 0:
        accuracy_output_path = os.path.join(output_dir, 'action_accuracy_statistics.csv')
        df_accuracy.to_csv(accuracy_output_path, index=False, encoding='utf-8-sig')
        logging.info(f"动作准确度统计结果已保存到: {accuracy_output_path}")
    
    return df_stats, reward_data, df_accuracy, action_data

def analyze_traces_rewards(traces_file, output_dir='../Figure'):
    """
    主分析函数
    
    Args:
        traces_file: traces文件路径
        output_dir: 输出目录
    """
    logging.info(f"开始分析traces文件: {traces_file}")
    
    # 1. 加载数据
    traces = load_traces(traces_file)
    
    # 2. 提取reward数据
    reward_data = extract_rewards(traces)
    
    # 3. 计算统计信息
    df_stats = calculate_statistics(reward_data)
    
    # 4. 打印汇总表格
    print_summary_table(df_stats)
    
    # 5. 保存统计结果到CSV
    stats_output_path = os.path.join(output_dir, 'reward_statistics.csv')
    os.makedirs(output_dir, exist_ok=True)
    df_stats.to_csv(stats_output_path, index=False, encoding='utf-8-sig')
    logging.info(f"统计结果已保存到: {stats_output_path}")
    
    return df_stats, reward_data

def analyze_action_accuracy_only(traces_file, output_dir='../Figure'):
    """
    仅分析动作准确度的函数
    
    Args:
        traces_file: traces文件路径
        output_dir: 输出目录
        
    Returns:
        df_accuracy: 动作准确度DataFrame
        action_data: 原始动作数据
    """
    logging.info(f"开始分析动作准确度，traces文件: {traces_file}")
    
    # 1. 加载数据
    traces = load_traces(traces_file)
    
    # 2. 提取action数据
    action_data = extract_actions(traces)
    
    # 3. 计算动作准确度
    df_accuracy = calculate_action_accuracy(action_data)
    
    # 4. 打印结果
    print_action_accuracy_table(df_accuracy)
    
    # 5. 保存结果
    if len(df_accuracy) > 0:
        os.makedirs(output_dir, exist_ok=True)
        accuracy_output_path = os.path.join(output_dir, 'action_accuracy_statistics.csv')
        df_accuracy.to_csv(accuracy_output_path, index=False, encoding='utf-8-sig')
        logging.info(f"动作准确度统计结果已保存到: {accuracy_output_path}")
    
    return df_accuracy, action_data

def analyze_custom_traces(traces_file, output_dir='../Figure'):
    """
    自定义分析函数，可以指定不同的traces文件
    
    Args:
        traces_file: 自定义的traces文件路径
        output_dir: 输出目录
        
    使用示例:
        analyze_custom_traces("../Data/pro_traces_with_ppo_predictions.pkl")
        analyze_custom_traces("../Data/test_traces.pkl", "../Results")
    """
    return analyze_traces_rewards(traces_file, output_dir)

def compare_multiple_files(file_list, labels=None, output_dir='../Figure'):
    """
    比较多个traces文件的性能
    
    Args:
        file_list: traces文件路径列表
        labels: 对应的标签列表，如果为None则使用文件名
        output_dir: 输出目录
        
    使用示例:
        compare_multiple_files([
            "../Data/pro_traces.pkl",
            "../Data/pro_traces_with_ppo_predictions.pkl"
        ], ["原始数据", "包含PPO预测"])
    """
    if labels is None:
        labels = [os.path.basename(f) for f in file_list]
    
    all_stats = []
    
    for file_path, label in zip(file_list, labels):
        if os.path.exists(file_path):
            logging.info(f"分析文件: {file_path} ({label})")
            traces = load_traces(file_path)
            reward_data = extract_rewards(traces)
            df_stats = calculate_statistics(reward_data)
            df_stats['Source'] = label
            all_stats.append(df_stats)
        else:
            logging.warning(f"文件不存在，跳过: {file_path}")
    
    if all_stats:
        # 合并所有统计结果
        combined_stats = pd.concat(all_stats, ignore_index=True)
        
        # 保存合并结果
        comparison_output = os.path.join(output_dir, 'multiple_files_comparison.csv')
        os.makedirs(output_dir, exist_ok=True)
        combined_stats.to_csv(comparison_output, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*100)
        print("多文件比较结果")
        print("="*100)
        print(combined_stats.to_string(index=False))
        
        logging.info(f"多文件比较结果已保存到: {comparison_output}")
        return combined_stats
    else:
        logging.error("没有有效的文件进行比较")
        return None

if __name__ == "__main__":
    # 默认分析文件
    traces_file = "../Data/pro_traces_test.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(traces_file):
        logging.error(f"Traces文件不存在: {traces_file}")
        print(f"请确保文件 {traces_file} 存在")
        exit(1)
    
    # 运行综合分析（包括reward和动作准确度）
    try:
        df_stats, reward_data, df_accuracy, action_data = analyze_traces_with_actions(traces_file)
        logging.info("综合数据分析完成！")
    except Exception as e:
        logging.error(f"分析过程中出现错误: {str(e)}")
        raise

# 使用示例（注释掉，需要时取消注释）
"""
# 示例1: 综合分析（reward + 动作准确度）
df_stats, reward_data, df_accuracy, action_data = analyze_traces_with_actions("../Data/pro_traces.pkl")

# 示例2: 仅分析动作准确度
df_accuracy, action_data = analyze_action_accuracy_only("../Data/pro_traces.pkl")

# 示例3: 仅分析reward（原有功能）
df_stats, reward_data = analyze_traces_rewards("../Data/pro_traces.pkl")

# 示例4: 分析自定义文件
analyze_custom_traces("../Data/pro_traces_with_ppo_predictions.pkl")

# 示例5: 比较多个文件
compare_multiple_files([
    "../Data/pro_traces.pkl",
    "../Data/pro_traces_with_ppo_predictions.pkl"
], ["训练前", "训练后"])
""" 