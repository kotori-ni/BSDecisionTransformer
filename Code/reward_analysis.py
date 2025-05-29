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
    traces_file = "../Data/pro_traces.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(traces_file):
        logging.error(f"Traces文件不存在: {traces_file}")
        print(f"请确保文件 {traces_file} 存在")
        exit(1)
    
    # 运行分析
    try:
        df_stats, reward_data = analyze_traces_rewards(traces_file)
        logging.info("数据分析完成！")
    except Exception as e:
        logging.error(f"分析过程中出现错误: {str(e)}")
        raise

# 使用示例（注释掉，需要时取消注释）
"""
# 示例1: 分析默认文件
df_stats, reward_data = analyze_traces_rewards("../Data/pro_traces.pkl")

# 示例2: 分析自定义文件
analyze_custom_traces("../Data/pro_traces_with_ppo_predictions.pkl")

# 示例3: 比较多个文件
compare_multiple_files([
    "../Data/pro_traces.pkl",
    "../Data/pro_traces_with_ppo_predictions.pkl"
], ["训练前", "训练后"])

# 示例4: 仅提取和查看统计信息，不绘图
traces = load_traces("../Data/pro_traces.pkl")
reward_data = extract_rewards(traces)
df_stats = calculate_statistics(reward_data)
print(df_stats)
""" 