import re
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def extract_scores_from_log(log_file):
    """从日志文件中提取训练分数和验证分数"""
    train_scores = []
    val_scores = []
    episodes = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配形如 "Episode X: Train score Y.Y, Avg validation score Z.Z" 的行
            match = re.search(r'Episode (\d+): Train score ([\d.]+), Avg validation score ([\d.]+)', line)
            if match:
                episode = int(match.group(1))
                train_score = float(match.group(2))
                val_score = float(match.group(3))
                
                episodes.append(episode)
                train_scores.append(train_score)
                val_scores.append(val_score)
    
    return episodes, train_scores, val_scores

def plot_training_curves(log_files, algorithm_names=None):
    """绘制多个算法的训练曲线"""
    if algorithm_names is None:
        algorithm_names = [os.path.basename(f).split('.')[0] for f in log_files]
    
    plt.figure(figsize=(12, 6))
    
    for log_file, name in zip(log_files, algorithm_names):
        episodes, train_scores, val_scores = extract_scores_from_log(log_file)
        
        # 计算运行平均值
        train_running_avg = np.zeros(len(train_scores))
        val_running_avg = np.zeros(len(val_scores))
        
        for i in range(len(train_scores)):
            train_running_avg[i] = np.mean(train_scores[max(0, i-10):(i+1)])
            val_running_avg[i] = np.mean(val_scores[max(0, i-10):(i+1)])
        
        # 绘制训练分数和验证分数
        plt.plot(episodes, train_running_avg, '--', label=f'{name} Train', alpha=0.7)
        plt.plot(episodes, val_running_avg, '-', label=f'{name} Validation', alpha=0.7)
    
    plt.title('Training and Validation Scores\n(Points: Raw Scores, Lines: 10-Episode Running Average)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整图形大小以适应图例
    plt.tight_layout()
    
    # 确保Figure目录存在
    os.makedirs('../Figure', exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figure_file = f'../Figure/training_curves_{timestamp}.png'
    
    plt.savefig(figure_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"训练曲线已保存到: {figure_file}")

if __name__ == "__main__":
    # 设置日志文件路径
    log_files = [
        '../Log/ppo.log',
    ]
    
    # 设置算法名称
    algorithm_names = ['PPO']
    
    # 绘制训练曲线
    plot_training_curves(log_files, algorithm_names) 