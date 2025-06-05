import pickle
import torch
import json
import numpy as np
from tqdm import tqdm
from BS_EV_Environment_Base import load_traces, BS_EV_Base
from BS_EV_Environment_PPO import Agent

# 配置
trace_file = "../Data/pro_traces_train.pkl"
actor_model_path = "../Models/actor_torch_ppo_best"
critic_model_path = "../Models/critic_torch_ppo_best"
config_path = "config.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取config
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
ppo_config = config.get("ppo", {})

# 初始化环境以获取状态和动作维度
env = BS_EV_Base(train_flag=False)
state_dim = env.n_states
action_dim = env.n_actions

# 初始化PPO代理
agent = Agent(
    n_actions=action_dim, 
    input_dims=state_dim,
    gamma=ppo_config.get('gamma', 0.99),
    alpha=ppo_config.get('alpha', 0.0003),
    gae_lambda=ppo_config.get('gae_lambda', 0.95),
    policy_clip=ppo_config.get('policy_clip', 0.2),
    batch_size=ppo_config.get('batch_size', 64),
    n_epochs=ppo_config.get('n_epochs', 10),
    use_lstm=True
)

# 加载训练好的模型
try:
    agent.actor.checkpoint_file_best = actor_model_path
    agent.critic.checkpoint_file_best = critic_model_path
    agent.load_models_best()
    print("成功加载PPO模型")
except Exception as e:
    print(f"加载模型失败: {str(e)}")
    exit(1)

# 设置为评估模式
agent.actor.eval()
agent.critic.eval()

# 读取traces
traces = load_traces(trace_file)

# 推理每个trace
for idx, trace in enumerate(tqdm(traces, desc="PPO推理进度", total=len(traces))):
    env = BS_EV_Base(train_flag=False)
    state = env.reset(trace)
    done = False
    action_seq = []
    t = 0
    
    while not done and t < len(trace["pro_trace"]):
        # 使用PPO模型选择动作
        with torch.no_grad():
            action, _, _ = agent.choose_action(
                state, env.SOC, env.min_SOC, env.SOC_charge_rate, env.SOC_discharge_rate
            )
        
        action_seq.append(action)
        
        # 环境交互
        next_state, reward, done, actual_action = env.step(action)
        state = next_state
        t += 1
    
    # 将动作序列写入trace
    trace["PPO_action"] = action_seq

    # 输出动作分布
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in action_seq:
        action_counts[action] += 1
    print(f"Trace {idx}: 动作分布 - 不动作: {action_counts[0]}, 充电: {action_counts[1]}, 放电: {action_counts[2]}")

# 保存带有PPO_action的traces
with open(trace_file, "wb") as f:
    pickle.dump(traces, f)
print("所有trace动作序列已写入PPO_action并保存。")

# 统计全局动作分布
total_action_counts = {0: 0, 1: 0, 2: 0}
for trace in traces:
    for action in trace["PPO_action"]:
        total_action_counts[action] += 1

total_actions = sum(total_action_counts.values())
action_distribution = {k: v/total_actions for k, v in total_action_counts.items()}
print(f"全局动作分布: 不动作: {action_distribution[0]:.3f}, 充电: {action_distribution[1]:.3f}, 放电: {action_distribution[2]:.3f}")

# 调用simulate_actions计算收益
# from BS_EV_Environment_Base import simulate_actions
# simulate_actions("PPO", trace_file) 