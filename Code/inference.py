import pickle
import torch
import json
from tqdm import tqdm
from DecisionTransformer import DecisionTransformer
from BS_EV_Environment_Base import simulate_actions, load_traces, BS_EV_Base

# 配置
trace_file = "../Data/pro_traces_test.pkl"
model_path = "../Models/dt_model_best.pth"
config_path = "config.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取config
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
dt_config = config.get("decision_transformer", {})

# 加载模型
state_dim = 145
action_dim = 3
model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, config=dt_config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 读取traces
traces = load_traces(trace_file)

# 推理每个trace
for idx, trace in enumerate(tqdm(traces, desc="推理进度", total=len(traces))):
    env = BS_EV_Base()
    state = env.reset(trace)
    done = False
    action_seq = []
    max_len = model.max_len if hasattr(model, 'max_len') else dt_config.get("max_len", 30)
    
    # 设定目标RTG
    target_return = 2000.0
    rtgs = [target_return]
    states = [state]
    actions = [0]  # dummy action，实际第一个不会被用到
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
        next_state, reward, done = env.step(action)
        # 更新序列
        states.append(next_state)
        actions.append(action)
        rtgs.append(rtgs[-1] - reward)
        t += 1
    trace["DT_action"] = action_seq

    # 输出动作分布
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in action_seq:
        action_counts[action] += 1
    print(f"动作分布: {action_counts}")


# 保存带有DT_action的traces
with open(trace_file, "wb") as f:
    pickle.dump(traces, f)
print("所有trace动作序列已写入DT_action并保存。")

# 调用simulate_actions计算收益
# simulate_actions("DT", trace_file)