import pickle

with open("../Data/pro_traces.pkl", "rb") as f:
    pro_traces = pickle.load(f)
    print(pro_traces[0])

