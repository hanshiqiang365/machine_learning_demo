#author:hanshiqiang365

from hmmlearn import hmm
import numpy as np

# 定义模型参数
states = ["Sunny", "Rainy"]
observations = ["Short-sleeved", "Long-sleeved"]

initial_probabilities = np.array([0.7, 0.3])

transition_probabilities = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

emission_probabilities = np.array([
    [0.6, 0.4],
    [0.3, 0.7]
])

# 创建并训练HMM模型
model = hmm.MultinomialHMM(n_components=2, n_trials=1)  # 设置n_trials为1
model.startprob_ = initial_probabilities
model.transmat_ = transition_probabilities
model.emissionprob_ = emission_probabilities

# 假设我们观察到的连续三天的衣服是: ["Short-sleeved", "Long-sleeved", "Long-sleeved"]
# 将其转化为多项分布的计数
obs_seq = np.array([
    [1, 0],  # Short-sleeved
    [0, 1],  # Long-sleeved
    [0, 1]   # Long-sleeved
])

# 使用Viterbi算法找到最可能的天气序列
logprob, state_seq = model.decode(obs_seq)

print("Observation sequence:", [observations[np.argmax(obs)] for obs in obs_seq])
print("Most likely weather sequence:", [states[i] for i in state_seq])

