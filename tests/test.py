

import numpy as np
from hmmlearn import hmm


# 创建模拟的多维观测序列
np.random.seed(42)
obs_seq = np.random.randn(100, 5)  # 100个时间步，每个时间步有5个特征

# 创建 GaussianHMM 模型
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, startprob_prior=np.array([1, 0, 0]),  transmat_prior=np.array([[0.9, 0.1, 0], [0, 0.9, 0.1], [0.1, 0, 0.9]]))

# 拟合模型
model.fit(obs_seq)
print(model.transmat_)
print(model.means_.shape)
print(model.covars_.shape)
# 预测隐藏状态序列
hidden_states = model.predict(obs_seq)

print("Hidden States:", hidden_states)
input('a')


# 生成一些模拟数据
np.random.seed(42)
observations = np.random.randint(0, 10, (100, 5))  # 观察状态

# 初始化MultinomialHMM模型，限制隐藏状态范围
model = hmm.MultinomialHMM(n_components=3, n_iter=200, tol=0.01, startprob_prior=np.array([1, 0, 0]), transmat_prior=np.array([[0.9, 0.1, 0], [0, 0.9, 0.1], [0.1, 0, 0.9]]))
print(observations.shape)

# 训练HMM模型
model.fit(observations)
# 预测隐藏状态序列
hidden_states = model.predict(observations)
print(hidden_states)
input('a')

from hmmlearn import hmm
import numpy as np

# 生成一些模拟数据
np.random.seed(42)
observations = np.random.randint(0, 10, (100, 5))  # 观察状态
# 构建一个简单的HMM模型
model = hmm.MultinomialHMM(n_components=3, n_iter=200, tol=0.01)
print(observations.shape)
# 训练HMM模型
model.fit(observations)

# 预测隐藏状态序列
hidden_states = model.predict(observations)

# 输出隐藏状态和观察状态序列
print("Hidden States:", hidden_states, len(hidden_states))
print("Observations:", observations, len(observations))
