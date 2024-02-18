
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import dbscan
# 假设你的数据集是X，它是一个二维数组，每一行是一个观察
np.random.seed(42)
X = np.random.randn(100, 5)  # 100个时间步，每个时间步有5个特征

# 训练GMM模型
n_components = 3  # GMM组件数量，即聚类的数量
gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
result = gmm.predict_proba(X)

# 提取GMM的参数来初始化HMM
weights = gmm.weights_
means = gmm.means_
covariances = gmm.covariances_

# 初始化HMM  n_components为HMM的簇数
model = hmm.GaussianHMM(n_components=n_components, covariance_type='full')

# 使用GMM的参数来初始化HMM的参数
model.startprob_ = weights
# 发射概率由GMM提供的高斯分布决定
model.means_ = means
model.covars_ = covariances

# 这里假设状态转移概率是均匀分布，也就是说每个状态转移到其他状态的概率是相同的，作为简单的开始。实际上可能需要更精细的方法来设置这个。
model.transmat_ = np.full((n_components, n_components), 1.0 / n_components)
print(f"model.transmat_{model.transmat_}")

# 拟合模型
model.fit(X)
print(model.transmat_)
print(model.means_.shape)
print(model.covars_.shape)
# 预测隐藏状态序列
hidden_states = model.predict(X)

print("Hidden States:", hidden_states)

...
input("s")

import numpy as np
from hmmlearn import hmm


# 创建模拟的多维观测序列
np.random.seed(42)
obs_seq = np.random.randn(100, 5)  # 100个时间步，每个时间步有5个特征

# 创建 GaussianHMM 模型
# model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, startprob_prior=np.array([1, 0, 0]),  transmat_prior=np.array([[0.9, 0.1, 0], [0, 0.9, 0.1], [0.1, 0, 0.9]]))
model = hmm.GMMHMM(n_components=3, covariance_type="full", n_iter=100, startprob_prior=np.array([1, 0, 0]),  transmat_prior=np.array([[0.9, 0.1, 0], [0, 0.9, 0.1], [0.1, 0, 0.9]]))
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
