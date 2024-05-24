from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def initialize_hmm_with_gmm(X, n_components=3):
    """
    利用gmm的结果初始化hmm
    :param X: 二维数据（样本个数，特征维度）
    :param n_components: 聚类数量
    :return:
    """
    # 训练GMM模型
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(X)
    # result = gmm.predict_proba(X)
    # 提取GMM的参数来初始化HMM
    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    # 初始化HMM  n_components为HMM的簇数
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='full')
    # # 使用GMM的参数来初始化HMM的参数
    model.startprob_ = weights
    # 发射概率由GMM提供的高斯分布决定
    model.means_ = means
    model.covars_ = covariances
    # 这里假设状态转移概率是均匀分布，也就是说每个状态转移到其他状态的概率是相同的，作为简单的开始。实际上可能需要更精细的方法来设置这个。
    model.transmat_ = np.full((n_components, n_components), 1.0 / n_components)
    # 拟合模型
    model.fit(X)
    # 预测隐藏状态序列
    hidden_states = model.predict(X)
    return hidden_states


def gmm_hmm(X, n_components=3):
    model = hmm.GMMHMM(n_components=n_components, covariance_type='full', random_state=42)
    # model.transmat_ = np.full((n_components, n_components), 1.0 / n_components)
    model.fit(X)
    hidden_states = model.predict(X)
    return hidden_states



