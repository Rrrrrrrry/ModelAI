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
    """
    使用GMM-HMM模型对时间序列数据进行建模，并返回最有可能的状态路径
    :param X:观测序列数据，通常是一个二维数组（n_sample,n_features）
    :param n_components:隐藏状态个数，通常为3
    :return:
    """
    # 初始化GMM-HMM
    model = hmm.GMMHMM(n_components=n_components, covariance_type='full', random_state=42)
    # model.transmat_ = np.full((n_components, n_components), 1.0 / n_components)

    # 模型训练
    model.fit(X)

    # # 给定预测序列X中，每个时间点上的最可能隐藏状态（基于前向-后向算法）
    # hidden_states = model.predict(X)
    #
    # # 给定观测序列X在给定模型下的出现对数似然概率，用于评估模型与观测数据的匹配程度
    # logprob = model.score(X)

    # 解码观测序列X，找到最有可能的状态路径（Viterbi路径）
    logprob, hidden_states = model.decode(X, algorithm="viterbi")

    return logprob, hidden_states



