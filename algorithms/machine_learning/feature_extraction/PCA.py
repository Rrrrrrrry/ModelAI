
from sklearn.decomposition import PCA

def pca(X, n_components=None):
    """
    对输入的数据进行 PCA 降维处理

    :param X: 输入的特征矩阵
    :param n_components: 降维后的维度
    :return: 降维后的特征矩阵
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca



