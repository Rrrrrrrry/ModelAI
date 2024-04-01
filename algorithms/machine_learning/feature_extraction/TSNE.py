from sklearn.manifold import TSNE

def tsne(X, n_components=2, **kwargs):
    """
    对输入的数据进行 t-SNE 降维处理

    :param X: 输入的特征矩阵
    :param n_components: 降维后的维度，默认为2
    :param kwargs: t-SNE 参数
    :return: 降维后的特征矩阵
    """
    tsne = TSNE(n_components=n_components, **kwargs)
    X_tsne = tsne.fit_transform(X)
    return X_tsne