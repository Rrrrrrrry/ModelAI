from sklearn.manifold import TSNE

def tsne(X, n_components=2, **kwargs):
    """
    对输入的数据进行 t-SNE 降维处理
    t-SNE 更适用于数据的可视化而不是数据的降维。
    虽然 t-SNE 本身可以用于降维，但它的主要优势在于保持数据点之间的局部结构，
    使得相似的数据点在降维后的空间中保持彼此靠近的关系。这使得 t-SNE 特别适用于可视化高维数据，帮助我们更好地理解数据的结构和关系。
    :param X: 输入的特征矩阵
    :param n_components: 降维后的维度，默认为2
    :param kwargs: t-SNE 参数
    :return: 降维后的特征矩阵
    """
    tsne = TSNE(n_components=n_components, **kwargs)
    X_tsne = tsne.fit_transform(X)
    return X_tsne