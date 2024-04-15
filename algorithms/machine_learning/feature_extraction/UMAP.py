import umap


def umap_dimensionality_reduction(X, n_components=2, **kwargs):
    """
    使用 UMAP 进行降维

    参数：
    X : array-like, shape (n_samples, n_features)
        输入的特征矩阵。
    n_components : int, optional (default=2)
        降维后的特征维度。
    **kwargs : dict, optional
        其他可选参数，传递给 UMAP。

    返回值：
    embedding : array-like, shape (n_samples, n_components)
        降维后的特征矩阵。
    """
    reducer = umap.UMAP(n_components=n_components, **kwargs)
    embedding = reducer.fit_transform(X)
    return embedding