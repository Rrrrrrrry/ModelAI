import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlation(df, method='pearson'):
    """
    计算相关性
    :param df:pd数据
    :param method:方法
    'pearson':皮尔逊相关系数
    'spearman':斯皮尔曼等级相关系数
    'kendall':肯德尔相关系数
    :return:
    """
    return df.corr(method=method)


def calculate_specific_correlation(feature1, feature2, method='spearman'):
    """"""
    if method == 'spearman':
        return spearmanr(feature1, feature2)
    elif method == 'kendall':
        return kendalltau(feature1, feature2)
    else:
        raise ValueError("Unsupported method. Use 'spearman' or 'kendall'.")

def plot_heatmap(corr_matrix, title='Correlation Matrix Heatmap'):
    """绘制相关性矩阵的热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()

def perform_pca(features, n_components=2):
    """
    主成分分析降维
    :param features:
    :param n_components:
    :return:
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    return pca_result

def plot_pca_result(pca_result, labels):
    """
    可视化主成分分析结果
    :param pca_result:
    :param labels:
    :return:
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.colorbar()
    plt.show()

def perform_factor_analysis(features, n_components=2):
    """执行因子分析（Factor Analysis）"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    fa = FactorAnalysis(n_components=n_components)
    fa_result = fa.fit_transform(features_scaled)
    return fa_result

def plot_factor_analysis_result(fa_result, labels):
    """可视化因子分析结果"""
    plt.figure(figsize=(10, 8))
    plt.scatter(fa_result[:, 0], fa_result[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('Factor Analysis Result')
    plt.colorbar()
    plt.show()

def perform_lda(features, labels, n_components=2):
    """执行线性判别分析（LDA）"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    lda = LDA(n_components=n_components)
    lda_result = lda.fit_transform(features_scaled, labels)
    return lda_result

def plot_lda_result(lda_result, labels):
    """可视化LDA结果"""
    plt.figure(figsize=(10, 8))
    plt.scatter(lda_result[:, 0], lda_result[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.title('LDA Result')
    plt.colorbar()
    plt.show()

def calculate_mutual_information(features, labels):
    """计算互信息"""
    mi = mutual_info_regression(features, labels)
    mi_series = pd.Series(mi, index=features.columns)
    return mi_series.sort_values(ascending=False)

# 示例代码，可以根据需要修改
if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_digits, load_iris

    # classes = np.arange(5)
    # feature_num = 8
    # X, y = make_classification(n_samples=100, n_features=feature_num, n_classes=10,  n_informative=4, n_clusters_per_class=1)

    digits = load_iris()
    X, y = digits.data, digits.target

    data = pd.DataFrame(X, columns=list(map(lambda x:f"feature{x}", range(X.shape[1]))))
    data['label'] = y

    df = pd.DataFrame(data)
    features = df.drop('label', axis=1)
    labels = df['label']

    # 计算并显示相关系数
    pearson_corr = calculate_correlation(df, method='pearson')
    spearman_corr = calculate_correlation(df, method='spearman')
    kendall_corr = calculate_correlation(df, method='kendall')
    # plot_heatmap(pearson_corr, 'Pearson Correlation Matrix Heatmap')
    # plot_heatmap(spearman_corr, 'Spearman Correlation Matrix Heatmap')
    # plot_heatmap(kendall_corr, 'Kendall Correlation Matrix Heatmap')

    # PCA分析并显示结果
    pca_result = perform_pca(features)
    plot_pca_result(pca_result, labels)

    # 因子分析并显示结果
    fa_result = perform_factor_analysis(features)
    plot_factor_analysis_result(fa_result, labels)

    # 线性判别分析并显示结果
    lda_result = perform_lda(features, labels)
    plot_lda_result(lda_result, labels)

    # 计算并显示互信息
    mi_series = calculate_mutual_information(features, labels)
    print(f"互信息：{mi_series}")
