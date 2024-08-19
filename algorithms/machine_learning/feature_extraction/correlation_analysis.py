import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    corr_matrix = df.corr(method=method)
    return corr_matrix


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


def select_variables_by_threshold(df, method='pearson', target_column=None, threshold=0.5):
    """
    设置一个相关系数的阈值，例如只选择绝对值大于0.5的相关变量。这可以帮助排除弱相关或不相关的变量。
    :param df:
    :param method:
    :param target_column:
    :param threshold:
    :return:
    """
    corr_matrix = calculate_correlation(df, method=method)
    if target_column:
        corr_matrix = corr_matrix[target_column].abs().sort_values(ascending=False)
        selected_vars = corr_matrix[corr_matrix > threshold].index.tolist()
    else:
        selected_vars = corr_matrix[(corr_matrix > threshold) | (corr_matrix < -threshold)].dropna(how='all').dropna(axis=1, how='all').columns.tolist()
    return selected_vars


def select_variables_by_vif(df, target_column, threshold=5):
    """
    在多元回归分析中，如果多个自变量之间高度相关，可能会导致多重共线性问题。
    可以通过方差膨胀因子（VIF）来检测和处理多重共线性，选择VIF较低的变量。
    :param df:
    :param target_column:
    :param threshold:
    :return:
    """
    X = df.drop(target_column, axis=1)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    selected_vars = vif_data[vif_data['VIF'] < threshold]['feature'].tolist()
    return selected_vars

def select_variables_by_stepwise(df, target_column, select_feature_num=5):
    """
    使用逐步回归技术，自动添加或删除变量基于其对模型的贡献，如AIC（赤池信息准则）、BIC（贝叶斯信息准则）或p值。
    :param df:
    :param target_column:
    :return:
    """
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=select_feature_num)
    rfe = rfe.fit(df.drop(target_column, axis=1), df[target_column])
    print(f"rfe{rfe}")
    selected_vars = df.drop(target_column, axis=1).columns[rfe.support_].tolist()
    return selected_vars

def select_variables_by_importance(df, target_column, n=5):
    """
    在决策树或随机森林等机器学习模型中，可以利用特征重要性指标来识别对模型预测能力贡献最大的变量。
    :param df:
    :param target_column:
    :param n:
    :return:
    """
    """使用随机森林特征重要性选择变量"""
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(df.drop(target_column, axis=1), df[target_column])
    importances = rf.feature_importances_
    print(f"importances{importances}")
    feature_importances = pd.DataFrame({'Feature': df.drop(target_column, axis=1).columns, 'Importance': importances})
    selected_vars = feature_importances.nlargest(n, 'Importance')['Feature'].tolist()
    return selected_vars

def select_variables_by_lasso(df, target_column, cv=5):
    """使用LASSO回归选择变量"""
    lasso = LassoCV(cv=cv)
    lasso.fit(df.drop(target_column, axis=1), df[target_column])
    print(f"lasso.coef_{lasso.coef_}")
    selected_vars = df.drop(target_column, axis=1).columns[lasso.coef_ != 0].tolist()
    return selected_vars



if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_digits, load_iris

    # classes = np.arange(5)
    # feature_num = 8
    # X, y = make_classification(n_samples=100, n_features=feature_num, n_classes=10,  n_informative=4, n_clusters_per_class=1)

    digits = load_iris()
    X, y = digits.data, digits.target

    # data = pd.DataFrame(X, columns=list(map(lambda x:f"feature{x}", range(X.shape[1]))))
    data = pd.DataFrame(X, columns=digits.feature_names)
    data['label'] = y

    df = pd.DataFrame(data)
    features = df.drop('label', axis=1)
    labels = df['label']

    # 计算并显示相关系数
    pearson_corr = calculate_correlation(df, method='pearson')
    spearman_corr = calculate_correlation(df, method='spearman')
    kendall_corr = calculate_correlation(df, method='kendall')

    # 特征选择
    # select_var = select_variables_by_threshold(df, method='pearson', target_column=None, threshold=0.5)
    # select_var = select_variables_by_vif(df, target_column='label', threshold=5)
    # select_var = select_variables_by_stepwise(df, 'label')
    # select_var = select_variables_by_importance(df, 'label', n=5)
    # select_var = select_variables_by_lasso(df, target_column='label')
    # print(select_var)



    # plot_heatmap(pearson_corr, 'Pearson Correlation Matrix Heatmap')
    # plot_heatmap(spearman_corr, 'Spearman Correlation Matrix Heatmap')
    # plot_heatmap(kendall_corr, 'Kendall Correlation Matrix Heatmap')

    # PCA分析并显示结果
    pca_result = perform_pca(features)
    print(f"labels{labels}")
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
