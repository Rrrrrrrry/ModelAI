import catboost as cb
import optuna
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
# 加载示例数据集
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义CatBoost分类器的目标函数
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'random_strength': trial.suggest_int('random_strength', 1, 10),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50)
    }

    # 创建CatBoost分类器并训练
    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    # 在验证集上计算准确率作为评估指标
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# 创建optuna优化器并运行参数寻优
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 打印最佳参数和对应的准确率
best_params = study.best_params
best_accuracy = study.best_value
print("Best parameters: ", best_params)
print("Best accuracy: ", best_accuracy)


