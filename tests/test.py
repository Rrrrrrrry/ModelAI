import optuna
import os

class OptunaOptimizer:
    def __init__(self, model_class, train_data, test_data, eval_metric='Accuracy'):
        self.model_class = model_class
        self.train_data = train_data
        self.test_data = test_data
        self.eval_metric = eval_metric
        self.best_trial_params = None

    def objective(self, trial):
        # Optuna优化参数的目标函数
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 1e2),
        }

        model = self.model_class(**params, eval_metric=self.eval_metric)
        model.fit(self.train_data, eval_set=self.test_data, verbose=100)
        score = model.score(self.test_data)
        return 1.0 - score  # Optuna最小化目标函数，因此返回1 - score

    def optimize_parameters(self, n_trials=5):
        # 使用Optuna优化模型参数
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        print("Best trial:")
        print(study.best_trial.params)
        print(f"Best {self.eval_metric}: {1.0 - study.best_value}")

        # 保存最佳模型
        self.best_trial_params = study.best_trial.params
        best_model = self.model_class(**self.best_trial_params, eval_metric=self.eval_metric)
        best_model.fit(self.train_data, eval_set=self.test_data, verbose=100)

        return best_model
