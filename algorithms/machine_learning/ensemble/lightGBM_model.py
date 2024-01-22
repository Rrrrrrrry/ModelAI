import os

import catboost
import optuna
import joblib  # For model serialization
from lightgbm import LGBMClassifier

class LightGBM_Manager:
    def __init__(self, X_train, y_train, X_val, y_val, eval_metric='Accuracy'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.eval_metric = eval_metric
        self.model = None
        self.best_trial_params = None

    def train(self, params):
        # Train the CatBoost model
        self.model = LGBMClassifier(**params, eval_metric=self.eval_metric)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data):
        # Evaluate the model on the test set
        if self.model is not None:
            return self.model.predict(test_data)
        else:
            print("Model not trained yet. Please train the model first.")




    def objective(self, trial):
        # Optuna objective function for parameter optimization
        params = {
            'objective': 'multiclass',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 100),
            'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
            'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 100.0),
        }
        # params = {
        #     'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
        #     'max_depth': trial.suggest_int('max_depth', 1, 100),
        #     'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
        #     'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        # }

        self.train(params)
        score = self.model.score(self.X_val, self.y_val)
        return 1.0 - score  # Optuna minimizes the objective function, so we return 1 - score

    def optimize_parameters(self, n_trials=5):
        # Optimize CatBoost parameters using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        print("Best trial:")
        print(study.best_trial.params)
        print(f"Best {self.eval_metric}: {1.0 - study.best_value}")
        # print(study.trials_dataframe())

        # Save the best model
        self.best_trial_params = study.best_trial.params
        self.train(self.best_trial_params)
