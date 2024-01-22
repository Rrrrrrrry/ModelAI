import os

import catboost
import optuna
import joblib  # For model serialization
from catboost import CatBoostClassifier


class CatBoostManager:
    def __init__(self, train_data, test_data, eval_metric='Accuracy'):
        self.train_data = train_data
        self.test_data = test_data
        self.eval_metric = eval_metric
        self.model = None
        self.best_trial_params = None

    def train(self, params):
        # Train the CatBoost model
        self.model = CatBoostClassifier(**params, eval_metric=self.eval_metric)
        self.model.fit(self.train_data, eval_set=self.test_data, verbose=100)

    def predict(self, test_data):
        # Evaluate the model on the test set
        if self.model is not None:
            return self.model.predict(test_data)
        else:
            print("Model not trained yet. Please train the model first.")

    def save_model(self, model_save_path='catboost_model.cbm'):
        # Save the trained CatBoost model
        if self.model is not None:
            self.model.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            print("Model not trained yet. Please train the model first.")

    def load_model(self, model_load_path='catboost_model.cbm'):
        # load the trained CatBoost model
        if self.model is None:
            self.model = CatBoostClassifier()
        if os.path.exists(model_load_path):
            self.model.load_model(model_load_path)
            print(f"Model saved to {model_load_path}")
            return self.model
        else:
            print(f"model load path is not existed. ")

    def objective(self, trial):
        # Optuna objective function for parameter optimization
        params = {
            'iterations': trial.suggest_int('iterations', 100, 3000),
            'depth': trial.suggest_int('depth', 1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
            'random_strength': trial.suggest_int('random_strength', 1, 100),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 50.0),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 200),
        }

        self.train(params)
        score = self.model.score(self.test_data)
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
