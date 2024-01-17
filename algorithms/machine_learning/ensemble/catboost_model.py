import catboost
import optuna
import joblib  # For model serialization
from catboost import CatBoostClassifier

class CatBoostManager:
    def __init__(self, train_data, test_data, eval_metric='Accuracy', model_save_path='catboost_model.cbm'):
        self.train_data = train_data
        self.test_data = test_data
        self.eval_metric = eval_metric
        self.model = None
        self.model_save_path = model_save_path
        self.best_trial_params = None
        print(model_save_path)

    def train(self, params):
        # Train the CatBoost model
        self.model = CatBoostClassifier(**params, eval_metric=self.eval_metric)
        self.model.fit(self.train_data, eval_set=self.test_data, verbose=100)

    def test(self):
        # Evaluate the model on the test set
        if self.model is not None:
            result = self.model.score(self.test_data)
            print(f"Test {self.eval_metric}: {result}")
        else:
            print("Model not trained yet. Please train the model first.")

    def save_model(self):
        # Save the trained CatBoost model
        if self.model is not None:
            self.model.save_model(self.model_save_path)
            print(f"Model saved to {self.model_save_path}")
        else:
            print("Model not trained yet. Please train the model first.")

    def objective(self, trial):
        # Optuna objective function for parameter optimization
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 1e2),
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

        # Save the best model
        self.best_trial_params = study.best_trial.params
        self.train(self.best_trial_params)
        self.save_model()


