from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut


class RandomForest:
    """
    Random Forest Classifier/Regressor
    """

    def __init__(self, model_type):
        """
        Constructor for initialize the Random Forest model

        Parameters:
        -----------
        model_type: The model type,can be 'classification' or 'regression'
        """
        self.model_type = model_type
        self.model = None
        self.best_params_ = None

    def train(self, X, y, param_grid=None, cv=None):
        """
        Train the Random Forest

        Parameters:
        ----------
        X(array-like): The feature dataset
        y(array-like): The target dataset
        param_grid(dict): The parameter to grid search, defaults to None
            can contain the following parameters:
            'n_estimators': The number of decision trees,
            'criterion': The criterion used to measure the quality of a split,
                        for classification problems, commonly used criteria are "gini" or "entropy",
                        for regression problems, commonly used "mse" or "mae",
            'max_depth': The maximum depth of the decision trees,
            'min_samples_split': The minimum number of samples required to split an internal node,
            'min_samples_leaf': The minimum number of samples required to be at a leaf node,
            'max_features': The number of features to consider when looking for the best split,
            'bootstrap':Whether to use bootstrap sampling with replacement to build the decision trees
        Example:
        param_grid = {
                        'n_estimators': [50, 100, 200],
                        'criterion': ['gini', 'entropy', 'mse', 'mae'],
                        'max_depth': [5, 10],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': [None],
                        'bootstrap':[True, False]
                    }

        cv(int):the number of cross-validation folds, defaults to None

        Returns
        self : object
            Returns the instance itself.
        -------
        """
        if self.model_type == 'classification':
            self.model = RandomForestClassifier()
        elif self.model_type == 'regression':
            self.model = RandomForestRegressor()
        else:
            print('Invalid model type.Supported model types are: classification, regression')
        if param_grid is None:
            self.model.fit(X, y)
            return
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, y)
        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def predict(self, X):
        """

        :param X:
        :return:
        """

        if self.model is None:
            raise ValueError('Model is not been trained yet. Please call the train method first.')
        return self.model.predict(X)
