from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class RandomForest:
    """
    Random Forest Classifier/Regressor
    """
    def __init__(self, model_type):
        """
        Constructor for intializing the Random Forest model

        Parameters:
        -----------
        model_type: The model type,can be 'classification' or 'regression'
        """
        self.model_type = model_type
        self.model = None
        self.model_params = None

    def train(self, X, y, param_grid=None, cv=None):
        """
        Train the Random Forest

        Parameters:
        ----------
        X(array-like): The feature dataset
        y(array-like): The target dataset
        param_grid(dict): The parameter to grid search, defaults to None
        cv(int):the number of cross-validation folds, defaults to 5


        Returns
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


if __name__ == '__main__':
    model = RandomForest(model_type='classification')

    X = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y = [0, 1, 2, 3]

    param_grid = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy', 'mse', 'mae'],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None],
        'bootstrap':[True, False]
    }

    model.train(X, y, param_grid=param_grid, cv=2)

    predictions = model.predict([[4, 4], [5, 5]])

    print(predictions)