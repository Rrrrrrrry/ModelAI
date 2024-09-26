import os
import sys

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0])))
sys.path.append(current_dir)
from utils.obj_op import *

class linearModel:
    def __init__(self, params=None, choose_model_name='Ridge'):
        self.choose_model_dict = {'Ridge': Ridge, 'Lasso': Lasso,
                                  'ElasticNet': ElasticNet, 'LinearRegression':LinearRegression}
        if choose_model_name not in self.choose_model_dict:
            raise KeyError(f"{choose_model_name} not in [Ridge, Lasso, ElasticNet, LinearRegression]")
        self.choose_model_name = self.choose_model_dict[choose_model_name]
        self.params = params if params else {}
        self.model = None

    def train(self, X_train, y_train):
        self.params = filter_param(self.params, self.choose_model_name)
        self.model = self.choose_model_name(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)