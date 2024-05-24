import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0])))
sys.path.append(current_dir)
from utils.obj_op import *
class MLPModel:
    def __init__(self, params=None):
        self.params = params if params else {}
        self.model = None

    def train(self, X_train, y_train):
        self.params = filter_param(self.params, MLPClassifier)
        self.model = MLPClassifier(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)