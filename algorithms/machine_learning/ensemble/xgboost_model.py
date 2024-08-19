import os
import sys
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0])))
sys.path.append(current_dir)
from utils.obj_op import *

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params else {}
        self.model = None

    def train(self, X_train, y_train):
        self.params = filter_param(self.params, XGBClassifier)
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_binary = [round(value) for value in y_pred]
        return accuracy_score(y_test, y_pred_binary)

    def get_feature_importance(self, importance_type='weight'):
        importances = self.model.get_booster().get_score(importance_type=importance_type)
        return importances
