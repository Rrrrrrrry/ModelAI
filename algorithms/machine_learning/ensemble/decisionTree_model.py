import pydotplus
from pydotplus import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

class DecisionTree:
    def __init__(self):
        self.model = None

    def grid_search(self, X_train, y_train, param_grid, cv=5):
        self.model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv)
        self.model.fit(X_train, y_train)

    def train(self, X_train, y_train, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
        self.model.fit(X_train, y_train)

    def predict(self, X_test, y_test=None):
        if self.model is None:
            print("Please train the model first.")
            return
        y_pred = self.model.predict(X_test)
        acc = None
        if y_test is not None:
            acc = accuracy_score(y_test, y_pred)
            print("Accuracy:", acc)
        return y_pred, acc

    def visualize_tree(self):
        if self.model is None:
            print("Please train the model first.")
            return
        plt.figure(figsize=(20, 10))
        tree.plot_tree(self.model, filled=True)
        plt.show()

    def save_tree_to_pdf(self, filename, **kwargs):
        if self.model is None:
            print("Please train the model first.")
            return
        # dot_data = export_graphviz(self.model, out_file=None, filled=True, feature_names=None, class_names=None)
        dot_data = export_graphviz(self.model, out_file=None, filled=True, **kwargs)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(filename)
        # plt.show()