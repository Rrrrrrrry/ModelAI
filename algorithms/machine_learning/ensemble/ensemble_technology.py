import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score


class EnsembleBase:
    def __init__(self):
        pass
    
    def calculate_accuracy(self, predictions, y):
        accuracy = accuracy_score(y, predictions)
        return accuracy


class CustomStackingClassifier(EnsembleBase):
    def __init__(self, base_classifiers, meta_classifier, cv=5):
        super().__init__()
        """
        指定基分类器和次级分类器，进行stacking集成
        :param base_classifiers:
        :param meta_classifier:
        :param cv:
        """
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.cv = cv
        self.stacking_classifier = StackingClassifier(estimators=self.base_classifiers,
                                                      final_estimator=self.meta_classifier, cv=self.cv)

    def fit(self, X, y):
        self.stacking_classifier.fit(X, y)

    def predict(self, X):
        return self.stacking_classifier.predict(X)

    def predict_proba(self, X):
        return self.stacking_classifier.predict_proba(X)

    # def calculate_accuracy(self, X, y):
    #     predictions = self.predict(X)
    #     accuracy = accuracy_score(y, predictions)
    #     return accuracy


class CustomVotingClassifier(EnsembleBase):
    def __init__(self, estimators, voting='hard'):
        super().__init__()
        self.estimators = estimators
        self.voting = voting
        self.voting_classifier = VotingClassifier(estimators=self.estimators, voting=self.voting)

    def fit(self, X, y):
        self.voting_classifier.fit(X, y)

    def predict(self, X):
        return self.voting_classifier.predict(X)

    def predict_proba(self, X):
        return self.voting_classifier.predict_proba(X)




class CustomBlendingClassifier(EnsembleBase):
    """
    首先在训练集上训练多个基本分类器，然后使用验证集上的预测结果来训练次级分类器。
    """
    def __init__(self, base_classifiers, meta_classifier):
        super().__init__()
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier

    def fit(self, X_train, y_train, X_val, y_val):
        for classifier in self.base_classifiers:
            classifier.fit(X_train, y_train)
        predictions = np.column_stack([classifier.predict(X_val) for classifier in self.base_classifiers])
        self.meta_classifier.fit(predictions, y_val)

    def predict(self, X_test):
        predictions = np.column_stack([classifier.predict(X_test) for classifier in self.base_classifiers])
        return self.meta_classifier.predict(predictions)

    def predict_proba(self, X_test):
        predictions = np.column_stack([classifier.predict(X_test) for classifier in self.base_classifiers])
        return self.meta_classifier.predict_proba(predictions)


