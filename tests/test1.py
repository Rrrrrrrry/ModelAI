from sklearn.ensemble import StackingClassifier

class CustomStackingClassifier:
    def __init__(self, base_classifiers, meta_classifier, cv=5):
        """
        指定基分类器和次级分类器，进行stacking集成
        :param base_classifiers:
        :param meta_classifier:
        :param cv:
        """
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.cv = cv
        self.stacking_classifier = StackingClassifier(estimators=self.base_classifiers, final_estimator=self.meta_classifier, cv=self.cv)

    def fit(self, X, y):
        self.stacking_classifier.fit(X, y)

    def predict(self, X):
        return self.stacking_classifier.predict(X)

    def predict_proba(self, X):
        return self.stacking_classifier.predict_proba(X)


if __name__ == '__main__':
    # Example usage:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Define base classifiers
    base_classifiers = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('knn', KNeighborsClassifier())
    ]

    # Define meta classifier
    meta_classifier = LogisticRegression()

    # Create CustomStackingClassifier instance
    custom_stacking_classifier = CustomStackingClassifier(base_classifiers, meta_classifier)

    # Train CustomStackingClassifier
    custom_stacking_classifier.fit(X, y)

    # Make predictions
    predictions = custom_stacking_classifier.predict(X)
    print("Predictions:", predictions)
