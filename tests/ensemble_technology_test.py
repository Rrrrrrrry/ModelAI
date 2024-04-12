# Example usage:
import os
import sys

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

currentdir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(currentdir)
from algorithms.machine_learning.ensemble.ensemble_technology import *


if __name__ == '__main__':
    # Stacking/Voting
    use_ensemble_t = 'Voting'
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 划分数据集为训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_blending_classifier = CustomBlendingClassifier(base_classifiers=[RandomForestClassifier(), SVC()], meta_classifier=LogisticRegression())
    custom_blending_classifier.fit(X_train, y_train, X_test, y_test)
    predictions = custom_blending_classifier.predict(X_test)

    if use_ensemble_t == 'Voting':
        """
        使用Voting集成
        """
        custom_voting_classifier = CustomVotingClassifier(estimators=[('rf', RandomForestClassifier()), ('svm', SVC()), ('knn', KNeighborsClassifier())])
        custom_voting_classifier.fit(X_train, y_train)
        predictions = custom_voting_classifier.predict(X_test)
        acc = custom_voting_classifier.calculate_accuracy(predictions, y_test)
        print(f"predictions:{predictions}")
        print(f"acc:{acc}")
    if use_ensemble_t == 'Stacking':
        """
        使用Stacking集成
        """
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
        custom_stacking_classifier.fit(X_train, y_train)

        # Make predictions
        predictions = custom_stacking_classifier.predict(X_test)
        print("Predictions:", predictions)

        # cal acc
        acc = custom_stacking_classifier.calculate_accuracy(predictions, y_test)
        print("Accuracy:", acc)