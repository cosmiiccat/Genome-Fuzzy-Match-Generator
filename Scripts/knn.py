import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the KNN classifier to the training data.

        Args:
            X_train (numpy.ndarray): The training feature vectors.
            y_train (numpy.ndarray): The training class labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two points.

        Args:
            x1 (numpy.ndarray): The first point.
            x2 (numpy.ndarray): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """
        Predict the class labels for the given feature vectors.

        Args:
            X_test (numpy.ndarray): The feature vectors to predict.

        Returns:
            list: The predicted class labels.
        """
        y_pred = []
        for sample in X_test:
            distances = []
            for x_train, y_train in zip(self.X_train, self.y_train):
                distance = self.euclidean_distance(sample, x_train)
                distances.append((distance, y_train))
            distances.sort(key=lambda x: x[0])  # Sort by distances
            k_nearest = distances[:self.k]
            k_nearest_labels = [label for _, label in k_nearest]
            majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
            y_pred.append(majority_vote)
        return y_pred
