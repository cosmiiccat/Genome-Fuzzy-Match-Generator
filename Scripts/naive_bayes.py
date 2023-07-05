import numpy as np


class NaiveBayes:
    def __init__(self):
        self.class_prior = {}        # Dictionary to store class prior probabilities
        # Dictionary to store class-conditional likelihood probabilities
        self.class_likelihood = {}
        self.classes = []            # List to store unique class labels

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Args:
            X (numpy.ndarray): The training feature vectors.
            y (numpy.ndarray): The training class labels.
        """
        # Compute class prior probabilities
        self.classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for class_label, count in zip(self.classes, class_counts):
            self.class_prior[class_label] = count / total_samples

        # Compute class-conditional likelihood
        num_features = X.shape[1]
        for class_label in self.classes:
            class_samples = X[y == class_label]
            feature_likelihood = {}
            for feature_index in range(num_features):
                feature_values, feature_counts = np.unique(
                    class_samples[:, feature_index], return_counts=True)
                feature_likelihood[feature_index] = dict(
                    zip(feature_values, feature_counts / count))
            self.class_likelihood[class_label] = feature_likelihood

    def predict(self, X):
        """
        Predict the class labels for the given feature vectors.

        Args:
            X (numpy.ndarray): The feature vectors to predict.

        Returns:
            list: The predicted class labels.
        """
        predictions = []
        for sample in X:
            posteriors = []
            for class_label in self.classes:
                prior = self.class_prior[class_label]
                likelihood = 1.0
                for feature_index, feature_value in enumerate(sample):
                    likelihood *= self.class_likelihood[class_label][feature_index].get(
                        feature_value, 0.0)
                posterior = prior * likelihood
                posteriors.append(posterior)
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        return predictions
