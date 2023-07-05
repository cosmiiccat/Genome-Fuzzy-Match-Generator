import numpy as np


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):
        """
        Fit the SVM model to the training data.

        Args:
            X_train (numpy.ndarray): The training feature vectors.
            y_train (numpy.ndarray): The training class labels.
        """
        num_samples, num_features = X_train.shape

        # Initialize parameters
        self.w = np.zeros(num_features)
        self.b = 0

        # Training the SVM model
        for _ in range(self.num_iterations):
            for i in range(num_samples):
                # Check if the sample is classified correctly
                if y_train[i] * (np.dot(self.w, X_train[i]) - self.b) >= 1:
                    # Correctly classified, update only the weight vector
                    self.w -= self.learning_rate * \
                        (2 * self.lambda_param * self.w)
                else:
                    # Misclassified, update both the weight vector and the bias term
                    self.w -= self.learning_rate * \
                        (2 * self.lambda_param * self.w -
                         np.dot(X_train[i], y_train[i]))
                    self.b -= self.learning_rate * y_train[i]

    def predict(self, X_test):
        """
        Predict the class labels for the given feature vectors.

        Args:
            X_test (numpy.ndarray): The feature vectors to predict.

        Returns:
            numpy.ndarray: The predicted class labels.
        """
        y_pred = np.sign(np.dot(X_test, self.w) - self.b)
        return y_pred.astype(int)
