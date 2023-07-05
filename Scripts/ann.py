import numpy as np


class ANN:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=0.01, num_epochs=1000):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = []
        self.biases = []

    def initialize_parameters(self):
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        num_layers = len(dims)

        for i in range(1, num_layers):
            # Initialize weights with random values
            weights = np.random.randn(dims[i-1], dims[i])
            self.weights.append(weights)

            # Initialize biases as zeros
            biases = np.zeros((1, dims[i]))
            self.biases.append(biases)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        activations = [X]
        z_values = []

        num_layers = len(self.weights) + 1

        for i in range(num_layers):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            z_values.append(z)
            activations.append(a)

        return activations, z_values

    def backward_propagation(self, X, y, activations, z_values):
        num_samples = X.shape[0]
        num_layers = len(self.weights) + 1

        deltas = [None] * num_layers
        gradients = [None] * num_layers

        deltas[num_layers - 1] = activations[num_layers] - y

        for i in reversed(range(num_layers - 1)):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * \
                (activations[i+1] * (1 - activations[i+1]))

        for i in range(num_layers - 1):
            gradients[i] = np.dot(activations[i].T, deltas[i+1]) / num_samples

        return gradients

    def update_parameters(self, gradients):
        num_layers = len(self.weights)

        for i in range(num_layers):
            self.weights[i] -= self.learning_rate * gradients[i]

    def fit(self, X, y):
        self.initialize_parameters()

        for epoch in range(self.num_epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)

            # Backward propagation
            gradients = self.backward_propagation(X, y, activations, z_values)

            # Update parameters
            self.update_parameters(gradients)

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        predictions = np.round(activations[-1])
        return predictions.astype(int)
