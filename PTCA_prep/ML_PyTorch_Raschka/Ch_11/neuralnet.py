import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def int_to_onehot(y, num_labels):
    array_ = np.zeros(y.shape[0], num_labels)
    for i, val in enumerate(y):
        array_[i, val] = 1

    return array_


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=42) -> None:
        super().__init__()
        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        y_onehot = int_to_onehot(y, self.num_classes)  # n_examples x n_classes

        dL_dA = 2.0 * (a_out - y_onehot) / y.shape[0]  # n_examples x n_classes
        dA_dZ = a_out * (1.0 - a_out)  # n_examples x n_classes
        delta_out = dL_dA * dA_dZ  # n_examples x n_classes

        dZ_dWo = a_h  # n_examples x n_hidden
        dL_dWo = np.dot(delta_out.T, dZ_dWo)  # n_classes x n_examples
        dL_dBo = np.sum(delta_out, axis=0)
