# Neural Network to distinguish digits with multilayer architecture support.

from typing import Callable

from abc import ABC, abstractmethod

import numpy as np



class Layer(ABC):
    input: np.ndarray
    output: np.ndarray

    @abstractmethod
    def forward_propagation(self, input: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        ...


class FCLayer(Layer):
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        self.bias = np.random.uniform(-0.5, 0.5, (1, output_size))
        # self.weights = np.random.rand(input_size, output_size) - 0.5
        # self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.input @ self.weights + self.bias
        # self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = output_error @ self.weights.T
        # input_error = np.dot(output_error, self.weights.T)
        weights_error = self.input.T @ output_error
        # weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


ActivationFunction = Callable[[np.ndarray], np.ndarray]

class ActivationLayer(Layer):
    activation: ActivationFunction
    activation_prime: ActivationFunction

    def __init__(self, activation: ActivationFunction, activation_prime: ActivationFunction):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return self.activation_prime(self.input) * output_error



def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.power(y_true-y_pred, 2)))

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2*(y_pred - y_true) / y_true.size


class NeuralNetwork:
    layers: list[Layer]
    loss: Callable[[np.ndarray, np.ndarray], float]
    loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(self, loss, loss_prime, layers: list[Layer]):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data: np.ndarray) -> list[np.ndarray]:
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs: int, learning_rate: float):
        samples = len(x_train)
        for i in range(epochs):
            err: float = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print(f"epoch: {i+1}/{epochs},   {err=}")


def main():
    POINTS = 1_000

    # training data
    x_train = np.linspace(0, 2*np.pi, num=POINTS)[np.newaxis].T
    # print(x_train)
    y_train = np.sin(x_train)

    # network
    nn = NeuralNetwork(mse, mse_prime, [
        FCLayer(1, 10),
        ActivationLayer(np.tanh, lambda x: 1.-np.tanh(x)**2),
        FCLayer(10, 1),
        ActivationLayer(np.tanh, lambda x: 1.-np.tanh(x)**2),
    ])

    # train
    nn.fit(x_train, y_train, epochs=10**3, learning_rate=0.1)

    # test
    print()
    res = nn.predict(x_train)
    # print(f"{res = }")
    print()
    for i in range(0, 20):
        expected = y_train[i, 0]
        observed = res[i][0,0]
        relative_error = abs(expected - observed) / expected
        print("x:", x_train[i,0], "->", "expected:", expected, "->", "observed:", observed, "->", "rel_err:", relative_error)
    print()
    for i in range(POINTS):
        print(x_train[i, 0], res[i][0,0], sep="\t")
    print()


if __name__ == "__main__":
    main()

