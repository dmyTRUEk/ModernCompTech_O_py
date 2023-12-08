# Neural Network to distinguish digits with multilayer architecture support.

from typing import Callable

from abc import ABC, abstractmethod

from math import isnan, sqrt

import numpy as np

import matplotlib.pyplot as plt


OUTPUT_PATH: str = "./plots/hidden_layers_2/fc_tanh_fc_tanh_fc"
# ARCH = 

LEARNING_RATE: float = 0.1
EPOCHS: int = 10**4


def main():
    for neurons_1 in [2,3,4,5,10,20]:
        for neurons_2 in [2,3,4,5,10,20]:
            print(f"Working on {neurons_1 = }, {neurons_2 = }")
            produce_plot(neurons_1, neurons_2)


POINTS = 10**2
X_TRAIN = np.linspace(0, 2*np.pi, num=POINTS)[np.newaxis].T
Y_TRAIN = np.sin(X_TRAIN)
def produce_plot(neurons_1, neurons_2):
    f = tanh
    f_prime = tanh_prime
    nn = NeuralNetwork(mse, mse_prime, [
        FCLayer(1, neurons_1),
        ActivationLayer(f, f_prime),
        FCLayer(neurons_1, neurons_2),
        ActivationLayer(f, f_prime),
        FCLayer(neurons_2, 1),
        # ActivationLayer(f, f_prime),
    ])

    # train
    res = nn.fit(X_TRAIN, Y_TRAIN, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    if res != 0: return

    # test
    res = nn.predict(X_TRAIN)

    plt.clf()
    plt.plot(X_TRAIN, np.zeros_like(Y_TRAIN), "k--")
    plt.plot(X_TRAIN, Y_TRAIN, "k")
    plt.plot(X_TRAIN, np.array(res).T[0,0], "ro")
    plt.title(f"Arch: Fc1-{neurons_1} Tanh Fc{neurons_1}-{neurons_2} Tanh Fc{neurons_2}-1")
    plt.savefig(f"{OUTPUT_PATH}/{neurons_1}-{neurons_2}.pdf")


abs = lambda x: np.abs(x)
abs_prime = lambda x: x / np.abs(x)

binary_step = lambda x: x > 0
binary_step_prime = lambda x: x == 0

relu = np.vectorize(lambda x: x if x > 0 else 0)
# relu = lambda x: 0.5*(np.sign(x)+1)
relu_prime = np.vectorize(lambda x: 1 if x > 0 else 0)

tanh = lambda x: np.tanh(x)
tanh_prime = lambda x: 1.-np.tanh(x)**2

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))

softplus = lambda x: np.log1p(np.exp(x))
softplus_prime = sigmoid

silu = lambda x: x / (1 + np.exp(-x))
silu_prime = lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x))**2

gaussian = lambda x: np.exp(-x**2)
gaussian_prime = lambda x: -2*x*gaussian(x)

sym_sqrt = lambda x: np.sign(x) * np.sqrt(np.abs(x))
SYM_SQRT_PRIME_MAX_VALUE = 1e0
sym_sqrt_prime = np.vectorize(lambda x: 0.5 * (1 if x>0 else -1) / min(SYM_SQRT_PRIME_MAX_VALUE, sqrt(abs(x))))

sym_ln = lambda x: np.sign(x) * np.log(np.abs(x)+1)
sym_ln_prime = lambda x: np.sign(x) * x / (np.abs(x) + x**2)


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
        self.weights = np.random.uniform(-0.01, 0.01, (input_size, output_size))
        self.bias = np.random.uniform(-0.01, 0.01, (1, output_size))
        # self.weights = np.random.rand(input_size, output_size) - 0.5
        # self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.input @ self.weights + self.bias
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = output_error @ self.weights.T
        weights_error = self.input.T @ output_error
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

    def fit(self, x_train, y_train, epochs: int, learning_rate: float) -> int:
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
            if isnan(err): return 1
            print(f"epoch: {i+1}/{epochs},   {err=}")
        return 0



if __name__ == "__main__":
    main()

