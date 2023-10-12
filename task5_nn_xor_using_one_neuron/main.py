# neural perceptron that calculates XOR using ONLY ONE neuron

from random import uniform as random_float
from math import exp

import numpy as np


EPOCHS: int = 100_000

LEARNING_RATE: float = 0.01


def main():
    X: list[np.ndarray] = list(map(np.array, [ (0,0), (0,1), (1,0), (1,1) ]))
    Y: list[int] = [0, 1, 1, 0]
    neuron = Neuron(2)
    # nn = NeuralNetwork()
    print("Training neuron...", end="")
    neuron.train(X, Y)
    print(" Finished.")

    print(f"Neural Network:")
    print(f"  weights: {neuron.weights}")
    print(f"  shift: {neuron.shift}")
    # print(f"  Neuron 0:")
    # print(f"    weights: {neuron.neurons[0].weights}")
    # print(f"    shift: {neuron.neurons[0].shift}")
    # print(f"  Neuron 1:")
    # print(f"    weights: {neuron.neurons[1].weights}")
    # print(f"    shift: {neuron.neurons[1].shift}")

    while (inp := input("Input input to try neuron: ")) != "":
        try:
            inp = list(map(float, inp.split()))
            x = np.array(inp)
            y = neuron.process_input(x)
            print(f"result: {y}")
        except Exception as e:
            print(f"Error: {e}")



class Neuron:
    weights: np.ndarray
    shift: float

    def __init__(self, n: int) -> None:
        self.weights = np.array([random_float(-1, 1) for _ in range(n)])
        self.shift = random_float(-1, 1)

    def activation_function(self, x: float) -> float:
        # return x
        return 1 if abs(x) > 0.5 else 0
        # return 1 / (1 + exp(-x))

    def process_input(self, input: np.ndarray) -> float:
        return self.activation_function(float(input @ self.weights + self.shift))

    def train(self, x: list[np.ndarray], y: list[int]):
        assert len(x) == len(y)
        for epoch in range(EPOCHS):
            for i in range(len(x)):
                xi = x[i]
                y_expected = y[i]
                y_actual = self.process_input(xi)
                y_error = y_expected - y_actual
                self.weights += LEARNING_RATE * y_error * xi
                self.shift += LEARNING_RATE * y_error



class NeuralNetwork:
    neurons: tuple[Neuron, Neuron]

    def __init__(self) -> None:
        self.neurons = (Neuron(2), Neuron(2))

    def process_input(self, input: np.ndarray) -> float:
        neuron_0_result = self.neurons[0].process_input(input)
        neuron_1_input = np.array((neuron_0_result, input[1]))
        return self.neurons[1].process_input(neuron_1_input)

    def train(self, x: list[np.ndarray], y: list[int]):
        assert len(x) == len(y)
        for epoch in range(EPOCHS):
            for i in range(len(x)):
                xi = x[i]
                y_expected = y[i]
                y_actual = self.process_input(xi)
                y_error = y_expected - y_actual
                raise NotImplemented()



if __name__ == "__main__":
    main()

