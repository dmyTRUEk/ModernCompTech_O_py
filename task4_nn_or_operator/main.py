# neural perceptron that predicts `or` operation

from random import uniform as random_float

import numpy as np


EPOCHS: int = 1000

LEARNING_RATE: float = 0.01


def main():
    X: list[np.ndarray] = list(map(np.array, [ (0,0), (0,1), (1,0), (1,1) ]))
    Y: list[int] = [0, 1, 1, 1]
    neuron = Neuron()
    print("Training neuron...", end="")
    neuron.train(X, Y)
    print(" Finished.")

    print(f"Neuron:")
    print(f"  {neuron.weights = }")
    print(f"  {neuron.shift = }")

    while (inp := input("Input input to try neuron: ")) != "":
        try:
            inp_0, inp_1 = map(float, inp.split())
            x = np.array([inp_0, inp_1])
            y = neuron.process_input(x)
            print(f"result: {y}")
        except Exception as e:
            print(f"Error: {e}")


class Neuron:
    weights: np.ndarray
    shift: float

    def __init__(self) -> None:
        self.weights = np.array([random_float(-1, 1), random_float(-1, 1)])
        self.shift = random_float(-1, 1)

    def process_input(self, inputs: np.ndarray) -> float:
        return float(inputs @ self.weights + self.shift)

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



if __name__ == "__main__":
    main()

