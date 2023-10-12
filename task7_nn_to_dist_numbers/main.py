# Neural Network that "sees" numbers.

from random import uniform as random_float
from math import exp

import numpy as np

from pipe import Pipe, map as map_


EPOCHS: int = 100

LEARNING_RATE: float = 0.01


def main():
    x_train, y_train, x_test, y_test = load_datasets(["./Data_train.csv", "./Data_test.csv"])

    nn = NeuralNetwork(64, 10)
    print("Training Neural Network...")
    train_nn(nn, x_train, y_train)
    print("Training finished.")
    print()
    print(nn)
    print()
    print("NeuralNetwork test:")
    test_nn(nn, x_test, y_test)
    print()

    while (inp := input("Input input to try Neural Network: ")) != "":
        try:
            inp = inp.split(',') | map_(float) | list_
            x = np.array(inp)
            y_raw = nn.process_input_raw(x)
            y = nn.process_raw_output(y_raw)
            print(f"result: {y_raw} => {y}")
        except Exception as e:
            print(f"Error: {e}")


def train_nn(nn: "NeuralNetwork", x: list[np.ndarray], y: list[int]):
    for neuron_number, neuron in enumerate(nn.neurons):
        neuron.train(x, y, neuron_number)


def test_nn(nn: "NeuralNetwork", x: list[np.ndarray], y: list[int]):
    confusion_matrix = np.zeros((10, 10), dtype=np.int32)
    for i in range(len(x)):
        y_prediction = nn.process_input(x[i])
        y_expected = y[i]
        confusion_matrix[y_expected, y_prediction] += 1
    print("Confusion matrix:")
    print(confusion_matrix)


def activation_function(x: float) -> float:
    # return x
    # return 1 if abs(x) > 0.5 else 0
    # return 1 if x > 0.5 else 0
    # return 1 if x > 0 else 0
    # return x if x > 0 else 0
    return 1 / (1 + exp(-x))


class Neuron:
    weights: np.ndarray
    shift: float

    def __init__(self, n: int) -> None:
        self.weights = np.array([random_float(-1, 1) for _ in range(n)])
        self.shift = random_float(-1, 1)

    def __repr__(self) -> str:
        return [
            f"- Neuron:",
            f"  - weights: {self.weights}",
            f"  - shift: {self.shift}",
        ] | join_

    def process_input(self, input: np.ndarray) -> float:
        return activation_function(float(input @ self.weights + self.shift))

    def train(self, x: list[np.ndarray], y: list[int], neuron_number: int):
        assert len(x) == len(y)
        for epoch in range(EPOCHS):
            for train_i in range(len(x)):
                # print(self)
                xi = x[train_i]
                y_expected = (y[train_i] | one_hot_)[neuron_number]
                y_actual = self.process_input(xi)
                y_error = y_expected - y_actual
                self.weights += LEARNING_RATE * y_error * xi
                self.shift += LEARNING_RATE * y_error



class NeuralNetwork:
    input_size: int
    neurons: list[Neuron]

    def __init__(self, input_size: int, neurons_number: int) -> None:
        self.input_size = input_size
        self.neurons = [Neuron(input_size) for _ in range(neurons_number)]

    def __repr__(self) -> str:
        s: str = f"Neural Network Neurons (input_size = {self.input_size}):\n"
        for i in range(len(self.neurons)):
            s += f"- Neuron {i=}:\n"
            neuron = self.neurons[i]
            s += f"  - weights: {neuron.weights}\n"
            s += f"  - shift: {neuron.shift}\n"
        s = s[:-1]
        return s

    def process_input_raw(self, input: np.ndarray) -> list[float]:
        return [neuron.process_input(input) for neuron in self.neurons]

    def process_raw_output(self, raw_output: list[float]) -> int:
        return raw_output | index_of_max_

    def process_input(self, input: np.ndarray) -> int:
        return self.process_raw_output(self.process_input_raw(input))


def load_datasets(filenames: list[str]) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    xy_all: list[tuple[np.ndarray, int]] = []
    for filename in filenames:
        with open(filename) as file:
            for line in file.readlines():
                y, x = line.strip().split(',') | split_at_(1)
                y = int(y[0])
                x = np.array(x | map_(float) | list_)
                xy_all.append((x, y))
    xy_train, xy_test = xy_all | split_at_percentage_(0.7)
    x_train, y_train = xy_train | unzip_
    x_test, y_test = xy_test | unzip_

    return x_train, y_train, x_test, y_test



# pipes:

list_ = Pipe(list)
sum_ = Pipe(sum)

def unzip(xy: list[tuple["X", "Y"]]) -> tuple[list["X"], list["Y"]]: # pyright: ignore[reportUndefinedVariable]
    x = [el for el, _ in xy]
    y = [el for _, el in xy]
    return x, y

unzip_ = Pipe(unzip)


def split_at(l: list["T"], index: int) -> tuple[list["T"], list["T"]]: # pyright: ignore[reportUndefinedVariable]
    return l[:index], l[index:]

split_at_ = Pipe(split_at)


def split_at_percentage(l: list["T"], p: float) -> tuple[list["T"], list["T"]]: # pyright: ignore[reportUndefinedVariable]
    return l | split_at_(round(len(l) * p))

split_at_percentage_ = Pipe(split_at_percentage)


def index_of_max(l: list["T"]) -> int: # pyright: ignore[reportUndefinedVariable]
    return max(enumerate(l), key=lambda x: x[1])[0]

index_of_max_ = Pipe(index_of_max)


def one_hot(n: int, n_max: int = 10) -> np.ndarray:
    '''
    if `n_max` == 2:
      0 -> [0,0,1],
      1 -> [0,1,0],
      2 -> [1,0,0].
    '''
    oh = np.zeros(n_max, dtype=np.float64)
    oh[n] = 1.
    return oh

one_hot_ = Pipe(one_hot)


def join(l: list[str], s: str = '\n') -> str:
    return s.join(l)

join_ = Pipe(join)



if __name__ == "__main__":
    main()

