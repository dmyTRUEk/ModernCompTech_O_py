# Neural Network that "sees" numbers.

from random import shuffle, uniform as random_float
from math import exp

import numpy as np

from pipe import Pipe, map as map_


EPOCHS: int = 1_000

LEARNING_RATE: float = 0.01


def main():
    x_train, y_train, x_test, y_test = load_datasets(0.7, ["./Data_train.csv", "./Data_test.csv"])
    # x_train, y_train = load_dataset("./Data_train.csv")
    # x_test, y_test = load_dataset("./Data_test.csv")

    nn = NeuralNetwork(64, 10)
    print("Training Neural Network...")
    nn.train(x_train, y_train)
    print("Training finished.\n")

    print(nn, "\n")

    print("NeuralNetwork test:")
    confusion_matrix = nn.test(x_test, y_test)
    print("Confusion matrix:")
    print(confusion_matrix)
    precision = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
    print(f"Neural Network Precision: {precision}")
    print()

    # return
    while (inp := input("Input input to try Neural Network: ")) != "":
        try:
            inp = inp.split(',') | map_(float) | list_
            x = np.array(inp)
            y_raw = nn.process_input_raw(x)
            y = nn.process_raw_output(y_raw)
            print(f"result: {y_raw} => {y}")
        except Exception as e:
            print(f"Error: {e}")



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
                xi = x[train_i]
                y_expected = y[train_i] | one_hot_at_(neuron_number)
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

    def train(self, x: list[np.ndarray], y: list[int]):
        assert len(x) == len(y)
        for neuron_number, neuron in enumerate(self.neurons):
            neuron.train(x, y, neuron_number)

    def test(self, x: list[np.ndarray], y: list[int]) -> np.ndarray:
        assert len(x) == len(y)
        confusion_matrix = np.zeros((10, 10), dtype=np.int32)
        for i in range(len(x)):
            y_prediction = self.process_input(x[i])
            y_expected = y[i]
            confusion_matrix[y_expected, y_prediction] += 1
        return confusion_matrix


def load_dataset(filename: str) -> tuple[list[np.ndarray], list[int]]:
    x: list[np.ndarray] = []
    y: list[int] = []
    with open(filename) as file:
        for line in file.readlines():
            yi, xi = line.strip().split(',') | split_at_(1)
            y.append(int(yi[0]))
            x.append(np.array(xi | map_(float) | list_))
    return x, y


def load_datasets(split_k: float, filenames: list[str]) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    xy_all: list[tuple[np.ndarray, int]] = []
    for filename in filenames:
        x, y = load_dataset(filename)
        for xi, yi in zip(x, y):
            xy_all.append((xi, yi))
    shuffle(xy_all)
    xy_train, xy_test = xy_all | split_at_percentage_(split_k)
    x_train, y_train = xy_train | unzip_
    x_test, y_test = xy_test | unzip_
    return x_train, y_train, x_test, y_test



# pipes:

list_ = Pipe(list)
sum_ = Pipe(sum)
# zip_ = Pipe(zip)

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


ONE_HOT_DELTA = 0.01

def one_hot(n: int, n_max: int = 10) -> np.ndarray:
    '''
    if `n_max` == 2:
      0 -> [1,0,0],
      1 -> [0,1,0],
      2 -> [0,0,1].
      but 0->0.01, 1->0.99
    '''
    # This code have to be consistent with `one_hot_at` function.
    # If code here is changed, `one_hot_at` should also be changed to produce same output.
    assert 0 <= n < n_max
    oh = np.full(n_max, ONE_HOT_DELTA, dtype=np.float64)
    oh[n] = 1. - ONE_HOT_DELTA
    return oh

one_hot_ = Pipe(one_hot)


def one_hot_at(n: int, index: int, n_max: int = 10) -> float:
    # Naive, unoptimized, but consistent to `one_hot` function solution:
    # return one_hot(n, n_max)[index]
    # Must faster solution:
    # This code have to be consistent with `one_hot` function.
    assert 0 <= n < n_max
    assert 0 <= index < n_max
    if n == index:
        return 1. - ONE_HOT_DELTA
    else:
        return ONE_HOT_DELTA

one_hot_at_ = Pipe(one_hot_at)


def join(l: list[str], s: str = '\n') -> str:
    return s.join(l)

join_ = Pipe(join)



if __name__ == "__main__":
    main()

