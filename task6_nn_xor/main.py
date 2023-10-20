# neural perceptron that calculates XOR

from abc import abstractmethod
from collections.abc import Iterator
from math import exp

import numpy as np

from pipe import Pipe, map as map_


EPOCHS: int = 1_000

LEARNING_RATE: float = 0.01


def main():
    x_train: list[np.ndarray] = [ (0,0), (0,1), (1,0), (1,1) ] | map_(np.array) | list_
    y_train: list[int] = [0, 1, 1, 0]
    x_test = x_train
    y_test = y_train

    nn = NeuralNetwork(2, [2, 5, 1])

    print("Training Neural Network...")
    nn.train(x_train, y_train)
    print("Finished.\n")

    print(nn, "\n")

    print("Neural Network Test:")
    nn.test(x_test, y_test)
    print()

    return
    while (inp := input("Input input to try Neural Network: ")) != "":
        try:
            inp = list(map(float, inp.split()))
            x = np.array(inp)
            y = nn.process_input(x, dbg=True)
            print(f"result: {y}")
        except Exception as e:
            print(f"Error: {e}")




class NeuralNetwork:
    input_size: int
    weights: list[np.ndarray]
    values: list[np.ndarray]

    def __init__(self, input_size: int, layers_sizes: list[int]) -> None:
        self.input_size = input_size
        sizes_all = [input_size, *layers_sizes]
        self.weights = [
            np.random.uniform(-1., 1., (layer_size_next, layer_size_prev))
            for layer_size_prev, layer_size_next in sizes_all | windows_(2)
        ]
        print(f"{self.weights = }")

    def __repr__(self) -> str:
        s: str = f"Neural Network Neurons (input_size = {self.input_size}):\n"
        for layer_index in range(len(self.weights)):
            s += f"- {layer_index=}:\n"
            s += f"{self.weights[layer_index]}\n"
        return s[:-1]

    def process_input(self, input: np.ndarray) -> np.ndarray:
        # TODO(optimization): maybe remove `.copy()`?
        input = input.copy()[np.newaxis].T
        self.values = [self.weights[0] @ input]
        for layer_index in range(1, len(self.weights)):
            # print(f"{self.weights[layer_index] = }")
            # print(f"{self.values[layer_index-1] = }")
            self.values.append(
                Sigmoid.eval_np_array(
                    self.weights[layer_index]
                    @
                    self.values[layer_index-1]
                )
            )
        return self.values[-1]

    def train(self, x: list[np.ndarray], y: list[int]):
        assert len(x) == len(y)
        for epoch in range(EPOCHS):
            for trainset_i in range(len(x)):
                y_expected = y[trainset_i]
                y_actual = self.process_input(x[trainset_i])
                values = [x[trainset_i], *self.values]
                err = y_actual - y_expected
                for layer_index in list(range(1, len(self.weights)))[::-1]:
                    print(3*"\n")
                    print(self)
                    print(f"{layer_index = }")
                    # w = self.weights[layer]
                    # print(f"w_matrix{w.shape}:\n", w)
                    # print(f"{err = }")
                    # print(f"{self.values = }")
                    print(f"{err * values[layer_index] * (1. - values[layer_index]) = }")
                    print(f"{values[layer_index-1] = }")
                    self.weights[layer_index] -= LEARNING_RATE * (
                        (err * values[layer_index] * (1. - values[layer_index]))
                        @
                        (values[layer_index-1].T)
                    )
                    err = self.weights[layer_index].T @ err

    def test(self, x: list[np.ndarray], y: list[int]):
        assert len(x) == len(y)
        for xi in x:
            yi = self.process_input(xi)
            print(f"{xi} -> {yi}")



class ActivationFunction:
    @classmethod
    @abstractmethod
    def eval_float(cls, x: float) -> float:
        ...
    @classmethod
    @abstractmethod
    def eval_np_array(cls, x: np.ndarray) -> np.ndarray:
        ...
    @classmethod
    @abstractmethod
    def eval_derivative_float(cls, x: float) -> float:
        ...
    @classmethod
    @abstractmethod
    def eval_derivative_np_array(cls, x: np.ndarray) -> np.ndarray:
        ...


class Sigmoid(ActivationFunction):
    @classmethod
    def eval_float(cls, x: float) -> float:
        return 1. / (1. + exp(-x))
    @classmethod
    def eval_np_array(cls, x: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-x))
    @classmethod
    def eval_derivative_float(cls, x: float) -> float:
        s = Sigmoid.eval_float(x)
        return s * (1. - s)
    @classmethod
    def eval_derivative_np_array(cls, x: np.ndarray) -> np.ndarray:
        s = Sigmoid.eval_np_array(x)
        return s * (1. - s)



# Pipes:

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
      0 -> [1,0,0],
      1 -> [0,1,0],
      2 -> [0,0,1].
    '''
    oh = np.zeros(n_max, dtype=np.float64)
    oh[n] = 1.
    return oh

one_hot_ = Pipe(one_hot)


def one_hot_at(n: int, index: int, n_max: int = 10) -> float:
    # return one_hot(n, n_max)[index]
    assert n <= n_max
    assert index <= n_max
    if n == index:
        return 1.
    else:
        return 0.

one_hot_at_ = Pipe(one_hot_at)


def join(l: list[str], s: str = '\n') -> str:
    return s.join(l)

join_ = Pipe(join)


def windows(it: Iterator["T"] | list["T"], window_size: int) -> Iterator[tuple["T", ...]] | list[tuple["T", ...]]: # pyright: ignore[reportUndefinedVariable]
    # simple impl for lists:
    # assert window_size <= len(l)
    # res = []
    # for i in range(len(l) - window_size + 1):
    #     this_window = tuple(l[i+j] for j in range(window_size))
    #     res.append(this_window)
    # return res
    # better impl for iterable:
    it = iter(it)
    res = tuple(next(it) for _ in range(window_size))
    yield res
    for el in it:
        res = res[1:] + (el,)
        yield res

windows_ = Pipe(windows)



if __name__ == "__main__":
    main()

