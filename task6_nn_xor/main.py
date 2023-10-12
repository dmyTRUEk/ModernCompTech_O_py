# neural perceptron that calculates XOR

from random import uniform as random_float
from math import exp

import numpy as np


EPOCHS: int = 1_000

LEARNING_RATE: float = 0.01


def main():
    X: list[np.ndarray] = list(map(np.array, [ (0,0), (0,1), (1,0), (1,1) ])) # pyright: ignore
    Y: list[int] = [0, 1, 1, 0]
    nn = NeuralNetwork(2, [2, 1])
    print("Training Neural Network...", end=" ", flush=True)
    train(nn, X, Y)
    print("Finished.\n")

    print(nn, "\n")

    print("Neural Network Results on dataset:")
    for x in X:
        y = nn.process_input(x)
        print(f"{x} -> {y}")
    print()

    # while (inp := input("Input input to try Neural Network: ")) != "":
    #     try:
    #         inp = list(map(float, inp.split()))
    #         x = np.array(inp)
    #         y = nn.process_input(x, dbg=True)
    #         print(f"result: {y}")
    #     except Exception as e:
    #         print(f"Error: {e}")


def train(nn: "NeuralNetwork", x: list[np.ndarray], y: list[int]):
    assert len(x) == len(y)
    for epoch in range(EPOCHS):
        for trainset_i in range(len(x)):
            y_expected = y[trainset_i]
            y_actual = nn.process_input(x[trainset_i])
            y_error = y_expected - y_actual
            # for j in range(len(nn.neurons[-1])):
            #     nn.neurons[-1][j].weights += LEARNING_RATE * y_error[j] * xi
            #     nn.neurons[-1][j].shift   += LEARNING_RATE * y_error[j]
            err = y_error
            # if epoch % 100 == 0: print(f"{y_error = }")
            for layer in list(range(0, len(nn.neurons)))[::-1]:
                # print(3*"\n")
                # print(nn)
                # print(f"{layer = }")
                # print(f"{len(self.neurons[0]) = }")
                # print(f"{len(self.neurons[layer]) = }")
                wh = 1 + (len(nn.neurons[layer-1]) if layer > 0 else nn.input_size)
                ww = len(nn.neurons[layer])
                # print(f"{w_matrix_w = }, {w_matrix_h = }")
                w = np.zeros((ww, wh))
                # print(f"w_matrix{w_matrix.shape}:\n", w_matrix)
                for i in range(len(nn.neurons[layer])):
                    neuron = nn.neurons[layer][i]
                    w[i] = np.append(neuron.weights, neuron.shift)
                # w_matrix[-1] = [self.neurons[layer+1][j].shift for j in range(len(self.neurons[layer+1]))]
                # print(f"w_matrix{w.shape}:\n", w)
                # print(f"err{err.shape}:\n", err)
                err_new = w.T @ err
                # err_new = err @ w
                # print(f"err_new{err_new.shape}:\n", err_new)
                # assert len(err) > 1
                # if epoch % 100 == 0: print(f"{err = }")
                for i in range(len(nn.neurons[layer])):
                    # print(f"{nn.neurons[layer][i].weights = }")
                    # print(f"{err_new = }")
                    nn.neurons[layer][i].weights -= LEARNING_RATE * err[i] * err_new[:-1]
                    nn.neurons[layer][i].shift   -= LEARNING_RATE * err[i] * err_new[-1]
                    # self.neurons[layer][j].shift = 0
                err = err_new[1:]
            # raise NotImplemented()



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


    def process_input(self, input: np.ndarray) -> float:
        return activation_function(float(input @ self.weights + self.shift))



class NeuralNetwork:
    input_size: int
    neurons: list[list[Neuron]]
    values: list[np.ndarray]

    def __init__(self, input_size: int, layers_sizes: list[int]) -> None:
        self.input_size = input_size
        self.neurons = [[Neuron(input_size) for _ in range(layers_sizes[0])]]
        for i in range(1, len(layers_sizes)):
            self.neurons.append([Neuron(layers_sizes[i-1]) for _ in range(layers_sizes[i])])

    def __repr__(self) -> str:
        s: str = f"Neural Network Neurons (input_size = {self.input_size}):\n"
        for layer in range(len(self.neurons)):
            for i in range(len(self.neurons[layer])):
                s += f"- {layer=}, {i=}:\n"
                neuron = self.neurons[layer][i]
                s += f"  - weights: {neuron.weights}\n"
                s += f"  - shift: {neuron.shift}\n"
        s = s[:-1]
        return s

    def process_input(self, input: np.ndarray, *, dbg: bool=False) -> np.ndarray:
        self.values = []
        for layer in range(len(self.neurons)):
            output = np.zeros(len(self.neurons[layer]))
            for i in range(len(self.neurons[layer])):
                output[i] = self.neurons[layer][i].process_input(input)
            self.values.append(output)
            input = output
        if dbg:
            print(f"Neurons values by layers:")
            for layer in range(len(self.neurons)):
                print(f"- {layer=} -> {self.values[layer]}")
        # return output
        return self.values[-1]



if __name__ == "__main__":
    main()

