# neural perceptron that predicts lyashch/shchuka width based on it's height

from random import uniform as random_float


EPOCHS: int = 1000

LEARNING_RATE: float = 0.001


def main():
    X: list[float] = [7, 8, 12, 15  , 16  , 18, 19, 20, 22  ]
    Y: list[float] = [6, 7, 10, 13.2, 16.8, 21, 23, 27, 31.5]
    neuron = Neuron()
    print("Training neuron...", end="")
    neuron.train(X, Y)
    print(" Finished.")

    while (inp := input("Input input to try neuron: ")) != "":
        x = float(inp)
        y = neuron.process_input(x)
        print(f"result: {y}")


class Neuron:
    weight: float
    shift: float

    def __init__(self) -> None:
        self.weight = random_float(-1, 1)
        self.shift = random_float(-1, 1)

    def process_input(self, input: float) -> float:
        return input * self.weight + self.shift

    def train(self, x: list[float], y: list[float]):
        assert len(x) == len(y)
        for epoch in range(EPOCHS):
            for i in range(len(x)):
                xi = x[i]
                y_expected = y[i]
                y_actual = self.process_input(xi)
                y_error = y_expected - y_actual
                self.weight += LEARNING_RATE * y_error * xi
                self.shift += LEARNING_RATE * y_error



if __name__ == "__main__":
    main()

