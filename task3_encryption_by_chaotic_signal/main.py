# encrypt data by chaotic signal (lorenz attractor)

from math import pi, sin
from random import uniform as random_float

import numpy as np


# info signal params:
POINTS_PER_BIT: int = 100
PERIODS_PER_BIT: float = 2
# POINTS_PER_PERIOD: int = 
AMPLITUDE_OF_0: float = 10
AMPLITUDE_OF_1: float = 20

# chaotic signal params:
AMPLITUDE_OF_CHAOTIC_SIGNAL: float = 15

# noise params:
AMPLITUDE_OF_PURE_NOISE: float = 15
NUMBER_OF_NOISE_SINES: int = 10
AMPLITUDE_OF_SINES_NOISE_MIN: float = 3
AMPLITUDE_OF_SINES_NOISE_MAX: float = 30
FREQUENCY_OF_SINES_NOISE_MIN: float = 3
FREQUENCY_OF_SINES_NOISE_MAX: float = 30


def main():
    # info = [1,0,1]
    info = [1,0,1,0,1,0]

    info_encoded = np.zeros(POINTS_PER_BIT * len(info))
    for bit_index in range(len(info)):
        bit = info[bit_index]
        for i in range(POINTS_PER_BIT):
            index = bit_index * POINTS_PER_BIT + i
            x = (i / POINTS_PER_BIT) * 2 * pi * PERIODS_PER_BIT
            amplitude = AMPLITUDE_OF_0 if bit == 0 else AMPLITUDE_OF_1
            info_encoded[index] = amplitude * sin(x)
    # print(info_encoded)
    # print(len(info_encoded))

    BETA: float = 8 / 3
    SIGMA: float = 10
    RHO: float = 27
    DELTA_TIME: float = 0.01
    X_MAX, Y_MAX, Z_MAX = 20.836354501114837, 28.273591057940802, 51.84361557477107
    X0, Y0, Z0 = 0.1, 0.2, 0.3
    x, y, z = X0, Y0, Z0
    for i in range(len(info_encoded)):
        dx = SIGMA * (y - x)
        dy = x * (RHO - z) - y
        dz = x * y - BETA * z
        x += dx * DELTA_TIME
        y += dy * DELTA_TIME
        z += dz * DELTA_TIME
        # print(x, y, z)
        xn = x / X_MAX
        yn = y / Y_MAX
        zn = z / Z_MAX
        info_encoded[i] += (xn + yn + zn)/3 * AMPLITUDE_OF_CHAOTIC_SIGNAL
    # print(info_encoded)

    # info_encoded = np.zeros(len(info_encoded))
    noise_sines_amplitudes = []
    noise_sines_frequencies = []
    noise_sines_shifts = []
    for i in range(NUMBER_OF_NOISE_SINES):
        noise_sines_amplitudes.append(random_float(AMPLITUDE_OF_SINES_NOISE_MIN, AMPLITUDE_OF_SINES_NOISE_MAX))
        noise_sines_frequencies.append(random_float(FREQUENCY_OF_SINES_NOISE_MIN, FREQUENCY_OF_SINES_NOISE_MAX))
        noise_sines_shifts.append(random_float(0, 2*pi))
    assert NUMBER_OF_NOISE_SINES == len(noise_sines_amplitudes)
    assert NUMBER_OF_NOISE_SINES == len(noise_sines_frequencies)
    for i in range(len(info_encoded)):
        x = i / len(info_encoded) * len(info) * PERIODS_PER_BIT
        for j in range(NUMBER_OF_NOISE_SINES):
            info_encoded[i] += noise_sines_amplitudes[j] * sin(noise_sines_frequencies[j] * x + noise_sines_shifts[j]) / NUMBER_OF_NOISE_SINES
    # print(info_encoded)

    for i in range(len(info_encoded)):
        info_encoded[i] += random_float(-AMPLITUDE_OF_PURE_NOISE, AMPLITUDE_OF_PURE_NOISE)
    # print(info_encoded)

    x, y, z = X0, Y0, Z0
    for i in range(len(info_encoded)):
        dx = SIGMA * (y - x)
        dy = x * (RHO - z) - y
        dz = x * y - BETA * z
        x += dx * DELTA_TIME
        y += dy * DELTA_TIME
        z += dz * DELTA_TIME
        # print(x, y, z)
        xn = x / X_MAX
        yn = y / Y_MAX
        zn = z / Z_MAX
        info_encoded[i] += (xn + yn + zn)/3 * AMPLITUDE_OF_CHAOTIC_SIGNAL
    print(info_encoded)



if __name__ == "__main__":
    main()

