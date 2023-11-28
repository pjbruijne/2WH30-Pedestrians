import numpy as np
from numba import njit


@njit()
def square_neighbour_sum(lattice: np.array, x: int, y: int) -> int:
    """Calculates the sum of the neighbours of the spin at position (x, y)
    given a square lattice."""

    width, height = lattice.shape
    neighbour_sum = 0

    if x > 0:  # a left beighbour exists
        neighbour_sum += lattice[x - 1, y]
    if x < width - 1:  # a right beighbour exists
        neighbour_sum += lattice[x + 1, y]
    if y > 0:  # an upper beighbour exists
        neighbour_sum += lattice[x, y - 1]
    if y < height - 1:  # a lower beighbour exists
        neighbour_sum += lattice[x, y + 1]

    return neighbour_sum


@njit()
def simulate(lattice: np.ndarray[int, int], J: float, B: float, T: float, steps: int):
    """Simulate ising model using the Metropolis algorithm.
    J : global spin-spin interaction constant,
    B : external magnetic field"""

    images = [lattice.copy()]
    width, height = lattice.shape

    for _ in range(steps):
        x = np.random.randint(width)
        y = np.random.randint(height)

        spin = lattice[x, y]

        neighbour_sum = square_neighbour_sum(lattice, x, y)

        dE = 2 * spin * (J * neighbour_sum - B)

        if (neighbour_sum * spin < 0) or (np.random.random(1) < np.exp(-dE / T)):
            lattice[x, y] = -lattice[x, y]

        images.append(lattice.copy())

    return images
