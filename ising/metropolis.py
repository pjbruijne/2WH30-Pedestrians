import numpy as np
from numba import njit

SquareLattice = np.ndarray[int, int]


@njit
def square_neighbour_sum(lattice: np.array, x: int, y: int) -> int:
    """Calculates the sum of the neighbours of the spin at position (x, y)
    given a square lattice."""

    n, m = lattice.shape
    neighbour_sum = 0

    if x > 0:  # a left beighbour exists
        neighbour_sum += lattice[x - 1, y]
    if x < n - 1:  # a right beighbour exists
        neighbour_sum += lattice[x + 1, y]
    if y > 0:  # an upper beighbour exists
        neighbour_sum += lattice[x, y - 1]
    if y < m - 1:  # a lower beighbour exists
        neighbour_sum += lattice[x, y + 1]

    return neighbour_sum


@njit
def simulate(
    lattice: np.ndarray[int, int], J: float, h: float, T: float, steps: int
) -> np.ndarray[SquareLattice]:
    """Simulate ising model using the Metropolis algorithm.
    J : global spin-spin interaction constant,
    h : external magnetic field"""

    n, m = lattice.shape
    lattices = np.empty((steps + 1, n, m))
    lattices[0] = lattice.copy()

    for i in range(1, steps + 1):
        x = np.random.randint(n)
        y = np.random.randint(m)

        spin = lattice[x, y]

        neighbour_sum = square_neighbour_sum(lattice, x, y)

        dE = 2 * spin * (J * neighbour_sum + h)

        if (np.random.random(1) < np.exp(-dE / T)):
            lattice[x, y] = -lattice[x, y]

        lattices[i] = lattice.copy()

    return lattices
