import numpy as np


def square_random_lattice(n: int, m: int, states: [int] = None) -> np.array:
    """Generates a square 'n' by 'm' lattice,
    each cell is in one of the possible 'states' with equal probability."""

    if states is None:
        states = [-1, 1]

    return np.random.choice(states, size=(n, m))


def triangular_random_lattice(n: int, m: int, states: [int] = None) -> np.array:
    """Generates a triangular 'n' by 'm' lattice,
    each cell is in one of the possible 'states' with equal probability."""

    if states is None:
        states = [-1, 1]

    # NEED TO DETERMINE NOTATIONAL CONVENTION FOR TRIANGULAR
    # (AND HEXAGONAL) LATTICES
