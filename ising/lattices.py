import numpy as np


def square_random_lattice(width: int, height: int, states: [int] = None) -> np.array:
    """Generates a square 'width' by 'heigth' lattice,
    each cell is in one of the possible 'states' with equal probability."""

    if states is None:
        states = [-1, 1]

    return np.random.choice(states, size=(width, height))


def triangular_random_lattice(
    width: int, height: int, states: [int] = None
) -> np.array:
    """Generates a triangular 'width' by 'heigth' lattice,
    each cell is in one of the possible 'states' with equal probability."""

    if states is None:
        states = [-1, 1]

    # NEEDS TO DETERMINE NOTATIONAL CONVENTION FOR TRIANGULAR
    # (AND HEXAGONAL) LATTICES
