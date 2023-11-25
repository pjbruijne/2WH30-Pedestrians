import numpy as np


def random_lattice(width: int, height: int, states: [int] = None) -> np.array:
    """Generates a 'width' by 'heigth' lattice,
    each cell is in one of the possible 'states' with equal probability."""

    if states is None:
        states = [-1, 1]

    return np.random.choice(states, size=(width, height))
