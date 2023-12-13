import numpy as np


def random_lattice(l, states=None):
    if states is None:
        states = [-1, 1]

    return np.random.choice(states, size=(l))
