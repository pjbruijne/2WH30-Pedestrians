import numpy as np


AdjustedMagneticField = float
LineLattice = np.ndarray
Lattices = np.ndarray
WhiteSquares = str
BlackSquares = str
SquareColor = {WhiteSquares, BlackSquares}




def neighbour_sum(lattice, pos):
    l, = lattice.shape
    return lattice[(pos-1) % l] + lattice[(pos+1) % l]


def simulate(lattice: LineLattice, h_J: float, T: float, steps: int):
    """Evolve the line lattice a given number of steps
    using the 1D variant of the line lattice."""


    (l,) = lattice.shape
    lattices = np.empty((steps + 1, l))
    lattices[0] = lattice.copy()

    for i in range(1, steps + 1):
        # update the white squares
        
        flip_position = np.random.random_integers(0, l-1)

        spin = lattice[flip_position]

        dE = 2 * spin * (neighbour_sum(lattice, flip_position) + h_J)

        if np.random.random(1) < np.exp(-dE / T):
            lattice[flip_position] = -lattice[flip_position]

        lattices[i] = lattice.copy()

    return lattices